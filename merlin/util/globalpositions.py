"""
Camera/stage global-position correction for a MERFISH FOV grid.

Ported from the sibling `MERci` project's `acquisition.camera_rotation`
module (`251225_LT027_saving_time/MERci`), then redesigned in the
`260926_LT074_stitching_test` investigation. A neighbouring FOV's nominal
(stage-reported) position disagrees with its true relative position by a
few microns, from two sources:

- one linear map shared by the whole grid (camera-vs-stage rotation,
  image-vs-stage scale), and
- a direction-dependent offset per step (stage backlash: e.g. y differs
  by 0.72 um between the two scan directions).

The method, in four parts:

1. register every 4-connected neighbour pair on its overlap band
   (:func:`register_neighbor_pair`);
2. reject outliers per direction, on the residual measured - nominal
   (:func:`filter_correspondence_outliers`);
3. fit one 2x2 affine ``A`` on the edge DISPLACEMENTS, ``m = A d``
   (:func:`fit_displacement_affine`). Fitting an affine on absolute
   (nominal, measured) positions instead returns about the identity;
4. solve every FOV's position by least squares on all kept edges, with a
   weak ``A d`` prior on every grid edge (:func:`fit_global_positions`).

On real data this leaves held-out edges 0.02-0.13 um off (nominal: ~3.3
um). Global coordinates stay in the camera frame with translation-only
per-FOV offsets; the stage grid appears rotated and scaled in that frame.

Not ported: MERci's orientation-detection helpers (MERlin already applies
`transpose`/`flip_horizontal`/`flip_vertical` at image-load time via
`ImageDataSet.load_image`, so raw images handed to this module are
already correctly oriented).
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import sparse
from scipy.ndimage import median_filter
from scipy.ndimage import shift as ndi_shift
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from scipy.spatial import KDTree
from skimage import registration

# (direction label, dx, dy) -- a fov's 4-connected neighbours in nominal
# grid-step units. Labels are arbitrary grouping tags for the per-direction
# reliability accounting below; they carry no assumption about physical
# up/down/left/right, since that mapping is microscope/mounting-specific.
_DIRECTIONS: Tuple[Tuple[str, float, float], ...] = (
    ('+x', 1.0, 0.0), ('-x', -1.0, 0.0), ('+y', 0.0, 1.0), ('-y', 0.0, -1.0),
)


@dataclass
class NeighborCorrespondence:
    """One anchor-neighbour pair's nominal vs. measured position.

    Attributes
    ----------
    anchor_fov, neighbor_fov : fov ids
    direction    : one of ``"+x"``/``"-x"``/``"+y"``/``"-y"`` (anchor -> neighbour)
    nominal_xy   : the neighbour's recorded grid position (microns)
    measured_xy  : the neighbour's true position, i.e. the anchor's own
                   (assumed-correct) recorded position plus the real
                   relative shift measured from image registration (microns)
    error        : phase_cross_correlation's own registration error for
                   this pair (lower = more confident)
    """
    anchor_fov:   int
    neighbor_fov: int
    direction:    str
    nominal_xy:   Tuple[float, float]
    measured_xy:  Tuple[float, float]
    error:        float


def estimate_step_size_um(positions: Dict[int, Tuple[float, float]]) -> float:
    """Median nearest-neighbour distance across every fov's nominal
    position -- the grid's real step size, measured directly rather than
    assumed, so this works regardless of whether the nominal spacing is
    documented anywhere else in the dataset.
    """
    if len(positions) < 2:
        return 0.0
    coords = np.array(list(positions.values()), dtype=float)
    distances, _ = KDTree(coords).query(coords, k=2)
    return float(np.median(distances[:, 1]))


def find_grid_neighbor(
    anchor_fov:         int,
    positions:          Dict[int, Tuple[float, float]],
    dx:                 float,
    dy:                 float,
    tolerance_fraction: float = 0.25,
) -> Optional[int]:
    """Find the fov (if any) sitting in the ``(dx, dy)`` direction from
    *anchor_fov*'s own nominal position -- the closest other fov whose
    displacement from the anchor is dominated by that axis and has the
    matching sign, rather than the fov closest to a target point at exactly
    ``anchor + (dx, dy) * (one dataset-wide step size)``.

    This matters on a non-rectangular grid where different regions have
    their own true step or phase along one axis (e.g. independently
    phase-shifted scan bands): a cross-region neighbour's real offset from
    the anchor isn't ``(dx, dy) * (one dataset-wide step)``, so matching
    against that exact target point systematically misses/mistolerances
    exactly those neighbours (confirmed on real data by the sibling MERci
    project -- see this module's own docstring).

    ``tolerance_fraction`` still guards against picking a real but distant
    fov when the anchor has no true neighbour on this side (e.g. it sits on
    the grid's exterior): the best directional candidate is only accepted
    if its distance from the anchor is within ``tolerance_fraction`` of the
    anchor's own local step size, i.e. its distance to its single nearest
    neighbour in ANY direction (not a dataset-wide step size).

    Returns ``None`` if no fov qualifies.
    """
    candidateIds = [f for f in positions if f != anchor_fov]
    if not candidateIds:
        return None

    anchorXY = np.array(positions[anchor_fov], dtype=float)
    displacements = np.array([positions[f] for f in candidateIds], dtype=float) - anchorXY
    distances = np.hypot(displacements[:, 0], displacements[:, 1])
    localStepUm = float(np.min(distances))
    if localStepUm <= 0:
        return None

    if dx != 0:
        matchesDirection = (
            (np.abs(displacements[:, 0]) >= np.abs(displacements[:, 1]))
            & (np.sign(displacements[:, 0]) == np.sign(dx)))
    else:
        matchesDirection = (
            (np.abs(displacements[:, 1]) > np.abs(displacements[:, 0]))
            & (np.sign(displacements[:, 1]) == np.sign(dy)))

    eligible = np.where(matchesDirection)[0]
    if len(eligible) == 0:
        return None
    best = eligible[np.argmin(distances[eligible])]
    if distances[best] <= (1.0 + tolerance_fraction) * localStepUm:
        return candidateIds[best]
    return None


def crop_overlap(
    anchor_img:       np.ndarray,
    neighbor_img:     np.ndarray,
    dx:               float,
    dy:               float,
    overlap_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop the expected overlapping strip from a pair of 4-connected-neighbour
    frames, ready for `skimage.registration.phase_cross_correlation`.

    *(dx, dy)* is the anchor -> neighbour direction (e.g. ``dx=1`` means the
    neighbour sits on the anchor's +x side, so the anchor's own +x edge
    should match the neighbour's -x edge). *overlap_fraction* is the
    expected overlap as a fraction of the frame's full width/height.

    Row-axis convention for the y direction: `SimpleGlobalAlignment.
    fov_coordinates_to_global` (`merlin/analysis/globalalign.py`) already
    adds a fov-local pixel coordinate directly to the fov's global (x, y)
    offset with no sign flip (``fovStart[1] + fovCoordinates[1] *
    micronsPerPixel``) -- i.e. MERlin's own established convention is that
    increasing row index maps directly (not inverted) to increasing global
    y. A neighbour on the +y side therefore touches the anchor's
    LARGEST-row-index edge. Getting this backwards would silently corrupt
    every y-direction registration (the exact class of bug the sibling
    MERci project's own camera_rotation.py documents having hit once for
    real) -- this mapping is derived from that already-adopted convention,
    not assumed independently.
    """
    h, w = anchor_img.shape
    if dx != 0:
        n = _overlap_px(w, overlap_fraction)
        if dx > 0:
            return anchor_img[:, w - n:], neighbor_img[:, :n]
        return anchor_img[:, :n], neighbor_img[:, w - n:]
    else:
        n = _overlap_px(h, overlap_fraction)
        if dy > 0:
            return anchor_img[h - n:, :], neighbor_img[:n, :]
        return anchor_img[:n, :], neighbor_img[h - n:, :]


def _overlap_px(length: int, overlap_fraction: float) -> int:
    return max(1, int(round(length * overlap_fraction)))


def crop_offset_um(
    frame_shape:      Tuple[int, int],
    dx:               float,
    dy:               float,
    overlap_fraction: float,
    pixel_size_um:    float,
) -> Tuple[float, float]:
    """
    The anchor -> neighbour offset (microns) at which the two strips from
    `crop_overlap` show the same area: one frame minus the overlap strip
    along the *(dx, dy)* axis, and zero across it.

    A registration shift measured on those strips is relative to this
    offset, not to the pair's nominal offset. The two differ whenever a
    pair's nominal spacing is not exactly the assumed step (e.g. a
    non-rectangular grid with offset scan bands).
    """
    h, w = frame_shape
    if dx != 0:
        return (float(np.sign(dx)) * (w - _overlap_px(w, overlap_fraction))
                * pixel_size_um, 0.0)
    return (0.0, float(np.sign(dy)) * (h - _overlap_px(h, overlap_fraction))
            * pixel_size_um)


def _within_registration_range(
    frame_shape:      Tuple[int, int],
    anchor_xy:        Tuple[float, float],
    neighbor_xy:      Tuple[float, float],
    dx:               float,
    dy:               float,
    overlap_fraction: float,
    pixel_size_um:    float,
) -> bool:
    """
    Whether the pair's nominal offset is close enough to the offset the
    crops assume (see `crop_offset_um`) for phase correlation to measure
    the difference: it can only report shifts up to half the crop size on
    each axis, so a pair further off than that (e.g. two fovs that do not
    overlap at all) would give a meaningless registration.
    """
    h, w = frame_shape
    if dx != 0:
        cropRows, cropCols = h, _overlap_px(w, overlap_fraction)
    else:
        cropRows, cropCols = _overlap_px(h, overlap_fraction), w
    cropDx, cropDy = crop_offset_um(
        frame_shape, dx, dy, overlap_fraction, pixel_size_um)
    diffXPx = abs(neighbor_xy[0] - anchor_xy[0] - cropDx) / pixel_size_um
    diffYPx = abs(neighbor_xy[1] - anchor_xy[1] - cropDy) / pixel_size_um
    return diffXPx < cropCols / 2 and diffYPx < cropRows / 2


def remove_hot_pixels(
    img:         np.ndarray,
    size:        int   = 3,
    ratio:       float = 5.0,
    sigma_floor: float = 5.0,
) -> np.ndarray:
    """
    Replace isolated hot/dead camera pixels with their local median before
    registration. Ported from MERci's `acquisition.alignment.remove_hot_pixels`.

    Hot pixels sit at a fixed detector position in every frame, so they
    dominate phase correlation and pin the shift to zero when the real
    signal (e.g. dim beads) is weak. A hot pixel is a single-pixel spike
    far above its local median; a bead spans several pixels (the PSF), so
    it stays close to its local median.

    Parameters
    ----------
    img         : 2-D registration image
    size        : local-median window (pixels); 3 isolates single-pixel spikes
    ratio       : flag pixels above ``ratio`` x local median (hot pixels
                  were observed at ~10-600x, bead cores at ~1-2x)
    sigma_floor : also require the excess over the local median to exceed
                  this many background-noise sigmas, so noise on near-zero
                  background is not flagged

    Returns
    -------
    Copy of *img* with hot pixels replaced by their local median (*img*
    itself, uncopied, if none are flagged).
    """
    if size in (3, 5) and img.dtype in (np.uint8, np.uint16, np.float32):
        # Same result as median_filter(mode='nearest'), ~50x faster (0.02 vs
        # 1.1 s on a 2048x2048 frame).
        localMedian = cv2.medianBlur(img, size).astype(np.float64)
    else:
        localMedian = median_filter(img, size=size, mode='nearest').astype(np.float64)
    excess = img.astype(np.float64) - localMedian
    # 1.4826 converts a median absolute deviation to a Gaussian sigma.
    noiseSigma = 1.4826 * float(np.median(np.abs(excess - np.median(excess))))
    hot = (img > ratio * np.maximum(localMedian, 1.0)) & (excess > sigma_floor * noiseSigma)
    if hot.any():
        img = img.copy()
        img[hot] = localMedian[hot].astype(img.dtype)
    return img


def _hann_windowed(crop: np.ndarray) -> np.ndarray:
    """Mean-subtracted *crop* times a 2-D Hann window. Without it, the
    band's non-periodic edges add a zero-shift peak to the phase
    correlation, which wins when the shared structure is weak."""
    crop = crop.astype(np.float64) - crop.mean()
    return crop * np.outer(np.hanning(crop.shape[0]), np.hanning(crop.shape[1]))


def register_neighbor_pair(
    anchor_img:       np.ndarray,
    neighbor_img:     np.ndarray,
    anchor_xy:        Tuple[float, float],
    dx:               float,
    dy:               float,
    overlap_fraction: float,
    pixel_size_um:    float,
    upsample_factor:  int = 100,
    hann_window:      bool = False,
) -> Tuple[Tuple[float, float], float]:
    """
    Measure the neighbour's TRUE position relative to the anchor, from the
    real pixel shift needed to align their overlapping border crop.

    *hann_window* multiplies each crop by a 2-D Hann window before
    correlating (see `_hann_windowed`).

    Returns
    -------
    (measured_neighbor_xy, error) -- the neighbour's measured true (x, y)
    position (microns), and the registration's error metric.
    """
    a_crop, n_crop = crop_overlap(anchor_img, neighbor_img, dx, dy, overlap_fraction)
    if hann_window:
        a_crop, n_crop = _hann_windowed(a_crop), _hann_windowed(n_crop)
    shift, error, _ = registration.phase_cross_correlation(
        a_crop, n_crop, upsample_factor=upsample_factor)
    dy_px, dx_px = float(shift[0]), float(shift[1])

    # The shift is relative to the offset the crops assume (see
    # crop_offset_um), not to the pair's nominal offset.
    cropDx, cropDy = crop_offset_um(
        anchor_img.shape, dx, dy, overlap_fraction, pixel_size_um)
    measDx = cropDx + dx_px * pixel_size_um
    measDy = cropDy + dy_px * pixel_size_um
    return (anchor_xy[0] + measDx, anchor_xy[1] + measDy), float(error)


class _BoundedFrameCache:
    """A small LRU cache over ``load_frame``, bounded to *maxsize* frames --
    used in place of an unbounded per-call dict so a whole-grid pass
    (`sample_neighbor_correspondences`/`compute_overlap_correlations`) can't
    grow to hold every fov's frame at once. That unbounded growth was the
    root cause of a real OOM kill on a real 1651-fov experiment (see
    FINDINGS.md in the MERlin repo, 2026-09-03): with every fov visited as
    both an anchor and a neighbour, an uncapped cache ends up holding
    nearly the whole dataset's frames simultaneously by the end of the
    loop. A neighbour fov reused across two anchors close together in
    iteration order still gets a cache hit; one far apart just gets
    reloaded -- a bounded amount of extra I/O traded for a bounded memory
    footprint, not a correctness issue either way. `merlin.analysis.
    globalalign.RegisterFovNeighbors` sidesteps this entirely by
    registering one fov's neighbours per (Slurm-parallel) fragment, where
    at most two frames (the anchor plus the current neighbour) are ever
    live at once -- this cache remains for the whole-grid entry points
    below, still used directly on small datasets and by
    `LeastSquaresGlobalAlignment`'s own final `compute_overlap_correlations`
    QC pass.
    """

    def __init__(self, load_frame: Callable[[int], np.ndarray], maxsize: int = 8):
        self._load_frame = load_frame
        self._maxsize = maxsize
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

    def get(self, fov: int) -> np.ndarray:
        if fov in self._cache:
            self._cache.move_to_end(fov)
            return self._cache[fov]
        frame = self._load_frame(fov)
        self._cache[fov] = frame
        if len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
        return frame


def register_fov_against_neighbors(
    anchor_fov:         int,
    positions:          Dict[int, Tuple[float, float]],
    load_frame:         Callable[[int], np.ndarray],
    pixel_size_um:      float,
    overlap_fraction:   float,
    tolerance_fraction: float = 0.25,
    upsample_factor:    int = 100,
    hann_window:        bool = False,
) -> List[NeighborCorrespondence]:
    """
    Register *anchor_fov* against each of its present 4-connected
    neighbours (see `sample_neighbor_correspondences` for why this is
    exhaustive, not a sparse sample, and why an interior fov's edges end
    up measured twice). Holds at most the anchor's own frame plus one
    neighbour frame at a time -- no cache needed, since within one call
    each neighbour is loaded at most once.

    This is the single-fov unit of work behind `merlin.analysis.
    globalalign.RegisterFovNeighbors`'s per-fov Slurm parallelism, and is
    also what `sample_neighbor_correspondences` below loops over for its
    whole-grid, non-parallel entry point.

    Parameters
    ----------
    anchor_fov  : the fov to use as anchor
    positions   : ``{fov_id: (x, y)}`` nominal grid positions (microns),
                  covering *anchor_fov* and its neighbours
    load_frame  : ``load_frame(fov_id) -> np.ndarray``, returning the 2-D
                  registration image for one fov
    pixel_size_um, overlap_fraction : grid/camera geometry (see
                  `find_grid_neighbor` for how each anchor's own local step
                  size is derived from *positions* directly, rather than
                  taken as a dataset-wide value)
    """
    anchorImg = load_frame(anchor_fov)
    correspondences: List[NeighborCorrespondence] = []
    for direction, dx, dy in _DIRECTIONS:
        neighborFov = find_grid_neighbor(
            anchor_fov, positions, dx, dy, tolerance_fraction)
        if neighborFov is None:
            continue
        if not _within_registration_range(
                anchorImg.shape, positions[anchor_fov], positions[neighborFov],
                dx, dy, overlap_fraction, pixel_size_um):
            continue
        neighborImg = load_frame(neighborFov)
        measuredXY, error = register_neighbor_pair(
            anchorImg, neighborImg, positions[anchor_fov],
            dx, dy, overlap_fraction, pixel_size_um, upsample_factor, hann_window)
        correspondences.append(NeighborCorrespondence(
            anchor_fov=anchor_fov, neighbor_fov=neighborFov, direction=direction,
            nominal_xy=positions[neighborFov], measured_xy=measuredXY, error=error))
    return correspondences


def sample_neighbor_correspondences(
    fov_ids:            List[int],
    positions:          Dict[int, Tuple[float, float]],
    load_frame:         Callable[[int], np.ndarray],
    pixel_size_um:      float,
    overlap_fraction:   float,
    tolerance_fraction: float = 0.25,
    upsample_factor:    int = 100,
    hann_window:        bool = False,
) -> List[NeighborCorrespondence]:
    """
    Register every fov in *fov_ids* against each of its present 4-connected
    neighbours (exhaustive, not a sparse sample -- the real-data comparison
    this module is ported from found the joint least-squares solve needs a
    dense, whole-grid correspondence set to have real redundant constraints
    per fov; see `fit_global_positions`'s own docstring). An interior fov's
    edges are measured twice, independently, once from each side -- a free
    redundancy check, not wasted work.

    Whole-grid, non-parallel entry point: delegates each anchor to
    `register_fov_against_neighbors`, sharing one `_BoundedFrameCache`
    (not an unbounded cache -- see that class's docstring) across the
    whole call. `merlin.analysis.globalalign.RegisterFovNeighbors` is the
    per-fov-parallel equivalent used in production instead of this
    function, for any dataset large enough that a single job's memory
    matters.

    Parameters
    ----------
    fov_ids     : the fovs to use as anchors (typically every fov in the
                  dataset)
    positions   : ``{fov_id: (x, y)}`` nominal grid positions (microns),
                  covering every id in *fov_ids* and its neighbours
    load_frame  : ``load_frame(fov_id) -> np.ndarray``, returning the 2-D
                  registration image for one fov
    pixel_size_um, overlap_fraction : grid/camera geometry (see
                  `find_grid_neighbor` for how each anchor's own local step
                  size is derived from *positions* directly, rather than
                  taken as a dataset-wide value)
    """
    cache = _BoundedFrameCache(load_frame)
    correspondences: List[NeighborCorrespondence] = []
    for anchorFov in fov_ids:
        correspondences.extend(register_fov_against_neighbors(
            anchorFov, positions, cache.get, pixel_size_um, overlap_fraction,
            tolerance_fraction, upsample_factor, hann_window))
    return correspondences


def filter_correspondence_outliers(
    correspondences: List[NeighborCorrespondence],
    mad_threshold:   float = 5.0,
) -> Tuple[List[NeighborCorrespondence], List[NeighborCorrespondence]]:
    """
    Split correspondences into (kept, rejected), separately per direction.

    Each correspondence's residual is ``r = measured_xy - nominal_xy`` (the
    measured minus the nominal displacement). The residual differs by
    direction (stage backlash), so pooling every direction would reject
    good edges of one direction and keep bad ones of another. Within one
    direction, a correspondence is rejected when its 2-D distance from
    that direction's median residual exceeds ``median + mad_threshold *
    1.4826 * MAD`` of those distances (1.4826 converts a median absolute
    deviation to a Gaussian sigma). This catches individual registrations
    that failed outright (weak fiducial signal, a bad phase-correlation
    peak).

    The direction is taken from the lower to the higher fov id (a
    measurement made from the higher id is flipped): with fovs numbered in
    acquisition order, that is the stage's move direction, which is what
    the backlash depends on. Both measurements of one edge therefore get
    the same decision.

    Limitation: if backlash splits one direction's residuals into two
    clusters (e.g. alternate scan columns) and the registration noise is
    far below the gap between them, a cluster holding under half of that
    direction's edges is rejected too.
    """
    flip = np.array([c.anchor_fov > c.neighbor_fov for c in correspondences], dtype=bool)
    residuals = np.array(
        [(c.measured_xy[0] - c.nominal_xy[0], c.measured_xy[1] - c.nominal_xy[1])
         for c in correspondences], dtype=float).reshape(-1, 2)
    residuals[flip] *= -1
    opposite = {'+x': '-x', '-x': '+x', '+y': '-y', '-y': '+y'}
    directions = np.array([opposite[c.direction] if f else c.direction
                           for c, f in zip(correspondences, flip)])

    keep = np.ones(len(correspondences), dtype=bool)
    for direction in np.unique(directions):
        idx = np.where(directions == direction)[0]
        deviation = np.hypot(*(residuals[idx] - np.median(residuals[idx], axis=0)).T)
        median = np.median(deviation)
        threshold = median + mad_threshold * 1.4826 * np.median(np.abs(deviation - median))
        keep[idx[deviation > threshold]] = False

    kept = [c for c, k in zip(correspondences, keep) if k]
    rejected = [c for c, k in zip(correspondences, keep) if not k]
    return kept, rejected


def _displacements_um(
    correspondences:   List[NeighborCorrespondence],
    nominal_positions: Dict[int, Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """``(d, m)``, each ``(n, 2)`` microns: every correspondence's nominal
    displacement ``nominal[neighbor] - nominal[anchor]`` and measured
    displacement ``measured_xy - nominal[anchor]``."""
    anchorXY = np.array([nominal_positions[c.anchor_fov] for c in correspondences],
                        dtype=float).reshape(-1, 2)
    d = np.array([nominal_positions[c.neighbor_fov] for c in correspondences],
                 dtype=float).reshape(-1, 2) - anchorXY
    m = np.array([c.measured_xy for c in correspondences],
                 dtype=float).reshape(-1, 2) - anchorXY
    return d, m


def fit_displacement_affine(
    correspondences:   List[NeighborCorrespondence],
    nominal_positions: Dict[int, Tuple[float, float]],
) -> np.ndarray:
    """
    Least-squares 2x2 ``A`` with ``m = A d`` over *correspondences* (see
    `_displacements_um`); no translation term, since a displacement has
    none. ``A`` captures the camera-vs-stage rotation and the
    image-vs-stage scale (see `rotation_and_scale`).

    Fitting on absolute positions (``measured = M nominal + t``) instead
    would return about the identity: every measured position is the
    anchor's own nominal position plus a few microns.

    A direction no correspondence spans (e.g. a single-row grid, or no
    correspondences at all) falls back to the identity, via a ridge term
    far too small to affect a direction that is measured.
    """
    d, m = _displacements_um(correspondences, nominal_positions)
    ridge = 1e-6 * np.eye(2)
    return np.linalg.solve(d.T @ d + ridge, d.T @ m + ridge).T


def rotation_and_scale(affine: np.ndarray) -> Tuple[float, np.ndarray]:
    """Polar split ``A = R S``: the rotation angle of ``R`` (degrees) and the
    singular values of ``A``."""
    u, singularValues, vt = np.linalg.svd(affine)
    rotation = u @ vt
    return float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0]))), singularValues


def apply_affine(
    affine:            np.ndarray,
    nominal_positions: Dict[int, Tuple[float, float]],
    origin_fov:        int,
) -> Dict[int, Tuple[float, float]]:
    """Positions from *affine* alone, fixed at *origin_fov*:
    ``p0 + A (nominal - p0)``, with ``p0`` = *origin_fov*'s nominal position."""
    p0 = np.array(nominal_positions[origin_fov], dtype=float)
    return {f: tuple(float(v) for v in p0 + affine @ (np.array(xy, dtype=float) - p0))
            for f, xy in nominal_positions.items()}


def grid_neighbor_pairs(
    nominal_positions:  Dict[int, Tuple[float, float]],
    tolerance_fraction: float = 0.25,
) -> List[Tuple[int, int]]:
    """Every 4-connected grid edge ``(a, b)``, ``a < b``, found with the
    same `find_grid_neighbor` lookup the registration uses."""
    pairs = set()
    for fov in nominal_positions:
        for _, dx, dy in _DIRECTIONS:
            neighbor = find_grid_neighbor(
                fov, nominal_positions, dx, dy, tolerance_fraction)
            if neighbor is not None:
                pairs.add((min(fov, neighbor), max(fov, neighbor)))
    return sorted(pairs)


def edge_errors(
    correspondences:   List[NeighborCorrespondence],
    positions:         Dict[int, Tuple[float, float]],
    nominal_positions: Dict[int, Tuple[float, float]],
) -> np.ndarray:
    """``|P[neighbor] - P[anchor] - m|`` (microns) per correspondence: how
    far *positions* leave each measured overlap from lining up."""
    _, m = _displacements_um(correspondences, nominal_positions)
    solved = np.array(
        [np.subtract(positions[c.neighbor_fov], positions[c.anchor_fov])
         for c in correspondences], dtype=float).reshape(-1, 2)
    return np.hypot(*(solved - m).T)


@dataclass
class GlobalPositionCorrection:
    """
    Per-fov positions from :func:`fit_global_positions`.

    Attributes
    ----------
    positions         : ``{fov_id: (x, y)}`` (microns) for every fov in the
                        nominal positions passed in
    anchor_fovs       : ``{component_id: fov_id}`` -- the one fov per
                        connected component pinned at its affine position
    n_fovs_solved     : ``len(positions)``
    n_correspondences : how many correspondences fed the solve
    n_components      : connected components of the graph of kept
                        correspondences plus prior grid edges
    residual_rms_um   : RMS of `edge_errors` over the correspondences that
                        fed the solve. In-sample, so it cannot flag
                        overfitting; see :func:`cross_validate_positions`.
    affine            : the 2x2 displacement affine
                        (:func:`fit_displacement_affine`)
    """
    positions:         Dict[int, Tuple[float, float]]
    anchor_fovs:       Dict[int, int]
    n_fovs_solved:     int
    n_correspondences: int
    n_components:      int
    residual_rms_um:   float
    affine:            np.ndarray


def fit_global_positions(
    correspondences:    List[NeighborCorrespondence],
    nominal_positions:  Dict[int, Tuple[float, float]],
    prior_weight:       float = 1e-3,
    grid_pairs:         Optional[List[Tuple[int, int]]] = None,
    tolerance_fraction: float = 0.25,
) -> GlobalPositionCorrection:
    """
    Solve every fov's position from the kept correspondences, with a weak
    affine prior on every grid edge.

    With ``A`` from :func:`fit_displacement_affine`, ``m_e`` a
    correspondence's measured displacement and ``d_ab`` a grid edge's
    nominal displacement (see `_displacements_um`), minimise over every
    fov's position ``P``::

        sum_{kept e}      |P_b - P_a - m_e|^2
      + prior_weight sum_{grid edges ab} |P_b - P_a - A d_ab|^2

    with one fov per connected component pinned at its affine position
    ``apply_affine(A, nominal, origin)`` (origin = the largest component's
    pin; each component pins the fov that anchors the most
    correspondences). The prior makes fovs and edges with no kept
    measurement follow the affine rather than raw nominal positions (which
    are ~3 um off), and keeps the grid one component.

    Solved directly (sparse LU), per axis, for the offset from the affine
    positions: the normal matrix is a weighted graph Laplacian, so there is
    no iterative-solver tolerance to tune.

    Parameters
    ----------
    correspondences   : already passed through
                        :func:`filter_correspondence_outliers`
    nominal_positions : ``{fov_id: (x, y)}`` -- every fov to solve for
    prior_weight      : weight of each grid edge's ``A d`` prior relative
                        to a measured edge; 0 disables the prior
    grid_pairs        : the grid edges for the prior (default: computed
                        by :func:`grid_neighbor_pairs`; pass them in to
                        reuse across several solves)
    """
    fovs = sorted(nominal_positions)
    index = {f: i for i, f in enumerate(fovs)}
    nFovs = len(fovs)
    nominal = np.array([nominal_positions[f] for f in fovs], dtype=float)

    affine = fit_displacement_affine(correspondences, nominal_positions)
    d, m = _displacements_um(correspondences, nominal_positions)
    i = [index[c.anchor_fov] for c in correspondences]
    j = [index[c.neighbor_fov] for c in correspondences]
    # Offsets from the affine positions: measured edges ask for m - A d,
    # prior edges for 0.
    target = [m - d @ affine.T]
    weight = [np.ones(len(correspondences))]
    if prior_weight > 0:
        if grid_pairs is None:
            grid_pairs = grid_neighbor_pairs(nominal_positions, tolerance_fraction)
        i += [index[a] for a, _ in grid_pairs]
        j += [index[b] for _, b in grid_pairs]
        target.append(np.zeros((len(grid_pairs), 2)))
        weight.append(np.full(len(grid_pairs), prior_weight))
    target, weight = np.vstack(target), np.concatenate(weight)
    nEdges = len(i)

    incidence = sparse.csr_matrix(
        (np.r_[-np.ones(nEdges), np.ones(nEdges)],
         (np.r_[np.arange(nEdges), np.arange(nEdges)], np.r_[i, j])),
        shape=(nEdges, nFovs))
    nComponents, labels = connected_components(
        sparse.csr_matrix((np.ones(nEdges), (i, j)), shape=(nFovs, nFovs)),
        directed=False)

    anchorCounts = np.bincount(
        [index[c.anchor_fov] for c in correspondences], minlength=nFovs)
    pins = []
    for component in range(nComponents):
        members = np.where(labels == component)[0]
        pins.append(int(members[np.argmax(anchorCounts[members])]))
    originFov = fovs[pins[int(np.argmax(np.bincount(labels)))]]
    affinePositions = apply_affine(affine, nominal_positions, originFov)

    # Heavily weighted relative to unit-weighted correspondence rows: pins
    # each component's offset to within numerical noise of zero without
    # needing an equality-constrained solver.
    PIN_WEIGHT = 1.0e6
    laplacian = (incidence.T @ sparse.diags(weight) @ incidence).tolil()
    for p in pins:
        laplacian[p, p] += PIN_WEIGHT
    laplacian = laplacian.tocsc()
    offsets = np.column_stack([
        spsolve(laplacian, incidence.T @ (weight * target[:, axis]))
        for axis in range(2)]).reshape(nFovs, 2)

    positions = {
        f: (affinePositions[f][0] + float(offsets[k, 0]),
            affinePositions[f][1] + float(offsets[k, 1]))
        for k, f in enumerate(fovs)}
    errors = edge_errors(correspondences, positions, nominal_positions)
    residualRmsUm = float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else 0.0

    return GlobalPositionCorrection(
        positions=positions,
        anchor_fovs={component: fovs[p] for component, p in enumerate(pins)},
        n_fovs_solved=nFovs, n_correspondences=len(correspondences),
        n_components=nComponents, residual_rms_um=residualRmsUm, affine=affine)


def cross_validate_positions(
    correspondences:    List[NeighborCorrespondence],
    nominal_positions:  Dict[int, Tuple[float, float]],
    n_folds:            int = 5,
    seed:               int = 0,
    prior_weight:       float = 1e-3,
    tolerance_fraction: float = 0.25,
) -> Dict[str, np.ndarray]:
    """
    Held-out edge error (microns): hide one fold of the correspondences,
    solve from the rest, and measure `edge_errors` on the hidden fold.

    Folds are drawn per PHYSICAL edge, so an edge measured from both ends
    has both measurements in the same fold (the two are mirror images, so
    one would otherwise predict the other perfectly).

    Returns ``{'final', 'affine_only', 'nominal': errors}``, over every
    held-out correspondence, for three position sets: the full solve,
    ``A`` alone (:func:`apply_affine`), and the nominal positions.
    """
    gridPairs = grid_neighbor_pairs(nominal_positions, tolerance_fraction)
    edgeKeys = [(min(c.anchor_fov, c.neighbor_fov), max(c.anchor_fov, c.neighbor_fov))
                for c in correspondences]
    uniqueKeys = sorted(set(edgeKeys))
    foldOf = dict(zip(uniqueKeys, np.random.default_rng(seed).integers(
        0, n_folds, len(uniqueKeys))))

    errors: Dict[str, list] = {'final': [], 'affine_only': [], 'nominal': []}
    for fold in range(n_folds):
        train = [c for c, k in zip(correspondences, edgeKeys) if foldOf[k] != fold]
        test = [c for c, k in zip(correspondences, edgeKeys) if foldOf[k] == fold]
        if not test:
            continue
        correction = fit_global_positions(
            train, nominal_positions, prior_weight, gridPairs)
        affinePositions = apply_affine(
            correction.affine, nominal_positions, min(nominal_positions))
        errors['final'].append(
            edge_errors(test, correction.positions, nominal_positions))
        errors['affine_only'].append(
            edge_errors(test, affinePositions, nominal_positions))
        errors['nominal'].append(
            edge_errors(test, nominal_positions, nominal_positions))
    return {name: np.concatenate(v) if v else np.zeros(0) for name, v in errors.items()}


def compute_overlap_correlations(
    correspondences:   List[NeighborCorrespondence],
    positions:         Dict[int, Tuple[float, float]],
    load_frame:        Callable[[int], np.ndarray],
    pixel_size_um:     float,
    overlap_fraction:  float,
) -> Dict[Tuple[int, int, str], float]:
    """
    Pearson correlation between each correspondence's anchor/neighbour
    overlap-band crop, evaluated AT *positions* -- typically the FINAL,
    corrected positions from :func:`fit_global_positions` -- rather than at
    whatever shift `register_neighbor_pair`'s own phase correlation happened
    to measure for that one pair. Since the joint least-squares solve uses
    every fov's correspondences at once, its implied relative offset for a
    given pair can differ from that one pair's own raw measurement -- this
    is how well adjacent fovs' real image content actually agrees once
    every position has been jointly solved.

    Ported from the sibling MERci project's
    `acquisition.camera_rotation.overlap_correlation`, which used this exact
    metric (mean overlap correlation across the whole grid) to pick the
    best of several candidate correction strategies on real data -- see
    this module's own docstring.

    Returns ``{(anchor_fov, neighbor_fov, direction): correlation}``. A
    degenerate (zero-variance) crop scores ``0.0``, not ``NaN`` -- a
    zero-variance crop has no real correlation to report, and ``NaN``
    would silently corrupt any downstream mean/plot.
    """
    cache = _BoundedFrameCache(load_frame)

    directionToDxDy = {label: (dx, dy) for label, dx, dy in _DIRECTIONS}
    correlations: Dict[Tuple[int, int, str], float] = {}
    for c in correspondences:
        dx, dy = directionToDxDy[c.direction]
        anchorFrame = cache.get(c.anchor_fov)
        anchorCrop, neighborCrop = crop_overlap(
            anchorFrame, cache.get(c.neighbor_fov), dx, dy, overlap_fraction)
        anchorCrop = anchorCrop.astype(np.float64)
        neighborCrop = neighborCrop.astype(np.float64)

        # The extra shift implied by *positions* beyond the offset
        # `crop_overlap` already assumes -- same convention
        # `register_neighbor_pair` uses to turn a measured pixel shift into
        # a position, just inverted here to turn a position back into a
        # shift to apply to the crop before correlating.
        cropOffset = np.array(crop_offset_um(
            anchorFrame.shape, dx, dy, overlap_fraction, pixel_size_um))
        finalOffset = np.subtract(positions[c.neighbor_fov], positions[c.anchor_fov])
        extraShiftUm = finalOffset - cropOffset
        if extraShiftUm[0] != 0.0 or extraShiftUm[1] != 0.0:
            neighborCrop = ndi_shift(
                neighborCrop,
                shift=(extraShiftUm[1] / pixel_size_um, extraShiftUm[0] / pixel_size_um),
                order=1, mode='nearest')

        anchorFlat, neighborFlat = anchorCrop.ravel(), neighborCrop.ravel()
        if anchorFlat.std() == 0.0 or neighborFlat.std() == 0.0:
            correlations[(c.anchor_fov, c.neighbor_fov, c.direction)] = 0.0
        else:
            correlations[(c.anchor_fov, c.neighbor_fov, c.direction)] = float(
                np.corrcoef(anchorFlat, neighborFlat)[0, 1])
    return correlations
