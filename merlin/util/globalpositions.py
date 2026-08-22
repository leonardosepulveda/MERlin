"""
Camera/stage global-position correction for a MERFISH FOV grid.

Ported from the sibling `MERci` project's `acquisition.camera_rotation`
module (`251225_LT027_saving_time/MERci`), which found -- on a real,
several-hundred-FOV whole-grid dataset -- that a neighbouring FOV's
nominal (stage-reported) position disagrees with its true relative
position by a small but real, highly direction-dependent amount (most
plausibly stage backlash/hysteresis, not a fixed camera-vs-stage
rotation angle: see :func:`fit_global_positions`'s own docstring for why
a single global affine transform cannot correct this class of error at
all, no matter how it's fit).

Comparing three correction strategies on that real dataset via mean
pixel-intensity correlation between every real 4-connected neighbour
pair's own overlapping border region, a joint least-squares solve of
every FOV's position at once ("global_lsq") was the clear winner: 0.79
mean overlap correlation on held-out edges, vs. 0.60 for a greedy
per-FOV walk and 0.09 for both the uncorrected nominal grid and a single
global affine fit. This module ports only that winning method (plus the
neighbour-pair registration and outlier filtering it needs) into MERlin,
as the basis for `merlin.analysis.globalalign.CorrelationGlobalAlignment`.
Deliberately NOT ported: the single-affine and greedy-walk methods
(both lost the real-data comparison), and the orientation-detection
helpers (MERlin already applies `transpose`/`flip_horizontal`/
`flip_vertical` at image-load time via `ImageDataSet.load_image`, so raw
images handed to this module are already correctly oriented).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import shift as ndi_shift
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
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
        n = max(1, int(round(w * overlap_fraction)))
        if dx > 0:
            return anchor_img[:, w - n:], neighbor_img[:, :n]
        return anchor_img[:, :n], neighbor_img[:, w - n:]
    else:
        n = max(1, int(round(h * overlap_fraction)))
        if dy > 0:
            return anchor_img[h - n:, :], neighbor_img[:n, :]
        return anchor_img[:n, :], neighbor_img[h - n:, :]


def register_neighbor_pair(
    anchor_img:       np.ndarray,
    neighbor_img:     np.ndarray,
    anchor_xy:        Tuple[float, float],
    neighbor_xy:      Tuple[float, float],
    dx:               float,
    dy:               float,
    overlap_fraction: float,
    pixel_size_um:    float,
    upsample_factor:  int = 100,
) -> Tuple[Tuple[float, float], float]:
    """
    Measure the neighbour's TRUE position relative to the anchor, from the
    real pixel shift needed to align their overlapping border crop.

    Returns
    -------
    (measured_neighbor_xy, error) -- the neighbour's measured true (x, y)
    position (microns), and the registration's error metric.
    """
    a_crop, n_crop = crop_overlap(anchor_img, neighbor_img, dx, dy, overlap_fraction)
    shift, error, _ = registration.phase_cross_correlation(
        a_crop, n_crop, upsample_factor=upsample_factor)
    dy_px, dx_px = float(shift[0]), float(shift[1])

    nomDx = neighbor_xy[0] - anchor_xy[0]
    nomDy = neighbor_xy[1] - anchor_xy[1]
    measDx = nomDx + dx_px * pixel_size_um
    measDy = nomDy + dy_px * pixel_size_um
    return (anchor_xy[0] + measDx, anchor_xy[1] + measDy), float(error)


def sample_neighbor_correspondences(
    fov_ids:            List[int],
    positions:          Dict[int, Tuple[float, float]],
    load_frame:         Callable[[int], np.ndarray],
    pixel_size_um:      float,
    overlap_fraction:   float,
    tolerance_fraction: float = 0.25,
    upsample_factor:    int = 100,
) -> List[NeighborCorrespondence]:
    """
    Register every fov in *fov_ids* against each of its present 4-connected
    neighbours (exhaustive, not a sparse sample -- the real-data comparison
    this module is ported from found the joint least-squares solve needs a
    dense, whole-grid correspondence set to have real redundant constraints
    per fov; see `fit_global_positions`'s own docstring). An interior fov's
    edges are measured twice, independently, once from each side -- a free
    redundancy check, not wasted work.

    Parameters
    ----------
    fov_ids     : the fovs to use as anchors (typically every fov in the
                  dataset)
    positions   : ``{fov_id: (x, y)}`` nominal grid positions (microns),
                  covering every id in *fov_ids* and its neighbours
    load_frame  : ``load_frame(fov_id) -> np.ndarray``, returning the 2-D
                  registration image for one fov (results are cached per
                  fov id, since a neighbour can also be sampled as another
                  anchor's neighbour)
    pixel_size_um, overlap_fraction : grid/camera geometry (see
                  `find_grid_neighbor` for how each anchor's own local step
                  size is derived from *positions* directly, rather than
                  taken as a dataset-wide value)
    """
    frameCache: Dict[int, np.ndarray] = {}

    def _get_frame(fov: int) -> np.ndarray:
        if fov not in frameCache:
            frameCache[fov] = load_frame(fov)
        return frameCache[fov]

    correspondences: List[NeighborCorrespondence] = []
    for anchorFov in fov_ids:
        anchorImg = _get_frame(anchorFov)
        for direction, dx, dy in _DIRECTIONS:
            neighborFov = find_grid_neighbor(
                anchorFov, positions, dx, dy, tolerance_fraction)
            if neighborFov is None:
                continue
            neighborImg = _get_frame(neighborFov)
            measuredXY, error = register_neighbor_pair(
                anchorImg, neighborImg, positions[anchorFov], positions[neighborFov],
                dx, dy, overlap_fraction, pixel_size_um, upsample_factor)
            correspondences.append(NeighborCorrespondence(
                anchor_fov=anchorFov, neighbor_fov=neighborFov, direction=direction,
                nominal_xy=positions[neighborFov], measured_xy=measuredXY, error=error))
    return correspondences


def filter_correspondence_outliers(
    correspondences: List[NeighborCorrespondence],
    mad_threshold:   float = 5.0,
) -> Tuple[List[NeighborCorrespondence], List[NeighborCorrespondence]]:
    """
    Split correspondences into (kept, rejected) by a robust outlier test on
    each one's ``|measured - nominal|`` shift magnitude: reject a
    correspondence if its shift exceeds ``median + mad_threshold *
    robust_sigma``, where ``robust_sigma = 1.4826 * median_absolute_deviation``
    (1.4826 = 1/norm.ppf(0.75), the standard MAD-to-Gaussian-sigma
    conversion). A handful of individual registrations can fail outright
    (weak fiducial signal, a bad phase-correlation peak) even with correct
    geometry -- this catches those without needing manual review.
    """
    if len(correspondences) < 3:
        return list(correspondences), []

    shiftsUm = np.array([
        np.hypot(c.measured_xy[0] - c.nominal_xy[0], c.measured_xy[1] - c.nominal_xy[1])
        for c in correspondences
    ])
    median = float(np.median(shiftsUm))
    mad = float(np.median(np.abs(shiftsUm - median)))
    robustSigma = 1.4826 * mad
    threshold = median + mad_threshold * robustSigma if robustSigma > 0 else median

    kept = [c for c, s in zip(correspondences, shiftsUm) if s <= threshold]
    rejected = [c for c, s in zip(correspondences, shiftsUm) if s > threshold]
    return kept, rejected


@dataclass
class GlobalPositionCorrection:
    """
    Per-fov positions from jointly solving every measured fov's own position
    against all its pairwise neighbour constraints at once (see
    :func:`fit_global_positions`).

    Attributes
    ----------
    positions         : ``{fov_id: (x, y)}`` (microns) -- only fovs that
                        appeared in at least one kept correspondence; merge
                        over the full nominal positions dict as a fallback
                        for every other fov.
    anchor_fovs       : ``{component_id: fov_id}`` -- the one fov in each
                        connected correspondence-graph component held fixed
                        at its own nominal position, removing that
                        component's translational null space.
    n_fovs_solved     : ``len(positions)``
    n_correspondences : how many correspondences fed the solve
    n_components      : how many disconnected correspondence-graph
                        components were solved independently
    residual_rms_um   : RMS of ``(p[B] - p[A]) - measured_relative_offset``
                        across every correspondence, evaluated at the
                        solved positions -- near 0.0 when every redundant
                        measurement agrees; only meaningful once some fov
                        is constrained by more than one correspondence.
    """
    positions:         Dict[int, Tuple[float, float]]
    anchor_fovs:       Dict[int, int]
    n_fovs_solved:     int
    n_correspondences: int
    n_components:      int
    residual_rms_um:   float


def _connected_components(correspondences: List[NeighborCorrespondence]) -> List[List[int]]:
    """Plain BFS connected components of the anchor<->neighbour graph --
    these graphs are typically at most a few thousand nodes, no graph
    library needed."""
    adjacency: Dict[int, set] = {}
    for c in correspondences:
        adjacency.setdefault(c.anchor_fov, set()).add(c.neighbor_fov)
        adjacency.setdefault(c.neighbor_fov, set()).add(c.anchor_fov)

    visited: set = set()
    components = []
    for start in sorted(adjacency):
        if start in visited:
            continue
        stack, component = [start], []
        visited.add(start)
        while stack:
            fov = stack.pop()
            component.append(fov)
            for neighbor in adjacency[fov]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def fit_global_positions(
    correspondences:   List[NeighborCorrespondence],
    nominal_positions: Dict[int, Tuple[float, float]],
    lsqr_atol:         float = 1.0e-12,
    lsqr_btol:         float = 1.0e-12,
) -> GlobalPositionCorrection:
    """
    Jointly solve for every measured fov's own real position from all kept
    pairwise neighbour correspondences, instead of fitting one global affine
    transform applied uniformly to the whole nominal grid.

    Why a single affine transform can't fix this: the real-data comparison
    this module is ported from found a per-STEP, direction-symmetric bias
    (e.g. "+y" edges deviate by `(+3.35, +1.14)` um on average while "-y"
    edges deviate by almost the exact mirror `(-3.35, -1.14)` um --
    consistent with stage backlash/hysteresis, not a fixed rotation angle).
    Since every physical edge is measured from both endpoints, the "+y" and
    "-y" populations are numerically forced to be near-perfect mirrors of
    each other, so pooling every direction together (what a single-affine
    fit does, from bare nominal/measured point pairs with no direction
    label attached) cancels the bias out almost exactly and the best-fit
    transform comes out near-identity -- confirmed directly on real data:
    the affine method isn't failing to detect a real rotation, there simply
    isn't a coherent one to detect this way. A per-fov solve that uses each
    correspondence's own direction explicitly is the only way to recover
    it. See the sibling MERci project's `acquisition/camera_rotation.py`
    module docstring and its `notebooks/tests/
    compare_stitching_correction_methods.ipynb` for the full real-data
    comparison this design choice is based on.

    Method
    ------
    For each kept correspondence (anchor A, neighbour B), the measured
    relative offset ``r_AB = measured_xy(B) - nominal_positions[A]`` is a
    direct, independent estimate of B's true position relative to A's own
    nominal position. Solving::

        minimize over every fov's unknown position p[F]:
            sum_AB || (p[B] - p[A]) - r_AB ||^2

    is a sparse linear least-squares problem that separates cleanly into
    two independent solves (x and y), via ``scipy.sparse.linalg.lsqr``.

    The correspondence graph may not be one connected mesh (e.g. a fov
    excluded everywhere by outlier filtering). Each connected component has
    its own 1-D-per-axis translational null space (uniformly shifting every
    position in it satisfies every constraint in that component equally),
    removed by pinning ONE fov per component -- whichever appears as an
    ``anchor_fov`` in the most correspondences -- to its own nominal
    position.

    ``lsqr``'s own default convergence tolerances are too loose for a
    dense, exhaustively-measured grid: confirmed directly on real data
    (476 fovs, 1662 kept correspondences, one connected component) --
    calling ``lsqr`` with scipy's own default tolerances declared
    convergence with ``residual_rms_um`` = 81.6, roughly 3300x the value
    (0.025 um) that ``atol=btol=1e-12`` converges to on the exact same
    input (confirmed via ``istop`` 1/2, genuine convergence, not 7 --
    iteration limit -- or 3/4 -- ill-conditioned). Do not loosen these
    below their own defaults without re-confirming convergence the same
    way.

    Parameters
    ----------
    correspondences   : from :func:`sample_neighbor_correspondences`,
                        already passed through
                        :func:`filter_correspondence_outliers`
    nominal_positions : ``{fov_id: (x, y)}`` -- the full experiment's
                        nominal grid positions
    lsqr_atol, lsqr_btol : passed straight through to
                        ``scipy.sparse.linalg.lsqr`` for both the x and y
                        solves -- see the convergence note above before
                        loosening these.

    Returns
    -------
    GlobalPositionCorrection -- merge ``.positions`` over the full nominal
    positions dict as a fallback for every fov not directly measured.
    """
    if not correspondences:
        return GlobalPositionCorrection(
            positions={}, anchor_fovs={}, n_fovs_solved=0,
            n_correspondences=0, n_components=0, residual_rms_um=0.0,
        )

    components = _connected_components(correspondences)
    allFovs = sorted({fov for comp in components for fov in comp})
    fovToIdx = {fov: i for i, fov in enumerate(allFovs)}
    n = len(allFovs)

    # Pin each component's most-sampled real anchor to its own nominal position.
    anchorCounts: Dict[int, int] = {}
    for c in correspondences:
        anchorCounts[c.anchor_fov] = anchorCounts.get(c.anchor_fov, 0) + 1
    anchorFovs = {
        compId: max(comp, key=lambda fov: anchorCounts.get(fov, 0))
        for compId, comp in enumerate(components)
    }

    nCorr = len(correspondences)
    nPins = len(anchorFovs)
    # Heavily weighted relative to unit-weighted correspondence rows -- pins
    # the component's reference fov to within numerical noise of its own
    # nominal position without needing a true equality-constrained solver.
    PIN_WEIGHT = 1.0e4

    def _solve_axis(axis: int) -> np.ndarray:
        A = lil_matrix((nCorr + nPins, n), dtype=float)
        b = np.zeros(nCorr + nPins, dtype=float)

        for row, c in enumerate(correspondences):
            iA, iB = fovToIdx[c.anchor_fov], fovToIdx[c.neighbor_fov]
            A[row, iB] += 1.0
            A[row, iA] += -1.0
            b[row] = c.measured_xy[axis] - nominal_positions[c.anchor_fov][axis]

        for offset, (compId, pinFov) in enumerate(anchorFovs.items()):
            row = nCorr + offset
            A[row, fovToIdx[pinFov]] = PIN_WEIGHT
            b[row] = PIN_WEIGHT * nominal_positions[pinFov][axis]

        return lsqr(A.tocsr(), b, atol=lsqr_atol, btol=lsqr_btol)[0]

    xSolution = _solve_axis(0)
    ySolution = _solve_axis(1)
    positions = {
        fov: (float(xSolution[i]), float(ySolution[i])) for fov, i in fovToIdx.items()
    }

    residualsUm = []
    for c in correspondences:
        pA, pB = positions[c.anchor_fov], positions[c.neighbor_fov]
        rAB = (
            c.measured_xy[0] - nominal_positions[c.anchor_fov][0],
            c.measured_xy[1] - nominal_positions[c.anchor_fov][1],
        )
        residualsUm.append(np.hypot(pB[0] - pA[0] - rAB[0], pB[1] - pA[1] - rAB[1]))
    residualRmsUm = float(np.sqrt(np.mean(np.square(residualsUm)))) if residualsUm else 0.0

    return GlobalPositionCorrection(
        positions=positions, anchor_fovs=anchorFovs, n_fovs_solved=len(positions),
        n_correspondences=nCorr, n_components=len(components), residual_rms_um=residualRmsUm,
    )


def compute_overlap_correlations(
    correspondences:   List[NeighborCorrespondence],
    positions:         Dict[int, Tuple[float, float]],
    nominal_positions: Dict[int, Tuple[float, float]],
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
    frameCache: Dict[int, np.ndarray] = {}

    def _get_frame(fov: int) -> np.ndarray:
        if fov not in frameCache:
            frameCache[fov] = load_frame(fov)
        return frameCache[fov]

    directionToDxDy = {label: (dx, dy) for label, dx, dy in _DIRECTIONS}
    correlations: Dict[Tuple[int, int, str], float] = {}
    for c in correspondences:
        dx, dy = directionToDxDy[c.direction]
        anchorCrop, neighborCrop = crop_overlap(
            _get_frame(c.anchor_fov), _get_frame(c.neighbor_fov), dx, dy, overlap_fraction)
        anchorCrop = anchorCrop.astype(np.float64)
        neighborCrop = neighborCrop.astype(np.float64)

        # The extra shift implied by *positions* beyond the nominal-grid
        # alignment `crop_overlap` already assumes -- same convention
        # `register_neighbor_pair` uses to turn a measured pixel shift into
        # a position, just inverted here to turn a position back into a
        # shift to apply to the crop before correlating.
        nominalOffset = np.subtract(
            nominal_positions[c.neighbor_fov], nominal_positions[c.anchor_fov])
        finalOffset = np.subtract(positions[c.neighbor_fov], positions[c.anchor_fov])
        extraShiftUm = finalOffset - nominalOffset
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
