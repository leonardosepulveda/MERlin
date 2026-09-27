"""
Camera fixed-pattern template for fiducial (bead) registration.

Every frame from a camera carries the same per-pixel pattern: offset, row
banding, illumination profile. When the beads are weak it dominates phase
correlation:

- two frames of the same FOV correlate at zero shift whatever the sample
  did (BC555_sample_05/disk: FOV 1 cells vs H01 and FOV 1 vs an unrelated
  FOV 300 gave the same zero-shift peak, 0.176 vs 0.171);
- the facing bands of two FOVs lock at a zero perpendicular shift.

The template is the pixel-wise median of one fiducial image type over
several FOVs: the camera pattern sits at the same pixels in every FOV and
survives the median, the beads mostly do not. Subtracting it before the
high-pass filter removes the pattern (same FOV 0.045 at (-2, -2) px,
unrelated FOV 0.003). On dense bead fields a 15-FOV median is a blur of
beads rather than a camera pattern; that was harmless on the samples
tested, but the option stays off by default.

Found in the `260926_LT074_stitching_test` investigation (its
`measure_rounds.py`, `template_fovs()` and `--template` mode).
"""
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np

from merlin.util import globalpositions

#: (fiducialImageType, fiducialImagingRound, fiducialFrame) -- data
#: channels of the same imaging round share one fiducial image, and so one
#: template.
FiducialSource = Tuple[str, int, int]


def fiducial_source(dataOrganization, dataChannel: int) -> FiducialSource:
    row = dataOrganization.data.loc[dataChannel]
    return (str(row['fiducialImageType']), int(row['fiducialImagingRound']),
            int(row['fiducialFrame']))


def fiducial_sources(dataOrganization) -> List[FiducialSource]:
    """Every distinct fiducial image of the data organization, sorted."""
    return sorted({fiducial_source(dataOrganization, c)
                   for c in dataOrganization.get_data_channels()})


def choose_template_fovs(fovs: Sequence[int], count: int, seed: int,
                         exclude: Iterable[int] = ()) -> np.ndarray:
    """*count* FOVs picked at random (fixed *seed*) from *fovs*, avoiding
    *exclude* (the FOVs being scored) unless too few would remain."""
    rest = np.setdiff1d(np.asarray(fovs), np.asarray(list(exclude), dtype=int))
    if len(rest) < count:
        rest = np.asarray(fovs)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(rest, size=min(count, len(rest)), replace=False))


def load_fiducial_frame(dataSet, dataChannel: int, fov: int,
                        frame: Union[None, str, int] = None,
                        removeHotPixels: bool = True) -> np.ndarray:
    """One fiducial frame of *dataChannel*'s fiducial file, oriented (as
    every `ImageDataSet.load_image`), hot pixels optionally removed, as
    float32.

    *frame*: None for the data organization's own fiducial frame, 'last'
    for the file's last frame (a second bead frame, where the acquisition
    records one there), or a frame index.
    """
    dataOrganization = dataSet.get_data_organization()
    path = dataOrganization.get_fiducial_filename(dataChannel, fov)
    if frame is None:
        frame = dataOrganization.get_fiducial_frame_index(dataChannel)
    elif frame == 'last':
        frame = dataSet.image_stack_size(path)[2] - 1
    image = dataSet.load_image(path, frame)
    if removeHotPixels:
        image = globalpositions.remove_hot_pixels(image)
    return image.astype(np.float32)


def median_template(images: Sequence[np.ndarray]) -> np.ndarray:
    return np.median(np.stack(images), axis=0).astype(np.float32)
