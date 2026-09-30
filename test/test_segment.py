import warnings

import numpy as np
import scipy.ndimage

import pytest

from merlin.analysis import segment


def _sam_task(dataSet, **parameters):
    parameters = {'warp_task': 'SimpleFiducialWarp',
                  'global_align_task': 'SimpleGlobalAlignment',
                  **parameters}
    return segment.CellPoseSegmentSAM(dataSet, parameters,
                                      analysisName='testCellPoseSAM')


@pytest.mark.parametrize('zPositions, micronsPerPixel, downsample, expected', [
    # BC553 disk: 0.25 um z steps, 0.0893 um/px, downsampled 4x
    ([0.0, 0.25, 0.5, 0.75], 0.0893, 4, 0.25 / (0.0893 * 4)),
    # MF3 epi: 1 um z steps, 0.1074 um/px, downsampled 4x
    ([0.0, 1.0, 2.0], 0.1074, 4, 1 / (0.1074 * 4)),
    # no downsampling, unsorted z positions
    ([2.0, 0.0, 1.0], 0.5, None, 2.0),
    # a single plane has no z spacing
    ([0.0], 0.1, 4, 1.0),
])
@pytest.mark.parametrize('anisotropy', ['auto', 'AUTO', None])
def test_cellpose_sam_anisotropy_computed(
        simple_merfish_data, monkeypatch, zPositions, micronsPerPixel,
        downsample, expected, anisotropy):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        task = _sam_task(simple_merfish_data, downsample_factor=downsample,
                         anisotropy=anisotropy)
    monkeypatch.setattr(simple_merfish_data, 'get_z_positions_segmentation',
                        lambda fov=None: zPositions)
    monkeypatch.setattr(simple_merfish_data, 'get_microns_per_pixel',
                        lambda: micronsPerPixel)
    assert task._get_anisotropy(0) == pytest.approx(expected)


def test_cellpose_sam_anisotropy_override(simple_merfish_data, monkeypatch):
    task = _sam_task(simple_merfish_data, anisotropy=1.5)
    monkeypatch.setattr(simple_merfish_data, 'get_z_positions_segmentation',
                        lambda fov=None: [0.0, 0.25, 0.5])
    assert task._get_anisotropy(0) == 1.5


def test_cellpose_sam_anisotropy_default_is_auto(simple_merfish_data):
    assert _sam_task(simple_merfish_data).parameters['anisotropy'] == 'auto'


def test_cellpose_sam_anisotropy_null_warns(simple_merfish_data):
    with pytest.warns(UserWarning, match='null is read as auto'):
        _sam_task(simple_merfish_data, anisotropy=None)


def test_cellpose_sam_anisotropy_bad_string_raises(simple_merfish_data):
    with pytest.raises(ValueError, match='automatic'):
        _sam_task(simple_merfish_data, anisotropy='automatic')


def test_cellpose_sam_anisotropy_uneven_steps_warns(
        simple_merfish_data, monkeypatch):
    task = _sam_task(simple_merfish_data, downsample_factor=None)
    monkeypatch.setattr(simple_merfish_data, 'get_z_positions_segmentation',
                        lambda fov=None: [0.0, 1.0, 2.0, 4.0])
    monkeypatch.setattr(simple_merfish_data, 'get_microns_per_pixel',
                        lambda: 1.0)
    with pytest.warns(UserWarning, match='not uniform'):
        assert task._get_anisotropy(0) == pytest.approx(1.0)


def _sphere_volume(zStep, radius=14.0):
    """Four spheres of the given radius (um) in a 60 x 128 x 128 um box,
    blurred by a PSF (sigma 1 um xy, 2 um z), sampled at 1 um in xy and
    zStep um in z, with camera-like noise.
    """
    rng = np.random.default_rng(0)
    z = (np.arange(int(60 / zStep)) + 0.5) * zStep
    xy = np.arange(128) + 0.5
    Z, Y, X = np.meshgrid(z, xy, xy, indexing='ij')
    volume = np.zeros(Z.shape, np.float32)
    for cy in (32, 96):
        for cx in (32, 96):
            volume += np.sqrt((Z - 30)**2 + (Y - cy)**2 + (X - cx)**2) < radius
    volume = scipy.ndimage.gaussian_filter(volume, (2 / zStep, 1, 1))
    volume = volume * 1000 + 100 + rng.normal(0, 20, volume.shape)
    return scipy.ndimage.gaussian_filter(volume, (0, 1, 1)).astype(np.float32)


@pytest.mark.slowtest
def test_cellpose_sam_anisotropy_keeps_spheres_round():
    """With z sampled 3x coarser than xy, passing the anisotropy gives
    round cells in microns. Without it the cells come out about 18% taller
    than wide; with it about 7%. The 12% cutoff separates the two, so the
    margin is thin. Cellpose-SAM segments isolated cells almost correctly
    either way at smaller z steps, so those do not discriminate.
    """
    cellposeModels = pytest.importorskip('cellpose.models')
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('cellpose-SAM 3D needs a GPU to run in reasonable time')

    zStep = 3.0
    model = cellposeModels.CellposeModel(gpu=True)
    masks = model.eval(_sphere_volume(zStep), do_3D=True, z_axis=0,
                       anisotropy=zStep)[0]

    labels = np.unique(masks)[1:]
    assert len(labels) == 4
    for label in labels:
        zz, yy, xx = np.nonzero(masks == label)
        zExtent = (np.ptp(zz) + 1) * zStep
        xyExtent = (np.ptp(yy) + np.ptp(xx) + 2) / 2
        assert zExtent / xyExtent == pytest.approx(1, abs=0.12)
