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
def test_cellpose_sam_anisotropy_computed(
        simple_merfish_data, monkeypatch, zPositions, micronsPerPixel,
        downsample, expected):
    task = _sam_task(simple_merfish_data, downsample_factor=downsample)
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


def test_cellpose_sam_anisotropy_uneven_steps_warns(
        simple_merfish_data, monkeypatch):
    task = _sam_task(simple_merfish_data, downsample_factor=None)
    monkeypatch.setattr(simple_merfish_data, 'get_z_positions_segmentation',
                        lambda fov=None: [0.0, 1.0, 2.0, 4.0])
    monkeypatch.setattr(simple_merfish_data, 'get_microns_per_pixel',
                        lambda: 1.0)
    with pytest.warns(UserWarning, match='not uniform'):
        assert task._get_anisotropy(0) == pytest.approx(1.0)
