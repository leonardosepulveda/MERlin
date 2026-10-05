import pandas
import pytest
from shapely import geometry

from merlin.analysis.filterbarcodes import AdaptiveFilterBarcodes


def _filter(dataSet, extra):
    parameters = {'decode_task': 'D', 'adaptive_task': 'A',
                  'remove_z_duplicated_barcodes': True}
    parameters.update(extra)
    return AdaptiveFilterBarcodes(dataSet, parameters,
                                  analysisName='filter_' + '_'.join(extra))


def test_z_duplicate_xy_threshold_in_pixels(simple_merfish_data):
    task = _filter(simple_merfish_data, {'z_duplicate_xy_pixel_threshold': 3})
    assert task._z_duplicate_xy_pixels() == 3


def test_z_duplicate_xy_threshold_in_microns(simple_merfish_data):
    task = _filter(simple_merfish_data, {'z_duplicate_xy_distance_um': 0.27,
                                         'z_duplicate_xy_pixel_threshold': 3})
    micronsPerPixel = simple_merfish_data.get_microns_per_pixel()
    assert task._z_duplicate_xy_pixels() == pytest.approx(
        0.27 / micronsPerPixel)


class _FakeAlignTask:

    def get_fov_boxes(self):
        return [geometry.box(0, 0, 120, 120), geometry.box(100, 0, 220, 120)]


class _FakeDecodeTask:
    parameters = {'global_align_task': 'G'}


def test_filter_keeps_overlap_barcodes_of_nearest_fov(simple_merfish_data,
                                                     monkeypatch):
    tasks = {'D': _FakeDecodeTask(), 'G': _FakeAlignTask()}
    monkeypatch.setattr(simple_merfish_data, 'load_analysis_task',
                        lambda name: tasks[name])
    monkeypatch.setattr(simple_merfish_data, 'get_fovs', lambda: [0, 1])
    task = _filter(simple_merfish_data, {})
    bc = pandas.DataFrame({'global_x': [105.0, 115.0], 'global_y': [50.0] * 2})
    # centres at x = 60 and 160: x = 105 is fov 0's, x = 115 is fov 1's
    assert list(task._keep_owned_barcodes(bc, 0)['global_x']) == [105.0]
    assert list(task._keep_owned_barcodes(bc, 1)['global_x']) == [115.0]
    task.parameters['remove_overlap_duplicates'] = False
    assert len(task._keep_owned_barcodes(bc, 0)) == 2
