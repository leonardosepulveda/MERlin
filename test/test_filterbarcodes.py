import pytest

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
