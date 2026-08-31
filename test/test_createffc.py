import numpy as np
import pytest

from merlin.analysis import createffc
from merlin.analysis import generatemosaic


def _run_ffc_task(dataSet, analysisName, **parameters):
    parameters.setdefault('smooth_sigma', 5)
    task = createffc.CreateFfc(
        dataSet, parameters=parameters, analysisName=analysisName)
    task.save()
    task._run_analysis()
    return task


def test_ffc_field_shape_and_floor(simple_merfish_data):
    task = _run_ffc_task(
        simple_merfish_data, 'createFfcShapeFloor', minimum_value=0.2)

    dataOrganization = simple_merfish_data.get_data_organization()
    imageDimensions = simple_merfish_data.get_image_dimensions()
    colors = {dataOrganization.get_data_channel_color(d)
              for d in dataOrganization.get_data_channels()}

    for color in colors:
        field = task.get_ffc_field(color)
        assert field.shape == tuple(imageDimensions)
        assert field.dtype == np.float32
        assert field.min() >= 0.2


def test_channels_sharing_color_get_same_field(simple_merfish_data):
    task = _run_ffc_task(simple_merfish_data, 'createFfcSharedColor')

    dataOrganization = simple_merfish_data.get_data_organization()
    # bit1 and bit4 are both imaged at color 650 in test_data_organization.csv
    bit1 = dataOrganization.get_data_channel_index('bit1')
    bit4 = dataOrganization.get_data_channel_index('bit4')
    assert dataOrganization.get_data_channel_color(bit1) \
        == dataOrganization.get_data_channel_color(bit4)

    field1 = task.get_ffc_field_for_channel(bit1)
    field4 = task.get_ffc_field_for_channel(bit4)
    np.testing.assert_array_equal(field1, field4)

    # bit2 is imaged at color 750, a different field
    bit2 = dataOrganization.get_data_channel_index('bit2')
    field2 = task.get_ffc_field_for_channel(bit2)
    assert not np.array_equal(field1, field2)


def test_apply_ffc_matches_manual_division():
    image = np.array([[10, 20], [0, 40]], dtype=np.uint16)
    field = np.array([[2, 4], [1, 8]], dtype=np.float32)

    corrected = createffc.CreateFfc.apply_ffc(image, field)

    expected = np.clip(image.astype(np.float32) / field, 0, None)
    np.testing.assert_array_almost_equal(corrected, expected)


def test_generatemosaic_get_ffc_field_none_without_ffc_task(simple_merfish_data):
    task = generatemosaic.GenerateMosaic(
        simple_merfish_data,
        parameters={'global_align_task': 'globalAlign', 'warp_task': 'warp'})
    assert task._get_ffc_field(0) is None


def test_generatemosaic_get_ffc_field_from_ffc_task(simple_merfish_data):
    ffcTask = _run_ffc_task(simple_merfish_data, 'createFfcForMosaicLookup')

    mosaicTask = generatemosaic.GenerateMosaic(
        simple_merfish_data,
        parameters={'global_align_task': 'globalAlign', 'warp_task': 'warp',
                    'ffc_task': ffcTask.analysisName})

    dataOrganization = simple_merfish_data.get_data_organization()
    bit1 = dataOrganization.get_data_channel_index('bit1')
    np.testing.assert_array_equal(
        mosaicTask._get_ffc_field(bit1),
        ffcTask.get_ffc_field_for_channel(bit1))


def test_generatemosaic_load_tile_applies_ffc(simple_merfish_data):
    ffcTask = _run_ffc_task(
        simple_merfish_data, 'createFfcForMosaicLoadTile', minimum_value=0.5)

    mosaicTask = generatemosaic.GenerateMosaic(
        simple_merfish_data,
        parameters={'global_align_task': 'globalAlign', 'warp_task': 'warp',
                    'ffc_task': ffcTask.analysisName, 'downsample': 1})

    dataOrganization = simple_merfish_data.get_data_organization()
    bit1 = dataOrganization.get_data_channel_index('bit1')
    imageDimensions = simple_merfish_data.get_image_dimensions()
    rawTile = np.full(tuple(imageDimensions), 100, dtype=np.uint16)

    class FakeWarpTask:
        def get_aligned_image(self, fov, dataChannel, zIndex,
                               chromaticCorrector=None):
            return rawTile

    mosaicTask.warpTask = FakeWarpTask()
    tile = mosaicTask.load_tile(0, 0, bit1)

    field = ffcTask.get_ffc_field_for_channel(bit1)
    expected = createffc.CreateFfc.apply_ffc(rawTile, field).astype(
        rawTile.dtype)
    np.testing.assert_array_equal(tile, expected)
