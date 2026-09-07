import numpy as np
import pytest
import skimage.transform

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


def _mosaic_tile_parameters(**overrides):
    parameters = {
        'global_align_task': 'globalAlign', 'warp_task': 'warp',
        'preprocess_task': 'preprocess', 'downsample': 2, 'z_index': 0,
        'data_channels': ['bit1']}
    parameters.update(overrides)
    return parameters


def test_generatemosaictile_requires_ffc_task(simple_merfish_data):
    # use_ffc defaults to true, and ffc_task has no default -- so a config
    # that omits ffc_task without explicitly opting out (use_ffc=False)
    # must fail rather than silently skipping the correction.
    task = generatemosaic.GenerateMosaicTile(
        simple_merfish_data, parameters=_mosaic_tile_parameters())
    with pytest.raises(KeyError):
        task.get_dependencies()


def test_generatemosaictile_use_ffc_false_skips_ffc_task_dependency(
        simple_merfish_data):
    # with use_ffc explicitly disabled, ffc_task is neither required nor a
    # dependency
    task = generatemosaic.GenerateMosaicTile(
        simple_merfish_data,
        parameters=_mosaic_tile_parameters(use_ffc=False))
    assert set(task.get_dependencies()) == {'globalAlign', 'warp', 'preprocess'}


def test_generatemosaictile_load_tile_applies_ffc(simple_merfish_data):
    ffcTask = _run_ffc_task(
        simple_merfish_data, 'createFfcForMosaicLoadTile', minimum_value=0.5)

    mosaicTask = generatemosaic.GenerateMosaicTile(
        simple_merfish_data,
        parameters=_mosaic_tile_parameters(ffc_task=ffcTask.analysisName))

    dataOrganization = simple_merfish_data.get_data_organization()
    bit1 = dataOrganization.get_data_channel_index('bit1')
    imageDimensions = simple_merfish_data.get_image_dimensions()
    rawTile = np.full(tuple(imageDimensions), 100, dtype=np.uint16)

    class FakeWarpTask:
        def get_aligned_image(self, fov, dataChannel, zIndex,
                               chromaticCorrector=None):
            return rawTile

    mosaicTask.warpTask = FakeWarpTask()
    mosaicTask.ffcTask = ffcTask
    tile = mosaicTask._load_tile(0, bit1, 2)

    field = ffcTask.get_ffc_field_for_channel(bit1)
    expected = skimage.transform.resize(
        createffc.CreateFfc.apply_ffc(rawTile, field),
        mosaicTask.get_tile_shape(2), anti_aliasing=True,
        preserve_range=True).astype(np.float32)
    np.testing.assert_array_almost_equal(tile, expected)


def test_generatemosaictile_use_ffc_false_skips_correction(simple_merfish_data):
    mosaicTask = generatemosaic.GenerateMosaicTile(
        simple_merfish_data,
        parameters=_mosaic_tile_parameters(use_ffc=False))

    dataOrganization = simple_merfish_data.get_data_organization()
    bit1 = dataOrganization.get_data_channel_index('bit1')
    imageDimensions = simple_merfish_data.get_image_dimensions()
    rawTile = np.full(tuple(imageDimensions), 100, dtype=np.uint16)

    class FakeWarpTask:
        def get_aligned_image(self, fov, dataChannel, zIndex,
                               chromaticCorrector=None):
            return rawTile

    mosaicTask.warpTask = FakeWarpTask()
    # ffcTask is never loaded/consulted when use_ffc is false
    mosaicTask.ffcTask = None
    tile = mosaicTask._load_tile(0, bit1, 2)

    expected = skimage.transform.resize(
        rawTile.astype(np.float32), mosaicTask.get_tile_shape(2),
        anti_aliasing=True, preserve_range=True).astype(np.float32)
    np.testing.assert_array_almost_equal(tile, expected)
