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


def test_apply_ffc_subtracts_dark():
    image = np.array([[110, 120], [100, 140]], dtype=np.uint16)
    field = np.array([[2, 4], [1, 8]], dtype=np.float32)
    dark = np.full((2, 2), 100, dtype=np.float32)

    corrected = createffc.CreateFfc.apply_ffc(image, field, dark)

    np.testing.assert_array_almost_equal(
        corrected, np.array([[5, 5], [0, 5]], dtype=np.float32))


def test_select_fovs_is_seeded_and_capped(simple_merfish_data):
    allFovs = list(simple_merfish_data.get_fovs())

    def select(analysisName, **parameters):
        return createffc.CreateFfc(
            simple_merfish_data, parameters=parameters,
            analysisName=analysisName)._select_fovs()

    assert select('selAll') == allFovs  # default 300 exceeds the fov count
    first = select('selSeedA', fov_count=1, random_seed=1)
    assert first == select('selSeedB', fov_count=1, random_seed=1)
    assert len(first) == 1
    assert set(first) <= set(allFovs)


def test_dark_frame_saved_and_subtracted(simple_merfish_data):
    task = _run_ffc_task(
        simple_merfish_data, 'createFfcDark', dark_frame_indices=[0, 1],
        dark_fov_count=2)

    dark = task.get_dark_frame()
    assert dark.shape == tuple(simple_merfish_data.get_image_dimensions())

    dataOrganization = simple_merfish_data.get_data_organization()
    channel = dataOrganization.get_data_channels()[0]
    fovs = simple_merfish_data.get_fovs()[:2]
    expected = np.mean([
        simple_merfish_data.load_image(
            dataOrganization.get_image_filename(channel, f), i)
        for f in fovs for i in (0, 1)], axis=0)
    np.testing.assert_array_almost_equal(dark, expected, decimal=3)


def test_no_dark_frame_without_indices(simple_merfish_data):
    task = _run_ffc_task(simple_merfish_data, 'createFfcNoDark')
    assert task.get_dark_frame() is None


def test_field_is_normalized_to_median_one(simple_merfish_data):
    task = _run_ffc_task(
        simple_merfish_data, 'createFfcMedian', minimum_value=0.0)
    field = task.get_ffc_field(
        simple_merfish_data.get_data_organization().get_data_channel_color(0))
    assert np.median(field) == pytest.approx(1.0, rel=1e-3)
