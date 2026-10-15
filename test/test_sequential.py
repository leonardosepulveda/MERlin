import numpy as np
import pytest

from merlin.core import analysistask
from merlin.analysis import globalalign
from merlin.analysis import sequential


@pytest.fixture(scope='module')
def sum_signal_global_align_task(simple_merfish_data):
    # SumSignal.__init__ loads this by name via load_analysis_task, which
    # requires a saved task.json on disk
    task = globalalign.SimpleGlobalAlignment(
        simple_merfish_data, parameters={},
        analysisName='sumSignalTestGlobalAlign')
    task.save()
    return task


def _make_sum_signal(simple_merfish_data, sum_signal_global_align_task,
                      channel_names=None, analysisName=None):
    parameters = {'global_align_task': sum_signal_global_align_task.analysisName}
    if channel_names is not None:
        parameters['channel_names'] = channel_names
    return sequential.SumSignal(
        simple_merfish_data, parameters=parameters, analysisName=analysisName)


def test_sumsignal_channel_names_default_selects_all(
        simple_merfish_data, sum_signal_global_align_task):
    task = _make_sum_signal(simple_merfish_data, sum_signal_global_align_task)
    assert task.parameters['channel_names'] is None

    channels, geneNames = simple_merfish_data.get_data_organization()\
        .get_sequential_rounds()
    selectedChannels, selectedNames = task._select_channels(
        channels, geneNames)
    assert selectedChannels == channels
    assert selectedNames == geneNames


def test_sumsignal_channel_names_filters_requested_subset(
        simple_merfish_data, sum_signal_global_align_task):
    channels, geneNames = simple_merfish_data.get_data_organization()\
        .get_sequential_rounds()
    assert 'DAPI' in geneNames and 'polyT' in geneNames

    task = _make_sum_signal(
        simple_merfish_data, sum_signal_global_align_task,
        channel_names=['DAPI'])
    selectedChannels, selectedNames = task._select_channels(
        channels, geneNames)
    assert selectedNames == ['DAPI']
    assert selectedChannels == [channels[geneNames.index('DAPI')]]


def test_sumsignal_channel_names_invalid_entry_raises(
        simple_merfish_data, sum_signal_global_align_task):
    with pytest.raises(analysistask.InvalidParameterException):
        _make_sum_signal(
            simple_merfish_data, sum_signal_global_align_task,
            channel_names=['not_a_real_channel'])


def _grid_cells(cellCount, zCount=1, side=4, pitch=6, columns=20):
    """Square cells on a grid, each with the same boundary on every plane,
    in fov pixel coordinates."""
    from shapely import geometry
    from merlin.util import spatialfeature
    cells = []
    for i in range(cellCount):
        x = (i % columns) * pitch + 1
        y = (i // columns) * pitch + 1
        square = geometry.box(x, y, x + side, y + side)
        cells.append(spatialfeature.SpatialFeature(
            [[square]] * zCount, 0, uniqueID=i))
    return cells


def _identity_sum_signal(parameters=None):
    """A SumSignal whose global alignment maps coordinates to themselves."""
    from unittest import mock
    task = sequential.SumSignal.__new__(sequential.SumSignal)
    task.parameters = parameters or {}
    task.alignTask = mock.Mock()
    task.alignTask.global_coordinates_to_fov = \
        lambda fov, xy: np.array(xy).astype(int)
    return task


def test_sumsignal_keeps_cells_after_the_254th_apart():
    # Each cell's pixels hold its own index + 1, so any merging of labels
    # changes the summed intensities.
    cells = _grid_cells(300)
    task = _identity_sum_signal()
    image = np.zeros((6 * 15 + 2, 6 * 20 + 2), np.float64)
    for i, cell in enumerate(cells):
        minx, miny, maxx, maxy = cell.get_boundaries()[0][0].bounds
        image[int(miny):int(maxy) + 1, int(minx):int(maxx) + 1] = i + 1

    signal = task._extract_signal(cells, image, 0)

    pixels = signal['Pixels'].to_numpy()
    assert (pixels == pixels[0]).all() and pixels[0] > 0
    np.testing.assert_allclose(
        signal['Intensity'].to_numpy(), pixels * np.arange(1, 301))


def test_sumsignal_all_z_sums_each_cell_over_its_own_planes():
    # 300 cells on 3 planes; the last cell has no boundary on plane 2.
    # On plane z, cell i holds (i + 1) * (z + 1) and the background holds
    # 10 * (z + 1).
    cells = _grid_cells(300, zCount=3)
    cells[-1].get_boundaries()[2] = []
    task = _identity_sum_signal({'z_index': 'all'})
    shape = (6 * 15 + 2, 6 * 20 + 2)

    def load_image(channel, z):
        image = np.full(shape, 10.0 * (z + 1))
        for i, cell in enumerate(cells):
            for region in cell.get_boundaries()[z]:
                minx, miny, maxx, maxy = region.bounds
                image[int(miny):int(maxy) + 1, int(minx):int(maxx) + 1] = \
                    (i + 1) * (z + 1)
        return image if channel == 'a' else 2 * image

    signal, background = task._sum_all_z(
        cells, ['a', 'b'], ['A', 'B'], 3, load_image)

    assert list(signal.columns) == [
        'A', 'B', 'A_background', 'B_background', 'Voxels', 'Planes']
    np.testing.assert_allclose(background['A'], [10, 20, 30])
    np.testing.assert_allclose(background['B'], [20, 40, 60])

    perPlane = signal['Voxels'].iloc[0] / 3
    planes = np.array([3] * 299 + [2])
    zSum = np.array([6] * 299 + [3])  # sum of z + 1 over the cell's planes
    np.testing.assert_array_equal(signal['Planes'], planes)
    np.testing.assert_array_equal(signal['Voxels'], perPlane * planes)
    np.testing.assert_allclose(
        signal['A'], perPlane * np.arange(1, 301) * zSum)
    np.testing.assert_allclose(signal['B'], 2 * signal['A'])
    np.testing.assert_allclose(signal['A_background'], perPlane * 10 * zSum)


def test_sumsignal_accepts_z_index_all(
        simple_merfish_data, sum_signal_global_align_task):
    task = sequential.SumSignal(
        simple_merfish_data, parameters={
            'global_align_task': sum_signal_global_align_task.analysisName,
            'z_index': 'all'})
    assert task.allZ


def test_prepare_spot_chunk_keeps_parquet_schema_stable(tmp_path):
    # One z plane has only integer thresholds and no anti-colocalization
    # hits (int -1 placeholder); the next has an automatic float threshold
    # and a real distance. Unfixed, the second write raised
    # "Table schema does not match schema used to create file".
    import pandas
    from merlin.core.dataset import ParquetChunkWriter

    def chunk(thresholds, antiDistance):
        return pandas.DataFrame({
            'geometry': [None] * len(thresholds),
            'x': [1] * len(thresholds),
            'threshold': thresholds,
            'intensity_threshold': thresholds,
            'distance_threshold': [2] * len(thresholds),
            'anti_colocalization_distance': antiDistance,
            'index_right': [-1] * len(thresholds)})

    savePath = str(tmp_path / 'spots.parquet')
    with ParquetChunkWriter(savePath) as writer:
        writer.write(sequential._prepare_spot_chunk(chunk([100, 100], -1)))
        writer.write(sequential._prepare_spot_chunk(
            chunk([100, 4.5], [-1, 3.25])))

    result = pandas.read_parquet(savePath)
    assert len(result) == 4
    assert 'geometry' not in result.columns
    assert list(result['threshold']) == [100.0, 100.0, 100.0, 4.5]
    assert list(result['anti_colocalization_distance']) == [-1, -1, -1, 3.25]
    assert list(result['index_right']) == ['-1'] * 4
