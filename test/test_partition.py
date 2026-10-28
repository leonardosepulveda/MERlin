import numpy as np
import pandas
import pytest

from merlin.analysis.partition import PartitionBarcodes


class _FakeCombineTask:

    def get_fov_z_offsets(self):
        return pandas.Series([1.0, -0.5], index=pandas.Index([0, 1],
                                                              name='fov'))


class _FakeAssignmentTask:

    def __init__(self, combineName):
        self.parameters = {'combine_cleaning_task': combineName}


def _partition(dataSet, monkeypatch, apply=True):
    monkeypatch.setattr(dataSet, 'load_analysis_task',
                        lambda name: _FakeCombineTask())
    monkeypatch.setattr(dataSet, 'get_z_positions',
                        lambda fov=None: [0.0, 0.5, 1.0, 1.5])
    return PartitionBarcodes(dataSet, {'filter_task': 'F',
                                       'assignment_task': 'A',
                                       'alignment_task': 'G',
                                       'apply_fov_z_offsets': apply},
                             analysisName='partition_z_shift')


def test_z_shift_moves_neighbour_barcodes_into_fov_planes(
        simple_merfish_data, monkeypatch):
    task = _partition(simple_merfish_data, monkeypatch)
    shift = task._z_shift_to_fov(_FakeAssignmentTask('Combine'), 0,
                                 np.array([0, 1, 1, 7]))
    # (offset of the barcode's fov - offset of fov 0) / 0.5 um planes; a
    # fov without an offset counts as 0
    assert shift == pytest.approx([0, -3, -3, -2])


@pytest.mark.parametrize('apply, combineName', [(False, 'Combine'),
                                                (True, None)])
def test_z_shift_is_zero_without_offsets(simple_merfish_data, monkeypatch,
                                         apply, combineName):
    task = _partition(simple_merfish_data, monkeypatch, apply)
    shift = task._z_shift_to_fov(_FakeAssignmentTask(combineName), 0,
                                 np.array([0, 1]))
    assert shift == pytest.approx([0, 0])


def test_z_shift_is_zero_for_an_older_combine_run(simple_merfish_data,
                                                 monkeypatch):
    task = _partition(simple_merfish_data, monkeypatch)

    class _OldCombineTask:
        def get_fov_z_offsets(self):
            raise FileNotFoundError

    monkeypatch.setattr(simple_merfish_data, 'load_analysis_task',
                        lambda name: _OldCombineTask())
    with pytest.warns(UserWarning, match='fov_z_offsets'):
        shift = task._z_shift_to_fov(_FakeAssignmentTask('Combine'), 0,
                                     np.array([0, 1]))
    assert shift == pytest.approx([0, 0])


# The per-cell loops PartitionBarcodes used before count_traditional and
# count_with_boundary_correction, kept as the reference for their output.

def _reference_traditional(cells, barcodes, barcodeCount):
    countsDF = pandas.DataFrame(
        data=np.zeros((len(cells), barcodeCount)),
        columns=range(barcodeCount),
        index=[x.get_feature_id() for x in cells])
    for cell in cells:
        contained = cell.contains_positions(
            barcodes.loc[:, ['global_x', 'global_y', 'z']].values)
        count = barcodes[contained].groupby('barcode_id').size()
        count = count.reindex(range(barcodeCount), fill_value=0)
        countsDF.loc[cell.get_feature_id(), :] = count.values.tolist()
    return countsDF


def _reference_boundary_correction(cells, barcodes, barcodeCount,
                                   bufferSize):
    index = [x.get_feature_id() for x in cells]
    countsDF = pandas.DataFrame(np.zeros((len(cells), barcodeCount)),
                                columns=range(barcodeCount), index=index)
    countsShrunk = countsDF.copy()
    expandedCells = [c.make_a_buffered_copy(bufferSize) for c in cells]
    shrunkCells = [c.make_a_buffered_copy(-bufferSize) for c in cells]
    n = barcodes.shape[0]
    hostNormal = [[] for _ in range(n)]
    hostShrunk = [[] for _ in range(n)]
    hostExpanded = [[] for _ in range(n)]
    positions = barcodes.loc[:, ['global_x', 'global_y', 'z']].values
    for i in range(len(cells)):
        inNormal = cells[i].contains_positions(positions)
        inShrunk = shrunkCells[i].contains_positions(positions)
        inExpanded = expandedCells[i].contains_positions(positions)
        for j in np.where(inNormal)[0]:
            hostNormal[j].append(i)
        for j in np.where(inShrunk)[0]:
            hostShrunk[j].append(i)
        for j in np.where(inExpanded)[0]:
            hostExpanded[j].append(i)
        count = barcodes[inShrunk].groupby('barcode_id').size()
        count = count.reindex(range(barcodeCount), fill_value=0)
        countsShrunk.loc[shrunkCells[i].get_feature_id(), :] = \
            count.values.tolist()

    types = list(barcodes['barcode_id'])
    for i in range(n):
        if len(hostExpanded[i]) == 0:
            continue
        t = types[i]
        if len(hostShrunk[i]) > 0:
            countsDF.iloc[hostShrunk[i][0], t] += 1
            continue
        filtered = [c for c in hostExpanded[i]
                    if countsShrunk.iloc[c, t] > 0]
        if len(hostNormal[i]) > 0:
            c0 = hostNormal[i][0]
            if countsShrunk.iloc[c0, t] > 0:
                countsDF.iloc[c0, t] += 1
            elif len(filtered) > 0:
                countsDF.iloc[np.random.choice(filtered), t] += 1
            else:
                countsDF.iloc[c0, t] += 1
            continue
        if len(filtered) > 0:
            countsDF.iloc[np.random.choice(filtered), t] += 1
    return countsDF


def _synthetic_cells_and_barcodes(seed):
    """Overlapping cells over 4 planes (one plane empty in some cells,
    one cell split in two parts on a plane) and barcodes around them:
    inside, on edges, in the buffer zones, outside, with fractional z,
    z exactly between planes and z outside the stack."""
    from shapely import geometry
    from merlin.util import spatialfeature
    rng = np.random.default_rng(seed)
    cells = []
    for k in range(12):
        cx, cy = rng.uniform(0, 20, 2)
        r = rng.uniform(1.5, 4)
        planes = []
        for z in range(4):
            if rng.random() < 0.15:
                planes.append([])
            elif rng.random() < 0.15:
                planes.append([geometry.Point(cx - r, cy).buffer(r / 2),
                               geometry.Point(cx + r, cy).buffer(r / 2)])
            else:
                planes.append([geometry.box(cx - r, cy - r, cx + r, cy + r)
                               if k % 2 else
                               geometry.Point(cx, cy).buffer(r)])
        cells.append(spatialfeature.SpatialFeature(
            planes, 0, np.arange(4, dtype=float), uniqueID=k))
    n = 3000
    x = rng.uniform(-3, 23, n)
    y = rng.uniform(-3, 23, n)
    # snap some barcodes to box edges and corners, which are not inside
    x[:200] = np.round(x[:200])
    y[:100] = np.round(y[:100])
    z = rng.choice([0, 1, 2, 3, 0.5, 1.5, 2.49, 2.51, -0.6, 3.6, 4.2],
                   n) + rng.normal(0, 0.05, n) * (rng.random(n) < 0.5)
    barcodes = pandas.DataFrame({
        'barcode_id': rng.integers(0, 5, n), 'global_x': x,
        'global_y': y, 'z': z})
    return cells, barcodes


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_count_traditional_matches_per_cell_loop(seed):
    from merlin.analysis import partition
    cells, barcodes = _synthetic_cells_and_barcodes(seed)
    zBefore = barcodes['z'].values.copy()
    new = partition.count_traditional(cells, barcodes, 5)
    pandas.testing.assert_frame_equal(
        new, _reference_traditional(cells, barcodes, 5))
    assert new.values.sum() > 0
    assert np.array_equal(barcodes['z'].values, zBefore)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_count_with_boundary_correction_matches_per_barcode_loop(seed):
    from merlin.analysis import partition
    cells, barcodes = _synthetic_cells_and_barcodes(seed)
    np.random.seed(seed)
    new = partition.count_with_boundary_correction(cells, barcodes, 5, 0.5)
    randomAfterNew = np.random.random()
    np.random.seed(seed)
    old = _reference_boundary_correction(cells, barcodes, 5, 0.5)
    # the same number of random draws, so the same random choices
    assert np.random.random() == randomAfterNew
    pandas.testing.assert_frame_equal(new, old)
    assert new.values.sum() > 0


def test_contained_barcodes_with_no_cells_or_barcodes():
    from merlin.analysis import partition
    cells, barcodes = _synthetic_cells_and_barcodes(0)
    b, c = partition.contained_barcodes([], *[barcodes[k].values for k in
                                              ('global_x', 'global_y', 'z')])
    assert len(b) == len(c) == 0
    counts = partition.count_with_boundary_correction(
        cells, barcodes.iloc[:0], 5, 0.5)
    assert counts.shape == (len(cells), 5) and counts.values.sum() == 0


def test_export_partitioned_barcodes_adds_cell_depth(simple_merfish_data,
                                                     monkeypatch):
    """Cell ids read back from csv are ints; depth is matched to them by
    id, not by row order."""
    from unittest import mock
    counts = pandas.DataFrame({'GeneA': [1, 2]}, index=[30, 10])
    pTask = mock.Mock(parameters={'assignment_task': 'FakeSegment'})
    pTask.get_partitioned_barcodes.return_value = counts
    depthTask = mock.Mock()
    depthTask.get_cell_depths.return_value = pandas.Series(
        {'10': 1.5, '30': 3.5})
    tasks = {'FakePartition': pTask, 'FakeSegment': mock.Mock(),
             'FakeDepth': depthTask}
    monkeypatch.setattr(simple_merfish_data, 'load_analysis_task',
                        tasks.__getitem__)
    from merlin.analysis.partition import ExportPartitionedBarcodes
    task = ExportPartitionedBarcodes(
        simple_merfish_data, {'partition_task': 'FakePartition',
                              'depth_task': 'FakeDepth'},
        analysisName='ExportPartitionedDepthTest')
    assert task.get_dependencies() == ['FakePartition', 'FakeDepth']
    task._run_analysis()
    exported = simple_merfish_data.load_dataframe_from_csv(
        'barcodes_per_feature', task.get_analysis_name(), index_col=0)
    assert exported.loc[30, 'depth'] == 3.5
    assert exported.loc[10, 'depth'] == 1.5
