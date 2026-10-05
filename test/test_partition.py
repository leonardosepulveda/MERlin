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
