import numpy as np
import pandas
import pytest

from merlin.analysis.exportbarcodes import ExportBarcodes

COLUMNS = ['barcode_id', 'global_x', 'fov']


def _fov_barcodes(fov, count):
    return pandas.DataFrame({
        'barcode_id': np.arange(count, dtype=np.uint16) % 4,
        'global_x': np.arange(count, dtype=np.float32) + 100 * fov,
        'fov': np.full(count, fov, dtype=np.uint16)})


class _FakeBarcodeDB:

    def __init__(self, perFov):
        self._perFov = perFov

    def get_barcodes(self, fov=None, columnList=None):
        return self._perFov[fov][columnList]


class _FakeCodebook:

    def get_coding_indexes(self):
        return [0, 1]


class _FakeFilterTask:

    def __init__(self, perFov):
        self._db = _FakeBarcodeDB(perFov)

    def get_barcode_database(self):
        return self._db

    def get_codebook(self):
        return _FakeCodebook()


def _run_export(dataSet, monkeypatch, perFov, excludeBlanks, name,
                outputFormat='parquet'):
    monkeypatch.setattr(dataSet, 'get_fovs', lambda: sorted(perFov))
    monkeypatch.setattr(dataSet, 'load_analysis_task',
                        lambda taskName: _FakeFilterTask(perFov))
    task = ExportBarcodes(dataSet, {'filter_task': 'FakeFilter',
                                    'columns': COLUMNS,
                                    'exclude_blanks': excludeBlanks,
                                    'format': outputFormat},
                          analysisName=name)
    task._run_analysis()
    if outputFormat == 'csv':
        return task
    return dataSet.load_dataframe_from_parquet('barcodes', task)


@pytest.mark.parametrize('excludeBlanks', [False, True])
def test_export_barcodes_matches_all_fovs_combined(
        simple_merfish_data, monkeypatch, excludeBlanks):
    """Writing one fov at a time gives the same table as combining every
    fov first, including an empty fov in the middle."""
    perFov = {0: _fov_barcodes(0, 7),
              1: pandas.DataFrame(columns=COLUMNS),
              2: _fov_barcodes(2, 5)}
    exported = _run_export(simple_merfish_data, monkeypatch, perFov,
                           excludeBlanks,
                           'ExportBarcodesTest%d' % excludeBlanks)

    expected = pandas.concat([perFov[0], perFov[2]], ignore_index=True)
    if excludeBlanks:
        expected = expected[expected['barcode_id'].isin([0, 1])]
    pandas.testing.assert_frame_equal(exported,
                                      expected.reset_index(drop=True))


def test_export_barcodes_no_barcodes_writes_empty_table(
        simple_merfish_data, monkeypatch):
    perFov = {0: pandas.DataFrame(columns=COLUMNS),
              1: pandas.DataFrame(columns=COLUMNS)}
    exported = _run_export(simple_merfish_data, monkeypatch, perFov, False,
                           'ExportBarcodesEmptyTest')

    assert len(exported) == 0
    assert list(exported.columns) == COLUMNS


@pytest.mark.parametrize('excludeBlanks', [False, True])
def test_export_barcodes_csv_matches_single_write(
        simple_merfish_data, monkeypatch, tmp_path, excludeBlanks):
    """The csv format, written one fov at a time, is byte-identical to
    writing all fovs combined in one to_csv call."""
    perFov = {0: _fov_barcodes(0, 7),
              1: pandas.DataFrame(columns=COLUMNS),
              2: _fov_barcodes(2, 5)}
    task = _run_export(simple_merfish_data, monkeypatch, perFov,
                       excludeBlanks, 'ExportBarcodesCsvTest%d' % excludeBlanks,
                       outputFormat='csv')

    combined = pandas.concat(list(perFov.values()), sort=False)
    if excludeBlanks:
        combined = combined[combined['barcode_id'].isin([0, 1])]
    expectedPath = tmp_path / 'expected.csv'
    combined.to_csv(expectedPath, index=False)
    exportedPath = simple_merfish_data._analysis_result_save_path(
        'barcodes', task, None, None, '.csv')
    with open(exportedPath) as exported, open(expectedPath) as expected:
        assert exported.read() == expected.read()


def test_export_barcodes_rejects_unknown_format(simple_merfish_data):
    with pytest.raises(ValueError):
        ExportBarcodes(simple_merfish_data,
                       {'filter_task': 'FakeFilter', 'format': 'tsv'},
                       analysisName='ExportBarcodesBadFormatTest')
