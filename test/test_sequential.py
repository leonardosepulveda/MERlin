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
