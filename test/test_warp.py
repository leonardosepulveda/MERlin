import pytest

from merlin.analysis import warp


def _make_restricted_warp_task(ragged_merfish_data, analysisName):
    # edge_width_to_remove/percentile_pixel_to_keep relaxed from their
    # defaults, matching conftest.ragged_warp_task -- the tiny synthetic
    # ragged test images are much smaller than the default edge crop.
    return warp.FiducialCorrelationWarp(
        ragged_merfish_data,
        parameters={
            'edge_width_to_remove': 0,
            'percentile_pixel_to_keep': 100,
            'channels_to_process': ['bit1', 'bit2'],
        },
        analysisName=analysisName)


def test_warp_channels_to_process_default_processes_every_channel(
        ragged_warp_task, ragged_merfish_data):
    dataOrg = ragged_merfish_data.get_data_organization()
    allChannels = list(dataOrg.get_data_channels())
    assert ragged_warp_task._channels_to_process() == allChannels
    # every channel's transformation is readable -- no restriction applied
    for channel in allChannels:
        ragged_warp_task.get_transformation(0, channel)


def test_warp_channels_to_process_restricts_computation(ragged_merfish_data):
    dataOrg = ragged_merfish_data.get_data_organization()
    task = _make_restricted_warp_task(ragged_merfish_data, 'restrictedWarp')
    task.save()

    expectedChannels = [dataOrg.get_data_channel_index('bit1'),
                        dataOrg.get_data_channel_index('bit2')]
    assert task._channels_to_process() == expectedChannels

    task._run_analysis(0)

    # the processed channels resolve to real (non-placeholder) transforms
    for channel in expectedChannels:
        task.get_transformation(0, channel)

    # a configured-but-unprocessed channel is guarded, not silently
    # returned as an identity placeholder
    otherChannel = dataOrg.get_data_channel_index('bit3')
    with pytest.raises(ValueError):
        task.get_transformation(0, otherChannel)
    with pytest.raises(ValueError):
        task.get_aligned_image(0, otherChannel, 0)
