import os
import shutil
import numpy as np
import pytest

import merlin
from merlin.core import dataset
from merlin.data import dataorganization


def test_dataorganization_get_channels(simple_merfish_data):
    assert np.array_equal(
            simple_merfish_data.get_data_organization().get_data_channels(),
            np.arange(18))


def test_dataorganization_get_channel_name(simple_merfish_data):
    for i in range(16):
        assert simple_merfish_data.get_data_organization()\
                .get_data_channel_name(i) == 'bit' + str(i+1)

    assert simple_merfish_data.get_data_organization()\
        .get_data_channel_name(16) == 'DAPI'
    assert simple_merfish_data.get_data_organization()\
        .get_data_channel_name(17) == 'polyT'


def test_dataorganization_get_channel_index(simple_merfish_data):
    for i in range(16):
        assert simple_merfish_data.get_data_organization() \
            .get_data_channel_index('bit' + str(i+1)) == i

    assert simple_merfish_data.get_data_organization() \
        .get_data_channel_index('DAPI') == 16
    assert simple_merfish_data.get_data_organization() \
        .get_data_channel_index('polyT') == 17


def test_dataorganization_get_fovs(simple_merfish_data):
    assert np.array_equal(
            simple_merfish_data.get_data_organization().get_fovs(),
            np.arange(2))


def test_dataorganization_get_z_positions(simple_merfish_data):
    assert np.array_equal(
            simple_merfish_data.get_data_organization().get_z_positions(),
            np.array([0]))


def test_dataorganization_get_fiducial_information(simple_merfish_data):
    data = simple_merfish_data.get_data_organization()
    for d in data.get_data_channels():
        assert data.get_fiducial_frame_index(d) == 2
    assert os.path.normpath(data.get_fiducial_filename(0, 0)) \
        == os.path.normpath(
        os.path.abspath('test_data/merfish_test/test_0_0.tif'))
    assert os.path.normpath(data.get_fiducial_filename(0, 1)) \
        == os.path.normpath(
        os.path.abspath('test_data/merfish_test/test_1_0.tif'))
    assert os.path.normpath(data.get_fiducial_filename(1, 1)) \
        == os.path.normpath(
        os.path.abspath('test_data/merfish_test/test_1_0.tif'))
    assert os.path.normpath(data.get_fiducial_filename(2, 1)) \
        == os.path.normpath(
        os.path.abspath('test_data/merfish_test/test_1_1.tif'))


def test_dataorganization_get_image_information(simple_merfish_data):
    data = simple_merfish_data.get_data_organization()
    assert data.get_image_frame_index(0, 0) == 1
    assert data.get_image_frame_index(1, 0) == 0
    assert data.get_image_frame_index(16, 0) == 3
    assert os.path.normpath(data.get_image_filename(0, 0)) \
        == os.path.normpath(
        os.path.abspath('test_data/merfish_test/test_0_0.tif'))
    assert os.path.normpath(data.get_image_filename(0, 1)) \
        == os.path.normpath(
        os.path.abspath('test_data/merfish_test/test_1_0.tif'))
    assert os.path.normpath(data.get_image_filename(1, 1)) \
        == os.path.normpath(
        os.path.abspath('test_data/merfish_test/test_1_0.tif'))
    assert os.path.normpath(data.get_image_filename(2, 1)) \
        == os.path.normpath(
        os.path.abspath('test_data/merfish_test/test_1_1.tif'))


def test_dataorganization_load_from_dataset(simple_merfish_data):
    originalOrganization = simple_merfish_data.get_data_organization()
    loadedOrganization = dataorganization.DataOrganization(simple_merfish_data)

    assert np.array_equal(
        originalOrganization.get_data_channels(),
        loadedOrganization.get_data_channels())
    assert np.array_equal(
        originalOrganization.get_fovs(), loadedOrganization.get_fovs())
    assert np.array_equal(
        originalOrganization.get_z_positions(),
        loadedOrganization.get_z_positions())

    for channel in originalOrganization.get_data_channels():
        assert originalOrganization.get_data_channel_name(channel) \
            == loadedOrganization.get_data_channel_name(channel)
        assert originalOrganization.get_fiducial_frame_index(channel) \
            == loadedOrganization.get_fiducial_frame_index(channel)

        for fov in originalOrganization.get_fovs():
            assert originalOrganization.get_fiducial_filename(channel, fov) \
                == loadedOrganization.get_fiducial_filename(channel, fov)
            assert originalOrganization.get_image_filename(channel, fov) \
                == loadedOrganization.get_image_filename(channel, fov)

        for z in originalOrganization.get_z_positions():
            assert originalOrganization.get_image_frame_index(channel, z) \
                == loadedOrganization.get_image_frame_index(channel, z)


def test_dataorganization_get_sequential_rounds(simple_merfish_data):
    dataOrganization = simple_merfish_data.get_data_organization()
    sequentialRounds, sequentialChannels = \
        dataOrganization.get_sequential_rounds()

    assert sequentialRounds == [16, 17]
    assert sequentialChannels == ['DAPI', 'polyT']


def test_dataorganization_get_sequential_rounds_two_codebooks(
        two_codebook_merfish_data):
    dataOrganization = two_codebook_merfish_data.get_data_organization()
    sequentialRounds, sequentialChannels = \
        dataOrganization.get_sequential_rounds()

    assert sequentialRounds == [16, 17]


def test_dataorganization_ragged_get_z_positions_default_unchanged(
        ragged_merfish_data):
    # fov=None must reproduce the exact dataset-wide behavior regardless of
    # any individual fov's raw file, matching every fov's full union
    dataOrg = ragged_merfish_data.get_data_organization()
    assert dataOrg.get_z_positions() == [0, 1, 2, 3]
    assert dataOrg.get_z_positions(fov=None) == [0, 1, 2, 3]
    assert dataOrg.get_z_positions_segmentation() == [0, 1, 2]


def test_dataorganization_ragged_get_z_positions_per_fov(ragged_merfish_data):
    dataOrg = ragged_merfish_data.get_data_organization()
    # fov 0: full depth in every round
    assert dataOrg.get_z_positions(0) == [0, 1, 2, 3]
    # fov 1: round 0 (bit1/bit2) full, round 1 (bit3/bit4) truncated to
    # z=[0,1] -- the overall available range is the intersection across
    # both rounds, not just the shallower round considered in isolation
    assert dataOrg.get_z_positions(1) == [0, 1]
    # fov 2: both bit rounds truncated uniformly to z=[0,1,2]
    assert dataOrg.get_z_positions(2) == [0, 1, 2]
    # fov 3: bit rounds are full depth; only the segmentation round is
    # truncated for this fov (see test below), so regular z is unaffected
    assert dataOrg.get_z_positions(3) == [0, 1, 2, 3]


def test_dataorganization_ragged_get_z_positions_segmentation_per_fov(
        ragged_merfish_data):
    dataOrg = ragged_merfish_data.get_data_organization()
    # fov 0-2: segmentation (DAPI/polyT) round is full depth for all three,
    # even though fov 1 and 2 have truncated *regular* channels -- the two
    # z lists are derived independently of each other
    assert dataOrg.get_z_positions_segmentation(0) == [0, 1, 2]
    assert dataOrg.get_z_positions_segmentation(1) == [0, 1, 2]
    assert dataOrg.get_z_positions_segmentation(2) == [0, 1, 2]
    # fov 3: segmentation round truncated to z=[0,1], independent of the
    # regular z list (which is full depth for this fov)
    assert dataOrg.get_z_positions_segmentation(3) == [0, 1]


def test_dataorganization_ragged_validate_file_map_requires_flag(
        ragged_merfish_files, tmp_path):
    with pytest.raises(dataorganization.InputDataError):
        dataset.MERFISHDataSet(
            'ragged_merfish_test',
            dataOrganizationName='test_data_organization_ragged.csv',
            codebookNames=['test_codebook.csv'],
            positionFileName='test_positions_ragged.csv',
            analysisHome=str(tmp_path / 'strict'),
            microscopeParametersName='test_microscope_parameters.json',
            allowRaggedZStacks=False)


def test_dataorganization_ragged_validate_file_map_tolerates_with_flag(
        ragged_merfish_files, tmp_path):
    with pytest.warns(UserWarning):
        raggedData = dataset.MERFISHDataSet(
            'ragged_merfish_test',
            dataOrganizationName='test_data_organization_ragged.csv',
            codebookNames=['test_codebook.csv'],
            positionFileName='test_positions_ragged.csv',
            analysisHome=str(tmp_path / 'lenient'),
            microscopeParametersName='test_microscope_parameters.json',
            allowRaggedZStacks=True)

    assert raggedData.get_data_organization().get_z_positions(1) == [0, 1]


def test_dataorganization_missing_type_map_image_files_requires_flag(
        merfish_files, tmp_path):
    # test_data_organization_missing_type.csv is test_data_organization.csv
    # plus one channel (bitMissingType) whose imageType/regex matches no
    # raw file anywhere -- e.g. a decode round that hasn't been imaged yet
    # at all. This is the earlier of the two gates a fully-unacquired
    # channel can hit (_map_image_files, before a file map even exists).
    with pytest.raises(dataset.DataFormatException):
        dataset.MERFISHDataSet(
            'merfish_test',
            dataOrganizationName='test_data_organization_missing_type.csv',
            codebookNames=['test_codebook.csv'],
            positionFileName='test_positions.csv',
            analysisHome=str(tmp_path / 'strict'),
            microscopeParametersName='test_microscope_parameters.json',
            allowMissingChannels=False)


def test_dataorganization_missing_type_map_image_files_tolerates_with_flag(
        merfish_files, tmp_path):
    with pytest.warns(UserWarning):
        missingTypeData = dataset.MERFISHDataSet(
            'merfish_test',
            dataOrganizationName='test_data_organization_missing_type.csv',
            codebookNames=['test_codebook.csv'],
            positionFileName='test_positions.csv',
            analysisHome=str(tmp_path / 'lenient'),
            microscopeParametersName='test_microscope_parameters.json',
            allowMissingChannels=True)

    dataOrg = missingTypeData.get_data_organization()
    # the channel is still configured (allowMissingChannels only affects
    # whether its raw files were mapped, not the channel list)
    assert len(dataOrg.get_data_channels()) == 19
    # an existing, actually-acquired channel is unaffected
    assert list(dataOrg.get_z_positions(0)) == [0]
    # actually requesting the unmapped channel's file still raises --
    # allowMissingChannels defers the error to first use, it doesn't
    # fabricate data for a channel that was never imaged
    missingTypeIndex = dataOrg.get_data_channel_index('bitMissingType')
    with pytest.raises(IndexError):
        dataOrg.get_image_filename(missingTypeIndex, 0)


def test_dataorganization_missing_round_validate_file_map_requires_flag(
        merfish_files, tmp_path):
    # test_data_organization_missing_channel.csv is test_data_organization.csv
    # plus one channel (bitMissingRound) sharing a real imageType/regex with
    # other channels, but whose specific imagingRound has no matching file
    # for any fov. This is the later of the two gates a fully-unacquired
    # channel can hit (_validate_file_map, once a file map already exists).
    with pytest.raises(FileNotFoundError):
        dataset.MERFISHDataSet(
            'merfish_test',
            dataOrganizationName='test_data_organization_missing_channel.csv',
            codebookNames=['test_codebook.csv'],
            positionFileName='test_positions.csv',
            analysisHome=str(tmp_path / 'strict'),
            microscopeParametersName='test_microscope_parameters.json',
            allowMissingChannels=False)


def test_dataorganization_missing_round_validate_file_map_tolerates_with_flag(
        merfish_files, tmp_path):
    # bitMissingRound's raw file is absent for both fovs -- confirm that's
    # reported as a single aggregated warning (see _validate_file_map),
    # not one warning per (data channel, fov) combination.
    with pytest.warns(UserWarning) as recorded:
        missingRoundData = dataset.MERFISHDataSet(
            'merfish_test',
            dataOrganizationName='test_data_organization_missing_channel.csv',
            codebookNames=['test_codebook.csv'],
            positionFileName='test_positions.csv',
            analysisHome=str(tmp_path / 'lenient'),
            microscopeParametersName='test_microscope_parameters.json',
            allowMissingChannels=True)
    assert len(recorded) == 1
    assert '2' in str(recorded[0].message)

    dataOrg = missingRoundData.get_data_organization()
    assert len(dataOrg.get_data_channels()) == 19
    assert list(dataOrg.get_z_positions(0)) == [0]
    missingRoundIndex = dataOrg.get_data_channel_index('bitMissingRound')
    with pytest.raises(IndexError):
        dataOrg.get_image_filename(missingRoundIndex, 0)


def test_dataorganization_corrupt_file_validate_file_map_requires_flag(
        corrupt_merfish_files, tmp_path):
    # test_0_0.tif (bit1, fov 0) exists and resolves to a real path, but
    # its content isn't a valid tiff (see corrupt_merfish_files) -- this is
    # the third gate: a channel/fov that passes _get_image_path and the
    # .exists() check but fails when _validate_file_map actually opens it
    # to read its image_stack_size, e.g. because the raw file is still
    # being written by the acquisition software.
    with pytest.raises(dataorganization.InputDataError):
        dataset.MERFISHDataSet(
            'corrupt_merfish_test',
            dataOrganizationName='test_data_organization.csv',
            codebookNames=['test_codebook.csv'],
            positionFileName='test_positions.csv',
            analysisHome=str(tmp_path / 'strict'),
            microscopeParametersName='test_microscope_parameters.json',
            allowMissingChannels=False)


def test_dataorganization_corrupt_file_validate_file_map_tolerates_with_flag(
        corrupt_merfish_files, tmp_path):
    with pytest.warns(UserWarning):
        corruptData = dataset.MERFISHDataSet(
            'corrupt_merfish_test',
            dataOrganizationName='test_data_organization.csv',
            codebookNames=['test_codebook.csv'],
            positionFileName='test_positions.csv',
            analysisHome=str(tmp_path / 'lenient'),
            microscopeParametersName='test_microscope_parameters.json',
            allowMissingChannels=True)

    dataOrg = corruptData.get_data_organization()
    assert len(dataOrg.get_data_channels()) == 18
    # an unaffected fov for the same channel, and other channels/fovs
    # entirely, are unaffected
    assert list(dataOrg.get_z_positions(1)) == [0]
    # unlike the missing-channel/missing-round cases, the corrupt file does
    # exist and its path is still resolvable -- allowMissingChannels only
    # skips validating it, it doesn't remove it from the file map
    bit1Index = dataOrg.get_data_channel_index('bit1')
    imagePath = dataOrg.get_image_filename(bit1Index, 0)
    assert os.path.basename(imagePath) == 'test_0_0.tif'
    # but actually reading it still fails -- allowMissingChannels defers
    # the error to first use, it doesn't fabricate a frame count for a
    # file that was never successfully read
    with pytest.raises(Exception):
        corruptData.image_stack_size(imagePath)


def test_dataorganization_recalculate_filemap_picks_up_new_files(
        merfish_files, tmp_path):
    # Simulates the two-phase workflow allowMissingChannels exists for:
    # construct once while bitMissingType's round hasn't been imaged yet
    # (caching a file map that doesn't cover it), "image" it by dropping a
    # matching raw file into place, then confirm a plain rerun keeps
    # reusing the stale cached file map while recalculateFileMap rebuilds
    # it and picks up the new file.
    analysisHome = str(tmp_path / 'recalc')
    datasetKwargs = dict(
        dataOrganizationName='test_data_organization_missing_type.csv',
        codebookNames=['test_codebook.csv'],
        positionFileName='test_positions.csv',
        analysisHome=analysisHome,
        microscopeParametersName='test_microscope_parameters.json')

    with pytest.warns(UserWarning):
        dataset.MERFISHDataSet(
            'merfish_test', allowMissingChannels=True, **datasetKwargs)

    merfishDataDirectory = os.path.join(merlin.DATA_HOME, 'merfish_test')
    newFiles = [os.path.join(merfishDataDirectory, 'missingtype_0_97.tif'),
               os.path.join(merfishDataDirectory, 'missingtype_1_97.tif')]
    try:
        for newFile in newFiles:
            shutil.copy(
                os.path.join(merfishDataDirectory, 'test_0_7.tif'), newFile)

        # a plain rerun against the same analysisHome reuses the stale
        # cached file map -- the newly "imaged" round still isn't seen
        staleData = dataset.MERFISHDataSet('merfish_test', **datasetKwargs)
        missingTypeIndex = staleData.get_data_organization()\
            .get_data_channel_index('bitMissingType')
        with pytest.raises(IndexError):
            staleData.get_data_organization().get_image_filename(
                missingTypeIndex, 0)

        # recalculateFileMap rebuilds it, picking up the new file --
        # allowMissingChannels is no longer even needed
        freshData = dataset.MERFISHDataSet(
            'merfish_test', recalculateFileMap=True, **datasetKwargs)
        assert freshData.get_data_organization().get_image_filename(
            missingTypeIndex, 0) == newFiles[0]
    finally:
        for newFile in newFiles:
            os.remove(newFile)
