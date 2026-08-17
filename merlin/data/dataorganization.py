import os
import re
import warnings
from typing import List
from typing import Tuple
import pandas
import numpy as np

import merlin
from merlin.core import dataset


def _parse_list(inputString: str, dtype=float):
    if ',' in inputString:
        return np.fromstring(inputString.strip('[] '), dtype=dtype, sep=',')
    else:
        return np.fromstring(inputString.strip('[] '), dtype=dtype, sep=' ')


def _parse_int_list(inputString: str):
    return _parse_list(inputString, dtype=int)


class InputDataError(Exception):
    pass


class DataOrganization(object):

    """
    A class to specify the organization of raw images in the original
    image files.
    """

    def __init__(self, dataSet, filePath: str = None,
                 allowRaggedZStacks: bool = False):
        """
        Create a new DataOrganization for the data in the specified data set.

        If filePath is not specified, a previously stored DataOrganization
        is loaded from the dataSet if it exists. If filePath is specified,
        the DataOrganization at the specified filePath is loaded and
        stored in the dataSet, overwriting any previously stored
        DataOrganization.

        Args:
            allowRaggedZStacks: if True, a fov whose raw file has fewer
                    frames than the deepest globally-configured z position
                    requires is tolerated (its available z range is
                    inferred from its own file's frame count, see
                    get_z_positions) instead of raising InputDataError.
                    Defaults to False to preserve prior behavior for
                    datasets where every fov shares the same z range.
        Raises:
            InputDataError: If the set of raw data is incomplete or the
                    format of the raw data deviates from expectations.
        """

        self._dataSet = dataSet
        self._allowRaggedZStacks = allowRaggedZStacks
        # caches the per-(imageType, imagingRound, fov) raw frame count so
        # repeated get_z_positions(fov) calls during analysis don't re-parse
        # the same file header multiple times
        self._fovFrameCountCache = {}

        if filePath is not None:
            if not os.path.exists(filePath):
                filePath = os.sep.join(
                        [merlin.DATA_ORGANIZATION_HOME, filePath])

            self.data = pandas.read_csv(
                filePath,
                converters={'frame': _parse_int_list,
                            'zPos': _parse_list}
                            )
                            
            # specific for 3d registration
            if 'fiducial3DStackFrames' in self.data.columns:
                self.data['fiducial3DStackFrames'] = self.data['fiducial3DStackFrames'].apply(_parse_int_list)
            if 'fiducial3DzPos' in self.data.columns:
                self.data['fiducial3DzPos'] = self.data['fiducial3DzPos'].apply(_parse_list)

            self.data['readoutName'] = self.data['readoutName'].str.strip()
            self._dataSet.save_dataframe_to_csv(
                    self.data, 'dataorganization', index=False)

        else:
            self.data = self._dataSet.load_dataframe_from_csv(
                'dataorganization',
                converters={'frame': _parse_int_list,
                            'zPos': _parse_list}
                            )

            # specific for 3d registration
            if 'fiducial3DStackFrames' in self.data.columns:
                self.data['fiducial3DStackFrames'] = self.data['fiducial3DStackFrames'].apply(_parse_int_list)
            if 'fiducial3DzPos' in self.data.columns:
                self.data['fiducial3DzPos'] = self.data['fiducial3DzPos'].apply(_parse_list)

        stringColumns = ['readoutName', 'channelName', 'imageType',
                         'imageRegExp', 'fiducialImageType', 'fiducialRegExp'
                         ]
                         
        self.data[stringColumns] = self.data[stringColumns].astype('str')
        
        if 'fiducial3DImageType' in self.data.columns:
                self.data['fiducial3DImageType'] = self.data['fiducial3DImageType'].astype('str')
        if 'fiducial3DRegExp' in self.data.columns:
                self.data['fiducial3DRegExp'] = self.data['fiducial3DRegExp'].astype('str')
        
        self._map_image_files()

    def get_data_channels(self) -> np.array:
        """Get the data channels for the MERFISH data set.

        Returns:
            A list of the data channel indexes
        """
        return np.array(self.data.index)

    def get_data_channel_readout_name(self, dataChannelIndex: int) -> str:
        """Get the name for the data channel with the specified index.

        Args:
            dataChannelIndex: The index of the data channel
        Returns:
            The name of the specified data channel
        """
        return self.data.iloc[dataChannelIndex]['readoutName']

    def get_data_channel_name(self, dataChannelIndex: int) -> str:
        """Get the name for the data channel with the specified index.

        Args:
            dataChannelIndex: The index of the data channel
        Returns:
            The name of the specified data channel,
            primarily relevant for non-multiplex measurements
        """
        return self.data.iloc[dataChannelIndex]['channelName']

    def get_data_channel_index(self, dataChannelName: str) -> int:
        """Get the index for the data channel with the specified name.

        Args:
            dataChannelName: the name of the data channel. The data channel
                name is not case sensitive.
        Returns:
            the index of the data channel where the data channel name is
                dataChannelName
        Raises:
            # TODO this should raise a meaningful exception if the data channel
            # is not found
        """
        return self.data[self.data['channelName'].apply(
            lambda x: str(x).lower()) == str(dataChannelName).lower()]\
            .index.values.tolist()[0]

    def get_data_channel_color(self, dataChannel: int) -> str:
        """Get the color used for measuring the specified data channel.

        Args:
            dataChannel: the data channel index
        Returns:
            the color for this data channel as a string
        """
        return str(self.data.at[dataChannel, 'color'])

    def get_data_channel_for_bit(self, bitName: str) -> int:
        """Get the data channel associated with the specified bit.

        Args:
            bitName: the name of the bit to search for
        Returns:
            The index of the associated data channel
        """
        return self.data[self.data['readoutName'] ==
                         bitName].index.values.item()

    def get_data_channel_with_name(self, channelName: str) -> int:
        """Get the data channel associated with a gene name.

        Args:
            channelName: the name of the gene to search for
        Returns:
            The index of the associated data channel
        """
        return self.data[self.data['channelName'] ==
                         channelName].index.values.item()

    def get_fiducial_filename(self, dataChannel: int, fov: int) -> str:
        """Get the path for the image file that contains the fiducial
        image for the specified dataChannel and fov.

        Args:
            dataChannel: index of the data channel
            fov: index of the field of view
        Returns:
            The full path to the image file containing the fiducials
        """

        imageType = self.data.loc[dataChannel, 'fiducialImageType']
        imagingRound = \
            self.data.loc[dataChannel, 'fiducialImagingRound']
        return self._get_image_path(imageType, fov, imagingRound)

    def get_fiducial_frame_index(self, dataChannel: int) -> int:
        """Get the index of the frame containing the fiducial image
        for the specified data channel.

        Args:
            dataChannel: index of the data channel
        Returns:
            The index of the fiducial frame in the corresponding image file
        """
        return self.data.iloc[dataChannel]['fiducialFrame']

    ### new for 3D registration

    def get_fiducial3D_filename(self, dataChannel: int, fov: int) -> str:
        """Get the path for the image file that contains the 3D fiducial
        image for the specified dataChannel and fov.

        Args:
            dataChannel: index of the data channel
            fov: index of the field of view
        Returns:
            The full path to the image file containing the fiducials
        """

        imageType = self.data.loc[dataChannel, 'fiducial3DImageType']
        imagingRound = \
            self.data.loc[dataChannel, 'fiducial3DImagingRound']
        return self._get_image_path(imageType, fov, imagingRound)
        
    def get_fiducial3D_stack_frame_indices(self, dataChannel: int) -> List[int]:
        """Get the indices of the frames containing the 3D fiducial image stack
        for the specified data channel.

        Args:
            dataChannel: index of the data channel
        Returns:
            The indices of the fiducial frame in the 3D stack
        """
        return self.data.iloc[dataChannel]['fiducial3DStackFrames']
        
    def get_fiducial3D_stack_frame_zPos(self, dataChannel: int) -> List[int]:
        """Get the zpos of the frames containing the 3D fiducial image stack
        for the specified data channel.
        Args:
            dataChannel: index of the data channel
        Returns:
            The zpos of the fiducial frames in 3D stack
        """
        return self.data.iloc[dataChannel]['fiducial3DzPos']

    ###

    def get_image_filename(self, dataChannel: int, fov: int) -> str:
        """Get the path for the image file that contains the
        images for the specified dataChannel and fov.

        Args:
            dataChannel: index of the data channel
            fov: index of the field of view
        Returns:
            The full path to the image file containing the fiducials
        """
        channelInfo = self.data.iloc[dataChannel]
        imagePath = self._get_image_path(
                channelInfo['imageType'], fov, channelInfo['imagingRound'])
        return imagePath

    def get_image_frame_index(self, dataChannel: int, zPosition: float) -> int:
        """Get the index of the frame containing the image
        for the specified data channel and z position.

        Args:
            dataChannel: index of the data channel
            zPosition: the z position
        Returns:
            The index of the frame in the corresponding image file
        """
        channelInfo = self.data.iloc[dataChannel]
        channelZ = channelInfo['zPos']
        if isinstance(channelZ, np.ndarray):
            zIndex = np.where(channelZ == zPosition)[0]
            if len(zIndex) == 0:
                raise Exception('Requested z position not found. Position ' +
                                'z=%0.2f not found for channel %i'
                                % (zPosition, dataChannel))
            else:
                frameIndex = zIndex[0]
        else:
            frameIndex = 0

        frames = channelInfo['frame']
        if isinstance(frames, np.ndarray):
            frame = frames[frameIndex]
        else:
            frame = frames

        return frame

    def get_z_positions(self, fov: int = None) -> List[float]:
        """Get the z positions present in this data organization.

        Args:
            fov: if provided, the z positions are restricted to those
                    actually available for this fov (a fov's raw file may
                    have fewer frames than the deepest z position
                    configured here, e.g. when acquisition was trimmed to
                    this fov's own tissue depth). If None (default), the
                    full, dataset-wide z position list is returned exactly
                    as before, regardless of individual fovs' raw files.
        Returns:
            A sorted list of unique z positions
        """
        # assume channels that contain dapi or polyt in the channelName are for segmentation
        sel = self.data['channelName'].str.contains('dapi|polyt', case = False, regex=True)
        if fov is None:
            zpos = self.data['zPos'][~sel]
            return sorted(np.unique([y for x in zpos for y in x]))
        return self._get_available_z_positions(self.data.index[~sel], fov)

    def get_z_positions_segmentation(self, fov: int = None) -> List[float]:
        """Get the z positions present in this data organization.

        Args:
            fov: if provided, the z positions are restricted to those
                    actually available for this fov (see get_z_positions).
                    If None (default), the full, dataset-wide z position
                    list is returned exactly as before.
        Returns:
            A sorted list of unique z positions
        """
        # assume channels that contain dapi or polyt in the channelName are for segmentation
        sel = self.data['channelName'].str.contains('dapi|polyt', case = False, regex=True)
        if fov is None:
            zpos = self.data['zPos'][sel]
            return sorted(np.unique([y for x in zpos for y in x]))
        return self._get_available_z_positions(self.data.index[sel], fov)

    def _get_available_z_positions(
            self, channelIndices, fov: int) -> List[float]:
        """Get the z positions common to the raw files of the specified
        fov across the specified data channels.

        A z position is considered available for a channel only if that
        channel's raw file for this fov actually contains the frame it
        maps to -- inferred from the file's own frame count, not from a
        separate per-fov metadata table (a fov whose acquisition was
        trimmed to a shallower depth simply has a shorter raw file).
        Different channels can point at different imaging rounds (and
        therefore different raw files) for the same fov, so nothing
        guarantees they were all trimmed to the same depth; the
        intersection across channels is the set of z positions genuinely
        usable by every channel that needs them.

        Args:
            channelIndices: the data channel indices to consider
            fov: index of the field of view
        Returns:
            A sorted list of the z positions available in every channel
        """
        availableZSets = []
        for dataChannel in channelIndices:
            channelInfo = self.data.loc[dataChannel]
            zPosArray = channelInfo['zPos']
            frames = channelInfo['frame']
            if not isinstance(zPosArray, np.ndarray) \
                    or not isinstance(frames, np.ndarray):
                # a channel with a single scalar zPos/frame has no z sweep
                # and so does not constrain which z positions are available
                continue
            frameCount = self._get_fov_frame_count(dataChannel, fov)
            availableZSets.append(set(zPosArray[frames < frameCount].tolist()))

        if not availableZSets:
            # no channel in this selection has a real z sweep to truncate;
            # fall back to the full configured z list for these channels
            zpos = self.data['zPos'].loc[channelIndices]
            return sorted(np.unique([y for x in zpos for y in x]))

        return sorted(set.intersection(*availableZSets))

    def _get_fov_frame_count(self, dataChannel: int, fov: int) -> int:
        """Get the number of frames actually present in the raw file for
        the specified data channel and fov.

        Results are cached per (imageType, imagingRound, fov) since
        multiple data channels commonly share the same underlying raw
        file/round, to avoid re-parsing the same file header repeatedly.

        Args:
            dataChannel: index of the data channel
            fov: index of the field of view
        Returns:
            The number of frames in the corresponding raw image file
        Raises:
            InputDataError: if the frame count cannot be determined, e.g.
                    the raw file is missing or corrupted
        """
        channelInfo = self.data.iloc[dataChannel]
        cacheKey = (channelInfo['imageType'], channelInfo['imagingRound'], fov)
        if cacheKey not in self._fovFrameCountCache:
            imagePath = self._get_image_path(
                channelInfo['imageType'], fov, channelInfo['imagingRound'])
            try:
                self._fovFrameCountCache[cacheKey] = \
                    self._dataSet.image_stack_size(imagePath)[2]
            except Exception as e:
                raise InputDataError(
                    ('Unable to determine image stack size for fov {0} from'
                     ' data channel {1} at {2}')
                    .format(fov, dataChannel, imagePath)) from e
        return self._fovFrameCountCache[cacheKey]

    def get_fovs(self) -> np.ndarray:
        return np.unique(self.fileMap['fov'])

    def get_sequential_rounds(self) -> Tuple[List[int], List[str]]:
        """ Get the rounds that are not present in your codebook

        Returns:
            A tuple of two lists, the first list contains the channel number
            for all the rounds not contained in the codebook, the second list
            contains the name associated with that channel in the data
            organization file.
        """
        multiplexBits = {b for x in self._dataSet.get_codebooks()
                         for b in x.get_bit_names()}
        sequentialChannels = [i for i in self.get_data_channels()
                              if self.get_data_channel_readout_name(i)
                              not in multiplexBits]
        sequentialGeneNames = [self.get_data_channel_name(x) for
                               x in sequentialChannels]
        return sequentialChannels, sequentialGeneNames

    def _get_image_path(
            self, imageType: str, fov: int, imagingRound: int) -> str:
        selection = self.fileMap[(self.fileMap['imageType'] == imageType) &
                                 (self.fileMap['fov'] == fov) &
                                 (self.fileMap['imagingRound'] == imagingRound)]
        filemapPath = selection['imagePath'].values[0]
        return os.path.join(self._dataSet.dataHome, self._dataSet.dataSetName,
                            filemapPath)

    def _truncate_file_path(self, path) -> str:
        # Store the path relative to the raw data directory rather than
        # just the bare filename, so that files in per-round subfolders
        # (e.g. data/hybs/H01/) are preserved rather than collapsed to a
        # name that no longer resolves back to the right file. Paths
        # already relative (e.g. loaded from an older cached filemap) are
        # left as-is.
        rawDataPath = self._dataSet.rawDataPath
        if path == rawDataPath or path.startswith(rawDataPath + os.sep):
            return os.path.relpath(path, rawDataPath)
        return path

    def _map_image_files(self) -> None:
        # TODO: This doesn't map the fiducial image types and currently assumes
        # that the fiducial image types and regular expressions are part of the
        # standard image types.

        try:
            self.fileMap = self._dataSet.load_dataframe_from_csv('filemap')
            self.fileMap['imagePath'] = self.fileMap['imagePath'].apply(
                self._truncate_file_path)

        except FileNotFoundError:
        
            # this should now handle adding fiducial files
            # note that some may get added twice - remove them later
            uniqueTypes = []
            uniqueRegExps = []
            
            # handle case when 3d registration images are present..
            if ('fiducial3DImageType' in self.data.columns) and ('fiducial3DRegExp' in self.data.columns):
                for imtype, imregex in zip(['imageType','fiducialImageType','fiducial3DImageType'],
                                  ['imageRegExp','fiducialRegExp','fiducial3DRegExp']):
                                  
                    uniqueEntries = self.data.drop_duplicates(
                        subset = [imtype, imregex], keep='first')
                                
                    uniqueTypes += list(uniqueEntries[imtype])
                    uniqueRegExps += list(uniqueEntries[imregex])
            else:
                for imtype, imregex in zip(['imageType','fiducialImageType'],
                                  ['imageRegExp','fiducialRegExp']):
                                  
                    uniqueEntries = self.data.drop_duplicates(
                        subset = [imtype, imregex], keep='first')
                                
                    uniqueTypes += list(uniqueEntries[imtype])
                    uniqueRegExps += list(uniqueEntries[imregex])
                

            fileNames = self._dataSet.get_image_file_names()
            if len(fileNames) == 0:
                raise dataset.DataFormatException(
                    'No image files found at %s.' % self._dataSet.rawDataPath)
            fileData = []
            for currentType, currentRegExp in zip(uniqueTypes, uniqueRegExps):
                matchRE = re.compile(currentRegExp)
                matchingFiles = False
                for currentFile in fileNames:
                    matchedName = matchRE.match(os.path.split(currentFile)[-1])
                    if matchedName is not None:
                        transformedName = matchedName.groupdict()
                        if transformedName['imageType'] == currentType:
                            if 'imagingRound' not in transformedName:
                                transformedName['imagingRound'] = -1
                            transformedName['imagePath'] = currentFile
                            matchingFiles = True
                            fileData.append(transformedName)

                if not matchingFiles:
                    raise dataset.DataFormatException(
                        'Unable to identify image files matching regular '
                        + 'expression %s for image type %s.'
                        % (currentRegExp,
                           currentType))
            
            # drop duplicates to remove duplicate file names
            self.fileMap = pandas.DataFrame(fileData).drop_duplicates() 
            
            self.fileMap[['imagingRound', 'fov']] = \
                self.fileMap[['imagingRound', 'fov']].astype(int)
            self.fileMap['imagePath'] = self.fileMap['imagePath'].apply(
                self._truncate_file_path)

            self._validate_file_map()

            self._dataSet.save_dataframe_to_csv(
                    self.fileMap, 'filemap', index=False)

    def _validate_file_map(self) -> None:
        """
        This function ensures that all the files specified in the file map
        of the raw images are present.

        Raises:
            InputDataError: If the set of raw data is incomplete or the
                    format of the raw data deviates from expectations.
        """

        expectedImageSize = None
        for dataChannel in self.get_data_channels():
            for fov in self.get_fovs():
                channelInfo = self.data.iloc[dataChannel]
                try:
                    imagePath = self._get_image_path(
                        channelInfo['imageType'], fov,
                        channelInfo['imagingRound'])
                except IndexError:
                    raise FileNotFoundError(
                        'Unable to find image path for %s, fov=%i, round=%i' %
                        (channelInfo['imageType'], fov,
                         channelInfo['imagingRound']))

                if not self._dataSet.rawDataPortal.open_file(
                        imagePath).exists():
                    raise InputDataError(
                        ('Image data for channel {0} and fov {1} not found. '
                         'Expected at {2}')
                        .format(dataChannel, fov, imagePath))

                try:
                    imageSize = self._dataSet.image_stack_size(imagePath)
                except Exception as e:
                    raise InputDataError(
                        ('Unable to determine image stack size for fov {0} from'
                         ' data channel {1} at {2}')
                        .format(dataChannel, fov, imagePath))

                # share this frame count with get_z_positions(fov)/
                # get_z_positions_segmentation(fov) so they don't need to
                # reopen the same file later
                self._fovFrameCountCache[
                    (channelInfo['imageType'], channelInfo['imagingRound'],
                     fov)] = imageSize[2]

                frames = channelInfo['frame']

                # this assumes fiducials are stored in the same image file
                requiredFrames = max(np.max(frames),
                                     channelInfo['fiducialFrame'])
                if requiredFrames >= imageSize[2]:
                    if not self._allowRaggedZStacks:
                        raise InputDataError(
                            ('Insufficient frames in data for channel {0} and '
                             'fov {1}. Expected {2} frames '
                             'but only found {3} in file {4}')
                            .format(dataChannel, fov, requiredFrames,
                                    imageSize[2], imagePath))
                    warnings.warn(
                        ('Fov {0} has fewer frames than expected for channel '
                         '{1} (expected at least {2}, found {3} in file {4}); '
                         'treating this fov as having a shallower z range '
                         'since allowRaggedZStacks is enabled.')
                        .format(fov, dataChannel, requiredFrames + 1,
                                imageSize[2], imagePath))

                if expectedImageSize is None:
                    expectedImageSize = [imageSize[0], imageSize[1]]
                else:
                    if expectedImageSize[0] != imageSize[0] \
                            or expectedImageSize[1] != imageSize[1]:
                        raise InputDataError(
                            ('Image data for channel {0} and fov {1} has '
                             'unexpected dimensions. Expected {1}x{2} but '
                             'found {3}x{4} in image file {5}')
                            .format(dataChannel, fov, expectedImageSize[0],
                                    expectedImageSize[1], imageSize[0],
                                    imageSize[1], imagePath))
