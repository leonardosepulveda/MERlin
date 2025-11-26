import pandas
import rtree
import networkx
import numpy as np
import numbers
import cv2
from skimage.measure import regionprops

from merlin.core import analysistask
from merlin.util import imagefilters

import geopandas as gpd

import bigfish.detection
import bigfish.stack
import bigfish.multistack

from shapely.geometry import Point

class SumSignal(analysistask.ParallelAnalysisTask):

    """
    An analysis task that calculates the signal intensity within the boundaries
    of a cell for all rounds not used in the codebook, useful for measuring
    RNA species that were stained individually.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'apply_highpass' not in self.parameters:
            self.parameters['apply_highpass'] = False
        if 'highpass_sigma' not in self.parameters:
            self.parameters['highpass_sigma'] = 5
        if 'z_index' not in self.parameters:
            self.parameters['z_index'] = 0

        if self.parameters['z_index'] >= len(self.dataSet.get_z_positions()):
            raise analysistask.InvalidParameterException(
                'Invalid z_index specified for %s. (%i > %i)'
                % (self.analysisName, self.parameters['z_index'],
                   len(self.dataSet.get_z_positions())))

        self.highpass = str(self.parameters['apply_highpass']).upper() == 'TRUE'
        self.alignTask = self.dataSet.load_analysis_task(
            self.parameters['global_align_task'])

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):
        return [self.parameters['warp_task'],
                self.parameters['segment_task'],
                self.parameters['global_align_task']]

    def _extract_signal(self, cells, inputImage, zIndex) -> pandas.DataFrame:
        cellCoords = []
        for cell in cells:
            regions = cell.get_boundaries()[zIndex]
            if len(regions) == 0:
                cellCoords.append([])
            else:
                pixels = []
                for region in regions:
                    coords = region.exterior.coords.xy
                    xyZip = list(zip(coords[0].tolist(), coords[1].tolist()))
                    pixels.append(np.array(
                                self.alignTask.global_coordinates_to_fov(
                                    cell.get_fov(), xyZip)))
                cellCoords.append(pixels)

        cellIDs = [str(cells[x].get_feature_id()) for x in range(len(cells))]
        mask = np.zeros(inputImage.shape, np.uint8)
        for i, cell in enumerate(cellCoords):
            cv2.drawContours(mask, cell, -1, i+1, -1)
        propsDict = {x.label: x for x in regionprops(mask, inputImage)}
        propsOut = pandas.DataFrame(
            data=[(propsDict[k].intensity_image.sum(),
                   propsDict[k].filled_area)
                  if k in propsDict else (0, 0)
                  for k in range(1, len(cellCoords) + 1)],
            index=cellIDs,
            columns=['Intensity', 'Pixels'])
        return propsOut

    def _get_sum_signal(self, fov, channels, zIndex):

        fTask = self.dataSet.load_analysis_task(self.parameters['warp_task'])
        sTask = self.dataSet.load_analysis_task(self.parameters['segment_task'])

        cells = sTask.get_feature_database().read_features(fov)

        signals = []
        for ch in channels:
            img = fTask.get_aligned_image(fov, ch, zIndex)
            if self.highpass:
                highPassSigma = self.parameters['highpass_sigma']
                highPassFilterSize = int(2 * np.ceil(3 * highPassSigma) + 1)
                img = imagefilters.high_pass_filter(img,
                                                    highPassFilterSize,
                                                    highPassSigma)
            signals.append(self._extract_signal(cells, img,
                                                zIndex).iloc[:, [0]])

        # adding num of pixels
        signals.append(self._extract_signal(cells, img, zIndex).iloc[:, [1]])

        compiledSignal = pandas.concat(signals, axis = 1)
        compiledSignal.columns = channels+['Pixels']

        return compiledSignal

    def get_sum_signals(self, fov: int = None) -> pandas.DataFrame:
        """Retrieve the sum signals calculated from this analysis task.

        Args:
            fov: the fov to get the sum signals for. If not specified, the
                sum signals for all fovs are returned.

        Returns:
            A pandas data frame containing the sum signal information.
        """
        if fov is None:
            return pandas.concat(
                [self.get_sum_signals(fov) for fov in self.dataSet.get_fovs()]
            )

        return self.dataSet.load_dataframe_from_csv(
            'sequential_signal', self.get_analysis_name(),
            fov, 'signals', index_col=0)

    def _run_analysis(self, fragmentIndex):
        zIndex = int(self.parameters['z_index'])
        channels, geneNames = self.dataSet.get_data_organization()\
            .get_sequential_rounds()

        fovSignal = self._get_sum_signal(fragmentIndex, channels, zIndex)
        normSignal = fovSignal.iloc[:, :-1].div(fovSignal.loc[:, 'Pixels'], 0)
        normSignal.columns = geneNames

        self.dataSet.save_dataframe_to_csv(
                normSignal, 'sequential_signal', self.get_analysis_name(),
                fragmentIndex, 'signals')


class SmfishSignal(analysistask.ParallelAnalysisTask):

    """
    An analysis task that detects smFISH spots at different threshold
    and can optionally assign them to segmented cells if segmentation is provided.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        # expect a list 
        if 'z_indexes' not in self.parameters:
            self.parameters['z_indexes'] = list(range(len(self.dataSet.get_z_positions()))) # do all planes
        if isinstance(self.parameters['z_indexes'], int):
            self.parameters['z_indexes'] = [self.parameters['z_indexes']]

        # these are the channels to analyze for smFISH spots
        if 'channel_names' not in self.parameters:
            raise ValueError("no list of channel names")

        if 'spot_radius_nm' not in self.parameters:
            self.parameters['spot_radius_nm'] = 150

        if 'segment_task' not in self.parameters:
            self.parameters['segment_task'] = None

        # make sure the threshold is a list
        # the spot finding will be run for each threshold in the list
        if 'spot_threshold' not in self.parameters:
            self.parameters['spot_threshold'] = [None] # this will trigger automatic thresholding
        if isinstance(self.parameters['spot_threshold'], numbers.Number):
            self.parameters['spot_threshold'] = [self.parameters['spot_threshold']]

        self.alignTask = self.dataSet.load_analysis_task(
            self.parameters['global_align_task'])

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):

        # want this to work with or without segmentation
        if self.parameters['segment_task'] is None:
            return [self.parameters['warp_task'],
                    self.parameters['global_align_task']]
        else:
            return [self.parameters['warp_task'],
                    self.parameters['segment_task'],
                    self.parameters['global_align_task']]

    def _run_analysis(self, fragmentIndex):

        voxel_size_nm = int(self.dataSet.micronsPerPixel * 1000) # in nanometers
        spot_radius_nm = int(self.parameters['spot_radius_nm'])

        # spot radius
        spot_radius_px = bigfish.detection.get_object_radius_pixel(
            voxel_size_nm = (voxel_size_nm, voxel_size_nm), 
            object_radius_nm = (spot_radius_nm, spot_radius_nm), 
            ndim = 2)

        fTask = self.dataSet.load_analysis_task(self.parameters['warp_task'])

        if self.parameters['segment_task'] is not None:
            sTask = self.dataSet.load_analysis_task(self.parameters['segment_task'])
            # load the cell features upfront
            cells = sTask.get_feature_database().read_features(fragmentIndex)
            cellids_all = [cell.get_feature_id() for cell in cells]

        # place to store output dataframes
        results = []

        # analysis loop
        for channel_name in self.parameters['channel_names']:
            ch = self.dataSet.get_data_organization().get_data_channel_index(channel_name) # channel id
            for zIndex in self.parameters['z_indexes']:
            
                # get the aligned image
                img = fTask.get_aligned_image(fragmentIndex, ch, zIndex)

                # bigfish analysis step by step
                # LoG filter
                img_log = bigfish.stack.log_filter(img, sigma = spot_radius_px)
                # local maximum detection
                img_mask = bigfish.detection.local_maximum_detection(img_log, min_distance = spot_radius_px)
                
                # determine threshold automatically or use provided one
                for threshold in self.parameters['spot_threshold']:
                    if threshold is None:
                        # determine an automatic thresholding
                        threshold = bigfish.detection.automated_threshold_setting(img_log, img_mask)
                        spots, _ = bigfish.detection.spots_thresholding(img_log, img_mask, threshold)
                    else:
                        spots, _ = bigfish.detection.spots_thresholding(img_log, img_mask, threshold)

                    # just in case we find no spots...
                    if len(spots) == 0:
                        print(f'No spots found in FOV {fragmentIndex}, channel {channel_name}, z {zIndex} at threshold {threshold}')
                        continue

                    # convert these spots to global coordinates
                    spots_global = np.array([self.alignTask.fov_coordinates_to_global(fragmentIndex,
                                                (spot[1], spot[0])) for spot in spots])
                    # convert to shapely Points
                    points = [Point(pt[0], pt[1]) for pt in spots_global]
                    gdf_pts = gpd.GeoDataFrame(geometry = points)

                    gdf_pts['fov'] = fragmentIndex
                    gdf_pts['zIndex'] = zIndex
                    gdf_pts['x'] = spots[:,1]
                    gdf_pts['y'] = spots[:,0]
                    gdf_pts['global_x'] = spots_global[:,0]
                    gdf_pts['global_y'] = spots_global[:,1]
                    gdf_pts['channel'] = channel_name
                    gdf_pts['threshold'] = threshold

                    if self.parameters['segment_task'] is None:
                        result = gdf_pts
                        result['index_right'] = -1  # no cell id
                    else: # do sjoin with segmentation
                        # get the polygons from a specific z index
                        polys = [cell.get_boundaries()[zIndex] for cell in cells]
                        # only take valid polygons
                        mask = [len(p) > 0 for p in polys]
                        cellids = [cid for cid, m in zip(cellids_all, mask) if m]
                        polys = [poly[0] for poly, m in zip(polys, mask) if m]

                        gdf_polys = gpd.GeoDataFrame(index = cellids, geometry = polys)

                        # points are assigned a cell id if they are within
                        result = gpd.sjoin(gdf_pts, gdf_polys, predicate='within', how='left')
                        result['index_right'] = result['index_right'].fillna(-1) #

                    results.append(result)

        # final dataframe
        df = pandas.concat(results, axis = 0, ignore_index = True)
        df.drop(columns = ['geometry'], inplace = True)

        self.dataSet.save_dataframe_to_csv(
                df, 'smfish_signal', self.get_analysis_name(),
                fragmentIndex, 'signals')

class SmfishColocalizationSignal(analysistask.ParallelAnalysisTask):

    """
    An analysis task that detects smFISH spots at attempts to find colocalized spots
    and can optionally assign them to segmented cells.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        # expect a list 
        if 'z_indexes' not in self.parameters:
            self.parameters['z_indexes'] = list(range(len(self.dataSet.get_z_positions())))
        if isinstance(self.parameters['z_indexes'], int):
            self.parameters['z_indexes'] = [self.parameters['z_indexes']]
        if 'spot_radius_nm' not in self.parameters:
            self.parameters['spot_radius_nm'] = 150
        if 'segment_task' not in self.parameters:
            self.parameters['segment_task'] = None

        if 'distance_threshold_nm' not in self.parameters:
            self.parameters['distance_threshold_nm'] = None # trigger automatic distance thresholding

        # these are the channels to analyze for smFISH colocalization
        if 'channel_1_names' not in self.parameters:
            raise ValueError("no list of channel 1 names")
        if 'channel_2_names' not in self.parameters:
            raise ValueError("no list of channel 2 names")
        # if only a single channel name is provided, make it a list
        if isinstance(self.parameters['channel_1_names'], str):
            self.parameters['channel_1_names'] = [self.parameters['channel_1_names']]
        if isinstance(self.parameters['channel_2_names'], str):
            self.parameters['channel_2_names'] = [self.parameters['channel_2_names']]

        # if a single threshold is provided, make it a list
        if isinstance(self.parameters['channel_1_spot_thresholds'], numbers.Number):
            self.parameters['channel_1_spot_thresholds'] = [self.parameters['channel_1_spot_thresholds']]
        if isinstance(self.parameters['channel_2_spot_thresholds'], numbers.Number):
            self.parameters['channel_2_spot_thresholds'] = [self.parameters['channel_2_spot_thresholds']]
        # if no thresholds are provided, set to None to trigger automatic thresholding
        if 'channel_1_spot_thresholds' not in self.parameters:
            self.parameters['channel_1_spot_thresholds'] = [None] * len(self.parameters['channel_1_spot_thresholds'])
        if 'channel_2_spot_thresholds' not in self.parameters:
            self.parameters['channel_2_spot_thresholds'] = [None] * len(self.parameters['channel_2_spot_thresholds'])

        # thresholds should be a list of same length as the channel name lists
        if len(self.parameters['channel_1_names']) != len(self.parameters['channel_2_names']):
            raise ValueError("channel 1 and channel 2 name lists must be of same length")
        if len(self.parameters['channel_1_names']) != len(self.parameters['channel_1_spot_thresholds']):
            raise ValueError("channel 1 names and channel 1 thresholds must be of same length")
        if len(self.parameters['channel_2_names']) != len(self.parameters['channel_2_spot_thresholds']):
            raise ValueError("channel 2 names and channel 2 thresholds must be of same length")

        self.alignTask = self.dataSet.load_analysis_task(self.parameters['global_align_task'])
        self.warpTask = self.dataSet.load_analysis_task(self.parameters['warp_task'])

        # define and find some useful parameters
        self.voxel_size_nm = int(self.dataSet.micronsPerPixel * 1000) # in nanometers
        self.spot_radius_nm = int(self.parameters['spot_radius_nm'])

        self.spot_radius_px = bigfish.detection.get_object_radius_pixel(
            voxel_size_nm = (self.voxel_size_nm, self.voxel_size_nm), 
            object_radius_nm = (self.spot_radius_nm, self.spot_radius_nm), 
            ndim = 2)

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):

        # want this to work with or without segmentation
        if self.parameters['segment_task'] is None:
            return [self.parameters['warp_task'],
                    self.parameters['global_align_task']]
        else:
            return [self.parameters['warp_task'],
                    self.parameters['segment_task'],
                    self.parameters['global_align_task']]

    def _load_feature_database(self, fragmentIndex):
        sTask = self.dataSet.load_analysis_task(self.parameters['segment_task'])
        self.cells = sTask.get_feature_database().read_features(fragmentIndex)
        self.cellids_all = [str(cell.get_feature_id()) for cell in self.cells]

    def _get_feature_database_zIndex(self, zIndex):
        polys = [cell.get_boundaries()[zIndex] for cell in self.cells]
        # only take valid polygons
        mask = [len(p) > 0 for p in polys]
        cellids = [cid for cid, m in zip(self.cellids_all, mask) if m]
        polys = [poly[0] for poly, m in zip(polys, mask) if m]
        gdf_polys = gpd.GeoDataFrame(index = cellids, geometry = polys)
        return gdf_polys

    def _make_geodataframe_points(self, fragmentIndex, spots):
        # convert these spots to global coordinates
        spots_global = np.array([self.alignTask.fov_coordinates_to_global(fragmentIndex,
                                    (spot[1], spot[0])) for spot in spots])
        # convert to shapely Points
        points = [Point(pt[0], pt[1]) for pt in spots_global]
        gdf_pts = gpd.GeoDataFrame(geometry = points)
        gdf_pts['x'] = spots[:,1]
        gdf_pts['y'] = spots[:,0]
        gdf_pts['global_x'] = spots_global[:,0]
        gdf_pts['global_y'] = spots_global[:,1]
        return gdf_pts

    def _run_analysis(self, fragmentIndex):

        if self.parameters['segment_task'] is not None:
            self._load_feature_database(fragmentIndex)

        # place to store output dataframes
        results = []
        results_coloc = []

        # analysis loop

        for c1_name, c2_name, c1_thresh, c2_thresh in zip(self.parameters['channel_1_names'],
                                                            self.parameters['channel_2_names'],
                                                            self.parameters['channel_1_spot_thresholds'],
                                                            self.parameters['channel_2_spot_thresholds'],
                                                            ):
            
            ch1 = self.dataSet.get_data_organization().get_data_channel_index(c1_name) # channel 1 id
            ch2 = self.dataSet.get_data_organization().get_data_channel_index(c2_name) # channel 2 id

            for zIndex in self.parameters['z_indexes']:
            
                # get the aligned image
                img_channel_1 = self.warpTask.get_aligned_image(fragmentIndex, ch1, zIndex)
                img_channel_2 = self.warpTask.get_aligned_image(fragmentIndex, ch2, zIndex)

                # bigfish analysis step by step
                # LoG filter
                img_log_channel_1 = bigfish.stack.log_filter(img_channel_1, sigma=self.spot_radius_px)
                img_log_channel_2 = bigfish.stack.log_filter(img_channel_2, sigma=self.spot_radius_px)

                # local maximum detection
                img_mask_channel_1 = bigfish.detection.local_maximum_detection(img_log_channel_1, min_distance=self.spot_radius_px)
                img_mask_channel_2 = bigfish.detection.local_maximum_detection(img_log_channel_2, min_distance=self.spot_radius_px)

                # determine threshold automatically or use provided one
                if c1_thresh is None:
                    # determine an automatic thresholding
                    c1_thresh = bigfish.detection.automated_threshold_setting(img_log_channel_1, img_mask_channel_1)
                    c1_spots, _ = bigfish.detection.spots_thresholding(img_log_channel_1, img_mask_channel_1, c1_thresh)
                if c2_thresh is None:
                    # determine an automatic thresholding
                    c2_thresh = bigfish.detection.automated_threshold_setting(img_log_channel_2, img_mask_channel_2)
                    c2_spots, _ = bigfish.detection.spots_thresholding(img_log_channel_2, img_mask_channel_2, c2_thresh)

                else:
                    c1_spots, _ = bigfish.detection.spots_thresholding(img_log_channel_1, img_mask_channel_1, c1_thresh)
                    c2_spots, _ = bigfish.detection.spots_thresholding(img_log_channel_2, img_mask_channel_2, c2_thresh)

                # colocalization analysis
                output = bigfish.multistack.detect_spots_colocalization(
                            spots_1=c1_spots, 
                            spots_2=c2_spots,
                            voxel_size=(self.voxel_size_nm, self.voxel_size_nm),
                            threshold=self.parameters['distance_threshold_nm'],
                            return_indices=True,
                            return_threshold=True)
                
                c1_spots_colocalized = output[0]
                c2_spots_colocalized = output[1]
                distances =  output[2]
                c1_indices = output[3]
                c2_indices = output[4]
                distance_threshold = output[5]

                # save the raw spots for both channels
                for spots, channel, thresh, indices in [(c1_spots, c1_name, c1_thresh, c1_indices), 
                                                        (c2_spots, c2_name, c2_thresh, c2_indices)]:
                    
                    if len(spots) == 0:
                        print(f'No spots found in FOV {fragmentIndex}, channel {channel}, z {zIndex} at threshold {thresh}')
                        continue
                    
                    # be careful we can double count since we are looping over both channels
                    gdf_pts = self._make_geodataframe_points(fragmentIndex, spots)
                    gdf_pts['experiment'] = f'{c1_name}_to_{c2_name}'
                    gdf_pts['channel'] = channel
                    gdf_pts['fov'] = fragmentIndex
                    gdf_pts['zIndex'] = zIndex
                    gdf_pts['colocalized'] = gdf_pts.index.isin(indices)
                    # add distance information
                    gdf_pts['distance'] = np.nan
                    gdf_pts.loc[indices,'distance'] = distances
                    # add threshold and distance threshold info
                    gdf_pts['intensity_threshold'] = thresh
                    gdf_pts['distance_threshold'] = distance_threshold
                    # assign to cells if segmentation is provided
                    if self.parameters['segment_task'] is None:
                        result = gdf_pts
                        result['index_right'] = -1  # no cell id
                    else: # do sjoin with segmentation
                        gdf_polys = self._get_feature_database_zIndex(zIndex)
                        # points are assigned a cell id if they are within
                        result = gpd.sjoin(gdf_pts, gdf_polys, predicate='within', how='left')
                        result['index_right'] = result['index_right'].fillna(-1)
                    results.append(result)

        # final dataframe of raw spots
        df = pandas.concat(results, axis = 0, ignore_index = True)
        df.drop(columns = ['geometry'], inplace = True)

        self.dataSet.save_dataframe_to_csv(
                df, 'smfish_signal', self.get_analysis_name(),
                fragmentIndex, 'signals')

class ExportSumSignals(analysistask.AnalysisTask):
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return [self.parameters['sequential_task']]

    def _run_analysis(self):
        sTask = self.dataSet.load_analysis_task(
                    self.parameters['sequential_task'])
        signals = sTask.get_sum_signals()

        self.dataSet.save_dataframe_to_csv(
                    signals, 'sequential_sum_signals',
                    self.get_analysis_name())
