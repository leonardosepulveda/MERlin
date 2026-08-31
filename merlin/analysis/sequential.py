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

        # optional subset of sequential channels to sum; None means all of
        # them (matches the pre-existing, backward-compatible behavior)
        if 'channel_names' not in self.parameters:
            self.parameters['channel_names'] = None

        if self.parameters['channel_names'] is not None:
            _, sequentialGeneNames = self.dataSet.get_data_organization()\
                .get_sequential_rounds()
            invalidChannels = [c for c in self.parameters['channel_names']
                                if c not in sequentialGeneNames]
            if invalidChannels:
                raise analysistask.InvalidParameterException(
                    'Invalid channel_names specified for %s: %s not found '
                    'among sequential channels %s'
                    % (self.analysisName, invalidChannels,
                       sequentialGeneNames))

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

    def _select_channels(self, channels, geneNames):
        """Filter the channels/geneNames pair returned by
        get_sequential_rounds() down to the 'channel_names' subset, if one
        was specified; otherwise return them unchanged."""
        if self.parameters['channel_names'] is None:
            return channels, geneNames
        selected = [(ch, name) for ch, name in zip(channels, geneNames)
                    if name in self.parameters['channel_names']]
        return [ch for ch, _ in selected], [name for _, name in selected]

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
        fovZPositions = self.dataSet.get_z_positions(fragmentIndex)
        if zIndex >= len(fovZPositions):
            raise analysistask.InvalidParameterException(
                'Invalid z_index specified for %s. Fov %i only has %i '
                'available z positions but z_index %i was requested.'
                % (self.analysisName, fragmentIndex, len(fovZPositions),
                   zIndex))
        channels, geneNames = self.dataSet.get_data_organization()\
            .get_sequential_rounds()
        channels, geneNames = self._select_channels(channels, geneNames)

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

        # expect a list; None is a sentinel meaning "do all planes", resolved
        # per-fov in _resolve_z_indexes since fovs can have different
        # available z ranges with ragged z-stacks
        if 'z_indexes' not in self.parameters:
            self.parameters['z_indexes'] = None
        elif isinstance(self.parameters['z_indexes'], int):
            self.parameters['z_indexes'] = [self.parameters['z_indexes']]

        # these are the channels to analyze for smFISH spots
        if 'channel_names' not in self.parameters:
            #raise ValueError("no list of channel names")
            pass # why is this raising and issue!?

        # this is supposed to be the PSF standard deviation according to bigfish
        # I think ~250nm/2.35 is probabably a better estimate?
        if 'spot_radius_nm' not in self.parameters:
            self.parameters['spot_radius_nm'] = 150

        # segmentation task is optional, None will not sjoin with cellids
        if 'segment_task' not in self.parameters:
            self.parameters['segment_task'] = None

        # make sure the threshold is a list
        # the spot finding will be run for each threshold in the list
        if 'spot_threshold' not in self.parameters:
            self.parameters['spot_threshold'] = None # this will trigger automatic thresholding
        if self.parameters['spot_threshold'] is None:
            self.parameters['spot_threshold'] = [None] # to handle null in analysis json in case...
        if isinstance(self.parameters['spot_threshold'], numbers.Number):
            self.parameters['spot_threshold'] = [self.parameters['spot_threshold']]

        # return gaussian fitting of spots
        if 'subpixel_fitting' not in self.parameters:
            self.parameters['subpixel_fitting'] = False

        self.alignTask = self.dataSet.load_analysis_task(self.parameters['global_align_task'])
        self.warpTask = self.dataSet.load_analysis_task(self.parameters['warp_task'])

        # define and find some useful parameters
        self.voxel_size_nm = int(self.dataSet.micronsPerPixel * 1000) # in nanometers
        self.spot_radius_nm = int(self.parameters['spot_radius_nm'])

        self.spot_radius_px = bigfish.detection.get_object_radius_pixel(
            voxel_size_nm = (self.voxel_size_nm, self.voxel_size_nm), 
            object_radius_nm = (self.spot_radius_nm, self.spot_radius_nm), 
            ndim = 2)

        #apply monkey patch... actually this function is simple enough just rewrite it here
        #bigfish.detection._fit_subpixel_2d = self._fit_subpixel_2d

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

    # this is a monkey patch to fix bigfish subpixel fitting to return all gaussian params
    # by default it only returns the coordinates...
    # a little clunky, is there a better way to do this?

    def _fit_subpixel_2d(self, image, coord, radius_to_crop, 
                         voxel_size_yx, spot_radius_yx):
        """Fit a 2-d gaussian on a detected spot.
        Parameters
        ----------
        image : np.ndarray
            Image with shape (y, x).
        coord : np.ndarray
            Coordinate of the spot detected, with shape (2,). One coordinate per
            dimension (yx coordinates).
        radius_to_crop : Tuple[float]
            Enlarged radius of a spot, in pixel, used to crop an image around it.
            Tuple with 2 scalars (one per dimension yx).
        voxel_size_yx : int or float
            Size of a voxel in the yx plan, in nanometer.
        spot_radius_yx : int or float
            Radius of the spot in the yx plan, in nanometer.

        Returns
        -------
        new_coord : List[float]
            Coordinates of the spot centroid with a subpixel accuracy (one element
            per dimension)
        *** patched to ouput***
        [mu_y, mu_x, sigma_yx, amplitude, background, successful fit]

        """
        # extract spot image
        image_spot, bbox_low = bigfish.detection.get_spot_surface(
            image=image, spot_y=coord[0], spot_x=coord[1], radius_yx=radius_to_crop[-1])
        # fit gaussian
        try:
            parameters = bigfish.detection.modelize_spot(
                reference_spot=image_spot,
                voxel_size=(voxel_size_yx, voxel_size_yx),
                spot_radius=(spot_radius_yx, spot_radius_yx),
                return_coord=True)
            # format coordinates and ensure it is fitted within the spot image
            y_max, x_max = image_spot.shape
            coord_y = parameters[0] / voxel_size_yx
            if coord_y < 0 or coord_y > y_max:
                coord_y = coord[0]
            else:
                coord_y += bbox_low[0]
            coord_x = parameters[1] / voxel_size_yx
            if coord_x < 0 or coord_x > x_max:
                coord_x = coord[1]
            else:
                coord_x += bbox_low[1]
            new_coord = [coord_y, coord_x]
            # sucessful fit
            good_fit = True
        # if a spot is ill-conditioned, we simply keep its original coordinates
        except RuntimeError:
            # bad fit
            good_fit = False
            new_coord = list(coord)
            parameters = (0,0,-1,-1,-1) # return dummy parameters
        # here are the params that are returned
        # [mu_y, mu_x, sigma_yx, amplitude, background, successful fit]
        return new_coord + list(parameters[2:]) + [good_fit]

    def fit_subpixel(self, image, spots, voxel_size, spot_radius):
        """Fit gaussian signal on every spot to find a subpixel coordinates.

        Parameters
        ----------
        image : np.ndarray
            Image with shape (z, y, x) or (y, x).
        spots : np.ndarray
            Coordinate of the spots detected, with shape (nb_spots, 3) or
            (nb_spots, 2). One coordinate per dimension (zyx or yx coordinates).
        voxel_size : int, float, Tuple(int, float) or List(int, float)
            Size of a voxel, in nanometer. One value per spatial dimension (zyx or
            yx dimensions). If it's a scalar, the same value is applied to every
            dimensions.
        spot_radius : int, float, Tuple(int, float) or List(int, float)
            Radius of the spot, in nanometer. One value per spatial dimension (zyx
            or yx dimensions). If it's a scalar, the same radius is applied to
            every dimensions.

        Returns
        -------
        spots_subpixel : np.ndarray
            Coordinate and fit params of the spots detected, with shape (nb_spots, 6)
            nb_spots x [mu_y, mu_x, sigma_yx, amplitude, background, successful fit]
        """
        # check consistency between parameters
        ndim = image.ndim
        # compute radius used to crop spot image
        radius_pixel = bigfish.detection.get_object_radius_pixel(
            voxel_size_nm=voxel_size,
            object_radius_nm=spot_radius,
            ndim=ndim)
        radius = [np.sqrt(ndim) * r for r in radius_pixel]
        radius = tuple(radius)
        # loop over every spot
        spots_subpixel = []
        for coord in spots[:, :ndim]:
            #subpixel_coord = bigfish.detection._fit_subpixel_2d(
            subpixel_coord = self._fit_subpixel_2d(
                image=image, 
                coord=coord,
                radius_to_crop=radius,
                voxel_size_yx=voxel_size,
                spot_radius_yx=spot_radius)
            spots_subpixel.append(subpixel_coord)
        # format results
        spots_subpixel = np.stack(spots_subpixel)
        return spots_subpixel

    def _get_feature_database_zIndex(self, fragmentIndex, zIndex):
        """Build a GeoDataFrame of this fov's segmented-cell boundaries at
        a single z index, reading only that z-plane's geometry from the
        feature database instead of the whole 3D cell set -- _run_analysis
        processes one z at a time, so only one z-plane's boundaries are
        ever needed in memory."""
        sTask = self.dataSet.load_analysis_task(self.parameters['segment_task'])
        cellids, boundaries = sTask.get_feature_database()\
            .read_feature_ids_and_boundaries_at_z(zIndex, fragmentIndex)
        polys = [b[0] for b in boundaries]
        return gpd.GeoDataFrame(index = cellids, geometry = polys)

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
        gdf_pts['fragmentIndex'] = fragmentIndex
        return gdf_pts

    def _resolve_z_indexes(self, fov: int) -> list:
        """Determine the z indexes to process for the specified fov.

        If the user did not specify 'z_indexes' (parameter is None), returns
        this fov's full own available z range (fovs can have different
        depths with ragged z-stacks). If the user specified an explicit
        list, returns it filtered to the z indexes actually available for
        this fov, printing a notice naming any indexes skipped for this fov.
        """
        fovZCount = len(self.dataSet.get_z_positions(fov))
        if self.parameters['z_indexes'] is None:
            return list(range(fovZCount))

        requested = self.parameters['z_indexes']
        valid = [z for z in requested if z < fovZCount]
        skipped = [z for z in requested if z >= fovZCount]
        if skipped:
            print('Fov %i only has %i available z positions; skipping '
                  'requested z_indexes %s for this fov.'
                  % (fov, fovZCount, skipped))
        return valid

    def _run_analysis(self, fragmentIndex):

        zIndexes = self._resolve_z_indexes(fragmentIndex)

        # z is the outer loop (not channel), and each z plane's results are
        # written to disk as soon as that z is done, so at most one z
        # plane's worth of detected spots and one z plane's worth of
        # segmentation boundaries are ever held in memory at once --
        # previously every z plane for every channel was accumulated in a
        # single list before one final write, which for a dense fov could
        # reach several GB before anything was ever written to disk.
        with self.dataSet.open_parquet_chunk_writer(
                'smfish_signal', self.get_analysis_name(), fragmentIndex,
                'signals') as writer:
            for zIndex in zIndexes:

                # place to store results for this z plane
                resultsZ = []

                # segmentation boundaries for this z only, read once and
                # reused across every channel at this z
                if self.parameters['segment_task'] is not None:
                    gdf_polys = self._get_feature_database_zIndex(
                        fragmentIndex, zIndex)

                # analysis loop
                for channel_name in self.parameters['channel_names']:
                    ch = self.dataSet.get_data_organization().get_data_channel_index(channel_name) # channel id

                    # get the aligned image
                    img = self.warpTask.get_aligned_image(fragmentIndex, ch, zIndex)

                    # bigfish analysis step by step
                    # LoG filter
                    img_log = bigfish.stack.log_filter(img, sigma = self.spot_radius_px)
                    # local maximum detection
                    img_mask = bigfish.detection.local_maximum_detection(img_log,
                                                                         min_distance = self.spot_radius_px)

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

                        # convert these spots to geodataframe
                        gdf_pts = self._make_geodataframe_points(fragmentIndex, spots)
                        gdf_pts['zIndex'] = zIndex
                        gdf_pts['channel'] = channel_name
                        gdf_pts['threshold'] = threshold

                        # do subpixel fitting if requested
                        if self.parameters['subpixel_fitting']:
                            spots_fitted = self.fit_subpixel(
                                image=img,
                                spots=spots,
                                voxel_size=self.voxel_size_nm,
                                spot_radius=self.spot_radius_nm)
                            # add fitting parameters to dataframe
                            gdf_pts['subpixel_y'] = spots_fitted[:,0]
                            gdf_pts['subpixel_x'] = spots_fitted[:,1]
                            gdf_pts['sigma_yx'] = spots_fitted[:,2]
                            gdf_pts['amplitude'] = spots_fitted[:,3]
                            gdf_pts['background'] = spots_fitted[:,4]
                            gdf_pts['fit_successful'] = spots_fitted[:,5]

                        if self.parameters['segment_task'] is None:
                            result = gdf_pts
                            result['index_right'] = -1  # no cell id
                        else: # do sjoin with segmentation
                            # points are assigned a cell id if they are within
                            result = gpd.sjoin(gdf_pts, gdf_polys, predicate='within', how='left')
                            result['index_right'] = result['index_right'].fillna(-1) #

                        resultsZ.append(result)

                if not resultsZ:
                    # no spots at all in this z plane, at any channel or
                    # threshold -- normal (e.g. a z near the tissue edge),
                    # unlike no spots anywhere in the whole fov (below)
                    continue

                dfZ = pandas.concat(resultsZ, axis = 0, ignore_index = True)
                dfZ.drop(columns = ['geometry'], inplace = True)
                # seems like parquet has some issues saving index as int
                dfZ['index_right'] = dfZ['index_right'].astype(str)
                writer.write(dfZ)

        if not writer.wrote_any:
            raise ValueError(
                'No spots detected in any z plane/channel/threshold for '
                'fov %i' % fragmentIndex)

class SmfishColocalizationSignal(SmfishSignal):

    """
    An analysis task that detects smFISH spots at attempts to find colocalized spots
    and can optionally assign them to segmented cells.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'distance_threshold_nm' not in self.parameters:
            self.parameters['distance_threshold_nm'] = 500
            #self.parameters['distance_threshold_nm'] = None # trigger automatic distance thresholding

        # these are the channels to analyze for smFISH colocalization
        if 'channel_1_names' not in self.parameters:
            raise ValueError("no list of channel 1 names")
        if 'channel_2_names' not in self.parameters:
            raise ValueError("no list of channel 2 names")
        
        ### Thresholds for spot detection ###
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
        ### End of setting thresholds ###

        if 'do_anti_colocalization' not in self.parameters:
            self.parameters['do_anti_colocalization'] = False

    def _run_analysis(self, fragmentIndex):

        # analysis loops
        # we are going to loop over z indexes first, writing each z plane's
        # results to disk as soon as it's done instead of accumulating
        # every z plane's spots in memory before one final write.
        with self.dataSet.open_parquet_chunk_writer(
                'smfish_signal', self.get_analysis_name(), fragmentIndex,
                'signals') as writer:
            self._run_analysis_streaming(fragmentIndex, writer)

        if not writer.wrote_any:
            raise ValueError(
                'No spots detected in any z plane/channel pair for fov '
                '%i' % fragmentIndex)

    def _run_analysis_streaming(self, fragmentIndex, writer):
        for zIndex in self._resolve_z_indexes(fragmentIndex):

            # place to store results for this z plane
            results_z = []

            # segmentation boundaries for this z only, read once and
            # reused across every channel pair at this z
            if self.parameters['segment_task'] is not None:
                gdf_polys = self._get_feature_database_zIndex(
                    fragmentIndex, zIndex)

            # then loop over channel pairs
            for c1_name, c2_name, c1_thresh, c2_thresh in zip(self.parameters['channel_1_names'],
                                                                self.parameters['channel_2_names'],
                                                                self.parameters['channel_1_spot_thresholds'],
                                                                self.parameters['channel_2_spot_thresholds'],
                                                                ):
                
                ch1 = self.dataSet.get_data_organization().get_data_channel_index(c1_name) # channel 1 id
                ch2 = self.dataSet.get_data_organization().get_data_channel_index(c2_name) # channel 2 id

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
                # auto distance threshold
                if self.parameters['distance_threshold_nm'] is None:
                    output = bigfish.multistack.detect_spots_colocalization(
                            spots_1=c1_spots, 
                            spots_2=c2_spots,
                            voxel_size=(self.voxel_size_nm, self.voxel_size_nm),
                            return_indices=True,
                            return_threshold=True)
                    distance_threshold = output[5]

                else: # use provided distance threshold
                    distance_threshold = self.parameters['distance_threshold_nm']
                    output = bigfish.multistack.detect_spots_colocalization(
                                spots_1=c1_spots, 
                                spots_2=c2_spots,
                                voxel_size=(self.voxel_size_nm, self.voxel_size_nm),
                                threshold=distance_threshold,
                                return_indices=True,
                                return_threshold=False)

                c1_spots_colocalized = output[0]
                c2_spots_colocalized = output[1]
                distances =  output[2]
                c1_indices = output[3]
                c2_indices = output[4]
                
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
                        # points are assigned a cell id if they are within
                        result = gpd.sjoin(gdf_pts, gdf_polys, predicate='within', how='left')
                        result['index_right'] = result['index_right'].fillna(-1)
                    results_z.append(result)

            # here we will check if channel 1 or 2 has any colocalization with any other channel
            # as an attempt to remove false positives... 

            # turn the results for a single z plane into a dataframe
            results_z = pandas.concat(results_z, axis = 0, ignore_index = True)

            # then do the anti-colocalization analysis if requested
            if self.parameters['do_anti_colocalization']:
                # remove the old index in case
                #results_z.reset_index(inplace=True, drop=True)
                results_z['anti_colocalization'] = False
                results_z['anti_colocalization_distance'] = -1

                for c1_name, c2_name in zip(self.parameters['channel_1_names'],
                                            self.parameters['channel_2_names'],):
                    

                    df_c1 = results_z[results_z['channel'] == c1_name]
                    df_c2 = results_z[results_z['channel'] == c2_name]
                    # these are all other points not in channel 1 or 2
                    df_c3 = results_z[((results_z['channel'] != c1_name) &
                                       (results_z['channel'] != c2_name))]

                    # these spots are in pixel coordinates
                    channel_1_spots = np.stack([df_c1['x'],df_c1['y']], axis = 1)
                    channel_2_spots = np.stack([df_c2['x'],df_c2['y']], axis = 1)
                    channel_3_spots = np.stack([df_c3['x'],df_c3['y']], axis = 1)

                    # just consider a single distance parameter here...
                    if self.parameters['distance_threshold_nm'] is None:
                        raise ValueError("automatic distance thresholding not supported for anti-colocalization")

                    # first check if channel 1 colozalizes wth any other channel besides channel 2
                    output = bigfish.multistack.detect_spots_colocalization(
                            spots_1=channel_1_spots, 
                            spots_2=channel_3_spots,
                            voxel_size=(self.voxel_size_nm, self.voxel_size_nm),
                            threshold=self.parameters['distance_threshold_nm'], # must have a distance threshold set...
                            return_indices=True,
                            return_threshold=False)
                    c1_spots_colocalized = output[0]
                    c3_spots_colocalized = output[1]
                    distances =  output[2]
                    c1_indices = output[3]
                    c3_indices = output[4]
                    results_z.loc[df_c1.index[c1_indices], 'anti_colocalization'] = True
                    results_z.loc[df_c1.index[c1_indices], 'anti_colocalization_distance'] = distances
                    
                    # then check if channel 2 colocalizes wth any other channel besides channel 1
                    output = bigfish.multistack.detect_spots_colocalization(
                            spots_1=channel_2_spots, 
                            spots_2=channel_3_spots,
                            voxel_size=(self.voxel_size_nm, self.voxel_size_nm),
                            threshold=self.parameters['distance_threshold_nm'], # must have a distance threshold set...
                            return_indices=True,
                            return_threshold=False)
                    c2_spots_colocalized = output[0]
                    c3_spots_colocalized = output[1]
                    distances =  output[2]
                    c2_indices = output[3]
                    c3_indices = output[4]
                    results_z.loc[df_c2.index[c2_indices], 'anti_colocalization'] = True
                    results_z.loc[df_c2.index[c2_indices], 'anti_colocalization_distance'] = distances
                # end of anti-colocalization loop

            # write this z plane's results to disk now instead of holding
            # every z plane's results in memory until the fov is done
            results_z.drop(columns = ['geometry'], inplace = True)
            # seems like parquet has some issues saving index as int
            results_z['index_right'] = results_z['index_right'].astype(str)
            writer.write(results_z)

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
