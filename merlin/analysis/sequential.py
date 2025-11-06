import pandas
import rtree
import networkx
import numpy as np
import cv2
from skimage.measure import regionprops

from merlin.core import analysistask
from merlin.util import imagefilters

import geopandas as gpd

import bigfish.detection

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
    An analysis task that calculates the spots within the boundaries
    of a cell for specified rounds, useful for measuring
    RNA species that were stained individually.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        # expect a list 
        if 'z_indexes' not in self.parameters:
            self.parameters['z_indexes'] = [0]
        if isinstance(self.parameters['z_indexes'], int):
            self.parameters['z_indexes'] = [self.parameters['z_indexes']]

        if 'channel_names' not in self.parameters:
            raise ValueError("no list of channel names")

        if 'spot_radius_nm' not in self.parameters:
            self.parameters['spot_radius_nm'] = 150

        self.alignTask = self.dataSet.load_analysis_task(
            self.parameters['global_align_task'])

        if 'segment_task' not in self.parameters:
            self.parameters['segment_task'] = None


    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 1

    def get_dependencies(self):
        return [self.parameters['warp_task'],
                self.parameters['global_align_task']]

    def _run_analysis(self, fragmentIndex):

        voxel_size_nm = int(self.dataSet.micronsPerPixel * 1000) # in nanometers
        spot_radius_nm = int(self.parameters['spot_radius_nm'])

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
            
                img = fTask.get_aligned_image(fragmentIndex, ch, zIndex)

                # do spot analysis
                spots, threshold = bigfish.detection.detect_spots(
                    images=img, 
                    return_threshold=True, 
                    voxel_size=(voxel_size_nm, voxel_size_nm),  # in nanometer (one value per dimension zyx)
                    spot_radius=(spot_radius_nm, spot_radius_nm))  # in nanometer (one value per dimension zyx)

                # convert these spots to global coordinates

                spots_global = np.array([self.alignTask.fov_coordinates_to_global(fragmentIndex,
                                            (spot[1],spot[0])) for spot in spots])
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
                    # get the refined polygons
                    polys = [cell.get_boundaries()[zIndex] for cell in cells]
                    # only take valid polygons
                    mask = [len(p) > 0 for p in polys]
                    cellids = [cid for cid, m in zip(cellids_all, mask) if m]
                    polys = [poly[0] for poly, m in zip(polys, mask) if m]

                    gdf_polys = gpd.GeoDataFrame(index = cellids, geometry = polys)

                    # points are assigned a cell id if they are within
                    result = gpd.sjoin(gdf_pts, gdf_polys, predicate='within', how='left')

                results.append(result)

        # final dataframe
        df = pandas.concat(results, axis = 0)
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
