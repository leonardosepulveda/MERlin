import os
import time
import warnings
import cv2
import numpy as np
from skimage import measure
from skimage import segmentation
from skimage import exposure
from skimage import transform
from skimage import color
from skimage import util
from skimage import io
import rtree
from shapely import geometry
from typing import List, Dict, Tuple
from scipy.spatial import cKDTree
import scipy.ndimage

# cellpose takes a long time to load do it run_analysis?
    
from merlin.core import dataset
from merlin.core import analysistask
from merlin.util import spatialfeature
from merlin.util import watershed
from merlin.util import resourceestimate
import pandas
import networkx as nx

class FeatureSavingAnalysisTask(analysistask.ParallelAnalysisTask):

    """
    An abstract analysis class that saves features into a spatial feature
    database.
    """

    def __init__(self, dataSet: dataset.DataSet, parameters=None,
                 analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def _reset_analysis(self, fragmentIndex: int = None) -> None:
        super()._reset_analysis(fragmentIndex)
        self.get_feature_database().empty_database(fragmentIndex)

    def get_feature_database(self) -> spatialfeature.SpatialFeatureDB:
        """ Get the spatial feature database this analysis task saves
        features into.

        Returns: The spatial feature database reference.
        """
        return spatialfeature.HDF5SpatialFeatureDB(self.dataSet, self)

    def _generate_verification_figures(self) -> None:
        """Generate the segmentationplots figures (segment_task-only) as
        soon as this task is done, instead of waiting for the separate
        PlotPerformance task. Covers every FeatureSavingAnalysisTask
        subclass (WatershedSegment, CellPoseSegment3D, CellPoseSegmentSAM,
        RefineCellDatabases).
        """
        self._generate_plots_for_role('segment_task')


class WatershedSegment(FeatureSavingAnalysisTask):

    """
    An analysis task that determines the boundaries of features in the
    image data in each field of view using a watershed algorithm.
    
    Since each field of view is analyzed individually, the segmentation results
    should be cleaned in order to merge cells that cross the field of
    view boundary.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'seed_channel_name' not in self.parameters:
            self.parameters['seed_channel_name'] = 'DAPI'
        if 'watershed_channel_name' not in self.parameters:
            self.parameters['watershed_channel_name'] = 'polyT'

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters['warp_task'],
                self.parameters['global_align_task']]

    def get_cell_boundaries(self) -> List[spatialfeature.SpatialFeature]:
        featureDB = self.get_feature_database()
        return featureDB.read_features()

    def _run_analysis(self, fragmentIndex):
        globalTask = self.dataSet.load_analysis_task(
                self.parameters['global_align_task'])

        seedIndex = self.dataSet.get_data_organization().get_data_channel_index(
            self.parameters['seed_channel_name'])
        seedImages = self._read_and_filter_image_stack(fragmentIndex,
                                                       seedIndex, 5)

        watershedIndex = self.dataSet.get_data_organization() \
            .get_data_channel_index(self.parameters['watershed_channel_name'])
        watershedImages = self._read_and_filter_image_stack(fragmentIndex,
                                                            watershedIndex, 5)
        seeds = watershed.separate_merged_seeds(
            watershed.extract_seeds(seedImages))
        normalizedWatershed, watershedMask = watershed.prepare_watershed_images(
            watershedImages)

        seeds[np.invert(watershedMask)] = 0
        watershedOutput = segmentation.watershed(
            normalizedWatershed, measure.label(seeds), mask=watershedMask,
            connectivity=np.ones((3, 3, 3)), watershed_line=True)

        zPos = np.array(self.dataSet.get_z_positions(fragmentIndex))
        featureList = [spatialfeature.SpatialFeature.feature_from_label_matrix(
            (watershedOutput == i), fragmentIndex,
            globalTask.fov_to_global_transform(fragmentIndex), zPos)
            for i in np.unique(watershedOutput) if i != 0]

        featureDB = self.get_feature_database()
        featureDB.write_features(featureList, fragmentIndex)

    def _read_and_filter_image_stack(self, fov: int, channelIndex: int,
                                     filterSigma: float) -> np.ndarray:
        filterSize = int(2*np.ceil(2*filterSigma)+1)
        warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])
        return np.array([cv2.GaussianBlur(
            warpTask.get_aligned_image(fov, channelIndex, z),
            (filterSize, filterSize), filterSigma)
            for z in range(len(self.dataSet.get_z_positions(fov)))])


class CellPoseSegment3D(FeatureSavingAnalysisTask):

    """
    An analysis task that determines the boundaries of features in the
    image data in each field of view using cellpose (https://github.com/
    MouseLand/cellpose).

    3D implementation for spinning disk
    One and two color 

    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'diameter' not in self.parameters:
            self.parameters['diameter'] = 120
        if 'channel_1_name' not in self.parameters:
            self.parameters['channel_1_name'] = 'DAPI' # maybe sharper to segment on
        if 'channel_2_name' not in self.parameters:
            self.parameters['channel_2_name'] = None # default no channel 2
        
        # better to supply a user trained model
        if 'model_type' not in self.parameters:
            self.parameters['model_type'] = 'cyto2' # specify cp model if not using user trained model path
        if 'path_to_user_model' not in self.parameters:
            self.parameters['path_to_user_model'] = False
            
        # save mask file
        if 'dump_segmented_masks' not in self.parameters:
            self.parameters['dump_segmented_masks'] = True
        # save the raw images
        if 'dump_segmented_images' not in self.parameters:
            self.parameters['dump_segmented_images'] = True
        # only save for certain FOVs?
        if 'dump_segmented_FOVs' not in self.parameters:
            self.parameters['dump_segmented_FOVs'] = list(range(self.fragment_count()))
        # save rgb masks
        if 'dump_rgb_masks' not in self.parameters:
            self.parameters['dump_rgb_masks'] = True
        if 'use_gpu' not in self.parameters:
            self.parameters['use_gpu'] = False

        if 'cellpose_2D_to_3D_stitching' not in self.parameters:
            self.parameters['cellpose_2D_to_3D_stitching'] = True # cellpose way of doing stitching
            # if this is true cellpose will stitch 2D planes into 3D
            # if this is false cellpose will run the model along the Z axis... I believe...

        if 'anisotropy' not in self.parameters:
            # not used for 2d-3d stitching
            # ex. 2 if z is samples half as dense as XY
            # so should be z step size / pixel size
            self.parameters['anisotropy'] = 1 
        if 'stitch_threshold' not in self.parameters:
            self.parameters['stitch_threshold'] = 0.4
            # CP model.eval documentation:
            # if stitch_threshold>0.0 and not do_3D, masks are stitched in 3D to return volume segmentation
            
        # downsample to save on memory for cellpose
            # ex downsample_factor 2 will reduce the image size in half, but keep the same number of z planes
        if 'downsample_factor' not in self.parameters:
            self.parameters['downsample_factor'] = None

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters['warp_task'],
                self.parameters['global_align_task']]

    def get_cell_boundaries(self) -> List[spatialfeature.SpatialFeature]:
        featureDB = self.get_feature_database()
        return featureDB.read_features()

    def _read_image_stack(self, fov: int, channelIndex: int) -> np.ndarray:
        warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])
        
        transformation = warpTask.get_transformation(fov, channelIndex)

        zPositions = self.dataSet.get_z_positions_segmentation(fov)

        stack = []
        for zPos in zPositions:
            rawImage = self.dataSet.get_raw_image(channelIndex, fov, zPos)
            warpedImage = transform.warp(rawImage, transformation, preserve_range=True)
            stack.append(warpedImage)

        return np.array(stack).astype(rawImage.dtype)

    def _save_tiff_images(self, fov, filename_prefix, image_stack, use_skimage = False):
        '''Save a stack of images as a tiff file.'''
        if use_skimage:
            image_name = self.dataSet._analysis_image_name(
                self, filename_prefix, fov)
            io.imsave(image_name, image_stack)
        else:
            with self.dataSet.writer_for_analysis_images(self, filename_prefix, fov) as outputTif:
                for frame in image_stack:
                        outputTif.write(frame,
                                    photometric='MINISBLACK',
                                    contiguous=True)

    ###
    # make a reader for segmented masks
    # is this the best place to do this?
    # probably should be in dataset...
    # note we need to load a zIndex not a frameIndex or a zPosition
    def _load_mask_image(self, fov, zIndex, filename_prefix = 'segmented_mask'):
        imagePath = self.dataSet._analysis_image_name(self, filename_prefix, fov)
        return self.dataSet.load_image(imagePath, zIndex, transform = False)
    ###

    def _run_analysis(self, fragmentIndex):
        
        # only import cellpose when we run the analysis
        import cellpose.models
        import cellpose.utils

        globalTask = self.dataSet.load_analysis_task(
                self.parameters['global_align_task'])

        # assume single channel first
        is_two_channel = False
        cellpose_channels = [0,0]
        if self.parameters['channel_2_name'] is not None:
            is_two_channel = True
            cellpose_channels = [1,2]

        # get channel index and read images
        channel_1_id = self.dataSet.get_data_organization().get_data_channel_index(
                self.parameters['channel_1_name'])
        seg_images_1 = self._read_image_stack(fragmentIndex, channel_1_id)

        if is_two_channel:
            channel_2_id = self.dataSet.get_data_organization().get_data_channel_index(
                self.parameters['channel_2_name'])
            seg_images_2 = self._read_image_stack(fragmentIndex, channel_2_id)

        # downsample the image to save on memory
        if self.parameters['downsample_factor'] is not None:
            factor = self.parameters['downsample_factor']
            num_frames, rows_i, cols_i = seg_images_1.shape
            rows_f = int(rows_i / factor)
            cols_f = int(cols_i / factor)
            seg_images_1 = transform.resize(seg_images_1, [num_frames,rows_f,cols_f],
                preserve_range = True).astype(seg_images_1.dtype)
            if is_two_channel:
                seg_images_2 = transform.resize(seg_images_2, [num_frames,rows_f,cols_f],
                    preserve_range = True).astype(seg_images_2.dtype)
            # scale the diameter factor too
            self.parameters['diameter'] = self.parameters['diameter']/factor

        # stack the images if necessary
        if is_two_channel:
            seg_images = np.stack([seg_images_1, seg_images_2], axis = 3) # this should be a [z,x,y,c] stack
        else:
            seg_images = seg_images_1

        # get ready for cellpose stuff
        # select the model    
        # if path_to_user_model exist, override and use user model
        if self.parameters['path_to_user_model']:
            model = cellpose.models.CellposeModel(
                            gpu = self.parameters['use_gpu'],
                            pretrained_model=self.parameters['path_to_user_model'])
        else:
            model = cellpose.models.Cellpose(gpu=self.parameters['use_gpu'],
                                             model_type=self.parameters['model_type'])

        # evaluate the model using one of two ways, 3d vs 2d stitching
        # 2d stitching seems smoother imo
        if self.parameters['cellpose_2D_to_3D_stitching']:
            # 2d-3d stitching method
            cellpose_output = model.eval(seg_images, 
                                            diameter = self.parameters['diameter'], 
                                            do_3D = False,
                                            channels = cellpose_channels,
                                            stitch_threshold=self.parameters['stitch_threshold']
                                            )
        else:
            # 3d anisotropy method
            cellpose_output = model.eval(seg_images, 
                                            diameter = self.parameters['diameter'], 
                                            do_3D = True,
                                            channels = cellpose_channels,
                                            anisotropy = self.parameters['anisotropy']
                                            )
                                            
        # only take the mask output of cellpose
        # do it this way since sometimes cellpose responds with 3 or 4 outputs... weird...
        masks = cellpose_output[0]
        
        # recall that the segmentation channel may have more z positions
        # only take those zpositions
        zPos = np.array(self.dataSet.get_z_positions(fragmentIndex))
        zPos_segment = np.array(self.dataSet.get_z_positions_segmentation(fragmentIndex))
        sel = np.isin(zPos_segment, zPos)
        if sel.sum() != len(zPos):
            warnings.warn(
                ('Segmentation z positions for fov {0} do not fully cover '
                 "this fov's regular z positions (expected {1} matches, "
                 'found {2}); feature z-coordinates for this fov are '
                 "derived from the segmentation channel's own retained z "
                 'positions instead.')
                .format(fragmentIndex, len(zPos), sel.sum()))
        # use the actually-retained z-values (not the unfiltered zPos) as the
        # z-coordinate for each retained mask plane -- these are guaranteed to
        # match masks/seg_images_1's length and order after the sel mask,
        # unlike zPos which only coincides with them when every zPos entry is
        # present in zPos_segment
        zPosRetained = zPos_segment[sel]
        masks = masks[sel]

        seg_images_1 = seg_images_1[sel]

        # upsample the images if they were downsampled
        if self.parameters['downsample_factor'] is not None:
            masks = transform.resize(masks, [len(masks),rows_i,cols_i],
                order = 0,
                preserve_range = True).astype(masks.dtype)

            seg_images_1 = transform.resize(seg_images_1, [len(masks),rows_i,cols_i],
                order = 1,
                preserve_range = True).astype(seg_images_1.dtype)
            if is_two_channel:
                seg_images_2 = seg_images_2[sel]
                seg_images_2 = transform.resize(seg_images_2, [len(masks),rows_i,cols_i],
                    order = 1,
                    preserve_range = True).astype(seg_images_2.dtype)

        # saving images
        if self.parameters['dump_segmented_masks'] and fragmentIndex in self.parameters['dump_segmented_FOVs']:
            self._save_tiff_images(fragmentIndex, 'segmented_mask_', masks)
        
        if self.parameters['dump_segmented_images'] and fragmentIndex in self.parameters['dump_segmented_FOVs']:
            self._save_tiff_images(fragmentIndex, 'segmented_images_1_', seg_images_1)
            if is_two_channel:
                self._save_tiff_images(fragmentIndex, 'segmented_images_2_', seg_images_2)
        
        if self.parameters['dump_rgb_masks'] and fragmentIndex in self.parameters['dump_segmented_FOVs']:
            rgb = color.label2rgb(masks)
            self._save_tiff_images(fragmentIndex, 'segmented_mask_rgb_', util.img_as_ubyte(rgb), use_skimage = True)

        # extract the features
        mask_values = np.unique(masks)[1:] # ignore the zero mask value

        featureList = [spatialfeature.SpatialFeature.feature_from_label_matrix(
                        (masks == val),
                        fragmentIndex,
                        globalTask.fov_to_global_transform(fragmentIndex),
                        zPosRetained) for val in mask_values]

        featureDB = self.get_feature_database()
        featureDB.write_features(featureList, fragmentIndex)

class CellPoseSegmentSAM(FeatureSavingAnalysisTask):

    """
    An analysis task that determines the boundaries of features in the
    image data in each field of view using cellpose (https://github.com/
    MouseLand/cellpose).

    Cellpose4 or CellposeSAM
    need up updated to cellpose>4.0
    also make sure tensorflow with cuda is enabled
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'diameter' not in self.parameters: # diameter is not important for cp4?
            self.parameters['diameter'] = None
        if 'channel_1_name' not in self.parameters:
            self.parameters['channel_1_name'] = 'DAPI' # maybe sharper to segment on
        if 'channel_2_name' not in self.parameters:
            self.parameters['channel_2_name'] = None # default no channel 2
        
        if 'path_to_user_model' not in self.parameters:
            self.parameters['path_to_user_model'] = False
            
        # save the raw images
        if 'dump_segmented_images' not in self.parameters:
            self.parameters['dump_segmented_images'] = False

        # we really need this for CP4...
        if 'use_gpu' not in self.parameters:
            self.parameters['use_gpu'] = True

        # use CP name
        if 'do_3D' not in self.parameters:
            self.parameters['do_3D'] = True

        if 'anisotropy' not in self.parameters:
            # not used for 2d-3d stitching
            # ex. 2 if z is samples half as dense as XY
            # so should be z step size / pixel size
            self.parameters['anisotropy'] = 1

        if 'stitch_threshold' not in self.parameters:
            self.parameters['stitch_threshold'] = 0.25 # only for 2d stitching
            
        # downsample to save on memory for cellpose
        # this may be critical for CP4 where the network time is very slow
        # recommended to keep at least 4
        # ex downsample_factor 4 will reduce the image size in half, but keep the same number of z planes
        # make sure to consider the anisotropy
        if 'downsample_factor' not in self.parameters:
            self.parameters['downsample_factor'] = 4

        # some other CP params
        if 'flow3D_smooth' not in self.parameters:
            self.parameters['flow3D_smooth'] = 0
        if 'min_size' not in self.parameters:
            self.parameters['min_size'] = 0
        if 'flow_threshold' not in self.parameters:
            self.parameters['flow_threshold'] = 0.4
        if 'cellprob_threshold' not in self.parameters:
            self.parameters['cellprob_threshold'] = 0.0
        if 'tile_norm_blocksize' not in self.parameters:
            self.parameters['tile_norm_blocksize'] = 0

        # this is in case sometimes the masks come out a bit patchy
        if 'expand_mask' not in self.parameters:
            self.parameters['expand_mask'] = 0

        # overwrite old masks, by default we will rewrite, but setting false could save time if segmentation is long...
        if 'use_old_segmentation' not in self.parameters:
            self.parameters['use_old_segmentation'] = False

        # number of worker processes used to extract per-cell spatial
        # features after segmentation (the slow, serial step for FOVs with
        # many cells). 1 keeps the original serial behavior; only raise
        # this if the cluster resource allocation for this rule also
        # requests that many CPUs, otherwise the extra processes just
        # compete for the single allocated core.
        if 'feature_extraction_processes' not in self.parameters:
            self.parameters['feature_extraction_processes'] = 1

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    #: _run_analysis builds one [z, x, y, c] array holding every
    #: (downsampled) z plane for both segmentation channels at once
    #: (cellpose needs the whole volume for 3D segmentation/stitching),
    #: so memory genuinely scales with frame/z geometry. Wall-clock time
    #: does not: per FINDINGS.md, GPU inference itself finishes in under a
    #: second and the real bottleneck is the serial per-cell
    #: post-processing loop afterward, so time is dominated by detected
    #: cell count -- data-dependent, unknown ahead of a run -- not frame
    #: geometry. Only memory gets a real estimate here.
    providesMemoryEstimate = True

    def get_estimated_memory(self):
        # Uncalibrated -- no CellPoseSegmentSAM job has been measured.
        # baselineMb is deliberately much higher than the plain-python
        # ~230 MB baseline measured elsewhere in this file's sibling
        # tasks, since a loaded cellpose model (GPU or CPU) alone
        # typically costs on the order of a GB; this is a rough guess,
        # not a measurement. downsample_factor shrinks the x/y footprint
        # (not z, per this task's own parameter docstring) before
        # cellpose ever sees the volume.
        channelCount = 2 if self.parameters['channel_2_name'] else 1
        zCount = len(self.dataSet.get_z_positions())
        downsampleFactor = self.parameters['downsample_factor'] or 1
        return resourceestimate.estimate_stack_memory_mb(
            self.dataSet, frameCount=channelCount * zCount,
            downsampleFactor=downsampleFactor, kTask=20, baselineMb=3000)

    def get_estimated_time(self):
        # TODO - refine estimate. Not geometry-driven (see
        # providesMemoryEstimate's comment above) -- stays a flat,
        # unused-by-anything placeholder like every other non-opted-in
        # task until this is estimated from cell density instead.
        return 5

    def get_dependencies(self):
        return [self.parameters['warp_task'],
                self.parameters['global_align_task']]

    def get_cell_boundaries(self) -> List[spatialfeature.SpatialFeature]:
        featureDB = self.get_feature_database()
        return featureDB.read_features()

    def _read_image_stack(self, fov: int, channelIndex: int) -> np.ndarray:
        warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])
        
        transformation = warpTask.get_transformation(fov, channelIndex)

        # here we get more z positions
        zPositions = self.dataSet.get_z_positions_segmentation(fov)

        stack = []
        for zPos in zPositions:
            rawImage = self.dataSet.get_raw_image(channelIndex, fov, zPos)
            warpedImage = transform.warp(rawImage, transformation, preserve_range=True)
            stack.append(warpedImage)

        return np.array(stack).astype(rawImage.dtype)

    def _save_tiff_images(self, fov, filename_prefix, image_stack, use_skimage = False):
        '''Save a stack of images as a tiff file.'''
        if use_skimage:
            image_name = self.dataSet._analysis_image_name(
                self, filename_prefix, fov)
            io.imsave(image_name, image_stack)
        else:
            with self.dataSet.writer_for_analysis_images(self, filename_prefix, fov) as outputTif:
                for frame in image_stack:
                        outputTif.write(frame,
                                    photometric='MINISBLACK',
                                    contiguous=True)

    ###
    # make a reader for segmented masks
    # is this the best place to do this?
    # probably should be in dataset...
    # note we need to load a zIndex not a frameIndex or a zPosition
    def _load_mask_image(self, fov, zIndex = None, filename_prefix = 'segmented_mask_'):
        imagePath = self.dataSet._analysis_image_name(self, filename_prefix, fov)
        return self.dataSet.load_image(imagePath, zIndex, transform = False)
    
        # it might be better to use the function 
        # dataset.get_analysis_image
    
    # could probably do this with recursion...
    def _load_mask_stack(self, fov, filename_prefix = 'segmented_mask_'):
            num_z = len(self.dataSet.get_z_positions(fov))
            masks = [self._load_mask_image(fov, zIndex, filename_prefix = 'segmented_mask_') for zIndex in range(num_z)]
            return np.array(masks)

    def _run_analysis(self, fragmentIndex):

        globalTask = self.dataSet.load_analysis_task(
                self.parameters['global_align_task'])

        # check if the mask is already generated - this may be caused by GPU time limit and spatialfeature bottleneck
        imagePath = self.dataSet._analysis_image_name(self, 'segmented_mask_', fragmentIndex)
        print(f'looking for image at {imagePath}')
        if os.path.exists(imagePath) and self.parameters['use_old_segmentation']:
            print(f'reading saved mask file at {imagePath}')
            masks = self._load_mask_stack(fragmentIndex)
            print(f'loaded saved mask with dimensions {masks.shape} and dtype {masks.dtype}')
            print('skipping resegmenting!')

        # else run the full analysis
        else:

            # only import cellpose when we really need it since it is slow
            import cellpose.models
            import cellpose.utils

            print(f'starting cellposeSAM')
            # assume single channel first
            is_two_channel = False
            if self.parameters['channel_2_name'] is not None:
                is_two_channel = True

            # get channel index and read images
            channel_1_id = self.dataSet.get_data_organization().get_data_channel_index(
                    self.parameters['channel_1_name'])
            seg_images_1 = self._read_image_stack(fragmentIndex, channel_1_id)

            if is_two_channel:
                channel_2_id = self.dataSet.get_data_organization().get_data_channel_index(
                    self.parameters['channel_2_name'])
                seg_images_2 = self._read_image_stack(fragmentIndex, channel_2_id)

            # downsample the image to save on memory
            if self.parameters['downsample_factor'] is not None:
                factor = self.parameters['downsample_factor']
                num_frames, rows_i, cols_i = seg_images_1.shape
                rows_f = int(rows_i / factor)
                cols_f = int(cols_i / factor)
                seg_images_1 = transform.resize(seg_images_1, [num_frames,rows_f,cols_f],
                    preserve_range = True).astype(seg_images_1.dtype)
                if is_two_channel:
                    seg_images_2 = transform.resize(seg_images_2, [num_frames,rows_f,cols_f],
                        preserve_range = True).astype(seg_images_2.dtype)

            # stack the images if necessary
            if is_two_channel:
                seg_images = np.stack([seg_images_1, seg_images_2], axis = 3) # this should be a [z,x,y,c] stack
            else:
                seg_images = seg_images_1

            # there could be an issue here if we have a two channel image but a single z stack... watch out...

            # axes for cellpose
            if seg_images.ndim == 2: # single image
                z_axis = None
                channel_axis = None
            if seg_images.ndim == 3:
                z_axis = 0
                channel_axis = None
            if seg_images.ndim == 4: # z stack and channels
                z_axis = 0
                channel_axis = 3

            # get ready for cellpose stuff
            # if path_to_user_model exist, override and use user model - must be trained in cellpose
            if self.parameters['path_to_user_model']:
                model = cellpose.models.CellposeModel(gpu = self.parameters['use_gpu'],
                                                    pretrained_model=self.parameters['path_to_user_model'])
            else:
                model = cellpose.models.CellposeModel(gpu=self.parameters['use_gpu'])

            if self.parameters['do_3D']:
                cellpose_output = model.eval(seg_images,
                                                do_3D = True,
                                                z_axis = 0,
                                                channel_axis = channel_axis,
                                                flow3D_smooth = self.parameters['flow3D_smooth'],
                                                flow_threshold = self.parameters['flow_threshold'], 
                                                cellprob_threshold = self.parameters['cellprob_threshold'],
                                                min_size = self.parameters['min_size'],
                                                normalize={"tile_norm_blocksize": self.parameters['tile_norm_blocksize']})
                masks = cellpose_output[0]

            else: # 2D mode is a little different from CP1-3 from what I can tell

                masks_shape = np.array(seg_images.shape)
                if is_two_channel:
                    masks_shape = masks_shape[:-1]

                masks_raw = np.zeros(masks_shape, dtype = np.uint16)
                # have to run this plane by plane
                for i,im in enumerate(seg_images):
                    cellpose_output = model.eval(im, 
                                                    do_3D = False,
                                                    flow_threshold = self.parameters['flow_threshold'], 
                                                    cellprob_threshold = self.parameters['cellprob_threshold'],
                                                    min_size = self.parameters['min_size'],
                                                    normalize={"tile_norm_blocksize": self.parameters['tile_norm_blocksize']})
                    masks_raw[i] = cellpose_output[0]
                masks = cellpose.utils.stitch3D(masks_raw, stitch_threshold=self.parameters['stitch_threshold'])

            # this will cause masks to grow with the hope that they are a little smoother...
            if self.parameters['expand_mask'] > 0:
                for i in range(len(masks)):
                    masks[i] = segmentation.expand_labels(masks[i], int(self.parameters['expand_mask']))


            # recall that the segmentation channel may have more z positions
            # only take those zpositions
            zPos = np.array(self.dataSet.get_z_positions(fragmentIndex))
            zPos_segment = np.array(self.dataSet.get_z_positions_segmentation(fragmentIndex))
            sel = np.isin(zPos_segment, zPos)
            if sel.sum() != len(zPos):
                warnings.warn(
                    ('Segmentation z positions for fov {0} do not fully '
                     "cover this fov's regular z positions (expected {1} "
                     'matches, found {2}); feature z-coordinates for this '
                     "fov are derived from the segmentation channel's own "
                     'retained z positions instead.')
                    .format(fragmentIndex, len(zPos), sel.sum()))
            # use the actually-retained z-values (not the unfiltered zPos) --
            # guaranteed to match masks/seg_images's length and order after
            # the sel mask, unlike zPos which only coincides with them when
            # every zPos entry is present in zPos_segment
            zPosRetained = zPos_segment[sel]
            masks = masks[sel]

            # upsample the images if they were downsampled
            if self.parameters['downsample_factor'] is not None:
                masks = transform.resize(masks, [len(masks),rows_i,cols_i],
                    order = 0,
                    preserve_range = True).astype(masks.dtype)

            self._save_tiff_images(fragmentIndex, 'segmented_mask_', masks)
            
            if self.parameters['dump_segmented_images']:

                seg_images_1 = seg_images_1[sel]
                seg_images_1 = transform.resize(seg_images_1, [len(masks),rows_i,cols_i],
                    order = 1,
                    preserve_range = True).astype(seg_images_1.dtype)
                
                if is_two_channel:
                    seg_images_2 = seg_images_2[sel]
                    seg_images_2 = transform.resize(seg_images_2, [len(masks),rows_i,cols_i],
                        order = 1,
                        preserve_range = True).astype(seg_images_2.dtype)
                    
                self._save_tiff_images(fragmentIndex, 'segmented_images_1_', seg_images_1)
                if is_two_channel:
                    self._save_tiff_images(fragmentIndex, 'segmented_images_2_', seg_images_2)

        # finish if/else and do the spatial feature part...
        # extract the features

        print('generating features')
        print(f'mask image shape: {masks.shape}')
        
        mask_values = np.unique(masks)[1:] # ignore the zero mask value

        # just in case we got here without loading z positions (the
        # use_old_segmentation branch above loads masks straight from disk
        # and never computes zPosRetained) -- _load_mask_stack sizes its
        # stack from this same fov-scoped regular z count, so it's the
        # correct z-coordinate source for a previously-saved mask stack too
        if 'zPosRetained' not in locals():
            zPosRetained = np.array(self.dataSet.get_z_positions(fragmentIndex))

        t0 = time.time()
        featureList = spatialfeature.SpatialFeature.features_from_label_matrix_stack(
                    masks,
                    mask_values,
                    fragmentIndex,
                    globalTask.fov_to_global_transform(fragmentIndex),
                    zPosRetained,
                    processes=self.parameters['feature_extraction_processes'])
        t1 = time.time()
        print(f'generated features for {len(mask_values)} masks in time {t1-t0}s')

        """
        [spatialfeature.SpatialFeature.feature_from_label_matrix(
                        (masks == val),
                        fragmentIndex,
                        globalTask.fov_to_global_transform(fragmentIndex),
                        zPos) for val in mask_values]
        """

        featureDB = self.get_feature_database()
        featureDB.write_features(featureList, fragmentIndex)


class CleanCellBoundaries(analysistask.ParallelAnalysisTask):
    '''
    A task to construct a network graph where each cell is a node, and overlaps
    are represented by edges. This graph is then refined to assign cells to the
    fov they are closest to (in terms of centroid). This graph is then refined
    to eliminate overlapping cells to leave a single cell occupying a given
    position.
    '''
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.segmentTask = self.dataSet.load_analysis_task(
            self.parameters['segment_task'])
        self.alignTask = self.dataSet.load_analysis_task(
            self.parameters['global_align_task'])

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['segment_task'],
                self.parameters['global_align_task']]

    def return_exported_data(self, fragmentIndex) -> nx.Graph:
        return self.dataSet.load_graph_from_pickle(
            'cleaned_cells', self, fragmentIndex)

    def _run_analysis(self, fragmentIndex) -> None:
        allFOVs = np.array(self.dataSet.get_fovs())
        fovBoxes = self.alignTask.get_fov_boxes()
        fovIntersections = sorted([i for i, x in enumerate(fovBoxes) if
                                   fovBoxes[fragmentIndex].intersects(x)])
        intersectingFOVs = list(allFOVs[np.array(fovIntersections)])

        spatialTree = rtree.index.Index()
        count = 0
        idToNum = dict()
        for currentFOV in intersectingFOVs:
            cells = self.segmentTask.get_feature_database()\
                .read_features(currentFOV)
            cells = spatialfeature.simple_clean_cells(cells)

            spatialTree, count, idToNum = spatialfeature.construct_tree(
                cells, spatialTree, count, idToNum)

        graph = nx.Graph()
        cells = self.segmentTask.get_feature_database()\
            .read_features(fragmentIndex)
        cells = spatialfeature.simple_clean_cells(cells)
        graph = spatialfeature.construct_graph(graph, cells,
                                               spatialTree, fragmentIndex,
                                               allFOVs, fovBoxes)

        self.dataSet.save_graph_as_pickle(
            graph, 'cleaned_cells', self, fragmentIndex)


class CombineCleanedBoundaries(analysistask.AnalysisTask):
    """
    A task to construct a network graph where each cell is a node, and overlaps
    are represented by edges. This graph is then refined to assign cells to the
    fov they are closest to (in terms of centroid). This graph is then refined
    to eliminate overlapping cells to leave a single cell occupying a given
    position.

    """
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.cleaningTask = self.dataSet.load_analysis_task(
            self.parameters['cleaning_task'])

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters['cleaning_task']]

    def return_exported_data(self):
        kwargs = {'index_col': 0}
        return self.dataSet.load_dataframe_from_csv(
            'all_cleaned_cells', analysisTask=self.analysisName, **kwargs)

    def _run_analysis(self):
        allFOVs = self.dataSet.get_fovs()
        graph = nx.Graph()
        for currentFOV in allFOVs:
            subGraph = self.cleaningTask.return_exported_data(currentFOV)
            graph = nx.compose(graph, subGraph)

        cleanedCells = spatialfeature.remove_overlapping_cells(graph)

        self.dataSet.save_dataframe_to_csv(cleanedCells, 'all_cleaned_cells',
                                           analysisTask=self)


class RefineCellDatabases(FeatureSavingAnalysisTask):
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.segmentTask = self.dataSet.load_analysis_task(
            self.parameters['segment_task'])
        self.cleaningTask = self.dataSet.load_analysis_task(
            self.parameters['combine_cleaning_task'])

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        # TODO - refine estimate
        return 2048

    def get_estimated_time(self):
        # TODO - refine estimate
        return 5

    def get_dependencies(self):
        return [self.parameters['segment_task'],
                self.parameters['combine_cleaning_task']]

    def _run_analysis(self, fragmentIndex):

        cleanedCells = self.cleaningTask.return_exported_data()
        originalCells = self.segmentTask.get_feature_database()\
            .read_features(fragmentIndex)
        featureDB = self.get_feature_database()
        cleanedC = cleanedCells[cleanedCells['originalFOV'] == fragmentIndex]
        cleanedGroups = cleanedC.groupby('assignedFOV')
        for k, g in cleanedGroups:
            cellsToConsider = g['cell_id'].values.tolist()
            featureList = [x for x in originalCells if
                           str(x.get_feature_id()) in cellsToConsider]
            featureDB.write_features(featureList, fragmentIndex)


class ExportCellMetadata(analysistask.AnalysisTask):
    """
    An analysis task exports cell metadata.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        self.segmentTask = self.dataSet.load_analysis_task(
            self.parameters['segment_task'])

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['segment_task']]

    def _run_analysis(self):
        df = self.segmentTask.get_feature_database().read_feature_metadata()

        self.dataSet.save_dataframe_to_csv(df, 'feature_metadata',
                                           self.analysisName)
