import numpy as np
import skimage.transform
from typing import Dict, List, Optional, Tuple

from merlin.core import analysistask
from merlin.analysis.createffc import CreateFfc
from merlin.util import globalpositions


ExtentTuple = Tuple[float, float, float, float]

#: (top, bottom, left, right) crop into a fov's own untrimmed tile, plus
#: that tile's (x, y) origin in mosaic-pixel space.
FovPlacement = Tuple[int, int, int, int, int, int]


def _mosaic_geometry(alignTask, mosaicMicronsPerPixel: float
                     ) -> Tuple[np.ndarray, Tuple[int, int]]:
    """The microns->mosaic-pixel transform and the (height, width) shape of
    the full mosaic canvas at the given pixel size -- shared by
    GenerateMosaicTile (per-fov placement) and CombineMosaicTiles (canvas
    allocation) so the two always agree on where a fov's tile belongs.
    """
    micronExtents = alignTask.get_global_extent()
    s = 1 / mosaicMicronsPerPixel
    transform = np.float32(
        [[s, 0, -s * micronExtents[0]],
         [0, s, -s * micronExtents[1]],
         [0, 0, 1]])
    dimensions = np.matmul(
        transform, np.append(micronExtents[-2:], 1)).astype(np.int32)[:2]
    return transform, tuple(np.flip(dimensions, axis=0))


def _check_axis_aligned(alignTask, sampleFov: int, globalAlignTaskName: str,
                        tolerance: float = 1e-6) -> None:
    """Tile placement below only handles translation/scale transforms; it
    does not support a global_align_task whose fov_to_global_transform
    includes rotation or shear (none of the currently implemented ones do
    -- CorrelationGlobalAlignment, the one sketched with rotation in mind,
    is unimplemented). Checked directly against fov_to_global_transform,
    without composing in a mosaic-pixel scale first -- a uniform scale
    changes every element of the transform's linear part by the same
    factor, so it cancels out of the off-diagonal/diagonal ratio checked
    here and would only make this check depend, pointlessly, on a
    particular downsample choice.
    """
    transform = alignTask.fov_to_global_transform(sampleFov)
    offDiagonal = max(abs(transform[0, 1]), abs(transform[1, 0]))
    diagonal = max(abs(transform[0, 0]), abs(transform[1, 1]))
    if offDiagonal > tolerance * diagonal:
        raise NotImplementedError(
            "Mosaic tile placement only supports translation/scale global "
            "alignments (e.g. SimpleGlobalAlignment, "
            "LeastSquaresGlobalAlignment); '%s' returns a rotated or "
            "sheared fov_to_global_transform." % globalAlignTaskName)


def _fov_placement(
        alignTask, fov: int, positions: Dict[int, Tuple[float, float]],
        micronToMosaicTransform: np.ndarray, tileShape: Tuple[int, int],
        toleranceFraction: float) -> FovPlacement:
    """A fov's placement in mosaic-pixel space: its own origin (rounded to
    the nearest pixel -- placement here is pixel-accurate, not
    sub-pixel-interpolated), plus the (top, bottom, left, right) crop into
    its tileShape-sized (height, width) tile that removes whatever half of
    the tile overlaps a present 4-connected neighbour, so two adjoining
    fovs partition the canvas exactly at their shared midpoint instead of
    both drawing into the same pixels. A side with no true neighbour (e.g.
    the tissue's outer edge) is left untrimmed.

    Neighbours are found with the fov's own corrected global position
    (alignTask.fov_coordinates_to_global) -- the same positions placement
    itself uses -- via globalpositions.find_grid_neighbor, the same
    4-connected-neighbour primitive RegisterFovNeighbors uses. Only
    4-connected neighbours are considered, matching that module's own
    convention; a fov that only overlaps another diagonally, with no
    direct N/S/E/W neighbour, keeps its corner untrimmed there.

    Shared by GenerateMosaicTile (called once per fragment) and
    CombineMosaicTiles (called once per fov, reused across every channel,
    since placement depends only on fov geometry, not on any channel's
    pixel content).
    """
    height, width = tileShape
    transform = np.matmul(
        micronToMosaicTransform, alignTask.fov_to_global_transform(fov))
    x0, y0 = int(round(transform[0, 2])), int(round(transform[1, 2]))

    top, bottom, left, right = 0, height, 0, width
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        neighbor = globalpositions.find_grid_neighbor(
            fov, positions, dx, dy, toleranceFraction)
        if neighbor is None:
            continue
        neighborTransform = np.matmul(
            micronToMosaicTransform, alignTask.fov_to_global_transform(neighbor))
        x1 = int(round(neighborTransform[0, 2]))
        y1 = int(round(neighborTransform[1, 2]))
        if dx > 0:
            right = min(right, round(((x0 + width) + x1) / 2) - x0)
        elif dx < 0:
            left = max(left, round((x0 + (x1 + width)) / 2) - x0)
        elif dy > 0:
            bottom = min(bottom, round(((y0 + height) + y1) / 2) - y0)
        else:
            top = max(top, round((y0 + (y1 + height)) / 2) - y0)

    return x0, y0, top, bottom, left, right


class GenerateMosaicTile(analysistask.ParallelAnalysisTask):

    """
    Computes one fov's contribution to the mosaic, for every requested
    data channel at a single, fixed z index -- one Slurm fragment per fov.

    This replaces the old, single-job GenerateMosaic, which recomputed the
    fiducial warp from raw data for every (fov, channel, z) combination in
    one serial process -- on a real ~1650-fov, 28-channel, 25-z experiment
    that is over a million tiles processed one at a time on a single core,
    which does not finish within any practical wall-clock limit. Splitting
    the per-fov work into its own Slurm fragment gets real cluster
    parallelism, and fixing z to one value (see z_index) removes the
    25-fold redundant re-processing the old task did across z for no
    reason a mosaic needs.

    Each channel's tile, at each requested downsample factor, is:
      - loaded already fiducial-warped, via warp_task's own already-
        computed offsets (Warp.get_aligned_image / get_transformation --
        no alignment is recomputed here), either directly (image_source=
        'raw', the default) or additionally deconvolved/filtered
        (image_source='preprocessed', via preprocess_task.
        get_processed_image, which itself calls warp_task internally, so
        the same fiducial warp is used either way);
      - flat-field corrected (ffc_task, via CreateFfc.apply_ffc) if
        use_ffc is true (the default) -- set use_ffc to false to skip FFC
        entirely, in which case ffc_task is not needed and not loaded;
      - downsampled to 1/downsample its native resolution
        (skimage.transform.resize, anti-aliased) -- downsample may be a
        single factor or a list of factors (e.g. [1, 2, 4]) to generate
        several resolutions of the same mosaic in one pass; each factor
        must evenly divide both of the input images' pixel dimensions,
        so every downsampled tile lines up on an exact pixel grid;
      - trimmed to this fov's own non-overlapping region (see module-level
        _fov_placement) so CombineMosaicTiles can place every fov's tile
        directly, with no blending needed for overlaps.

    Each (fov, channel, downsample) tile is cached to disk (see
    get_cached_tile) before moving to the next one, so a fragment that is
    resubmitted after a timeout or failure resumes from whichever tiles it
    had already finished instead of recomputing them.

    preprocess_task is always a dependency (and always loaded), regardless
    of image_source, so the DAG reflects that a preprocessed mosaic is
    always a real option for this task without needing separate wiring.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        downsample = self.parameters['downsample']
        if isinstance(downsample, int):
            downsample = [downsample]
        downsample = list(downsample)
        if not downsample:
            raise ValueError(
                'downsample must specify at least one downsampling factor.')
        width, height = self.dataSet.get_image_dimensions()
        for factor in downsample:
            if not isinstance(factor, int) or factor < 1:
                raise ValueError(
                    'Each downsample factor must be a positive integer.')
            if width % factor != 0 or height % factor != 0:
                raise ValueError(
                    "downsample factor %d does not evenly divide the "
                    "image dimensions (%d x %d)." % (factor, width, height))
        self.parameters['downsample'] = downsample

        if 'use_ffc' not in self.parameters:
            self.parameters['use_ffc'] = True

        if 'image_source' not in self.parameters:
            self.parameters['image_source'] = 'raw'
        if self.parameters['image_source'] not in ('raw', 'preprocessed'):
            raise ValueError("image_source must be 'raw' or 'preprocessed'.")

        if not isinstance(self.parameters['z_index'], int):
            raise ValueError('z_index must specify a single integer z index.')

        if not self.parameters.get('data_channels'):
            raise ValueError(
                'data_channels must specify which channels to generate.')

        if 'neighbor_tolerance_fraction' not in self.parameters:
            self.parameters['neighbor_tolerance_fraction'] = 0.25

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        return 2000

    def get_estimated_time(self):
        return 10

    def get_dependencies(self):
        dependencies = [self.parameters['global_align_task'],
                        self.parameters['warp_task'],
                        self.parameters['preprocess_task']]
        if self.parameters['use_ffc']:
            dependencies.append(self.parameters['ffc_task'])
        return dependencies

    def get_mosaic_microns_per_pixel(self, downsample: int) -> float:
        return self.dataSet.get_microns_per_pixel() * downsample

    def get_data_channels(self) -> List[int]:
        dataOrganization = self.dataSet.get_data_organization()
        channels = self.parameters['data_channels']
        if isinstance(channels, str):
            return [dataOrganization.get_data_channel_index(channels)]
        if isinstance(channels, int):
            return [channels]
        return [dataOrganization.get_data_channel_index(x)
                if isinstance(x, str) else x for x in channels]

    def get_tile_shape(self, downsample: int) -> Tuple[int, int]:
        """The (height, width) of a single fov tile at the given downsample
        factor, before any overlap trim -- shared with CombineMosaicTiles
        so both agree on the untrimmed tile's extent. downsample is
        guaranteed (see __init__) to evenly divide both image dimensions,
        so this division is always exact.
        """
        width, height = self.dataSet.get_image_dimensions()
        return height // downsample, width // downsample

    def get_cached_tile(self, fov: int, dataChannel: int, downsample: int
                        ) -> Optional[np.ndarray]:
        """This fragment's already-computed, already-trimmed tile for the
        given channel and downsample factor, if cached from a previous
        (possibly interrupted) run of this fov's fragment -- None if it
        hasn't been computed yet.
        """
        return self.dataSet.load_numpy_analysis_result_if_available(
            'tile_%d_ds%d' % (dataChannel, downsample), self, None,
            resultIndex=fov, subdirectory='cache')

    def _load_tile(self, fov: int, dataChannel: int, downsample: int) -> np.ndarray:
        """Load, optionally ffc-correct, and downsample this fov/channel's
        tile (at this task's fixed z index) to get_tile_shape(downsample)'s
        shape -- untrimmed, at this fov's own placement; the overlap crop
        is applied by the caller.
        """
        zIndex = self.parameters['z_index']
        if zIndex < len(self.dataSet.get_z_positions(fov)):
            if self.parameters['image_source'] == 'preprocessed':
                rawImage = self.preprocessTask.get_processed_image(
                    fov, dataChannel, zIndex)
            else:
                rawImage = self.warpTask.get_aligned_image(fov, dataChannel, zIndex)
        else:
            # this fov has no data at this depth (ragged z-stack) -- a blank
            # tile keeps the ffc/downsample/placement below unchanged, since
            # an all-zero contribution is already treated as "not imaged here"
            rawImage = np.zeros(
                self.dataSet.get_image_dimensions(), dtype=np.uint16)

        if self.parameters['use_ffc']:
            ffcField = self.ffcTask.get_ffc_field_for_channel(dataChannel)
            correctedImage = CreateFfc.apply_ffc(rawImage, ffcField)
        else:
            correctedImage = rawImage.astype(np.float32)

        if downsample == 1:
            return correctedImage.astype(np.float32)
        return skimage.transform.resize(
            correctedImage, self.get_tile_shape(downsample), anti_aliasing=True,
            preserve_range=True).astype(np.float32)

    def _run_analysis(self, fragmentIndex):
        alignTask = self.dataSet.load_analysis_task(
            self.parameters['global_align_task'])
        self.warpTask = self.dataSet.load_analysis_task(
            self.parameters['warp_task'])
        self.preprocessTask = self.dataSet.load_analysis_task(
            self.parameters['preprocess_task'])
        self.ffcTask = self.dataSet.load_analysis_task(
            self.parameters['ffc_task']) if self.parameters['use_ffc'] else None

        _check_axis_aligned(
            alignTask, fragmentIndex, self.parameters['global_align_task'])

        # a fov's position in microns doesn't depend on the mosaic's pixel
        # size, so this is computed once and reused across every
        # downsample factor below.
        positions = {f: tuple(alignTask.fov_coordinates_to_global(f, (0, 0)))
                    for f in self.dataSet.get_fovs()}

        for downsample in self.parameters['downsample']:
            micronToMosaicTransform, _ = _mosaic_geometry(
                alignTask, self.get_mosaic_microns_per_pixel(downsample))
            _, _, top, bottom, left, right = _fov_placement(
                alignTask, fragmentIndex, positions, micronToMosaicTransform,
                self.get_tile_shape(downsample),
                self.parameters['neighbor_tolerance_fraction'])

            for dataChannel in self.get_data_channels():
                if self.get_cached_tile(
                        fragmentIndex, dataChannel, downsample) is not None:
                    continue
                tile = self._load_tile(fragmentIndex, dataChannel, downsample)
                self.dataSet.save_numpy_analysis_result(
                    tile[top:bottom, left:right],
                    'tile_%d_ds%d' % (dataChannel, downsample), self,
                    resultIndex=fragmentIndex, subdirectory='cache')


class CombineMosaicTiles(analysistask.AnalysisTask):

    """
    Assembles the final mosaic tiff(s) from GenerateMosaicTile's cached,
    already-trimmed per-fov tiles. Every fov's tile occupies disjoint
    pixels in the canvas (see GenerateMosaicTile's docstring), so placing
    it is a direct array write, not a warp or a blend -- no image
    processing happens in this task, which is what makes it safe to leave
    serial even though the per-fov computation is fully parallel.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'output_format' not in self.parameters:
            self.parameters['output_format'] = 'imagej'
        if self.parameters['output_format'] not in ('imagej', 'ome'):
            raise ValueError("output_format must be 'imagej' or 'ome'.")

        if self.parameters['output_format'] == 'ome':
            # OME output is always written one channel per file (pyramid
            # subresolutions are per-image), so it implies separate_files.
            if self.parameters.get('separate_files') is False:
                raise ValueError(
                    "separate_files=False is not supported with "
                    "output_format='ome'.")
            self.parameters['separate_files'] = True
        elif 'separate_files' not in self.parameters:
            self.parameters['separate_files'] = False

        if 'write_pyramidal_tiff' not in self.parameters:
            self.parameters['write_pyramidal_tiff'] = False
        if 'pyramidal_levels' not in self.parameters:
            self.parameters['pyramidal_levels'] = 4
        if self.parameters['write_pyramidal_tiff'] \
                and self.parameters['output_format'] != 'ome':
            raise ValueError("write_pyramidal_tiff requires output_format='ome'.")

    def get_estimated_memory(self):
        return 10000

    def get_estimated_time(self):
        return 30

    def get_dependencies(self):
        return [self.parameters['tile_task']]

    def get_mosaic(self) -> np.ndarray:
        """Get the mosaic generated by this analysis task, at its tile_task's
        first requested downsample factor.

        Returns:
            a 3-dimensional array containing the mosaic, arranged as
            [channel, x, y] -- a single z index, fixed by this task's
            tile_task (GenerateMosaicTile.parameters['z_index']).
        """
        return self.dataSet.get_analysis_image_set(self, 'mosaic')

    def _fov_placements(self, tileTask, alignTask, micronToMosaicTransform,
                        downsample: int) -> Dict[int, FovPlacement]:
        """Every fov's placement (see _fov_placement) at the given downsample
        factor, computed once and reused across every channel below --
        placement depends only on fov geometry, not on any channel's pixel
        content.
        """
        tileShape = tileTask.get_tile_shape(downsample)
        positions = {f: tuple(alignTask.fov_coordinates_to_global(f, (0, 0)))
                    for f in self.dataSet.get_fovs()}
        tolerance = tileTask.parameters['neighbor_tolerance_fraction']
        return {
            fov: _fov_placement(alignTask, fov, positions,
                                micronToMosaicTransform, tileShape, tolerance)
            for fov in self.dataSet.get_fovs()}

    def _build_channel_mosaic(self, tileTask, fovPlacements: Dict[int, FovPlacement],
                              mosaicShape: Tuple[int, int], dataChannel: int,
                              downsample: int) -> np.ndarray:
        mosaic = np.zeros(mosaicShape, dtype=np.float32)
        height, width = mosaicShape
        for fov, (x0, y0, top, bottom, left, right) in fovPlacements.items():
            tile = tileTask.get_cached_tile(fov, dataChannel, downsample)
            if tile is None:
                print(f'warning: no cached tile for fov {fov}, data channel '
                     f'{dataChannel}, downsample {downsample} -- leaving '
                     f'that region blank.', flush=True)
                continue
            yStart, xStart = max(y0 + top, 0), max(x0 + left, 0)
            yEnd, xEnd = min(y0 + bottom, height), min(x0 + right, width)
            if yEnd <= yStart or xEnd <= xStart:
                continue
            tileRowStart, tileColStart = yStart - (y0 + top), xStart - (x0 + left)
            mosaic[yStart:yEnd, xStart:xEnd] = tile[
                tileRowStart:tileRowStart + (yEnd - yStart),
                tileColStart:tileColStart + (xEnd - xStart)]
        return mosaic.astype(np.uint16)

    def _write_imagej_mosaics(self, tileTask, fovPlacements, mosaicShape,
                              dataOrganization, dataChannels, downsample):
        zIndex = tileTask.parameters['z_index']
        suffix = '_ds%d' % downsample
        if self.parameters['separate_files']:
            imageDescription = self.dataSet.analysis_tiff_description(1, 1)
            for d in dataChannels:
                with self.dataSet.writer_for_analysis_images(
                        self, 'mosaic_%s_%i%s' % (
                            dataOrganization.get_data_channel_name(d), zIndex,
                            suffix),
                        imagej=True) as outputTif:
                    mosaic = self._build_channel_mosaic(
                        tileTask, fovPlacements, mosaicShape, d, downsample)
                    outputTif.write(mosaic, photometric='MINISBLACK',
                                    contiguous=True, metadata=imageDescription)
        else:
            imageDescription = self.dataSet.analysis_tiff_description(
                1, len(dataChannels))
            with self.dataSet.writer_for_analysis_images(
                    self, 'mosaic%s' % suffix, imagej=True) as outputTif:
                for d in dataChannels:
                    print(f'assembling data channel {d}, downsample '
                         f'{downsample}.', flush=True)
                    mosaic = self._build_channel_mosaic(
                        tileTask, fovPlacements, mosaicShape, d, downsample)
                    outputTif.write(mosaic, photometric='MINISBLACK',
                                    contiguous=True, metadata=imageDescription)

    def _write_ome_mosaics(self, tileTask, fovPlacements, mosaicShape,
                          dataOrganization, dataChannels, downsample):
        zIndex = tileTask.parameters['z_index']
        for d in dataChannels:
            dataChannelName = dataOrganization.get_data_channel_name(d)
            mosaic_name = f'mosaic_{dataChannelName}_{zIndex}_ds{downsample}'
            print(f'assembling data channel {d}, downsample {downsample}.',
                 flush=True)
            with self.dataSet.writer_for_analysis_images(
                    self, mosaic_name, imagej=False, ome=True) as outputTif:
                mosaic = self._build_channel_mosaic(
                    tileTask, fovPlacements, mosaicShape, d, downsample)

                metadata = {'axes': 'YX'}
                options = dict(
                    photometric='MINISBLACK', tile=(256, 256),
                    compression='lzw', contiguous=True)

                if self.parameters['write_pyramidal_tiff']:
                    subresolutions = int(self.parameters['pyramidal_levels'])
                    outputTif.write(mosaic[:, :], subifds=subresolutions,
                                    metadata=metadata, **options)
                    for level in range(subresolutions):
                        print(f'writing channel {d} mosaic level {level}',
                             flush=True)
                        mag = 2 ** (level + 1)
                        outputTif.write(mosaic[::mag, ::mag], subfiletype=1,
                                        **options)
                else:
                    outputTif.write(mosaic[:, :], metadata=metadata, **options)

    def _run_analysis(self):
        tileTask = self.dataSet.load_analysis_task(self.parameters['tile_task'])
        alignTask = self.dataSet.load_analysis_task(
            tileTask.parameters['global_align_task'])
        _check_axis_aligned(
            alignTask, self.dataSet.get_fovs()[0],
            tileTask.parameters['global_align_task'])

        dataOrganization = self.dataSet.get_data_organization()
        dataChannels = tileTask.get_data_channels()

        for downsample in tileTask.parameters['downsample']:
            micronToMosaicTransform, mosaicShape = _mosaic_geometry(
                alignTask, tileTask.get_mosaic_microns_per_pixel(downsample))
            self.dataSet.save_numpy_txt_analysis_result(
                micronToMosaicTransform,
                'micron_to_mosaic_pixel_transform_ds%d' % downsample, self)

            fovPlacements = self._fov_placements(
                tileTask, alignTask, micronToMosaicTransform, downsample)

            print('assembling mosaic for \nchannels: {}\nz index: {}\n'
                 'downsample: {}'.format(
                     dataChannels, tileTask.parameters['z_index'], downsample),
                 flush=True)

            if self.parameters['output_format'] == 'ome':
                self._write_ome_mosaics(tileTask, fovPlacements, mosaicShape,
                                        dataOrganization, dataChannels,
                                        downsample)
            else:
                self._write_imagej_mosaics(tileTask, fovPlacements, mosaicShape,
                                           dataOrganization, dataChannels,
                                           downsample)
