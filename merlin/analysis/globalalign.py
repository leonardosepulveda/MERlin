from abc import abstractmethod
import numpy as np
import pandas as pd
from typing import Tuple
from typing import List
from shapely import geometry

from merlin.core import analysistask
from merlin.util import globalpositions
from merlin.util import resourceestimate


def _nominal_positions_and_overlap(dataSet, overlapFractionParam):
    """Shared setup for `RegisterFovNeighbors` and `LeastSquaresGlobalAlignment`:
    every fov's nominal (stage-reported) position, and the overlap fraction
    to register/score against (the given value, or -- if None -- inferred
    from the measured grid step size vs. the fov's own width, same
    convention the real-data comparison `merlin.util.globalpositions` is
    ported from used). Factored out so both tasks derive this identically
    rather than each keeping their own copy to drift apart.
    """
    fovs = dataSet.get_fovs()
    nominalPositions = {f: tuple(dataSet.get_fov_offset(f)) for f in fovs}
    micronsPerPixel = dataSet.get_microns_per_pixel()
    frameWidthUm = dataSet.get_image_dimensions()[0] * micronsPerPixel

    stepSizeUm = globalpositions.estimate_step_size_um(nominalPositions)
    overlapFraction = overlapFractionParam
    if overlapFraction is None:
        overlapFraction = max(
            0.0, 1.0 - stepSizeUm / frameWidthUm) if frameWidthUm > 0 else 0.0

    return fovs, nominalPositions, micronsPerPixel, stepSizeUm, overlapFraction


class GlobalAlignment(analysistask.AnalysisTask):

    """
    An abstract analysis task that determines the relative position of
    different field of views relative to each other in order to construct
    a global alignment.
    """
    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    @abstractmethod
    def fov_coordinates_to_global(
            self, fov: int, fovCoordinates: Tuple[float, float]) \
            -> Tuple[float, float]:
        """Calculates the global coordinates based on the local coordinates
        in the specified field of view.

        Args:
            fov: the fov where the coordinates are measured
            fovCoordinates: a tuple containing the x and y coordinates
                or z, x, and y coordinates (in pixels) in the specified fov.
        Returns:
            A tuple containing the global x and y coordinates or
            z, x, and y coordinates (in microns)
        """
        pass

    @abstractmethod
    def global_coordinates_to_fov(
            self, fov: int, globalCoordinates: List[Tuple[float, float]]) \
            -> List[Tuple[float, float]]:
        """Calculates the fov pixel coordinates for a list of global coordinates
        in the specified field of view.

        Args:
            fov: the fov where the coordinates are measured
            globalCoordinates: a list of tuples containing the x and
                               y coordinates (in pixels) in the specified fov.
        Returns:
            A list of tuples containing the global x and y coordinates
            (in microns)
        """
        pass
        # TODO this can be updated to take either a list or a single coordinate
        # and to convert z position

    @abstractmethod
    def fov_to_global_transform(self, fov: int) -> np.ndarray:
        """Calculates the transformation matrix for an affine transformation
        that transforms the fov coordinates to global coordinates.

        Args:
            fov: the fov to calculate the transformation
        Returns:
            a numpy array containing the transformation matrix
        """
        pass

    @abstractmethod
    def get_global_extent(self) -> Tuple[float, float, float, float]:
        """Get the extent of the global coordinate system.

        Returns:
            a tuple where the first two indexes correspond to the minimum
            and x and y extents and the last two indexes correspond to the
            maximum x and y extents. All are in units of microns.
        """
        pass

    @abstractmethod
    def fov_coordinate_array_to_global(self, fov: int,
                                       fovCoordArray: np.array) -> np.array:
        """A bulk transformation of a list of fov coordinates to
           global coordinates.
        Args:
            fov: the fov of interest
            fovCoordArray: numpy array of the [z, x, y] positions to transform
        Returns:
            numpy array of the global [z, x, y] coordinates
        """
        pass

    def get_fov_boxes(self) -> List:
        """
        Creates a list of shapely boxes for each fov containing the global
        coordinates as the box coordinates.

        Returns:
            A list of shapely boxes
        """
        fovs = self.dataSet.get_fovs()
        boxes = [geometry.box(*self.fov_global_extent(f)) for f in fovs]

        return boxes


class SimpleGlobalAlignment(GlobalAlignment):

    """A global alignment that uses the theoretical stage positions in
    order to determine the relative positions of each field of view.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_estimated_memory(self):
        return 1

    def get_estimated_time(self):
        return 0

    def _run_analysis(self):
        # This analysis task does not need computation
        pass

    def get_dependencies(self):
        return []

    def _get_fov_offset(self, fov: int) -> Tuple[float, float]:
        """The fov's (x, y) offset in the global coordinate system, in
        microns. Factored out (rather than calling
        ``self.dataSet.get_fov_offset`` directly below) so a subclass
        (e.g. LeastSquaresGlobalAlignment) can substitute a corrected
        offset while reusing every other coordinate-transform method
        unchanged.
        """
        return self.dataSet.get_fov_offset(fov)

    def fov_coordinates_to_global(self, fov, fovCoordinates):
        fovStart = self._get_fov_offset(fov)
        micronsPerPixel = self.dataSet.get_microns_per_pixel()
        if len(fovCoordinates) == 2:
            return (fovStart[0] + fovCoordinates[0]*micronsPerPixel,
                    fovStart[1] + fovCoordinates[1]*micronsPerPixel)
        elif len(fovCoordinates) == 3:
            zPositions = self.dataSet.get_z_positions(fov)
            return (np.interp(fovCoordinates[0], np.arange(len(zPositions)),
                              zPositions),
                    fovStart[0] + fovCoordinates[1]*micronsPerPixel,
                    fovStart[1] + fovCoordinates[2]*micronsPerPixel)

    def fov_coordinate_array_to_global(self, fov: int,
                                       fovCoordArray: np.array) -> np.array:
        tForm = self.fov_to_global_transform(fov)
        toGlobal = np.ones(fovCoordArray.shape)
        toGlobal[:, [0, 1]] = fovCoordArray[:, [1, 2]]
        globalCentroids = np.matmul(tForm, toGlobal.T).T[:, [2, 0, 1]]
        globalCentroids[:, 0] = fovCoordArray[:, 0]
        return globalCentroids

    def fov_global_extent(self, fov: int) -> List[float]:
        """
        Returns the global extent of a fov, output interleaved as
        xmin, ymin, xmax, ymax

        Args:
            fov: the fov of interest
        Returns:
            a list of four floats, representing the xmin, xmax, ymin, ymax
        """

        return [x for y in (self.fov_coordinates_to_global(fov, (0, 0)),
                            self.fov_coordinates_to_global(fov, (2048, 2048))) # this seems like a bug if the image is not 2048x2048
                for x in y]

    def global_coordinates_to_fov(self, fov, globalCoordinates):
        tform = np.linalg.inv(self.fov_to_global_transform(fov))

        def convert_coordinate(coordinateIn):
            coords = np.array([coordinateIn[0], coordinateIn[1], 1])
            return np.matmul(tform, coords).astype(int)[:2]
        pixels = [convert_coordinate(x) for x in globalCoordinates]
        return pixels

    def fov_to_global_transform(self, fov):
        micronsPerPixel = self.dataSet.get_microns_per_pixel()
        globalStart = self.fov_coordinates_to_global(fov, (0, 0))

        return np.float32([[micronsPerPixel, 0, globalStart[0]],
                           [0, micronsPerPixel, globalStart[1]],
                           [0, 0, 1]])

    def get_global_extent(self):
        fovSize = self.dataSet.get_image_dimensions()
        fovBounds = [self.fov_coordinates_to_global(x, (0, 0))
                     for x in self.dataSet.get_fovs()] + \
                    [self.fov_coordinates_to_global(x, fovSize)
                     for x in self.dataSet.get_fovs()]

        minX = np.min([x[0] for x in fovBounds])
        maxX = np.max([x[0] for x in fovBounds])
        minY = np.min([x[1] for x in fovBounds])
        maxY = np.max([x[1] for x in fovBounds])

        return minX, minY, maxX, maxY


class CorrelationGlobalAlignment(GlobalAlignment):

    """
    A global alignment that uses the cross-correlation between
    overlapping regions in order to determine the relative positions
    of each field of view.
    """

    # TODO - implement.  I expect rotation might be needed for this alignment
    # if the x-y orientation of the camera is not perfectly oriented with
    # the microscope stage

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_estimated_memory(self):
        return 1000

    def get_estimated_time(self):
        return 60

    def fov_coordinates_to_global(self, fov, fovCoordinates):
        raise NotImplementedError

    def fov_to_global_transform(self, fov):
        raise NotImplementedError

    def get_global_extent(self):
        raise NotImplementedError

    def fov_coordinate_array_to_global(self, fov: int,
                                       fovCoordArray: np.array) -> np.array:
        raise NotImplementedError

    @staticmethod
    def _calculate_overlap_area(x1, y1, x2, y2, width, height):
        """Calculates the overlapping area between two rectangles with
        equal dimensions.
        """

        dx = min(x1+width, x2+width) - max(x1, x2)
        dy = min(y1+height, y2+height) - max(y1, y2)

        if dx > 0 and dy > 0:
            return dx*dy
        else:
            return 0

    def _get_overlapping_regions(self, fov: int, minArea: int = 2000):
        """Get a list of all the fovs that overlap with the specified fov.
        """
        positions = self.dataSet.get_stage_positions()
        pixelToMicron = self.dataSet.get_microns_per_pixel()
        fovMicrons = [x*pixelToMicron
                      for x in self.dataSet.get_image_dimensions()]
        fovPosition = positions.loc[fov]
        overlapAreas = [i for i, p in positions.iterrows()
                        if self._calculate_overlap_area(
                p['X'], p['Y'], fovPosition['X'], fovPosition['Y'],
                fovMicrons[0], fovMicrons[1]) > minArea and i != fov]

        return overlapAreas

    def _run_analysis(self):
        fov1 = self.dataSet.get_fiducial_image(0, 0)
        fov2 = self.dataSet.get_fiducial_image(0, 1)

        return fov1, fov2


class RegisterFovNeighbors(analysistask.ParallelAnalysisTask):

    """
    Registers one fov against each of its present 4-connected neighbours
    (`merlin.util.globalpositions.register_fov_against_neighbors`), one
    fragment per fov.

    This is the per-fov, Slurm-parallel map half of `LeastSquaresGlobalAlignment`'s
    two-task split -- see that class's own docstring, and FINDINGS.md
    (2026-09-03), for why: the original single-job
    `LeastSquaresGlobalAlignment._run_analysis` cached every fov's fiducial
    frame for the life of the call, OOM-killing on a real 1651-fov
    experiment. Splitting the frame-heavy registration step into one
    fragment per fov bounds each fragment's own memory to at most two
    frames (the fov's own plus its current neighbour's) instead of the
    whole dataset, and gets real cluster parallelism for what was
    previously a single serial job.

    Writes one CSV per fov of that fov's own correspondences as anchor
    (empty, header-only, if the fov has no surviving neighbour -- e.g. an
    isolated fov); `return_exported_data` reads it back as a list of
    `NeighborCorrespondence`.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'fiducial_data_channel' not in self.parameters:
            self.parameters['fiducial_data_channel'] = 0
        if 'overlap_fraction' not in self.parameters:
            # None -> inferred in _run_analysis (see
            # _nominal_positions_and_overlap).
            self.parameters['overlap_fraction'] = None
        if 'tolerance_fraction' not in self.parameters:
            self.parameters['tolerance_fraction'] = 0.25
        if 'upsample_factor' not in self.parameters:
            self.parameters['upsample_factor'] = 100

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    #: A fov's own frame plus (up to) one neighbour frame are ever held at
    #: once -- see `globalpositions.register_fov_against_neighbors` -- so
    #: peak memory doesn't scale with the experiment's total fov count,
    #: only with a single frame's size.
    providesMemoryEstimate = True
    providesTimeEstimate = True

    def get_estimated_memory(self):
        # Uncalibrated -- no real job measured yet. Reuses
        # FiducialCorrelationWarp's own calibrated kTask (warp.py; same
        # skimage.registration.phase_cross_correlation algorithm family)
        # as a conservative proxy against 2 full frames (anchor +
        # neighbour) -- this task's crops are smaller than a full frame
        # (just the overlap band), so real usage should be lower.
        return resourceestimate.estimate_stack_memory_mb(
            self.dataSet, frameCount=2, kTask=59, baselineMb=230)

    def get_estimated_time(self):
        # Uncalibrated -- no real job measured yet. Up to 4 neighbour
        # registrations per fov (one per direction), each assumed to cost
        # about as much as one FiducialCorrelationWarp channel registration
        # (same secondsPerFrame guess, warp.py).
        return resourceestimate.estimate_stack_time_minutes(
            frameCount=4, secondsPerFrame=3, baselineMinutes=2)

    def get_dependencies(self):
        return []

    _CORRESPONDENCE_COLUMNS = [
        'anchor_fov', 'neighbor_fov', 'direction',
        'nominal_x', 'nominal_y', 'measured_x', 'measured_y', 'error']

    def return_exported_data(
            self, fragmentIndex) -> List[globalpositions.NeighborCorrespondence]:
        df = self.dataSet.load_dataframe_from_csv(
            'neighbor_correspondences_raw', self, resultIndex=fragmentIndex)
        return [
            globalpositions.NeighborCorrespondence(
                anchor_fov=int(row.anchor_fov), neighbor_fov=int(row.neighbor_fov),
                direction=row.direction, nominal_xy=(row.nominal_x, row.nominal_y),
                measured_xy=(row.measured_x, row.measured_y), error=row.error)
            for row in df.itertuples()
        ]

    def _run_analysis(self, fragmentIndex):
        _, nominalPositions, micronsPerPixel, _, overlapFraction = \
            _nominal_positions_and_overlap(
                self.dataSet, self.parameters['overlap_fraction'])

        fiducialChannel = self.parameters['fiducial_data_channel']

        def load_frame(fov):
            return self.dataSet.get_fiducial_image(fiducialChannel, fov)

        correspondences = globalpositions.register_fov_against_neighbors(
            fragmentIndex, nominalPositions, load_frame,
            pixel_size_um=micronsPerPixel, overlap_fraction=overlapFraction,
            tolerance_fraction=self.parameters['tolerance_fraction'],
            upsample_factor=self.parameters['upsample_factor'])

        self.dataSet.save_dataframe_to_csv(
            pd.DataFrame([
                {'anchor_fov': c.anchor_fov, 'neighbor_fov': c.neighbor_fov,
                 'direction': c.direction,
                 'nominal_x': c.nominal_xy[0], 'nominal_y': c.nominal_xy[1],
                 'measured_x': c.measured_xy[0], 'measured_y': c.measured_xy[1],
                 'error': c.error}
                for c in correspondences
            ], columns=self._CORRESPONDENCE_COLUMNS),
            'neighbor_correspondences_raw', self, resultIndex=fragmentIndex)


class LeastSquaresGlobalAlignment(SimpleGlobalAlignment):

    """
    A global alignment that corrects each fov's nominal (stage-reported)
    position by jointly solving a sparse least-squares system built from
    real pairwise image-registration measurements between every 4-connected
    neighbouring fov -- the "global_lsq" method identified, on real
    several-hundred-fov data, as the most accurate of several candidate
    correction strategies compared in the sibling MERci project
    (`251225_LT027_saving_time/MERci/notebooks/tests/
    compare_stitching_correction_methods.ipynb`; algorithm ported in
    `merlin.util.globalpositions`). See that module's docstring for why a
    single global affine transform (this task's `CorrelationGlobalAlignment`
    sibling was originally sketched as, per its own TODO comment) cannot
    correct this class of error at all, and why a joint per-fov solve is
    needed instead.

    Reuses every coordinate-transform method from `SimpleGlobalAlignment`
    unchanged (`fov_coordinates_to_global`, `fov_to_global_transform`,
    `get_global_extent`, etc.) by overriding only `_get_fov_offset` to
    return the corrected position once `_run_analysis` has computed it.

    The per-fov pairwise registration this task's joint fit consumes is
    computed by its `RegisterFovNeighbors` dependency (see that class's
    own docstring for why this is a separate, per-fov-parallel task and
    not inlined here as it originally was) -- this task itself only
    gathers each fov's already-registered correspondences, runs the joint
    least-squares solve (which, unlike registration, is not parallelizable
    per-fov: it needs every fov's correspondences at once), and scores the
    result.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        self._correctedPositions = None

        if 'neighbor_registration_task' not in self.parameters:
            self.parameters['neighbor_registration_task'] = 'RegisterFovNeighbors'
        self.registrationTask = self.dataSet.load_analysis_task(
            self.parameters['neighbor_registration_task'])

        if 'fiducial_data_channel' not in self.parameters:
            self.parameters['fiducial_data_channel'] = 0
        if 'overlap_fraction' not in self.parameters:
            # None -> inferred in _run_analysis from the measured grid step
            # size vs. the fov's own width, same convention the real-data
            # comparison this method is ported from used.
            self.parameters['overlap_fraction'] = None
        if 'mad_threshold' not in self.parameters:
            self.parameters['mad_threshold'] = 5.0
        if 'lsqr_atol' not in self.parameters:
            self.parameters['lsqr_atol'] = 1e-12
        if 'lsqr_btol' not in self.parameters:
            self.parameters['lsqr_btol'] = 1e-12

    #: No frame is ever held beyond the bounded `_BoundedFrameCache` used
    #: by this task's own final `compute_overlap_correlations` QC pass
    #: (globalpositions.py) -- the frame-heavy registration step now lives
    #: entirely in `RegisterFovNeighbors`. So, unlike before this task's
    #: split, memory here is roughly constant in fov count rather than
    #: scaling with it.
    providesMemoryEstimate = True
    providesTimeEstimate = True

    def get_estimated_memory(self):
        # Uncalibrated -- no real job measured yet. `_BoundedFrameCache`'s
        # default maxsize (8) full frames, plus a higher baseline than
        # FiducialCorrelationWarp's measured 230 MB to cover pandas/scipy
        # (the sparse lsqr solve, correspondence dataframes) -- kTask=2
        # rather than 1 since compute_overlap_correlations promotes crops
        # to float64.
        return resourceestimate.estimate_stack_memory_mb(
            self.dataSet, frameCount=8, kTask=2, baselineMb=500)

    def get_estimated_time(self):
        # Uncalibrated -- no real job measured yet. Dominated by reading
        # every fov's small RegisterFovNeighbors CSV plus the final
        # compute_overlap_correlations QC pass over the kept
        # correspondences -- both roughly linear in fov count, not frame-
        # count, so this (ab)uses frameCount as a fov-count proxy rather
        # than a real frame count.
        return resourceestimate.estimate_stack_time_minutes(
            frameCount=len(self.dataSet.get_fovs()), secondsPerFrame=0.2,
            baselineMinutes=5)

    def get_dependencies(self):
        return [self.parameters['neighbor_registration_task']]

    def _get_fov_offset(self, fov: int) -> Tuple[float, float]:
        return self._load_corrected_positions()[fov]

    def _load_corrected_positions(self) -> dict:
        if self._correctedPositions is None:
            positionsDF = self.dataSet.load_dataframe_from_csv(
                'corrected_positions', self)
            self._correctedPositions = {
                int(row.fov): (float(row.x), float(row.y))
                for row in positionsDF.itertuples()
            }
        return self._correctedPositions

    def _run_analysis(self):
        fovs, nominalPositions, micronsPerPixel, stepSizeUm, overlapFraction = \
            _nominal_positions_and_overlap(
                self.dataSet, self.parameters['overlap_fraction'])

        fiducialChannel = self.parameters['fiducial_data_channel']

        def load_frame(fov):
            return self.dataSet.get_fiducial_image(fiducialChannel, fov)

        correspondences = []
        for fov in fovs:
            correspondences.extend(self.registrationTask.return_exported_data(fov))

        kept, rejected = globalpositions.filter_correspondence_outliers(
            correspondences, mad_threshold=self.parameters['mad_threshold'])

        correction = globalpositions.fit_global_positions(
            kept, nominalPositions,
            lsqr_atol=self.parameters['lsqr_atol'],
            lsqr_btol=self.parameters['lsqr_btol'])

        # Merge over the full nominal grid as a fallback for any fov outside
        # the solved component(s) -- e.g. an isolated fov with no surviving
        # neighbour correspondence.
        correctedPositions = {**nominalPositions, **correction.positions}

        self.dataSet.save_dataframe_to_csv(
            pd.DataFrame([
                {'fov': f, 'x': correctedPositions[f][0], 'y': correctedPositions[f][1]}
                for f in fovs
            ]),
            'corrected_positions', self)

        if correspondences:
            keptKeys = {(c.anchor_fov, c.neighbor_fov, c.direction) for c in kept}
            # Only kept correspondences fed the correction fit, so only
            # those are meaningful to score against the FINAL positions;
            # rejected ones get no correlation (NaN).
            correlations = globalpositions.compute_overlap_correlations(
                kept, correctedPositions, nominalPositions, load_frame,
                pixel_size_um=micronsPerPixel, overlap_fraction=overlapFraction)
            self.dataSet.save_dataframe_to_csv(
                pd.DataFrame([
                    {'anchor_fov': c.anchor_fov, 'neighbor_fov': c.neighbor_fov,
                     'direction': c.direction,
                     'nominal_x': c.nominal_xy[0], 'nominal_y': c.nominal_xy[1],
                     'measured_x': c.measured_xy[0], 'measured_y': c.measured_xy[1],
                     'error': c.error,
                     'kept': (c.anchor_fov, c.neighbor_fov, c.direction) in keptKeys,
                     'correlation': correlations.get(
                         (c.anchor_fov, c.neighbor_fov, c.direction), np.nan)}
                    for c in correspondences
                ]),
                'neighbor_correspondences', self)

        self.dataSet.save_json_analysis_result(
            {'n_correspondences': len(correspondences), 'n_kept': len(kept),
             'n_rejected': len(rejected), 'n_components': correction.n_components,
             'residual_rms_um': correction.residual_rms_um,
             'step_size_um': stepSizeUm, 'overlap_fraction': overlapFraction},
            'correction_summary', self.analysisName)

    def _generate_verification_figures(self) -> None:
        """Generate this task's verification figures -- see
        merlin.plots.globalalignplots for the actual plotting code.
        """
        from merlin.plots import globalalignplots
        globalalignplots.generate_all(self)
