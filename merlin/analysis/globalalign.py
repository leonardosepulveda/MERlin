from abc import abstractmethod
import logging
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from typing import Tuple
from typing import List
from shapely import geometry

from merlin.core import analysistask
from merlin.util import globalpositions


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
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        self._correctedPositions = None

        if 'fiducial_data_channel' not in self.parameters:
            self.parameters['fiducial_data_channel'] = 0
        if 'overlap_fraction' not in self.parameters:
            # None -> inferred in _run_analysis from the measured grid step
            # size vs. the fov's own width, same convention the real-data
            # comparison this method is ported from used.
            self.parameters['overlap_fraction'] = None
        if 'tolerance_fraction' not in self.parameters:
            self.parameters['tolerance_fraction'] = 0.25
        if 'mad_threshold' not in self.parameters:
            self.parameters['mad_threshold'] = 5.0
        if 'upsample_factor' not in self.parameters:
            self.parameters['upsample_factor'] = 100
        if 'lsqr_atol' not in self.parameters:
            self.parameters['lsqr_atol'] = 1e-12
        if 'lsqr_btol' not in self.parameters:
            self.parameters['lsqr_btol'] = 1e-12

    def get_estimated_memory(self):
        return 1000

    def get_estimated_time(self):
        return 60

    def get_dependencies(self):
        return []

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
        fovs = self.dataSet.get_fovs()
        nominalPositions = {f: tuple(self.dataSet.get_fov_offset(f))
                            for f in fovs}
        micronsPerPixel = self.dataSet.get_microns_per_pixel()
        frameWidthUm = self.dataSet.get_image_dimensions()[0] * micronsPerPixel

        stepSizeUm = globalpositions.estimate_step_size_um(nominalPositions)
        overlapFraction = self.parameters['overlap_fraction']
        if overlapFraction is None:
            overlapFraction = max(
                0.0, 1.0 - stepSizeUm / frameWidthUm) if frameWidthUm > 0 else 0.0

        fiducialChannel = self.parameters['fiducial_data_channel']

        def load_frame(fov):
            return self.dataSet.get_fiducial_image(fiducialChannel, fov)

        correspondences = globalpositions.sample_neighbor_correspondences(
            fov_ids=fovs, positions=nominalPositions, load_frame=load_frame,
            pixel_size_um=micronsPerPixel,
            overlap_fraction=overlapFraction,
            tolerance_fraction=self.parameters['tolerance_fraction'],
            upsample_factor=self.parameters['upsample_factor'])

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
            self.dataSet.save_dataframe_to_csv(
                pd.DataFrame([
                    {'anchor_fov': c.anchor_fov, 'neighbor_fov': c.neighbor_fov,
                     'direction': c.direction,
                     'nominal_x': c.nominal_xy[0], 'nominal_y': c.nominal_xy[1],
                     'measured_x': c.measured_xy[0], 'measured_y': c.measured_xy[1],
                     'error': c.error,
                     'kept': (c.anchor_fov, c.neighbor_fov, c.direction) in keptKeys}
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
        # Each plot is independently guarded -- e.g. an isolated fov with no
        # surviving neighbour correspondence at all still gets a grid-overlay
        # figure even though the direction-reliability one has nothing to show.
        for plotMethod in (self._plot_direction_reliability, self._plot_grid_overlay):
            try:
                plotMethod()
            except Exception:
                logging.getLogger(self.analysisName).exception(
                    'Failed to generate a %s verification figure' % plotMethod.__name__)

    def _plot_direction_reliability(self) -> None:
        """Per-direction deviation scatter -- ``measured_xy - nominal_xy``
        for every KEPT correspondence, colored by direction. A tight cluster
        for a direction means that direction's registration is reproducible;
        mirrors the real-data comparison in the sibling MERci project's
        `notebooks/tests/compare_stitching_correction_methods.ipynb` (section
        7), which is what first showed this dataset's error is a real,
        per-direction bias rather than noise.
        """
        correspondenceDF = self.dataSet.load_dataframe_from_csv(
            'neighbor_correspondences', self)
        kept = correspondenceDF[correspondenceDF['kept']]
        if kept.empty:
            return

        dx = kept['measured_x'] - kept['nominal_x']
        dy = kept['measured_y'] - kept['nominal_y']
        directionColors = {'+x': 'tab:orange', '-x': 'tab:green',
                           '+y': 'tab:blue', '-y': 'tab:red'}

        fig, ax = plt.subplots(figsize=(6, 6))
        for direction, color in directionColors.items():
            mask = (kept['direction'] == direction).to_numpy()
            if mask.any():
                ax.scatter(dx[mask], dy[mask], s=14, alpha=0.6, color=color,
                          label=direction)
        ax.axhline(0, color='0.85', lw=0.8, zorder=0)
        ax.axvline(0, color='0.85', lw=0.8, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlabel('dx = measured - nominal (microns)')
        ax.set_ylabel('dy = measured - nominal (microns)')
        ax.set_title('%s: per-direction deviation (%i kept correspondences)'
                     % (self.analysisName, len(kept)))
        ax.legend()
        self.dataSet.save_task_figure(self, fig, 'direction_reliability')
        plt.close(fig)

    def _plot_grid_overlay(self) -> None:
        """Overlay each fov's corrected boundary on its original (nominal)
        boundary, at true physical scale -- mirrors the real-data comparison
        in the sibling MERci project's `notebooks/tests/
        compare_stitching_correction_methods.ipynb` (section 11), where a
        visible gap between the gray (original) and colored (corrected)
        squares IS the real, physical size of the correction.
        """
        fovs = self.dataSet.get_fovs()
        nominalPositions = {f: tuple(self.dataSet.get_fov_offset(f)) for f in fovs}
        correctedPositions = self._load_corrected_positions()
        micronsPerPixel = self.dataSet.get_microns_per_pixel()
        frameWidthUm = self.dataSet.get_image_dimensions()[0] * micronsPerPixel
        half = frameWidthUm / 2

        nominalXY = np.array([nominalPositions[f] for f in fovs])
        correctedXY = np.array([correctedPositions[f] for f in fovs])
        shiftUm = np.hypot(correctedXY[:, 0] - nominalXY[:, 0],
                           correctedXY[:, 1] - nominalXY[:, 1])
        margin = frameWidthUm * 2

        fig, ax = plt.subplots(figsize=(8, 8))
        for x, y in nominalXY:
            ax.add_patch(mpatches.Rectangle(
                (x - half, y - half), frameWidthUm, frameWidthUm,
                fill=False, edgecolor='0.75', linewidth=0.6, zorder=1))
        for x, y in correctedXY:
            ax.add_patch(mpatches.Rectangle(
                (x - half, y - half), frameWidthUm, frameWidthUm,
                fill=False, edgecolor='tab:green', linewidth=0.6, zorder=2))

        ax.set_xlim(nominalXY[:, 0].min() - margin, nominalXY[:, 0].max() + margin)
        # Inverted, matching the reference plot's image-like display
        # convention (row 0 / smallest y at the top) -- a display choice
        # only, not a correctness one (unlike crop_overlap's row convention).
        ax.set_ylim(nominalXY[:, 1].max() + margin, nominalXY[:, 1].min() - margin)
        ax.set_aspect('equal')
        ax.set_xlabel('x (microns)')
        ax.set_ylabel('y (microns)')
        ax.legend(handles=[
            mpatches.Patch(facecolor='none', edgecolor='0.75',
                          label='original (nominal) fov boundary'),
            mpatches.Patch(facecolor='none', edgecolor='tab:green',
                          label='corrected fov boundary'),
        ], loc='upper right')
        ax.set_title(
            '%s: corrected vs. nominal fov grid (mean shift=%.3f um, max=%.3f um)'
            % (self.analysisName, shiftUm.mean(), shiftUm.max()))
        self.dataSet.save_task_figure(self, fig, 'grid_overlay')
        plt.close(fig)
