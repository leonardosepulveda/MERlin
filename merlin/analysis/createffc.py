import numpy as np
from scipy.ndimage import gaussian_filter

from merlin.core import analysistask


class CreateFfc(analysistask.AnalysisTask):

    """
    An analysis task that estimates a flat-field correction (FFC) field --
    a per-pixel illumination/vignetting profile -- for each imaging color
    used in the experiment.

    Vignetting is a fixed property of the microscope/objective/color, not of
    any particular imaging round or data channel, so one field is estimated
    per color (see DataOrganization.get_data_channel_color) rather than per
    data channel; every data channel sharing a color reuses the same field.
    This task reads raw frames directly (no dependency on Warp/GlobalAlign)
    from a small sample of fovs -- the fovs farthest from the imaged
    footprint's centroid, a layout-agnostic proxy for "exterior fovs" (least
    likely to be dominated by real tissue signal near the frame edges) that
    needs nothing beyond the stage positions MERlin already loads.

    Consumption is opt-in: a task that wants FFC applied loads this task by
    name and calls get_ffc_field/get_ffc_field_for_channel on it (mirroring
    the existing optimize_task/ChromaticCorrector pattern), then divides it
    out of the images it reads. This task itself only computes and caches
    the fields; wiring is done separately in each consuming task.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'fov_count' not in self.parameters:
            self.parameters['fov_count'] = 10
        if 'smooth_sigma' not in self.parameters:
            self.parameters['smooth_sigma'] = 50
        if 'normalize_percentile' not in self.parameters:
            self.parameters['normalize_percentile'] = 99.99
        if 'minimum_value' not in self.parameters:
            self.parameters['minimum_value'] = 0.1

    def get_estimated_memory(self):
        return 2000

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return []

    def _select_fovs(self):
        """The fov_count fovs farthest from the imaged footprint's centroid.
        """
        fovs = self.dataSet.get_fovs()
        positions = np.array(
            [self.dataSet.get_fov_offset(f) for f in fovs])
        centroid = positions.mean(axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        order = np.argsort(distances)[::-1]
        n = min(self.parameters['fov_count'], len(fovs))
        return [fovs[i] for i in order[:n]]

    @staticmethod
    def _representative_channel_for_color(dataOrganization, color):
        for d in dataOrganization.get_data_channels():
            if dataOrganization.get_data_channel_color(d) == color:
                return d
        raise ValueError('No data channel found for color %s' % color)

    def _compute_field_for_color(self, dataChannel, fovs):
        total = None
        for fov in fovs:
            zPositions = self.dataSet.get_z_positions(fov)
            zPosition = zPositions[len(zPositions) // 2]
            frame = self.dataSet.get_raw_image(
                dataChannel, fov, zPosition).astype(np.float64)
            total = frame if total is None else total + frame

        field = (total / len(fovs)).astype(np.float32)
        field = gaussian_filter(field, sigma=self.parameters['smooth_sigma'])

        normValue = np.percentile(
            field, self.parameters['normalize_percentile'])
        if normValue > 0:
            field = field / normValue
        return np.clip(
            field, self.parameters['minimum_value'], None).astype(np.float32)

    def _run_analysis(self):
        dataOrganization = self.dataSet.get_data_organization()
        colors = sorted({dataOrganization.get_data_channel_color(d)
                          for d in dataOrganization.get_data_channels()})
        fovs = self._select_fovs()

        for color in colors:
            dataChannel = self._representative_channel_for_color(
                dataOrganization, color)
            field = self._compute_field_for_color(dataChannel, fovs)
            self.dataSet.save_numpy_analysis_result(
                field, 'ffc_field_%s' % color, self)

    def get_ffc_field(self, color: str) -> np.ndarray:
        """Get the cached flat-field-correction field for the specified
        color.

        Args:
            color: the color, as returned by
                DataOrganization.get_data_channel_color.
        Returns:
            the flat-field-correction field for that color.
        """
        return self.dataSet.load_numpy_analysis_result(
            'ffc_field_%s' % color, self)

    def get_ffc_field_for_channel(self, dataChannel: int) -> np.ndarray:
        """Get the cached flat-field-correction field for the color used by
        the specified data channel.
        """
        color = self.dataSet.get_data_organization()\
            .get_data_channel_color(dataChannel)
        return self.get_ffc_field(color)

    @staticmethod
    def apply_ffc(image: np.ndarray, field: np.ndarray) -> np.ndarray:
        """Divide image by the flat-field-correction field, clipping
        negative results (from any upstream dark-offset subtraction) to
        zero.
        """
        return np.clip(image.astype(np.float32) / field, 0, None)
