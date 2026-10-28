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
    from fov_count fovs drawn at random (fixed seed) across the whole
    footprint, and averages them with a running sum. Few frames, or frames
    from tissue-edge fovs only, leave the tissue texture in the mean.

    Optionally (dark_frame_indices) the camera's dark offset is estimated
    from blank frames of the same movies -- frames with no illumination,
    which the acquisition's frame table marks with no color -- and saved
    as 'dark_frame'. It does not depend on color. It is subtracted from
    every frame before averaging, and apply_ffc subtracts it too.

    Consumption is opt-in: a task that wants FFC applied loads this task by
    name and calls get_ffc_field/get_ffc_field_for_channel on it (mirroring
    the existing optimize_task/ChromaticCorrector pattern), then divides it
    out of the images it reads. This task itself only computes and caches
    the fields; wiring is done separately in each consuming task.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'fov_count' not in self.parameters:
            self.parameters['fov_count'] = 300
        if 'random_seed' not in self.parameters:
            self.parameters['random_seed'] = 0
        if 'dark_frame_indices' not in self.parameters:
            self.parameters['dark_frame_indices'] = None
        if 'dark_fov_count' not in self.parameters:
            self.parameters['dark_fov_count'] = 40
        if 'smooth_sigma' not in self.parameters:
            self.parameters['smooth_sigma'] = 50
        if 'normalize_percentile' not in self.parameters:
            self.parameters['normalize_percentile'] = 50
        if 'minimum_value' not in self.parameters:
            self.parameters['minimum_value'] = 0.1

    def get_estimated_memory(self):
        return 2000

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return []

    def _select_fovs(self):
        """fov_count fovs drawn at random across the whole footprint with
        random_seed, or all fovs if there are fewer.
        """
        fovs = self.dataSet.get_fovs()
        n = min(self.parameters['fov_count'], len(fovs))
        rng = np.random.RandomState(self.parameters['random_seed'])
        return [fovs[i] for i in np.sort(
            rng.choice(len(fovs), n, replace=False))]

    def _compute_dark_frame(self, dataChannel, fovs):
        """Per-pixel mean over the blank frames of the first dark_fov_count
        fovs, read from the raw file of dataChannel.
        """
        fovs = fovs[:self.parameters['dark_fov_count']]
        total = None
        count = 0
        for fov in fovs:
            filename = self.dataSet.get_data_organization()\
                .get_image_filename(dataChannel, fov)
            for index in self.parameters['dark_frame_indices']:
                frame = self.dataSet.load_image(
                    filename, index).astype(np.float64)
                total = frame if total is None else total + frame
                count += 1
        return (total / count).astype(np.float32)

    @staticmethod
    def _representative_channel_for_color(dataOrganization, color):
        for d in dataOrganization.get_data_channels():
            if dataOrganization.get_data_channel_color(d) == color:
                return d
        raise ValueError('No data channel found for color %s' % color)

    def _compute_field_for_color(self, dataChannel, fovs, dark=None):
        total = None
        for fov in fovs:
            zPositions = self.dataSet.get_z_positions(fov)
            zPosition = zPositions[len(zPositions) // 2]
            frame = self.dataSet.get_raw_image(
                dataChannel, fov, zPosition).astype(np.float64)
            if dark is not None:
                frame -= dark
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

        dark = None
        if self.parameters['dark_frame_indices'] is not None:
            dark = self._compute_dark_frame(
                self._representative_channel_for_color(
                    dataOrganization, colors[0]), fovs)
            self.dataSet.save_numpy_analysis_result(dark, 'dark_frame', self)

        for color in colors:
            dataChannel = self._representative_channel_for_color(
                dataOrganization, color)
            field = self._compute_field_for_color(dataChannel, fovs, dark)
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

    def get_dark_frame(self):
        """Get the cached dark frame, or None if the task ran without
        dark_frame_indices.
        """
        return self.dataSet.load_numpy_analysis_result_if_available(
            'dark_frame', self, None)

    @staticmethod
    def apply_ffc(image: np.ndarray, field: np.ndarray,
                  dark: np.ndarray = None) -> np.ndarray:
        """Compute (image - dark) / field, clipping negative results to
        zero. dark=None skips the dark subtraction.
        """
        image = image.astype(np.float32)
        if dark is not None:
            image = image - dark
        return np.clip(image / field, 0, None)
