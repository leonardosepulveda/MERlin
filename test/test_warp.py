import numpy as np
import pytest
from skimage import registration

from merlin.analysis import warp
from merlin.util import fixedpattern


def _make_restricted_warp_task(ragged_merfish_data, analysisName):
    # edge_width_to_remove/percentile_pixel_to_keep relaxed from their
    # defaults, matching conftest.ragged_warp_task -- the tiny synthetic
    # ragged test images are much smaller than the default edge crop.
    return warp.FiducialCorrelationWarp(
        ragged_merfish_data,
        parameters={
            'edge_width_to_remove': 0,
            'percentile_pixel_to_keep': 100,
            'channels_to_process': ['bit1', 'bit2'],
        },
        analysisName=analysisName)


def test_warp_channels_to_process_default_processes_every_channel(
        ragged_warp_task, ragged_merfish_data):
    dataOrg = ragged_merfish_data.get_data_organization()
    allChannels = list(dataOrg.get_data_channels())
    assert ragged_warp_task._channels_to_process() == allChannels
    # every channel's transformation is readable -- no restriction applied
    for channel in allChannels:
        ragged_warp_task.get_transformation(0, channel)


def test_warp_channels_to_process_restricts_computation(ragged_merfish_data):
    dataOrg = ragged_merfish_data.get_data_organization()
    task = _make_restricted_warp_task(ragged_merfish_data, 'restrictedWarp')
    task.save()

    expectedChannels = [dataOrg.get_data_channel_index('bit1'),
                        dataOrg.get_data_channel_index('bit2')]
    assert task._channels_to_process() == expectedChannels

    task._run_analysis(0)

    # the processed channels resolve to real (non-placeholder) transforms
    for channel in expectedChannels:
        task.get_transformation(0, channel)

    # a configured-but-unprocessed channel is guarded, not silently
    # returned as an identity placeholder
    otherChannel = dataOrg.get_data_channel_index('bit3')
    with pytest.raises(ValueError):
        task.get_transformation(0, otherChannel)
    with pytest.raises(ValueError):
        task.get_aligned_image(0, otherChannel, 0)


def _synthetic_fiducial_frames(shift, fovCount, seed=0, size=256):
    """Frames sharing one camera fixed pattern (row offsets + per-pixel
    offsets + a smooth illumination), each with its own sparse Gaussian
    beads of amplitude ~0.5x the pattern std, plus noise. Returns (frame
    of FOV 0, the same FOV's beads moved by *shift* (dy, dx), frames of
    *fovCount* other FOVs)."""
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[:size, :size].astype(float)
    pattern = (rng.normal(0, 20, (size, 1)) + rng.normal(0, 20, (size, size))
               + 200 * np.exp(-((rows - size / 2) ** 2 + (cols - size / 2) ** 2)
                              / (2 * (size / 3) ** 2)))
    amplitude = 0.5 * pattern.std()

    def beads(centers):
        image = np.zeros((size, size))
        for y, x in centers:
            image += amplitude * np.exp(-((rows - y) ** 2 + (cols - x) ** 2) / (2 * 1.5 ** 2))
        return image

    def frame(beadImage):
        return (1000 + pattern + beadImage
                + rng.normal(0, 5, (size, size))).astype(np.float32)

    centers = rng.uniform(20, size - 20, (80, 2))
    fixed = frame(beads(centers))
    moving = frame(beads(centers + np.asarray(shift)))
    others = [frame(beads(rng.uniform(20, size - 20, (80, 2))))
              for _ in range(fovCount)]
    return fixed, moving, others


def test_fiducial_template_removes_camera_pattern_lock():
    # the moving frame's beads sit at +shift, so phase_cross_correlation
    # (fixed, moving) returns -shift
    trueShift = np.array([3.4, -2.7])
    fixed, moving, others = _synthetic_fiducial_frames(trueShift, fovCount=18)
    filterTask = warp.FiducialCorrelationWarp(
        None, parameters={'edge_width_to_remove': 0})

    def measured_shift(a, b):
        return registration.phase_cross_correlation(
            filterTask._filter(a), filterTask._filter(b),
            upsample_factor=100)[0]

    rawShift = measured_shift(fixed, moving)
    np.testing.assert_allclose(rawShift, [0, 0], atol=0.1)

    # one template per round, as FiducialTemplate builds them: subtracting
    # one shared template would leave its own estimation noise in both
    # frames, locked at zero shift again
    fixedTemplate = fixedpattern.median_template(others[:9])
    movingTemplate = fixedpattern.median_template(others[9:])
    templateShift = measured_shift(fixed - fixedTemplate, moving - movingTemplate)
    np.testing.assert_allclose(templateShift, -trueShift, atol=0.1)


def test_fiducial_template_task_feeds_warp(simple_merfish_data):
    templateTask = warp.FiducialTemplate(
        simple_merfish_data, parameters={'n_fovs': 2},
        analysisName='fiducialTemplate')
    templateTask.save()
    for fragment in range(templateTask.fragment_count()):
        templateTask._run_analysis(fragment)

    template = templateTask.get_template(0)
    expected = fixedpattern.median_template([
        fixedpattern.load_fiducial_frame(simple_merfish_data, 0, fov)
        for fov in simple_merfish_data.get_fovs()])
    np.testing.assert_array_equal(template, expected)

    warpTask = warp.FiducialCorrelationWarp(
        simple_merfish_data,
        parameters={'fiducial_template_task': 'fiducialTemplate',
                    'remove_hot_pixels': True},
        analysisName='templateWarp')
    assert warpTask.get_dependencies() == ['fiducialTemplate']
    image = warpTask._registration_image(0, 0)
    np.testing.assert_allclose(
        image, fixedpattern.load_fiducial_frame(simple_merfish_data, 0, 0)
        - template)

    # the template was built with hot pixels removed; a warp that keeps
    # them would not match it
    mismatched = warp.FiducialCorrelationWarp(
        simple_merfish_data,
        parameters={'fiducial_template_task': 'fiducialTemplate'},
        analysisName='mismatchedTemplateWarp')
    with pytest.raises(ValueError):
        mismatched._registration_image(0, 0)
