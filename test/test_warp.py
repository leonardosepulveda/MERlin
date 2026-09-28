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


def _warp_with_round_shifts(ragged_merfish_data, analysisName, parameters):
    """A FiducialCorrelationWarp whose fiducial images are one bead field
    moved by a known amount per imaging round (the synthetic raw fiducials
    have no shift between rounds)."""
    dataOrg = ragged_merfish_data.get_data_organization()
    rng = np.random.default_rng(0)
    beads = np.zeros((128, 128))
    for y, x in rng.integers(10, 118, (40, 2)):
        beads[y - 1:y + 2, x - 1:x + 2] = 1000
    roundShifts = {'bit1': (0, 0), 'bit2': (0, 0), 'bit3': (3, -2),
                   'bit4': (3, -2), 'DAPI': (-4, 5), 'polyT': (-4, 5)}

    task = warp.FiducialCorrelationWarp(
        ragged_merfish_data,
        parameters={'edge_width_to_remove': 0, 'percentile_pixel_to_keep': 100,
                    **parameters},
        analysisName=analysisName)
    task._registration_image = lambda channel, fov: np.roll(
        beads, roundShifts[dataOrg.get_data_channel_name(channel)],
        axis=(0, 1)).astype(np.float32)
    task.save()
    task._run_analysis(0)
    return task


def test_warp_restricted_instance_shares_default_reference(ragged_merfish_data):
    """Unset, the reference is the first data channel in every instance, so
    a DAPI-only warp gives DAPI the same transform as the full warp does.
    Using the first listed channel instead made it the identity."""
    dataOrg = ragged_merfish_data.get_data_organization()
    dapi = dataOrg.get_data_channel_index('DAPI')
    full = _warp_with_round_shifts(ragged_merfish_data, 'shiftedFullWarp', {})
    restricted = _warp_with_round_shifts(
        ragged_merfish_data, 'shiftedDapiWarp', {'channels_to_process': ['DAPI']})

    expected = full.get_transformation(0, dapi).params
    assert not np.allclose(expected, np.eye(3))
    np.testing.assert_allclose(
        restricted.get_transformation(0, dapi).params, expected, atol=1e-6)


def test_warp_reference_channel(ragged_merfish_data):
    dataOrg = ragged_merfish_data.get_data_organization()
    dapi = dataOrg.get_data_channel_index('DAPI')
    bit1 = dataOrg.get_data_channel_index('bit1')
    task = _warp_with_round_shifts(
        ragged_merfish_data, 'dapiReferenceWarp', {'reference_channel': 'DAPI'})

    np.testing.assert_allclose(
        task.get_transformation(0, dapi).params, np.eye(3), atol=1e-6)
    assert not np.allclose(task.get_transformation(0, bit1).params, np.eye(3))


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


def test_fiducial_template_task_feeds_warp(simple_merfish_data, monkeypatch):
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

    # unset, remove_hot_pixels follows the template (built with them
    # removed); an explicit mismatch would subtract, or leave, hot pixels
    inheriting = warp.FiducialCorrelationWarp(
        simple_merfish_data,
        parameters={'fiducial_template_task': 'fiducialTemplate'},
        analysisName='inheritingTemplateWarp')
    assert inheriting.parameters['remove_hot_pixels'] is None
    hotPixelCalls = []
    removeHotPixels = warp.globalpositions.remove_hot_pixels
    monkeypatch.setattr(warp.globalpositions, 'remove_hot_pixels',
                        lambda x: hotPixelCalls.append(1) or removeHotPixels(x))
    np.testing.assert_array_equal(inheriting._registration_image(0, 0), image)
    assert hotPixelCalls == [1]
    monkeypatch.undo()
    mismatched = warp.FiducialCorrelationWarp(
        simple_merfish_data,
        parameters={'fiducial_template_task': 'fiducialTemplate',
                    'remove_hot_pixels': False},
        analysisName='mismatchedTemplateWarp')
    with pytest.raises(ValueError):
        mismatched._registration_image(0, 0)
    # without a template the default is unchanged
    assert warp.FiducialCorrelationWarp(
        simple_merfish_data).parameters['remove_hot_pixels'] is False
