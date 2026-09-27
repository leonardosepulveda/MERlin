import os
import numpy as np
import pandas as pd
import pytest

from merlin.analysis import globalalign


def test_simple_global_alignment_fov_coordinates_to_global(simple_merfish_data):
    task = globalalign.SimpleGlobalAlignment(simple_merfish_data, parameters={})
    micronsPerPixel = simple_merfish_data.get_microns_per_pixel()
    fov0Offset = simple_merfish_data.get_fov_offset(0)
    global00 = task.fov_coordinates_to_global(0, (0, 0))
    assert global00[0] == pytest.approx(fov0Offset[0])
    assert global00[1] == pytest.approx(fov0Offset[1])

    global1010 = task.fov_coordinates_to_global(0, (10, 10))
    assert global1010[0] == pytest.approx(fov0Offset[0] + 10 * micronsPerPixel)
    assert global1010[1] == pytest.approx(fov0Offset[1] + 10 * micronsPerPixel)


def test_least_squares_global_alignment_runs_and_persists(simple_merfish_data):
    # test_positions.csv places the 2 fovs 195um apart in y, far more than
    # the tiny 128x128 synthetic frames are wide, so they do not overlap at
    # all. This still exercises the full code path -- per-fov neighbour
    # lookup (RegisterFovNeighbors, run first since
    # LeastSquaresGlobalAlignment only consumes its output), the joint
    # solve and the corrected_positions CSV round-trip -- and checks that a
    # non-overlapping pair is not registered, so both fovs keep their
    # nominal positions.
    registrationTask = globalalign.RegisterFovNeighbors(
        simple_merfish_data, parameters={},
        analysisName='registerFovNeighbors_runsAndPersists')
    registrationTask.save()
    registrationTask.run()

    task = globalalign.LeastSquaresGlobalAlignment(
        simple_merfish_data,
        parameters={'neighbor_registration_task': registrationTask.analysisName},
        analysisName='leastSquaresGlobalAlign')
    task.save()
    task._run_analysis()

    summary = simple_merfish_data.load_json_analysis_result(
        'correction_summary', task.analysisName)
    assert summary['n_correspondences'] == 0

    for fov in (0, 1):
        nominal = simple_merfish_data.get_fov_offset(fov)
        corrected = task._get_fov_offset(fov)
        assert corrected[0] == pytest.approx(nominal[0], abs=1e-6)
        assert corrected[1] == pytest.approx(nominal[1], abs=1e-6)

    # fov_coordinates_to_global reuses the corrected offset transparently.
    corrected0 = task._get_fov_offset(0)
    global00 = task.fov_coordinates_to_global(0, (0, 0))
    assert global00[0] == pytest.approx(corrected0[0])
    assert global00[1] == pytest.approx(corrected0[1])


def test_least_squares_global_alignment_requires_run_before_offset_lookup(
        simple_merfish_data):
    # LeastSquaresGlobalAlignment.__init__ loads its neighbor_registration_task
    # dependency eagerly, so that dependency must exist (saved) even though
    # this test never runs either task.
    registrationTask = globalalign.RegisterFovNeighbors(
        simple_merfish_data, parameters={},
        analysisName='registerFovNeighbors_requiresRun')
    registrationTask.save()

    task = globalalign.LeastSquaresGlobalAlignment(
        simple_merfish_data,
        parameters={'neighbor_registration_task': registrationTask.analysisName},
        analysisName='leastSquaresGlobalAlignUnrun')
    task.save()
    with pytest.raises(FileNotFoundError):
        task._get_fov_offset(0)


def test_least_squares_global_alignment_generates_verification_figures(
        simple_merfish_data):
    # Full task.run() (not _run_analysis() directly), so the
    # generate-figures-after-completion hook actually fires. The fixture's
    # two fovs do not overlap, so registration itself finds nothing to
    # measure; known correspondences are written in its place so every
    # figure has data to plot.
    registrationTask = globalalign.RegisterFovNeighbors(
        simple_merfish_data, parameters={},
        analysisName='registerFovNeighbors_figures')
    registrationTask.save()
    nominal = {f: simple_merfish_data.get_fov_offset(f) for f in (0, 1)}
    for anchor, neighbor, direction in ((0, 1, '+y'), (1, 0, '-y')):
        simple_merfish_data.save_dataframe_to_csv(
            pd.DataFrame([{
                'anchor_fov': anchor, 'neighbor_fov': neighbor,
                'direction': direction,
                'nominal_x': nominal[neighbor][0],
                'nominal_y': nominal[neighbor][1],
                'measured_x': nominal[neighbor][0] + 0.5,
                'measured_y': nominal[neighbor][1] + 0.2, 'error': 0.1}]),
            'neighbor_correspondences_raw', registrationTask,
            resultIndex=anchor)

    task = globalalign.LeastSquaresGlobalAlignment(
        simple_merfish_data,
        parameters={'neighbor_registration_task': registrationTask.analysisName},
        analysisName='leastSquaresGlobalAlignFigures')
    task.save()
    task.run()
    assert task.is_complete()

    figuresDir = simple_merfish_data.figuresPath
    for figureName in ('direction_reliability', 'grid_overlay',
                       'overlap_correlation_grid', 'overlap_correlation_histogram'):
        figurePath = os.sep.join(
            [figuresDir,
             '.'.join(['merlin', task.analysisName, figureName]) + '.png'])
        assert os.path.exists(figurePath), figurePath


def test_registration_image_max_projection_by_channel_name(simple_merfish_data):
    # The fixture's channels have one z plane each, so the max projection
    # is that plane; the channel is given by name.
    parameters = {'max_projection_data_channel': 'bit2',
                  'fiducial_data_channel': 0, 'remove_hot_pixels': False}
    image = globalalign._load_registration_image(
        simple_merfish_data, parameters, 0)
    expected = simple_merfish_data.get_raw_image(1, 0, 0)
    np.testing.assert_array_equal(image, expected)

    parameters['max_projection_data_channel'] = None
    image = globalalign._load_registration_image(
        simple_merfish_data, parameters, 0)
    np.testing.assert_array_equal(
        image, simple_merfish_data.get_fiducial_image(0, 0))
