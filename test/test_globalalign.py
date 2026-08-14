import os
import numpy as np
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
    # test_positions.csv places the 2 fovs 195um apart in y with 0 offset in
    # x -- the synthetic fixture images have no real overlapping tissue
    # content, but this still exercises the full real code path: neighbour
    # lookup, registration, outlier filtering, the joint lsqr solve, and the
    # corrected_positions CSV round-trip.
    task = globalalign.LeastSquaresGlobalAlignment(
        simple_merfish_data, parameters={}, analysisName='leastSquaresGlobalAlign')
    task.save()
    task._run_analysis()

    summary = simple_merfish_data.load_json_analysis_result(
        'correction_summary', task.analysisName)
    assert summary['n_correspondences'] == 2  # fov0's +y pass and fov1's -y pass
    assert summary['n_components'] == 1

    nominal0 = simple_merfish_data.get_fov_offset(0)
    nominal1 = simple_merfish_data.get_fov_offset(1)
    corrected0 = task._get_fov_offset(0)
    corrected1 = task._get_fov_offset(1)

    # The fixture images' overlap crop is exactly 1 row tall (overlap_fraction
    # rounds to 0 given how far apart these fovs are relative to their tiny
    # 128x128 synthetic frames), which leaves phase_cross_correlation no room
    # to report a row (y) shift at all -- the y offset between the two fovs
    # must therefore come through completely unchanged.
    assert corrected0[1] == pytest.approx(nominal0[1], abs=1e-6)
    assert corrected1[1] == pytest.approx(nominal1[1], abs=1e-6)
    assert corrected1[1] - corrected0[1] == pytest.approx(nominal1[1] - nominal0[1], abs=1e-6)

    # x can shift by at most half the fiducial frame's own width in microns
    # (phase_cross_correlation's column-shift search range on a 1-row crop).
    micronsPerPixel = simple_merfish_data.get_microns_per_pixel()
    frameWidthUm = simple_merfish_data.get_image_dimensions()[0] * micronsPerPixel
    assert abs(corrected0[0] - nominal0[0]) <= frameWidthUm / 2 + 1e-6
    assert abs(corrected1[0] - nominal1[0]) <= frameWidthUm / 2 + 1e-6

    # fov_coordinates_to_global reuses the corrected offset transparently.
    global00 = task.fov_coordinates_to_global(0, (0, 0))
    assert global00[0] == pytest.approx(corrected0[0])
    assert global00[1] == pytest.approx(corrected0[1])


def test_least_squares_global_alignment_requires_run_before_offset_lookup(
        simple_merfish_data):
    task = globalalign.LeastSquaresGlobalAlignment(
        simple_merfish_data, parameters={}, analysisName='leastSquaresGlobalAlignUnrun')
    task.save()
    with pytest.raises(FileNotFoundError):
        task._get_fov_offset(0)


def test_least_squares_global_alignment_generates_verification_figures(
        simple_merfish_data):
    # Full task.run() (not _run_analysis() directly), so the
    # generate-figures-after-completion hook actually fires.
    task = globalalign.LeastSquaresGlobalAlignment(
        simple_merfish_data, parameters={}, analysisName='leastSquaresGlobalAlignFigures')
    task.save()
    task.run()
    assert task.is_complete()

    figuresDir = os.sep.join([simple_merfish_data.analysisPath, 'figures'])
    for figureName in ('direction_reliability', 'grid_overlay'):
        figurePath = os.sep.join(
            [figuresDir, '.'.join([task.analysisName, figureName]) + '.png'])
        assert os.path.exists(figurePath), figurePath
