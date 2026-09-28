import warnings

import pytest

from merlin.analysis import registrationdiagnostics
from merlin.analysis import warp


def test_registration_diagnostics_writes_report(simple_merfish_data):
    # A smoke test: the fixture has 2 small, non-overlapping fovs, so this
    # checks the report's structure, not its verdicts (those were checked
    # on real data, see the class docstring). A saved warp with no edge
    # crop supplies the filter, since the default 200 px crop would blank
    # these 128 px frames.
    warp.FiducialCorrelationWarp(
        simple_merfish_data, parameters={'edge_width_to_remove': 0},
        analysisName='diagnosticsWarp').save()
    task = registrationdiagnostics.RegistrationDiagnostics(
        simple_merfish_data,
        parameters={'n_fovs': 2, 'n_template_fovs': 2,
                    'warp_task': 'diagnosticsWarp'},
        analysisName='registrationDiagnostics')
    task.save()
    task._run_analysis()

    report = simple_merfish_data.load_json_analysis_result(
        'registration_diagnostics', task.analysisName)
    assert report['comparison_data_channel'] == 2   # bit3, the next round
    assert set(report['check1_pattern_lock']) >= {'raw', 'template', 'flagged'}
    assert set(report['check2_first_vs_last']) == {'raw', 'template'}
    assert report['check3_stitching_bands']['raw']['n_real_edges'] == 0
    assert report['check4_global_alignment'] == {'available': False}
    assert set(report['recommendation']) == {
        'FiducialCorrelationWarp', 'FiducialCorrelationWarp_reasons',
        'RegisterFovNeighbors', 'RegisterFovNeighbors_reasons'}


def test_registration_diagnostics_fail_on_warning(simple_merfish_data):
    task = registrationdiagnostics.RegistrationDiagnostics(
        simple_merfish_data,
        parameters={'n_fovs': 2, 'n_template_fovs': 2,
                    'second_fiducial_frame': None,
                    'warp_task': 'diagnosticsWarp', 'fail_on_warning': True},
        analysisName='registrationDiagnosticsStrict')
    task.save()
    report = {'warning': 'forced'}
    task._check4 = lambda: dict(report, available=True, flagged=True)
    with pytest.raises(RuntimeError, match='forced'):
        task._run_analysis()


@pytest.mark.parametrize(
    'rotation, cameraRotation, final, affineOnly, flagged, problems', [
        # BC555 disk beads (LT074): rotation and held-out error both failed
        (-0.146, -0.285, 6.37, 1.27, True, ['rotation', 'held-out']),
        # rotation alone, held-out fine
        (-0.146, -0.285, 0.05, 0.5, True, ['rotation']),
        # held-out alone, rotation fine
        (-0.28, -0.285, 6.37, 1.27, True, ['held-out']),
        # BC555 epi beads: passes
        (-0.944, -0.95, 0.052, 0.544, False, []),
    ])
def test_check4_flags_failed_alignment(
        simple_merfish_data, monkeypatch, rotation, cameraRotation, final,
        affineOnly, flagged, problems):
    monkeypatch.setattr(simple_merfish_data, 'cameraRotationDeg',
                        cameraRotation)
    simple_merfish_data.save_json_analysis_result(
        {'affine_rotation_deg': rotation,
         'heldout_edge_error_um': {
             'final': {'median': final, 'p90': 2 * final},
             'affine_only': {'median': affineOnly, 'p90': 2 * affineOnly}}},
        'correction_summary', 'syntheticAlignment')
    task = registrationdiagnostics.RegistrationDiagnostics(
        simple_merfish_data,
        parameters={'global_alignment_task': 'syntheticAlignment'})

    out = task._check4()

    assert out['flagged'] is flagged
    assert out['camera_rotation_deg'] == cameraRotation
    assert 'rotation_test' not in out
    warning = out.get('warning', '')
    assert ('rotation' in warning) is ('rotation' in problems)
    assert ('held-out' in warning) is ('held-out' in problems)


def test_check4_without_camera_rotation_skips_rotation_test(
        simple_merfish_data, monkeypatch):
    monkeypatch.setattr(simple_merfish_data, 'cameraRotationDeg', None)
    simple_merfish_data.save_json_analysis_result(
        {'affine_rotation_deg': -0.146,
         'heldout_edge_error_um': {
             'final': {'median': 0.05, 'p90': 0.1},
             'affine_only': {'median': 0.5, 'p90': 1.0}}},
        'correction_summary', 'syntheticAlignmentNoRotation')
    task = registrationdiagnostics.RegistrationDiagnostics(
        simple_merfish_data,
        parameters={'global_alignment_task': 'syntheticAlignmentNoRotation'})

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        out = task._check4()

    assert out['flagged'] is False
    assert 'warning' not in out
    assert out['rotation_test'] == ('skipped: the microscope parameters '
                                    'have no camera_rotation_deg')
