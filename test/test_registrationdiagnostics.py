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
