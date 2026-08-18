from merlin.util.naming import task_initials


def test_task_initials_default_length():
    assert task_initials('FiducialCorrelationWarp') == 'FidCorWar'
    assert task_initials('PlotPerformance') == 'PloPer'
    assert task_initials('FiducialCorrelationWarpDone') == 'FidCorWarDon'
    assert task_initials('DeconvolutionPreprocess') == 'DecPre'
    assert task_initials('Optimize01') == 'Opt01'
    assert task_initials('Decode') == 'Dec'
    assert task_initials('GenerateAdaptiveThreshold') == 'GenAdaThr'
    assert task_initials('AdaptiveFilterBarcodes') == 'AdaFilBar'


def test_task_initials_custom_length():
    assert task_initials('FiducialCorrelationWarp', length=4) == \
        'FiduCorrWarp'
    assert task_initials('Decode', length=1) == 'D'
