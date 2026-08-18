import logging

from snakemake_interface_logger_plugins.common import LogEvent

from merlin.merlin import _SlurmDriverLogFilter


def _record(msg, **extra):
    record = logging.LogRecord(
        'snakemake.logging', logging.INFO, __file__, 1, msg, None, None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_slurm_driver_log_filter_submission_with_fragment():
    logFilter = _SlurmDriverLogFilter('/out/data/snakemake')
    record = _record(
        'Job 539 has been submitted with SLURM jobid 40127567 (log: '
        '/out/data/snakemake/slurm_logs/rule_Optimize02/20/40127567.log).')
    logFilter.filter(record)
    message = record.getMessage()
    assert message.startswith('[')
    assert message.endswith(
        'Submitted jobid: 539 (slurm_id: 40127567) '
        '(Rule: Optimize02) (Fragment: 20)')


def test_slurm_driver_log_filter_submission_without_fragment():
    logFilter = _SlurmDriverLogFilter('/out/data/snakemake')
    record = _record(
        'Job 4 has been submitted with SLURM jobid 40133944 (log: '
        '/out/data/snakemake/slurm_logs/rule_AdaptiveFilterBarcodesDone/'
        '40133944.log).')
    logFilter.filter(record)
    assert record.getMessage().endswith(
        'Submitted jobid: 4 (slurm_id: 40133944) '
        '(Rule: AdaptiveFilterBarcodesDone)')


def test_slurm_driver_log_filter_finish_uses_submission_info():
    logFilter = _SlurmDriverLogFilter('/out/data/snakemake')
    logFilter.filter(_record(
        'Job 539 has been submitted with SLURM jobid 40127567 (log: '
        '/out/data/snakemake/slurm_logs/rule_Optimize02/20/40127567.log).'))
    finishRecord = _record(
        'Finished jobid: 539 (Rule: Optimize02)',
        event=LogEvent.JOB_FINISHED, job_id=539)
    logFilter.filter(finishRecord)
    assert finishRecord.getMessage() == (
        'Finished jobid: 539 (slurm_id: 40127567) '
        '(Rule: Optimize02) (Fragment: 20)')


def test_slurm_driver_log_filter_finish_without_prior_submission():
    # e.g. the local 'all' rule, which is never submitted via SLURM.
    logFilter = _SlurmDriverLogFilter('/out/data/snakemake')
    finishRecord = _record(
        'Finished jobid: 0 (Rule: all)',
        event=LogEvent.JOB_FINISHED, job_id=0)
    logFilter.filter(finishRecord)
    assert finishRecord.getMessage() == 'Finished jobid: 0 (Rule: all)'


def test_slurm_driver_log_filter_reformats_command():
    logFilter = _SlurmDriverLogFilter('/out/data/snakemake')
    cmd = ('/n/x/bin/merlin -k /n/x/parameters.json '
           '-a /n/x/analysis.json -n 4 -e /n/x/experiments '
           '-s /n/x/merlin BC553_sample_02_test/epi/data')
    record = _record('', event=LogEvent.WORKFLOW_STARTED, cmd=cmd)
    logFilter.filter(record)
    assert record.cmd == (
        '\n/n/x/bin/merlin'
        '\n\t-k /n/x/parameters.json'
        '\n\t-a /n/x/analysis.json'
        '\n\t-n 4'
        '\n\t-e /n/x/experiments'
        '\n\t-s /n/x/merlin BC553_sample_02_test/epi/data')
