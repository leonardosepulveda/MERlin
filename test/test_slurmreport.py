import subprocess

import pandas
import pytest

from merlin.analysis import slurmreport
from merlin.analysis import testtask


_SACCT_HEADER = slurmreport.SlurmReport._SACCT_FORMAT
_SACCT_COLUMNS = _SACCT_HEADER.split(',')


def _fake_sacct_row(jobId, maxRss, elapsed, submit='2026-08-30T10:00:00',
                     start='2026-08-30T10:01:00'):
    """One pipe-separated sacct output row, in _SACCT_FORMAT column order,
    with placeholder values for every column _clean_slurm_dataframe
    doesn't read."""
    values = {
        'AssocID': '1', 'Account': 'zhuang_lab', 'Cluster': 'test',
        'User': 'test', 'JobID': jobId, 'JobName': 'test',
        'NodeList': 'node1', 'AveCPU': '00:00:01', 'AveCPUFreq': '1G',
        'MaxPages': '0', 'MaxDiskRead': '1M', 'MaxDiskWrite': '1M',
        'MaxRSS': maxRss, 'ReqMem': '8000Mn', 'CPUTime': elapsed,
        'Elapsed': elapsed, 'Submit': submit, 'Start': start,
        'End': start, 'Timelimit': '3:00:00',
    }
    return '|'.join(values[c] for c in _SACCT_COLUMNS)


def _fake_sacct_output(rows):
    return '|'.join(_SACCT_COLUMNS) + '\n' + '\n'.join(rows) + '\n'


@pytest.fixture
def slurm_report_task(simple_data):
    return slurmreport.SlurmReport(simple_data, parameters={})


@pytest.fixture
def upstream_task(simple_data):
    # fragment_count() == 5, independent of simple_data's own fovs
    return testtask.SimpleParallelAnalysisTask(simple_data, parameters={})


def test_fragment_slurm_job_ids_skips_incomplete(
        monkeypatch, slurm_report_task, simple_data, upstream_task):
    # fragments 0 and 2 completed with a recorded environment; 1, 3, 4
    # never ran (still pending, or permanently failed) -- no
    # SLURM_JOB_ID recorded for those.
    envByFragment = {
        0: {'SLURM_JOB_ID': '1001'},
        2: {'SLURM_JOB_ID': '1002'},
    }

    def fake_get_analysis_environment(self, task, fragmentIndex=None):
        return envByFragment.get(fragmentIndex)

    monkeypatch.setattr(
        type(simple_data), 'get_analysis_environment',
        fake_get_analysis_environment)

    result = slurm_report_task._fragment_slurm_job_ids(upstream_task)
    assert result == {0: '1001', 2: '1002'}


def test_generate_per_fov_slurm_usage_empty_when_no_fragments_done(
        monkeypatch, slurm_report_task, simple_data, upstream_task):
    monkeypatch.setattr(
        type(simple_data), 'get_analysis_environment',
        lambda self, task, fragmentIndex=None: None)

    def fail_if_called(*args, **kwargs):
        raise AssertionError('sacct should not be queried with no jobs')
    monkeypatch.setattr(subprocess, 'run', fail_if_called)

    usage = slurm_report_task._generate_per_fov_slurm_usage(upstream_task)
    assert usage.empty
    assert list(usage.columns) == ['mem_mb', 'time_min']


def test_generate_per_fov_slurm_usage_parses_sacct_output(
        monkeypatch, slurm_report_task, simple_data, upstream_task):
    envByFragment = {
        0: {'SLURM_JOB_ID': '1001'},
        3: {'SLURM_JOB_ID': '1002'},
    }
    monkeypatch.setattr(
        type(simple_data), 'get_analysis_environment',
        lambda self, task, fragmentIndex=None: envByFragment.get(
            fragmentIndex))

    # real sacct output has one row per job *step* -- MaxRSS is usually
    # only populated on the '.batch' step, not the bare job id row, and
    # this mix of dotted/undotted JobID values is what keeps pandas from
    # inferring JobID as numeric (a bare '1001'/'1002'-only column would,
    # breaking _clean_slurm_dataframe's .str accessor use)
    fakeOutput = _fake_sacct_output([
        _fake_sacct_row('1001', '', '00:05:30'),
        _fake_sacct_row('1001.batch', '512.5M', '00:05:30'),
        _fake_sacct_row('1002', '', '00:10:00'),
        _fake_sacct_row('1002.batch', '1024M', '00:10:00'),
    ])

    class FakeResult:
        stdout = fakeOutput.encode('utf-8')

    recordedArgs = {}

    def fake_run(args, stdout=None):
        recordedArgs['args'] = args
        return FakeResult()

    monkeypatch.setattr(subprocess, 'run', fake_run)

    usage = slurm_report_task._generate_per_fov_slurm_usage(upstream_task)

    assert list(usage.index) == [0, 3]
    assert usage.loc[0, 'mem_mb'] == pytest.approx(512.5)
    assert usage.loc[0, 'time_min'] == pytest.approx(5.5)
    assert usage.loc[3, 'mem_mb'] == pytest.approx(1024)
    assert usage.loc[3, 'time_min'] == pytest.approx(10.0)
    # both job ids were queried in one sacct call
    assert '1001,1002' in recordedArgs['args'][-1] \
        or '1002,1001' in recordedArgs['args'][-1]


def test_generate_per_fov_slurm_usage_takes_max_rss_across_job_steps(
        monkeypatch, slurm_report_task, simple_data, upstream_task):
    # real sacct output for a job that runs 'srun python ...' inside an
    # sbatch script has (at least) a 'batch' step and a numbered srun
    # sub-step -- the actual analysis code runs in the sub-step, which
    # commonly uses far more memory than the outer 'batch' wrapper.
    # Confirmed against a real BC555_sample_05 CellPoseSegmentSAM job:
    # batch=120.53M, the real python step=3452.07M.
    monkeypatch.setattr(
        type(simple_data), 'get_analysis_environment',
        lambda self, task, fragmentIndex=None: (
            {'SLURM_JOB_ID': '1001'} if fragmentIndex == 0 else None))

    fakeOutput = _fake_sacct_output([
        _fake_sacct_row('1001', '', '00:05:00'),
        _fake_sacct_row('1001.batch', '120.53M', '00:05:00'),
        _fake_sacct_row('1001.0', '3452.07M', '00:04:50'),
    ])

    class FakeResult:
        stdout = fakeOutput.encode('utf-8')

    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: FakeResult())

    usage = slurm_report_task._generate_per_fov_slurm_usage(upstream_task)
    assert usage.loc[0, 'mem_mb'] == pytest.approx(3452.07)


def test_save_per_fov_resource_table_combines_tasks_and_skips_others(
        monkeypatch, slurm_report_task, simple_data, upstream_task):
    upstream_task.save()
    otherTask = testtask.SimpleAnalysisTask(
        simple_data, parameters={}, analysisName='NotParallel')
    otherTask.save()

    def fake_load_analysis_task(self, name):
        return {upstream_task.get_analysis_name(): upstream_task,
                'NotParallel': otherTask}[name]

    monkeypatch.setattr(
        type(simple_data), 'load_analysis_task', fake_load_analysis_task)

    fakeUsage = pandas.DataFrame(
        {'mem_mb': [100.0, 200.0], 'time_min': [1.0, 2.0]},
        index=pandas.Index([0, 1], name='fov'))
    monkeypatch.setattr(
        slurmreport.SlurmReport, '_generate_per_fov_slurm_usage',
        lambda self, task: fakeUsage)

    slurm_report_task._save_per_fov_resource_table(
        [upstream_task.get_analysis_name(), 'NotParallel'])

    loaded = simple_data.load_dataframe_from_parquet(
        'per_fov_resource_usage', slurm_report_task, subdirectory='reports')
    taskName = upstream_task.get_analysis_name()
    assert list(loaded.columns) == [
        taskName + '_mem_mb', taskName + '_time_min']
    assert list(loaded.index) == [0, 1]
    assert loaded.loc[0, taskName + '_mem_mb'] == 100.0

    simple_data.delete_analysis(upstream_task)
    simple_data.delete_analysis(otherTask)
    simple_data.delete_analysis(slurm_report_task)
