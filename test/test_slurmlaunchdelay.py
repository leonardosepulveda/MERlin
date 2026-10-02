import pandas as pd

from merlin.util import slurmlaunchdelay


def test_launch_delays_from_sacct_rows():
    rows = [
        # job 1: step .0 starts 5 min after the job
        ('1', 's2-BC553d_abc', '2026-10-01T16:00:00', '2026-10-01T16:30:00'),
        ('1.batch', 'batch', '2026-10-01T16:00:00', '2026-10-01T16:30:00'),
        ('1.0', 'merlin', '2026-10-01T16:05:00', '2026-10-01T16:30:00'),
        # job 2: starts while job 1 runs, step never created
        ('2', 's2-BC553d_abc', '2026-10-01T16:10:00', '2026-10-01T16:35:00'),
        # cancelled while pending: never started, ignored
        ('3', 's2-BC553d_abc', 'None', '2026-10-01T16:40:00'),
        # not a pipeline job
        ('4', 'vscode.job', '2026-10-01T16:10:00', '2026-10-01T16:20:00'),
    ]
    sacct = pd.DataFrame(
        [r + ('COMPLETED',) for r in rows],
        columns=['id', 'name', 'start', 'end', 'state'])

    jobs = slurmlaunchdelay.launch_delays(sacct, r'^s\d+-').set_index('id')

    assert list(jobs.index) == ['1', '2']
    assert jobs.loc['1', 'delay'] == 5
    assert pd.isna(jobs.loc['2', 'delay'])
    assert list(jobs['running']) == [1, 2]
    assert list(jobs['run']) == ['s2-BC553d', 's2-BC553d']
