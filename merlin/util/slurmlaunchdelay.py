"""Measure how long snakemake-launched SLURM jobs wait before their merlin
step starts, and how that wait relates to how many of the same user's
pipeline jobs were running or starting at the time.

snakemake-executor-plugin-slurm runs each rule as an sbatch job whose batch
script launches the actual merlin command as step `.0` via srun. The time
between the job's start and step `.0`'s start counts against the job's time
limit but is not covered by any task's time estimate (see
snakewriter.SNAKEMAKE_LAUNCH_MINUTES). Large waits coincided with many
concurrent jobs, which is what the `nodes` cap in the snakemake parameters
file controls.

Usage:
    python -m merlin.util.slurmlaunchdelay --since 2026-10-01T15:00
"""
import argparse
import io
import subprocess

import numpy as np
import pandas as pd

RUNNING_BINS = [0, 100, 250, 500, 1000, 1500, 2500, 5000]
BURST_BINS = [0, 10, 50, 100, 200, 500, 2000]


def load_sacct(since: str, until: str = 'now') -> pd.DataFrame:
    """Return sacct's job and step rows for the current user."""
    out = subprocess.run(
        ['sacct', '-n', '-P', '-S', since, '-E', until,
         '-o', 'JobID,JobName,Start,End,State'],
        check=True, capture_output=True, text=True).stdout
    return pd.read_csv(io.StringIO(out), sep='|', header=None, dtype=str,
                       names=['id', 'name', 'start', 'end', 'state'])


def launch_delays(sacct: pd.DataFrame, jobNamePattern: str) -> pd.DataFrame:
    """One row per started pipeline job: its run (job-name prefix), start,
    the delay in minutes until step `.0` started (NaN if it never did), the
    number of pipeline jobs running at its start, and the number started in
    the same minute.

    Args:
        sacct: the output of load_sacct.
        jobNamePattern: regex that pipeline job names match; the run is the
            part of the name before the first '_' (the plugin names jobs
            '<job_name_prefix>_<uuid>').
    """
    d = sacct.copy()
    d['start'] = pd.to_datetime(d['start'], errors='coerce')
    d['end'] = pd.to_datetime(d['end'], errors='coerce')

    isStep = d['id'].str.contains(r'\.')
    jobs = d[~isStep & d['name'].str.match(jobNamePattern)
             & d['start'].notna()].copy()
    jobs['run'] = jobs['name'].str.split('_').str[0]
    # still running: treat as running until now
    jobs['end'] = jobs['end'].fillna(pd.Timestamp.now())

    steps = d[d['id'].str.endswith('.0')][['id', 'start']].copy()
    steps['id'] = steps['id'].str[:-2]
    steps = steps.rename(columns={'start': 'stepStart'})
    jobs = jobs.merge(steps, on='id', how='left')
    jobs['delay'] = (
        jobs['stepStart'] - jobs['start']).dt.total_seconds() / 60

    starts = np.sort(jobs['start'].values)
    ends = np.sort(jobs['end'].values)
    t = jobs['start'].values
    jobs['running'] = (np.searchsorted(starts, t, side='right')
                       - np.searchsorted(ends, t, side='right'))
    jobs['burst'] = jobs.groupby(
        jobs['start'].dt.floor('min'))['id'].transform('count')
    return jobs[['id', 'run', 'start', 'delay', 'running', 'burst']]


def _binned(x: pd.DataFrame, column: str, bins) -> pd.DataFrame:
    g = x.groupby(pd.cut(x[column], bins), observed=True)['delay']
    return pd.DataFrame({
        'n': g.size(), 'median': g.median(), 'p90': g.quantile(0.9),
        'max': g.max(), 'frac_over_10min': g.apply(lambda s: (s > 10).mean()),
    }).round(2)


def report(jobs: pd.DataFrame) -> str:
    x = jobs.dropna(subset=['delay'])
    parts = [
        'jobs with a merlin step: %d, without: %d' % (
            len(x), jobs['delay'].isna().sum()),
        '\nDelay (min) per run:',
        x.groupby('run')['delay'].describe(
            percentiles=[.5, .9, .99]).round(1).to_string(),
        '\nDelay (min) by pipeline jobs running at the job\'s start:',
        _binned(x, 'running', RUNNING_BINS).to_string(),
        '\nDelay (min) by pipeline jobs started in the same minute:',
        _binned(x, 'burst', BURST_BINS).to_string(),
        '\nSpearman correlation of delay with running: %.2f, burst: %.2f' % (
            x['delay'].corr(x['running'], method='spearman'),
            x['delay'].corr(x['burst'], method='spearman')),
    ]
    return '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--since', required=True,
                        help='sacct start time, e.g. 2026-10-01T15:00')
    parser.add_argument('--until', default='now', help='sacct end time')
    parser.add_argument('--job-name-pattern', default=r'^s\d+-',
                        help='regex matching pipeline job names (default: '
                        'the s<N>-<sample> prefixes MERci generates)')
    args = parser.parse_args()
    jobs = launch_delays(load_sacct(args.since, args.until),
                         args.job_name_pattern)
    print(report(jobs))


if __name__ == '__main__':
    main()
