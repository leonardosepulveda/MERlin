import argparse
import cProfile
import logging
import os
import json
import re
import sys
import time
from pathlib import Path
import yaml
from typing import TextIO
from typing import Dict

from snakemake.api import SnakemakeApi
from snakemake.settings.types import (
    ExecutionSettings, OutputSettings, ResourceSettings)
from snakemake_executor_plugin_slurm import (
    ExecutorSettings as SlurmExecutorSettings)
from snakemake_interface_logger_plugins.common import LogEvent

import merlin as m
from merlin.core import dataset
from merlin.core import executor
from merlin.util import snakewriter


def build_parser():
    parser = argparse.ArgumentParser(description='Decode MERFISH data.')

    parser.add_argument('--profile', action='store_true',
                        help='enable profiling')
    parser.add_argument('--generate-only', action='store_true',
                        help='only generate the directory structure and ' +
                        'do not run any analysis.')
    parser.add_argument('--configure', action='store_true',
                        help='configure MERlin environment by specifying ' +
                        ' data, analysis, and parameters directories.')
    parser.add_argument('dataset',
                        help='directory where the raw data is stored')
    parser.add_argument('-a', '--analysis-parameters',
                        help='name of the analysis parameters file to use '
                        '(.json or .yaml/.yml, detected by extension)')
    parser.add_argument('-o', '--data-organization',
                        help='name of the data organization file to use')
    parser.add_argument('-c', '--codebook', nargs='+',
                        help='name of the codebook to use')
    parser.add_argument('-m', '--microscope-parameters',
                        help='name of the microscope parameters to use')                  
    parser.add_argument('-p', '--positions',
                        help='name of the position file to use')
    parser.add_argument('-n', '--core-count', type=int,
                        help='number of cores to use for the analysis')
    parser.add_argument('--check-done', action='store_true',
                        help='flag to only check if the analysis task is ' +
                        'done')
    parser.add_argument(
        '-t', '--analysis-task',
        help='the name of the analysis task to execute. If no '
             + 'analysis task is provided, all tasks are executed.')
    parser.add_argument(
        '-i', '--fragment-index', type=int,
        help='the index of the fragment of the analysis task to execute')
    parser.add_argument('-e', '--data-home',
                        help='the data home directory')
    parser.add_argument('-s', '--analysis-home',
                        help='the analysis home directory')
    parser.add_argument('-x', '--analysis-name',
                        help='the subdirectory name under analysis-home to '
                        'store analysis results in, if it should differ '
                        'from the dataset directory (e.g. a short fixed '
                        'name instead of a long/nested raw-data path). '
                        'Defaults to the dataset directory.')
    parser.add_argument('-q', '--parameters-home',
                        help='the parameters home directory')
    parser.add_argument('-f', '--figures-path',
                        help='the directory to save analysis task '
                        'verification figures into, used as-is (no dataset '
                        'subfolder is appended). Defaults to '
                        '<analysis-home>/<dataset>/figures')
    parser.add_argument('-k', '--snakemake-parameters',
                        help='the name of the snakemake parameters file, '
                        'for distributed execution on a SLURM cluster via '
                        'snakemake-executor-plugin-slurm')
    parser.add_argument('--allow-ragged-z-stacks', action='store_true',
                        help='tolerate fovs whose raw files have fewer '
                        'z frames than the deepest z position configured '
                        'in the data organization, instead of raising an '
                        'error (e.g. when acquisition was trimmed to each '
                        "fov's own tissue depth)")

    return parser


def _clean_string_arg(stringIn):
    if stringIn is None:
        return None
    return stringIn.strip('\'').strip('\"')


def _get_input_path(prompt):
    while True:
        pathString = str(input(prompt))
        if not pathString.startswith('s3://') \
                and not os.path.exists(os.path.expanduser(pathString)):
            print('Directory %s does not exist. Please enter a valid path.'
                  % pathString)
        else:
            return pathString


def configure_environment():
    dataHome = _get_input_path('DATA_HOME=')
    analysisHome = _get_input_path('ANALYSIS_HOME=')
    parametersHome = _get_input_path('PARAMETERS_HOME=')
    m.store_env(dataHome, analysisHome, parametersHome)


def merlin():
    print('MERlin - the MERFISH decoding pipeline')
    parser = build_parser()
    args, argv = parser.parse_known_args()

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

    if args.configure:
        print('Configuring MERlin environment')
        configure_environment()
        return

    if args.parameters_home is not None:
        print('specifying parameter file in arg parser')
        m.PARAMETERS_HOME = _clean_string_arg(args.parameters_home)
        m.ANALYSIS_PARAMETERS_HOME = os.sep.join(
                [m.PARAMETERS_HOME, 'analysis'])
        m.CODEBOOK_HOME = os.sep.join(
                [m.PARAMETERS_HOME, 'codebooks'])
        m.DATA_ORGANIZATION_HOME = os.sep.join(
                [m.PARAMETERS_HOME, 'dataorganization'])
        m.POSITION_HOME = os.sep.join(
                [m.PARAMETERS_HOME, 'positions'])
        m.MICROSCOPE_PARAMETERS_HOME = os.sep.join(
                [m.PARAMETERS_HOME, 'microscope'])
        m.FPKM_HOME = os.sep.join([m.PARAMETERS_HOME, 'fpkm'])
        m.SNAKEMAKE_PARAMETERS_HOME = os.sep.join(
            [m.PARAMETERS_HOME, 'snakemake'])
            

    dataSet = dataset.MERFISHDataSet(
        args.dataset,
        codebookNames=args.codebook,
        dataOrganizationName=_clean_string_arg(args.data_organization),
        positionFileName=_clean_string_arg(args.positions),
        dataHome=_clean_string_arg(args.data_home),
        analysisHome=_clean_string_arg(args.analysis_home),
        microscopeParametersName=_clean_string_arg(args.microscope_parameters),
        allowRaggedZStacks=args.allow_ragged_z_stacks,
        figuresPath=_clean_string_arg(args.figures_path),
        analysisName=_clean_string_arg(args.analysis_name)
    )
    
    parametersHome = m.ANALYSIS_PARAMETERS_HOME
    e = executor.LocalExecutor(coreCount=args.core_count)

    # Loaded ahead of snakefile generation (rather than only when actually
    # running) because clusterConfig's per-task resource overrides need to
    # be baked into each rule's `resources:` block at generation time -- see
    # snakewriter.SnakefileGenerator.
    snakemakeParameters = {}
    clusterConfig = None
    if args.snakemake_parameters:
        snakemakeParametersPath = args.snakemake_parameters \
                if os.path.exists(args.snakemake_parameters) \
                else os.sep.join([m.SNAKEMAKE_PARAMETERS_HOME,
                                  args.snakemake_parameters])
        with open(snakemakeParametersPath) as f:
            snakemakeParameters = json.load(f)
        if snakemakeParameters.get('cluster_config'):
            with open(snakemakeParameters['cluster_config']) as f:
                clusterConfig = _load_json_or_yaml(f)

    snakefilePath = None
    if args.analysis_parameters:
        # This is run in all cases that analysis parameters are provided
        # so that new analysis tasks are generated to match the new parameters
        analysisParametersPath = args.analysis_parameters \
                if os.path.exists(args.analysis_parameters) \
                else os.sep.join([parametersHome, args.analysis_parameters])
        with open(analysisParametersPath, 'r') as f:
            snakefilePath = generate_analysis_tasks_and_snakefile(
                dataSet, f, clusterConfig)

    if not args.generate_only:
        if args.analysis_task:
            task = dataSet.load_analysis_task(args.analysis_task)
            if args.check_done:
                # checking completion creates the .done file for parallel tasks
                # where completion has not yet been checked
                if task.is_complete():
                    print('Task %s is complete' % args.analysis_task)
                else:
                    print('Task %s is not complete' % args.analysis_task)

            else:
                print('Running %s' % args.analysis_task)
                e.run(task, index=args.fragment_index)
        elif snakefilePath:
            run_with_snakemake(dataSet, snakefilePath, args.core_count,
                               snakemakeParameters, clusterConfig)


def _load_json_or_yaml(fileObj: TextIO) -> Dict:
    """Parse an open file handle as YAML or JSON depending on its own
    extension (`.yaml`/`.yml` vs anything else, parsed as JSON as
    before). Shared by analysis-parameter recipes and
    cluster-resource-allocation configs -- both are plain JSON/YAML-
    compatible mapping/sequence structures, so this is purely a choice
    of parser, and dispatching on extension keeps every existing .json
    file (and any caller that doesn't set an extension) working
    unchanged. YAML's native `#` comments are the main reason to use it
    for a cluster-resource-allocation file: a per-task calculated
    mem/time value can be left as a commented-out reference line,
    uncommented to override it.
    """
    _, extension = os.path.splitext(fileObj.name)
    if extension.lower() in ('.yaml', '.yml'):
        return yaml.safe_load(fileObj)
    return json.load(fileObj)


def _load_analysis_parameters(parametersFile: TextIO) -> Dict:
    """Parse an analysis-parameters recipe from an open file handle. See
    _load_json_or_yaml for the format-dispatch details.
    """
    return _load_json_or_yaml(parametersFile)


def generate_analysis_tasks_and_snakefile(
        dataSet: dataset.MERFISHDataSet, parametersFile: TextIO,
        clusterConfig: Dict = None) -> str:
    print('Generating analysis tasks from %s' % parametersFile.name)
    analysisParameters = _load_analysis_parameters(parametersFile)
    snakeGenerator = snakewriter.SnakefileGenerator(
        analysisParameters, dataSet, sys.executable, clusterConfig)
    snakefilePath = snakeGenerator.generate_workflow()
    print('Snakefile generated at %s' % snakefilePath)
    return snakefilePath


class _SlurmDriverLogFilter(logging.Filter):
    """Reformats several snakemake/snakemake-executor-plugin-slurm driver
    log messages so the driver's log is easier to scan by eye:

    - The one-line job-submission message ('Job N has been submitted with
      SLURM jobid J (log: ...).') is rewritten as
      'Submitted jobid: N (slurm_id: J) (Rule: R) (Fragment: F)', with the
      rule name and fragment number recovered from the log path (which
      snakewriter lays out as .../slurm_logs/rule_<R>/[<F>/]<J>.log) --
      Fragment is omitted for rules that don't run per-fragment.
    - The matching 'Finished jobid: N (Rule: R)' message (emitted later,
      once the job completes) is enriched with the same slurm_id/Fragment,
      looked up by N from the submission message above.
    - The startup 'Command: ...' line is reformatted to one flag per line.

    In all cases, the common output-directory prefix is abbreviated as
    '$OUTPUT_DIR'.
    """

    _SUBMIT_RE = re.compile(
        r'^Job (?P<jobid>\S+) has been submitted with SLURM jobid '
        r'(?P<slurm_jobid>\S+) \(log: (?P<log>.+)\)\.$')
    _FINISH_RE = re.compile(
        r'^Finished jobid: (?P<jobid>\S+) \(Rule: (?P<rule>.+)\)$')
    _LOGPATH_RE = re.compile(
        r'slurm_logs/rule_(?P<rule>[^/]+)/(?:(?P<fragment>\d+)/)?'
        r'(?P<slurmid>\d+)\.log$')
    _FLAG_RE = re.compile(r'(?=\s-{1,2}[A-Za-z])')

    def __init__(self, outputDir):
        super().__init__()
        self._outputDir = str(outputDir)
        self._jobinfo = {}

    @staticmethod
    def _tag(rule: str, fragment: str) -> str:
        tag = ' (Rule: %s)' % rule
        if fragment is not None:
            tag += ' (Fragment: %s)' % fragment
        return tag

    def filter(self, record: logging.LogRecord) -> bool:
        if getattr(record, 'event', None) == LogEvent.WORKFLOW_STARTED \
                and hasattr(record, 'cmd'):
            flags = [t.strip() for t in
                     self._FLAG_RE.split(record.cmd) if t.strip()]
            record.cmd = '\n' + flags[0] + '\n' + \
                '\n'.join('\t%s' % f for f in flags[1:])
            return True

        submitMatch = self._SUBMIT_RE.match(record.getMessage())
        if submitMatch:
            log = submitMatch.group('log').replace(
                self._outputDir, '$OUTPUT_DIR')
            logMatch = self._LOGPATH_RE.search(log)
            rule = logMatch.group('rule') if logMatch else '?'
            fragment = logMatch.group('fragment') if logMatch else None
            jobid = submitMatch.group('jobid')
            slurmJobid = submitMatch.group('slurm_jobid')
            self._jobinfo[jobid] = (slurmJobid, rule, fragment)
            record.msg = (
                '[%s]\nSubmitted jobid: %s (slurm_id: %s)%s'
                % (time.asctime(time.localtime(record.created)), jobid,
                   slurmJobid, self._tag(rule, fragment)))
            record.args = None
            return True

        finishMatch = self._FINISH_RE.match(record.getMessage())
        if finishMatch:
            jobid = finishMatch.group('jobid')
            info = self._jobinfo.get(jobid)
            if info:
                slurmJobid, rule, fragment = info
                record.msg = (
                    'Finished jobid: %s (slurm_id: %s)%s'
                    % (jobid, slurmJobid, self._tag(rule, fragment)))
                record.args = None
        return True


def run_with_snakemake(
        dataSet: dataset.MERFISHDataSet, snakefilePath: str, coreCount: int,
        snakemakeParameters: Dict = {}, clusterConfig: Dict = None):
    """Execute a generated Snakefile through snakemake's Python API, either
    locally (the default) or, when snakemakeParameters (-k/
    --snakemake-parameters) is provided, distributed across a SLURM cluster
    via snakemake-executor-plugin-slurm.

    snakemake 8.0 removed the pre-8.0 generic `cluster:` sbatch
    command-template mechanism that snakemakeParameters/clusterConfig's JSON
    shapes originally targeted. Rather than change those JSON shapes (kept
    unchanged so existing -k/cluster_config files still work), the
    translation to the executor-plugin's per-rule `resources:` model happens
    internally: clusterConfig's per-task resource overrides are baked into
    the generated Snakefile's rules (see snakewriter.SnakefileGenerator),
    and snakemakeParameters' top-level `nodes`/`restart_times` plus
    clusterConfig's `__default__.requeue` are translated below into the
    equivalent executor/resource settings. snakemakeParameters'
    `job_name_prefix` (default 'merlin') becomes the SLURM job-name prefix
    (snakemake-executor-plugin-slurm always uses a per-run UUID as the job
    name itself, to support its own `sacct`/`squeue --name`-based status
    polling, so this prefix is the only per-run customization the plugin
    exposes -- it does not vary per rule/fragment).

    keep_going=True is set unconditionally so a failure in one job (e.g. a
    bad fragment in Decode) doesn't stop snakemake from scheduling any
    further new jobs: unrelated branches (segmentation, global alignment,
    ...) keep running to completion, and only jobs actually downstream of
    the failure are skipped. Without it, an error partway through an
    unattended overnight run leaves every other branch untouched even
    though nothing about them failed. The run still ends with a nonzero
    exit/a reported failure, so a genuine error is not hidden -- it just no
    longer stops healthy branches from finishing.
    """
    print('Running MERlin pipeline through snakemake')
    if snakemakeParameters:
        executorName = 'slurm'
        executionSettings = ExecutionSettings(
            lock=False, latency_wait=10, keep_going=True,
            retries=snakemakeParameters.get('restart_times', 0))
        resourceSettings = ResourceSettings(
            cores=coreCount, nodes=snakemakeParameters.get('nodes'))
        defaultResources = (clusterConfig or {}).get('__default__', {})
        executorSettings = SlurmExecutorSettings(
            requeue=bool(defaultResources.get('requeue', False)),
            jobname_prefix=snakemakeParameters.get(
                'job_name_prefix', 'merlin'),
            logdir=Path(dataSet.get_snakemake_path()) / 'slurm_logs')
        print('$OUTPUT_DIR = %s' % dataSet.get_snakemake_path())
        logging.getLogger('snakemake.logging').addFilter(
            _SlurmDriverLogFilter(dataSet.get_snakemake_path()))
    else:
        executorName = 'local'
        executionSettings = ExecutionSettings(
            lock=False, latency_wait=10, keep_going=True)
        resourceSettings = ResourceSettings(cores=coreCount)
        executorSettings = None

    with SnakemakeApi(OutputSettings()) as snakemakeApi:
        workflowApi = snakemakeApi.workflow(
            resource_settings=resourceSettings,
            snakefile=Path(snakefilePath),
            workdir=Path(dataSet.get_snakemake_path()))
        dagApi = workflowApi.dag(dag_settings=None)
        dagApi.execute_workflow(
            executor=executorName,
            execution_settings=executionSettings,
            executor_settings=executorSettings)
