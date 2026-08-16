import argparse
import cProfile
import os
import json
import sys
from pathlib import Path
from typing import TextIO
from typing import Dict

from snakemake.api import SnakemakeApi
from snakemake.settings.types import (
    ExecutionSettings, OutputSettings, ResourceSettings)

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
                        help='name of the analysis parameters file to use')
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
    parser.add_argument('-q', '--parameters-home',
                        help='the parameters home directory')
    parser.add_argument('-k', '--snakemake-parameters',
                        help='the name of the snakemake parameters file '
                        '(cluster/remote execution settings; not yet '
                        'supported on the current snakemake version, see '
                        'run_with_snakemake)')
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
        allowRaggedZStacks=args.allow_ragged_z_stacks
    )
    
    parametersHome = m.ANALYSIS_PARAMETERS_HOME
    e = executor.LocalExecutor(coreCount=args.core_count)
    snakefilePath = None
    if args.analysis_parameters:
        # This is run in all cases that analysis parameters are provided
        # so that new analysis tasks are generated to match the new parameters
        with open(os.sep.join(
                [parametersHome, args.analysis_parameters]), 'r') as f:
            snakefilePath = generate_analysis_tasks_and_snakefile(
                dataSet, f)

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
            snakemakeParameters = {}
            if args.snakemake_parameters:
                with open(os.sep.join([m.SNAKEMAKE_PARAMETERS_HOME,
                                      args.snakemake_parameters])) as f:
                    snakemakeParameters = json.load(f)

            run_with_snakemake(dataSet, snakefilePath, args.core_count,
                               snakemakeParameters)


def generate_analysis_tasks_and_snakefile(dataSet: dataset.MERFISHDataSet,
                                          parametersFile: TextIO) -> str:
    print('Generating analysis tasks from %s' % parametersFile.name)
    analysisParameters = json.load(parametersFile)
    snakeGenerator = snakewriter.SnakefileGenerator(
        analysisParameters, dataSet, sys.executable)
    snakefilePath = snakeGenerator.generate_workflow()
    print('Snakefile generated at %s' % snakefilePath)
    return snakefilePath


def run_with_snakemake(
        dataSet: dataset.MERFISHDataSet, snakefilePath: str, coreCount: int,
        snakemakeParameters: Dict = {}):
    """Execute a generated Snakefile locally through snakemake's Python API.

    Only local execution is supported. `snakemakeParameters` (from
    -k/--snakemake-parameters) previously carried a generic `cluster:`
    sbatch command-template string for remote/cluster submission, using
    snakemake's pre-8.0 API; snakemake 8.0 removed that generic
    cluster-submission mechanism in favor of a separate executor-plugin
    system (e.g. snakemake-executor-plugin-slurm) with its own resource
    model, which is a real, untested-here migration of its own -- so a
    non-empty snakemakeParameters is rejected explicitly rather than
    silently mishandled.
    """
    if snakemakeParameters:
        raise NotImplementedError(
            'Cluster/remote snakemake execution via -k/--snakemake-'
            'parameters is not supported with the current snakemake '
            'version: snakemake 8.0 replaced the generic `cluster:` '
            'command-template mechanism this parameters file format '
            'assumed with a separate executor-plugin system. Run without '
            '-k for local execution, or port the cluster path to an '
            'executor plugin first.')

    print('Running MERlin pipeline through snakemake')
    with SnakemakeApi(OutputSettings()) as snakemakeApi:
        workflowApi = snakemakeApi.workflow(
            resource_settings=ResourceSettings(cores=coreCount),
            snakefile=Path(snakefilePath),
            workdir=Path(dataSet.get_snakemake_path()))
        dagApi = workflowApi.dag(dag_settings=None)
        dagApi.execute_workflow(
            executor='local',
            execution_settings=ExecutionSettings(
                lock=False, latency_wait=10))
