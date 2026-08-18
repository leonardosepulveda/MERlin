import importlib
from typing import Dict
import networkx
from merlin.core import analysistask
from merlin.core import dataset


def _parse_slurm_time_to_minutes(timeString: str) -> int:
    """Convert an sbatch-style time limit ('[D-]H(H):MM:SS') to the number
    of minutes expected by snakemake-executor-plugin-slurm's `runtime`
    resource, rounding up any partial minute so the job is never
    under-provisioned.
    """
    days = 0
    if '-' in timeString:
        dayPart, timeString = timeString.split('-', 1)
        days = int(dayPart)
    timeParts = [int(x) for x in timeString.split(':')]
    while len(timeParts) < 3:
        timeParts = [0] + timeParts
    hours, minutes, seconds = timeParts
    totalMinutes = days*24*60 + hours*60 + minutes
    if seconds:
        totalMinutes += 1
    return totalMinutes


def _requests_nonzero_gres(gresString: str) -> bool:
    """Whether a gres string (e.g. 'gpu:0', 'gpu:1') actually requests a
    nonzero amount of something.

    'gpu:0' was the harmless __default__ for every task (GPU or not) under
    the old sbatch `--gres={cluster.gres}` template -- requesting zero of a
    resource is a no-op there. snakemake-executor-plugin-slurm instead
    treats any 'gpu' substring in `gres` as a GPU job (regardless of count)
    and unconditionally adds `--ntasks-per-gpu=1`, which sbatch then
    rejects when combined with a zero count (confirmed against a real
    submission: 'sbatch: error: gres_job_state_validate: --ntasks-per-tres
    needs either a GRES GPU specification or a node/ntask specification').
    So a zero count must be omitted entirely, not passed through.
    """
    countPart = gresString.rsplit(':', 1)[-1]
    try:
        return int(countPart) != 0
    except ValueError:
        return True  # not a numeric count (e.g. bare 'gpu') -- treat as real


def _translate_cluster_resources(rawResources: Dict) -> Dict:
    """Translate one merged (__default__ + per-task override) entry from a
    legacy cluster_resource_allocation_*.json (the sbatch `--cluster`
    command-template era) into snakemake-executor-plugin-slurm per-rule
    `resources:` keys. See FINDINGS.md's 'snakemake >=8 migration' section
    for the full old-key -> new-key mapping this implements.
    """
    resources = {}
    if rawResources.get('mem'):
        resources['mem_mb'] = rawResources['mem']
    if rawResources.get('partition'):
        resources['slurm_partition'] = rawResources['partition']
    if rawResources.get('account'):
        resources['slurm_account'] = rawResources['account']
    if rawResources.get('time'):
        resources['runtime'] = _parse_slurm_time_to_minutes(
            rawResources['time'])
    if rawResources.get('constraint'):
        resources['constraint'] = rawResources['constraint']
    if rawResources.get('gres') and _requests_nonzero_gres(
            rawResources['gres']):
        resources['gres'] = rawResources['gres']
    if rawResources.get('exclude'):
        resources['slurm_extra'] = '--exclude=%s' % rawResources['exclude']
    return resources


class SnakemakeRule(object):

    def __init__(self, analysisTask: analysistask.AnalysisTask,
                 pythonPath=None, clusterConfig: Dict = None):
        self._analysisTask = analysisTask
        self._pythonPath = pythonPath
        self._clusterConfig = clusterConfig

    @staticmethod
    def _add_quotes(stringIn):
        return '\'%s\'' % stringIn

    @staticmethod
    def _clean_string(stringIn):
        return stringIn.replace('\\', '/')

    def _expand_as_string(self, taskName, indexCount) -> str:
        return 'expand(%s, g=list(range(%i)))' % (self._add_quotes(
            self._analysisTask.dataSet.analysis_done_filename(taskName, '{g}')),
            indexCount)

    def _generate_output(self) -> str:
        if isinstance(self._analysisTask, analysistask.ParallelAnalysisTask):
            return self._clean_string(
                self._add_quotes(
                    self._analysisTask.dataSet.analysis_done_filename(
                        self._analysisTask, '{i}')))
        else:
            return self._clean_string(
                self._add_quotes(
                    self._analysisTask.dataSet.analysis_done_filename(
                        self._analysisTask)))

    def _generate_current_task_inputs(self):
        inputTasks = [self._analysisTask.dataSet.load_analysis_task(x)
                      for x in self._analysisTask.get_dependencies()]
        if len(inputTasks) > 0:
            
            # Using the ancient keyword causes incorrect scheduling order by snakemake
            # https://github.com/snakemake/snakemake/issues/946
            # May revert this keyword when the bug in snakemake is fixed.
            #inputString = ','.join(['ancient(' + self._add_quotes(
            #    x.dataSet.analysis_done_filename(x)) + ')'
            #                        for x in inputTasks])
            
            inputString = ','.join([self._add_quotes(
                x.dataSet.analysis_done_filename(x))
                                    for x in inputTasks])
        else:
            inputString = ''

        return self._clean_string(inputString)

    def _generate_message(self) -> str:
        messageString = \
            ''.join(['Running ', self._analysisTask.get_analysis_name()])
        if isinstance(self._analysisTask, analysistask.ParallelAnalysisTask):
            messageString += ' {wildcards.i}'
        return self._add_quotes(messageString)

    def _cluster_resources_for_rule(self, ruleName: str) -> Dict:
        """Merge __default__ with any override keyed by ruleName (mirroring
        snakemake<8's --cluster-config lookup, which fell back to
        __default__ for rules -- e.g. the 'Done' check rules below -- with
        no entry of their own), then translate to executor-plugin keys.
        """
        if not self._clusterConfig:
            return {}
        merged = dict(self._clusterConfig.get('__default__', {}))
        merged.update(self._clusterConfig.get(ruleName, {}))
        return _translate_cluster_resources(merged)

    @staticmethod
    def _generate_resources(resources: Dict) -> str:
        if not resources:
            return ''
        resourceItems = ', '.join(
            '%s=%s' % (k, repr(v)) for k, v in resources.items())
        return '\n\tresources: ' + resourceItems

    def _base_shell_command(self) -> str:
        if self._pythonPath is None:
            shellString = 'python '
        else:
            shellString = self._clean_string(self._pythonPath) + ' '
        shellString += ''.join(
            ['-m merlin -t ',
             self._clean_string(self._analysisTask.analysisName),
             ' -e \"',
             self._clean_string(self._analysisTask.dataSet.dataHome), '\"',
             ' -s \"',
             self._clean_string(self._analysisTask.dataSet.analysisHome),
             '\"'])
        return shellString

    def _generate_shell(self) -> str:
        shellString = self._base_shell_command()
        if isinstance(self._analysisTask, analysistask.ParallelAnalysisTask):
            shellString += ' -i {wildcards.i}'
        shellString += ' ' + self._clean_string(
            self._analysisTask.dataSet.dataSetName)
        return self._add_quotes(shellString)

    def _generate_done_shell(self) -> str:
        """ Check done shell command for parallel analysis tasks
        """
        shellString = self._base_shell_command()
        shellString += ' --check-done'
        shellString += ' ' + self._clean_string(
            self._analysisTask.dataSet.dataSetName)
        return self._add_quotes(shellString)

    def as_string(self) -> str:
        ruleName = self._analysisTask.get_analysis_name()
        fullString = ('rule %s:\n\tinput: %s\n\toutput: %s\n\tmessage: %s%s\n\t'
                      + 'shell: %s\n\n') \
                     % (ruleName,
                        self._generate_current_task_inputs(),
                        self._generate_output(),
                        self._generate_message(),
                        self._generate_resources(
                            self._cluster_resources_for_rule(ruleName)),
                        self._generate_shell())
        # for parallel tasks, add a second snakemake task to reduce the time
        # it takes to generate DAGs
        if isinstance(self._analysisTask, analysistask.ParallelAnalysisTask):
            doneRuleName = ruleName + 'Done'
            fullString += \
                ('rule %s:\n\tinput: %s\n\toutput: %s\n\tmessage: %s%s\n\t'
                 + 'shell: %s\n\n')\
                % (doneRuleName,
                   self._clean_string(self._expand_as_string(
                       self._analysisTask,
                       self._analysisTask.fragment_count())),
                   self._add_quotes(self._clean_string(
                       self._analysisTask.dataSet.analysis_done_filename(
                           self._analysisTask))),
                   self._add_quotes(
                       'Checking %s done' % self._analysisTask.analysisName),
                   self._generate_resources(
                       self._cluster_resources_for_rule(doneRuleName)),
                   self._generate_done_shell())
        return fullString

    def full_output(self) -> str:
        if isinstance(self._analysisTask, analysistask.ParallelAnalysisTask):
            return self._clean_string(self._expand_as_string(
                self._analysisTask.get_analysis_name(),
                self._analysisTask.fragment_count()))
        else:
            return self._clean_string(
                self._add_quotes(
                    self._analysisTask.dataSet.analysis_done_filename(
                        self._analysisTask)))


class SnakefileGenerator(object):

    def __init__(self, analysisParameters, dataSet: dataset.DataSet,
                 pythonPath: str = None, clusterConfig: Dict = None):
        self._analysisParameters = analysisParameters
        self._dataSet = dataSet
        self._pythonPath = pythonPath
        self._clusterConfig = clusterConfig

    def _parse_parameters(self):
        analysisTasks = {}
        for tDict in self._analysisParameters['analysis_tasks']:
            analysisModule = importlib.import_module(tDict['module'])
            analysisClass = getattr(analysisModule, tDict['task'])
            analysisParameters = tDict.get('parameters')
            analysisName = tDict.get('analysis_name')
            newTask = analysisClass(
                    self._dataSet, analysisParameters, analysisName)
            if newTask.get_analysis_name() in analysisTasks:
                raise Exception('Analysis tasks must have unique names. ' +
                                newTask.get_analysis_name() + ' is redundant.')
            # TODO This should be more careful to not overwrite an existing
            # analysis task that has already been run.
            newTask.save()
            analysisTasks[newTask.get_analysis_name()] = newTask
        return analysisTasks

    def _identify_terminal_tasks(self, analysisTasks):
        taskGraph = networkx.DiGraph()
        for x in analysisTasks.keys():
            taskGraph.add_node(x)

        for x, a in analysisTasks.items():
            for d in a.get_dependencies():
                taskGraph.add_edge(d, x)

        return [k for k, v in taskGraph.out_degree if v == 0]

    def generate_workflow(self) -> str:
        """Generate a snakemake workflow for the analysis parameters
        of this SnakemakeGenerator and save the workflow into the dataset.

        Returns:
            the path to the generated snakemake workflow
        """
        analysisTasks = self._parse_parameters()
        terminalTasks = self._identify_terminal_tasks(analysisTasks)

        ruleList = {k: SnakemakeRule(v, self._pythonPath, self._clusterConfig)
                    for k, v in analysisTasks.items()}

        workflowString = 'rule all: \n\tinput: ' + \
            ','.join([ruleList[x].full_output()
                      for x in terminalTasks]) + '\n\n'
        workflowString += '\n'.join([x.as_string() for x in ruleList.values()])

        return self._dataSet.save_workflow(workflowString)
