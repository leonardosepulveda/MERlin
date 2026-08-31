import subprocess
import pandas
import io
import requests
import time
import json
from matplotlib import pyplot as plt
import numpy as np

from merlin.core import analysistask


class SlurmReport(analysistask.AnalysisTask):

    """
    An analysis task that generates reports on previously completed analysis
    tasks using Slurm.

    This analysis task only works when Merlin is run through Slurm
    with every analysis task fragment run as a separate job.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

        if 'codebook_index' not in self.parameters:
            self.parameters['codebook_index'] = 0

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return [self.parameters['run_after_task']]

    _SACCT_FORMAT = (
        'AssocID,Account,Cluster,User,JobID,JobName,'
        'NodeList,AveCPU,AveCPUFreq,MaxPages,MaxDiskRead,MaxDiskWrite,'
        'MaxRSS,ReqMem,CPUTime,Elapsed,Submit,Start,End,Timelimit')

    def _query_sacct(self, jobIds) -> pandas.DataFrame:
        """Run sacct for the given SLURM job IDs and return the cleaned,
        per-JobID dataframe (see _clean_slurm_dataframe). Shared by
        _generate_slurm_report and _generate_per_fov_slurm_usage so both
        query/parse sacct output the same way.
        """
        queryResult = subprocess.run(
            ['sacct', '--format=' + self._SACCT_FORMAT,
             '--units=M', '-P', '-j', ','.join(jobIds)],
            stdout=subprocess.PIPE)

        slurmJobDF = pandas.read_csv(
            io.StringIO(queryResult.stdout.decode('utf-8')), sep='|')

        return self._clean_slurm_dataframe(slurmJobDF)

    def _generate_slurm_report(self, task: analysistask.AnalysisTask):
        if isinstance(task, analysistask.ParallelAnalysisTask):
            idList = [
                self.dataSet.get_analysis_environment(task, i)['SLURM_JOB_ID']
                for i in range(task.fragment_count())]
        else:
            idList = [
                self.dataSet.get_analysis_environment(task)['SLURM_JOB_ID']]

        return self._query_sacct(idList)

    @staticmethod
    def _clean_slurm_dataframe(inputDataFrame):
        outputDF = inputDataFrame[
            ~inputDataFrame['JobID'].str.contains('.extern')]
        outputDF = outputDF.assign(
            JobID=outputDF['JobID'].str.partition('.')[0])

        def get_not_nan(listIn):
            # a group can be entirely NaN for a given column (e.g. a
            # requeued or otherwise irregularly-accounted job) --
            # .dropna().iloc[0] on an empty result used to raise
            # IndexError and abort the whole query; NaN is a legitimate
            # "unknown" for that one job/column instead.
            nonNan = listIn.dropna()
            return nonNan.iloc[0] if len(nonNan) > 0 else np.nan

        def get_max_mb(listIn):
            # a job's real peak MaxRSS is the max across its steps, not
            # the first non-nan one: e.g. a 'python3.12' srun sub-step
            # commonly uses far more memory than the outer 'batch' step,
            # and sacct lists 'batch' first -- get_not_nan alone would
            # silently report the batch step's much lower value as the
            # job's usage (confirmed against a real BC555_sample_05
            # CellPoseSegmentSAM job: batch=120.53M vs the actual
            # python3.12 step's 3452.07M).
            numeric = pandas.to_numeric(
                listIn.dropna().str.rstrip('M'), errors='coerce').dropna()
            if numeric.empty:
                return np.nan
            return '%gM' % numeric.max()

        aggregators = {c: get_max_mb if c == 'MaxRSS' else get_not_nan
                      for c in outputDF.columns if c != 'JobID'}
        outputDF = outputDF.groupby('JobID').aggregate(aggregators)

        def reformat_timedelta(elapsedIn):
            if pandas.isna(elapsedIn):
                return np.nan
            splitElapsed = elapsedIn.split('-')
            if len(splitElapsed) > 1:
                return splitElapsed[0] + ' days ' + splitElapsed[1]
            else:
                return splitElapsed[0]

        outputDF = outputDF.assign(Elapsed=pandas.to_timedelta(
            outputDF['Elapsed'].apply(reformat_timedelta)))
        outputDF = outputDF.assign(Timelimit=pandas.to_timedelta(
            outputDF['Timelimit'].apply(reformat_timedelta)))
        outputDF = outputDF.assign(Queued=pandas.to_timedelta(
            pandas.to_datetime(outputDF['Start']) -
            pandas.to_datetime(outputDF['Submit'])))

        return outputDF.reindex()

    def _plot_slurm_report(self, slurmDF, analysisName):
        fig = plt.figure(figsize=(15, 4))

        plt.subplot(1, 4, 1)
        plt.boxplot([slurmDF['MaxRSS'].str[:-1].astype(float),
                     slurmDF['ReqMem'].str[:-2].astype(int)], widths=0.5)
        plt.xticks([1, 2], ['Max used', 'Requested'])
        plt.ylabel('Memory (mb)')
        plt.title('RAM')
        plt.subplot(1, 4, 2)
        plt.boxplot([slurmDF['Queued'] / np.timedelta64(1, 'm'),
                     slurmDF['Elapsed'] / np.timedelta64(1, 'm'),
                     slurmDF['Timelimit'] / np.timedelta64(1, 'm')],
                    widths=0.5)
        plt.xticks([1, 2, 3], ['Queued', 'Elapsed', 'Requested'])
        plt.ylabel('Time (min)')
        plt.title('Run time')
        plt.subplot(1, 4, 3)
        plt.boxplot([slurmDF['MaxDiskRead'].astype(str).str.strip('BKMG').astype(float)],
                    widths=0.25)
        plt.xticks([1], ['MaxDiskRead'])
        plt.ylabel('Number of mb read')
        plt.title('Disk usage')
        plt.subplot(1, 4, 4)
        plt.boxplot([slurmDF['MaxDiskWrite'].astype(str).str.strip('BKMG').astype(float)],
                    widths=0.25)
        plt.xticks([1], ['MaxDiskWrite'])
        plt.ylabel('Number of mb written')
        plt.suptitle(analysisName)
        plt.tight_layout(pad=1)
        self.dataSet.save_figure(self, fig, analysisName)

    def _plot_slurm_summary(self, reportDict):

        def setBoxColors(bPlot, c):
            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians',
                            'caps']:
                plt.setp(bPlot[element], color=c)

        # Plot memory requested and used for each task
        fig = plt.figure(figsize=(15, 12))

        bp = plt.boxplot([d['MaxRSS'].str[:-1].astype(float)
                         for d in reportDict.values()],
                         positions=np.arange(len(reportDict))-0.15,
                         widths=0.25)
        setBoxColors(bp, 'r')
        bp = plt.boxplot([d['ReqMem'].str[:-2].astype(float)
                         for d in reportDict.values()],
                         positions=np.arange(len(reportDict))+0.15,
                         widths=0.25)
        setBoxColors(bp, 'b')
        plt.xticks(np.arange(len(reportDict)), list(reportDict.keys()),
                   rotation='vertical')
        plt.yscale('log')
        hB, = plt.plot([1, 1], 'b-')
        hR, = plt.plot([1, 1], 'r-')
        plt.legend((hB, hR), ('Requested', 'Max used'))
        hB.set_visible(False)
        hR.set_visible(False)
        plt.ylabel('Memory per job (mb)')
        plt.title('Memory summary')
        plt.ylim([100, plt.ylim()[1]])
        plt.xlim([-0.5, len(reportDict)-0.5])
        plt.vlines(np.arange(0.5, len(reportDict)), ymin=plt.ylim()[0],
                   ymax=plt.ylim()[1], linestyles='dashed')
        plt.tight_layout(pad=1)
        self.dataSet.save_figure(self, fig, 'memory_summary')

        # Plot time requested, queued and used for each task
        fig = plt.figure(figsize=(15, 12))
        bp = plt.boxplot([d['Elapsed'] / np.timedelta64(1, 'm')
                         for d in reportDict.values()],
                         positions=np.arange(len(reportDict))-0.15,
                         widths=0.25)
        setBoxColors(bp, 'r')
        bp = plt.boxplot([d['Timelimit'] / np.timedelta64(1, 'm')
                         for d in reportDict.values()],
                         positions=np.arange(len(reportDict))+0.15,
                         widths=0.25)
        setBoxColors(bp, 'b')
        bp = plt.boxplot([d['Queued'] / np.timedelta64(1, 'm')
                         for d in reportDict.values()],
                         positions=np.arange(len(reportDict))+0.15,
                         widths=0.25)
        setBoxColors(bp, 'g')
        plt.xticks(np.arange(len(reportDict)), list(reportDict.keys()),
                   rotation='vertical')
        plt.yscale('log')
        hB, = plt.plot([1, 1], 'b-')
        hR, = plt.plot([1, 1], 'r-')
        hG, = plt.plot([1, 1], 'g-')
        plt.legend((hB, hR, hG), ('Requested', 'Used', 'Queued'))
        hB.set_visible(False)
        hR.set_visible(False)
        hG.set_visible(False)
        plt.ylabel('Time per job (min)')
        plt.title('Time summary')
        plt.xlim([-0.5, len(reportDict)+0.5])
        plt.vlines(np.arange(0.5, len(reportDict)), ymin=plt.ylim()[0],
                   ymax=plt.ylim()[1], linestyles='dashed')
        plt.tight_layout(pad=1)
        self.dataSet.save_figure(self, fig, 'time_summary')

    def _fragment_slurm_job_ids(self, task: analysistask.ParallelAnalysisTask):
        """Map each fragment (fov) of a ParallelAnalysisTask to its SLURM
        job ID, skipping any fragment with no recorded environment (still
        pending, or never run) instead of raising -- unlike
        _generate_slurm_report, which assumes every fragment finished and
        errors out entirely otherwise (real, partially-complete
        experiments routinely have a handful of such fragments).
        """
        fragmentJobIds = {}
        for i in range(task.fragment_count()):
            env = self.dataSet.get_analysis_environment(task, i)
            if env is not None and 'SLURM_JOB_ID' in env:
                fragmentJobIds[i] = env['SLURM_JOB_ID']
        return fragmentJobIds

    def _generate_per_fov_slurm_usage(
            self, task: analysistask.ParallelAnalysisTask) -> pandas.DataFrame:
        """Per-fragment (fov) memory/time usage for one
        ParallelAnalysisTask, as a small dataframe indexed by fov with
        numeric 'mem_mb'/'time_min' columns.

        Unlike _generate_slurm_report's per-task report (grouped by
        JobID, string-formatted, no fov column, and only for fully
        complete tasks), this keeps each row tagged with the fov it came
        from and tolerates incomplete fragments -- meant to be combined
        across tasks into one per-experiment table (see
        _save_per_fov_resource_table) for comparing real usage against
        AnalysisTask.get_estimated_memory()/get_estimated_time().
        """
        fragmentJobIds = self._fragment_slurm_job_ids(task)
        if not fragmentJobIds:
            return pandas.DataFrame(columns=['mem_mb', 'time_min'])

        jobIdToFragment = {v: k for k, v in fragmentJobIds.items()}
        slurmDF = self._query_sacct(list(fragmentJobIds.values()))

        fov = slurmDF.index.to_series().map(jobIdToFragment)
        memMb = pandas.to_numeric(
            slurmDF['MaxRSS'].astype(str).str.extract(r'([\d.]+)')[0],
            errors='coerce')
        timeMin = slurmDF['Elapsed'] / np.timedelta64(1, 'm')

        result = pandas.DataFrame(
            {'fov': fov, 'mem_mb': memMb, 'time_min': timeMin})
        return result.dropna(subset=['fov']).set_index('fov').sort_index()

    def _save_per_fov_resource_table(self, taskList) -> None:
        """Write one parquet table -- rows are fov, and each
        ParallelAnalysisTask in taskList that has any completed fragment
        contributes two columns, '<task>_mem_mb'/'<task>_time_min' -- so
        real, per-fov SLURM memory/time usage across every task in the
        run can be compared directly against
        AnalysisTask.get_estimated_memory()/get_estimated_time() (or a
        candidate replacement formula) for a fov of known frame/z-stack
        size, instead of guessing at calibration constants.

        A task with no completed fragments (or that isn't a
        ParallelAnalysisTask -- e.g. GenerateAdaptiveThreshold,
        ExportBarcodes, which run once for the whole dataset rather than
        once per fov) simply contributes no columns rather than blocking
        the rest of the table.
        """
        columns = {}
        for t in taskList:
            currentTask = self.dataSet.load_analysis_task(t)
            if not isinstance(currentTask, analysistask.ParallelAnalysisTask):
                continue
            try:
                usage = self._generate_per_fov_slurm_usage(currentTask)
            except Exception:
                continue
            if usage.empty:
                continue
            columns[t + '_mem_mb'] = usage['mem_mb']
            columns[t + '_time_min'] = usage['time_min']

        if not columns:
            return
        wideDF = pandas.DataFrame(columns)
        wideDF.index.name = 'fov'
        self.dataSet.save_dataframe_to_parquet(
            wideDF, 'per_fov_resource_usage', self, subdirectory='reports')

    def _run_analysis(self):
        taskList = self.dataSet.get_analysis_tasks()

        self._save_per_fov_resource_table(taskList)

        reportTime = int(time.time())
        reportDict = {}
        analysisParameters = {}
        for t in taskList:
            currentTask = self.dataSet.load_analysis_task(t)
            try:
                if currentTask.is_complete():
                    slurmDF = self._generate_slurm_report(currentTask)
                    self.dataSet.save_dataframe_to_csv(slurmDF, t, self,
                                                       'reports')
                    dfStream = io.StringIO()
                    slurmDF.to_csv(dfStream, sep='|')
                    self._plot_slurm_report(slurmDF, t)
                    reportDict[t] = slurmDF
                    analysisParameters[t] = currentTask.get_parameters()

                    try:
                        requests.post('http://merlin.georgeemanuel.com/post',
                                      files={'file': (
                                          '.'.join([t, self.dataSet.dataSetName,
                                                    str(reportTime)]) + '.csv',
                                          dfStream.getvalue())},
                                      timeout=10)
                    except requests.exceptions.RequestException:
                        pass
            except Exception:
                pass

        self._plot_slurm_summary(reportDict)

        datasetMeta = {
            'image_width': self.dataSet.get_image_dimensions()[0],
            'image_height': self.dataSet.get_image_dimensions()[1],
            'barcode_length': self.dataSet.get_codebook(
                self.parameters['codebook_index']).get_bit_count(),
            'barcode_count': self.dataSet.get_codebook(
                self.parameters['codebook_index']).get_barcode_count(),
            'fov_count': len(self.dataSet.get_fovs()),
            'z_count': len(self.dataSet.get_z_positions()),
            'sequential_count': len(self.dataSet.get_data_organization()
                                    .get_sequential_rounds()),
            'dataset_name': self.dataSet.dataSetName,
            'report_time': reportTime,
            'analysis_parameters': analysisParameters
        }
        try:
            requests.post('http://merlin.georgeemanuel.com/post',
                          files={'file': ('.'.join(
                              [self.dataSet.dataSetName, str(reportTime)])
                                         + '.json', json.dumps(datasetMeta))},
                          timeout=10)
        except requests.exceptions.RequestException:
            pass
