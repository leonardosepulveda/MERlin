import os
import pytest
from matplotlib import pyplot as plt

from merlin.analysis import testtask


class FigureGeneratingTask(testtask.SimpleAnalysisTask):
    """A plain (non-parallel) task that saves one verification figure."""

    def _generate_verification_figures(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        self.dataSet.save_task_figure(self, fig, 'a_figure')
        plt.close(fig)


class FailingFigureTask(testtask.SimpleAnalysisTask):
    """A plain task whose figure generation always raises -- run() must
    still complete successfully despite this."""

    def _generate_verification_figures(self):
        raise ValueError('deliberate failure in verification figure generation')


class FigureGeneratingParallelTask(testtask.SimpleParallelAnalysisTask):
    """A parallel task that saves one verification figure, expected to fire
    exactly once, only once every fragment is complete."""

    def _generate_verification_figures(self):
        fig, ax = plt.subplots()
        ax.plot([0, 1], [1, 0])
        self.dataSet.save_task_figure(self, fig, 'a_parallel_figure')
        plt.close(fig)


def _figure_path(dataSet, taskName, figureName):
    return os.sep.join([dataSet.figuresPath,
                        '.'.join(['merlin', taskName, figureName]) + '.png'])


def test_default_generate_verification_figures_is_a_noop(simple_merfish_data):
    task = testtask.SimpleAnalysisTask(
        simple_merfish_data, parameters={}, analysisName='noopFigureTask')
    task.save()
    task.run()
    assert task.is_complete()
    # The shared figures folder is dataset-wide (other tasks in this same
    # session may have already created it) -- what matters is that THIS
    # task's own no-op default didn't add anything to it.
    figuresDir = simple_merfish_data.figuresPath
    if os.path.isdir(figuresDir):
        assert not any(name.startswith('merlin.noopFigureTask.')
                       for name in os.listdir(figuresDir))


def test_generate_verification_figures_runs_after_task_completes(
        simple_merfish_data):
    task = FigureGeneratingTask(
        simple_merfish_data, parameters={}, analysisName='figureGeneratingTask')
    task.save()
    task.run()
    assert task.is_complete()
    assert os.path.exists(
        _figure_path(simple_merfish_data, 'figureGeneratingTask', 'a_figure'))


def test_failing_verification_figure_does_not_break_the_task(
        simple_merfish_data):
    task = FailingFigureTask(
        simple_merfish_data, parameters={}, analysisName='failingFigureTask')
    task.save()
    task.run()  # must not raise
    assert task.is_complete()
    assert not task.is_error()


def test_custom_figures_path_is_used_as_is(custom_figures_merfish_data):
    task = FigureGeneratingTask(
        custom_figures_merfish_data, parameters={},
        analysisName='customFiguresPathTask')
    task.save()
    task.run()
    assert task.is_complete()

    # figuresPath is used exactly as given, not nested under analysisPath.
    assert custom_figures_merfish_data.figuresPath \
        != os.sep.join([custom_figures_merfish_data.analysisPath, 'figures'])
    assert os.path.exists(_figure_path(
        custom_figures_merfish_data, 'customFiguresPathTask', 'a_figure'))


def test_parallel_task_generates_figure_once_after_last_fragment(
        simple_merfish_data):
    task = FigureGeneratingParallelTask(
        simple_merfish_data, parameters={}, analysisName='figureGeneratingParallelTask')
    task.save()
    figurePath = _figure_path(
        simple_merfish_data, 'figureGeneratingParallelTask', 'a_parallel_figure')

    for i in range(task.fragment_count() - 1):
        task.run(i)
        assert not os.path.exists(figurePath), \
            'figure must not appear before every fragment is complete'

    task.run(task.fragment_count() - 1)
    assert task.is_complete()
    assert os.path.exists(figurePath)
