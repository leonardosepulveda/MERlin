import os
import pytest
import shutil
import glob
from merlin.core import dataset
from merlin.analysis import testtask
import merlin


root = os.path.join(os.path.dirname(merlin.__file__), '..', 'test')
merlin.DATA_HOME = os.path.abspath('test_data')
merlin.ANALYSIS_HOME = os.path.abspath('test_analysis')
merlin.ANALYSIS_PARAMETERS_HOME = os.path.abspath('test_analysis_parameters')
merlin.CODEBOOK_HOME = os.path.abspath('test_codebooks')
merlin.DATA_ORGANIZATION_HOME = os.path.abspath('test_dataorganization')
merlin.POSITION_HOME = os.path.abspath('test_positions')
merlin.MICROSCOPE_PARAMETERS_HOME = os.path.abspath('test_microcope_parameters')


dataDirectory = os.sep.join([merlin.DATA_HOME, 'test'])
merfishDataDirectory = os.sep.join([merlin.DATA_HOME, 'merfish_test'])
raggedMerfishDataDirectory = os.sep.join(
    [merlin.DATA_HOME, 'ragged_merfish_test'])


@pytest.fixture(scope='session')
def base_files():
    folderList = [merlin.DATA_HOME, merlin.ANALYSIS_HOME,
                  merlin.ANALYSIS_PARAMETERS_HOME, merlin.CODEBOOK_HOME,
                  merlin.DATA_ORGANIZATION_HOME, merlin.POSITION_HOME,
                  merlin.MICROSCOPE_PARAMETERS_HOME]
    for folder in folderList:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    shutil.copyfile(
        os.sep.join(
            [root, 'auxiliary_files', 'test_data_organization.csv']),
        os.sep.join(
            [merlin.DATA_ORGANIZATION_HOME, 'test_data_organization.csv']))
    shutil.copyfile(
        os.sep.join(
            [root, 'auxiliary_files', 'test_data_organization_ragged.csv']),
        os.sep.join(
            [merlin.DATA_ORGANIZATION_HOME,
             'test_data_organization_ragged.csv']))
    shutil.copyfile(
        os.sep.join(
            [root, 'auxiliary_files', 'test_codebook.csv']),
        os.sep.join(
            [merlin.CODEBOOK_HOME, 'test_codebook.csv']))
    shutil.copyfile(
        os.sep.join(
            [root, 'auxiliary_files', 'test_codebook2.csv']),
        os.sep.join(
            [merlin.CODEBOOK_HOME, 'test_codebook2.csv']))
    shutil.copyfile(
        os.sep.join(
            [root, 'auxiliary_files', 'test_codebook_ragged.csv']),
        os.sep.join(
            [merlin.CODEBOOK_HOME, 'test_codebook_ragged.csv']))
    shutil.copyfile(
        os.sep.join(
            [root, 'auxiliary_files', 'test_positions.csv']),
        os.sep.join(
            [merlin.POSITION_HOME, 'test_positions.csv']))
    shutil.copyfile(
        os.sep.join(
            [root, 'auxiliary_files', 'test_positions_ragged.csv']),
        os.sep.join(
            [merlin.POSITION_HOME, 'test_positions_ragged.csv']))
    shutil.copyfile(
        os.sep.join(
            [root, 'auxiliary_files', 'test_analysis_parameters.json']),
        os.sep.join(
            [merlin.ANALYSIS_PARAMETERS_HOME, 'test_analysis_parameters.json']))
    shutil.copyfile(
        os.sep.join(
            [root, 'auxiliary_files', 'test_analysis_parameters.yaml']),
        os.sep.join(
            [merlin.ANALYSIS_PARAMETERS_HOME, 'test_analysis_parameters.yaml']))
    shutil.copyfile(
        os.sep.join(
            [root, 'auxiliary_files', 'test_microscope_parameters.json']),
        os.sep.join(
            [merlin.MICROSCOPE_PARAMETERS_HOME,
             'test_microscope_parameters.json']))

    yield

    for folder in folderList:
        shutil.rmtree(folder)


@pytest.fixture(scope='session')
def merfish_files(base_files):
    os.mkdir(merfishDataDirectory)

    for imageFile in glob.iglob(
            os.sep.join([root, 'auxiliary_files', '*.tif'])):
        if os.path.isfile(imageFile):
            shutil.copy(imageFile, merfishDataDirectory)

    yield

    shutil.rmtree(merfishDataDirectory)


@pytest.fixture(scope='session')
def simple_data(base_files):
    os.mkdir(dataDirectory)
    testData = dataset.DataSet('test')

    yield testData

    shutil.rmtree(dataDirectory)


@pytest.fixture(scope='session')
def simple_merfish_data(merfish_files):
    testMERFISHData = dataset.MERFISHDataSet(
            'merfish_test',
            dataOrganizationName='test_data_organization.csv',
            codebookNames=['test_codebook.csv'],
            positionFileName='test_positions.csv',
            microscopeParametersName='test_microscope_parameters.json')
    yield testMERFISHData


@pytest.fixture(scope='session')
def two_codebook_merfish_data(merfish_files):
    testMERFISHData = dataset.MERFISHDataSet(
            'merfish_test',
            dataOrganizationName='test_data_organization.csv',
            codebookNames=['test_codebook2.csv', 'test_codebook.csv'],
            positionFileName='test_positions.csv',
            analysisHome=os.path.join(merlin.ANALYSIS_HOME, '..',
                                      'test_analysis_two_codebook'),
            microscopeParametersName='test_microscope_parameters.json')
    yield testMERFISHData

    shutil.rmtree('test_analysis_two_codebook')


@pytest.fixture(scope='session')
def custom_figures_merfish_data(merfish_files):
    figuresPath = os.path.abspath('test_figures_custom')
    testMERFISHData = dataset.MERFISHDataSet(
            'merfish_test',
            dataOrganizationName='test_data_organization.csv',
            codebookNames=['test_codebook.csv'],
            positionFileName='test_positions.csv',
            analysisHome=os.path.join(merlin.ANALYSIS_HOME, '..',
                                      'test_analysis_custom_figures'),
            microscopeParametersName='test_microscope_parameters.json',
            figuresPath=figuresPath)
    yield testMERFISHData

    shutil.rmtree('test_analysis_custom_figures')
    shutil.rmtree(figuresPath, ignore_errors=True)


@pytest.fixture(scope='session')
def ragged_merfish_files(base_files):
    os.mkdir(raggedMerfishDataDirectory)

    for imageFile in glob.iglob(
            os.sep.join([root, 'auxiliary_files', 'ragged_tifs', '*.tif'])):
        if os.path.isfile(imageFile):
            shutil.copy(imageFile, raggedMerfishDataDirectory)

    yield

    shutil.rmtree(raggedMerfishDataDirectory)


@pytest.fixture(scope='session')
def ragged_merfish_data(ragged_merfish_files):
    # fov 0 is full depth everywhere; fov 1, 2, and 3 each have at least one
    # raw file shorter than the deepest globally-configured z position (see
    # test_data_organization_ragged.csv / ragged_tifs), so this requires
    # allowRaggedZStacks to construct without raising.
    testMERFISHData = dataset.MERFISHDataSet(
            'ragged_merfish_test',
            dataOrganizationName='test_data_organization_ragged.csv',
            codebookNames=['test_codebook_ragged.csv'],
            positionFileName='test_positions_ragged.csv',
            analysisHome=os.path.join(merlin.ANALYSIS_HOME, '..',
                                      'test_analysis_ragged'),
            microscopeParametersName='test_microscope_parameters.json',
            allowRaggedZStacks=True)
    yield testMERFISHData

    shutil.rmtree('test_analysis_ragged')


@pytest.fixture(scope='session')
def ragged_warp_task(ragged_merfish_data):
    from merlin.analysis import warp
    # edge_width_to_remove/percentile_pixel_to_keep are relaxed from their
    # defaults since the tiny synthetic ragged test images are much smaller
    # than the default edge crop; must be .save()d so downstream tasks can
    # load_analysis_task('raggedWarp') by name
    task = warp.FiducialCorrelationWarp(
        ragged_merfish_data,
        parameters={'edge_width_to_remove': 0, 'percentile_pixel_to_keep': 100},
        analysisName='raggedWarp')
    task.save()
    for fov in ragged_merfish_data.get_fovs():
        task._run_analysis(int(fov))
    return task


@pytest.fixture(scope='session')
def ragged_global_align_task(ragged_merfish_data):
    from merlin.analysis import globalalign
    task = globalalign.SimpleGlobalAlignment(
        ragged_merfish_data, parameters={}, analysisName='raggedGlobalAlign')
    task.save()
    task._run_analysis()
    return task


@pytest.fixture(scope='session')
def ragged_preprocess_task(ragged_merfish_data, ragged_warp_task):
    from merlin.analysis import preprocess
    # save_pixel_histogram=False avoids needing histogram files on disk,
    # which _calculate_initial_scale_factors would otherwise require
    task = preprocess.DeconvolutionPreprocess(
        ragged_merfish_data,
        parameters={'warp_task': 'raggedWarp', 'codebook_index': 0,
                    'save_pixel_histogram': False},
        analysisName='raggedPreprocess')
    task.save()
    return task


@pytest.fixture(scope='session')
def ragged_optimize_task(ragged_merfish_data, ragged_preprocess_task):
    from merlin.analysis import optimize
    task = optimize.OptimizeIterationFOV(
        ragged_merfish_data, parameters={'preprocess_task': 'raggedPreprocess'},
        analysisName='raggedOptimize')
    task.save()
    # .run() (not _run_analysis directly) so completion is recorded --
    # get_scale_factors()/get_backgrounds() check is_complete() before
    # returning results
    for fragmentIndex in range(len(task.parameters['fov_index'])):
        task.run(fragmentIndex)
    return task


@pytest.fixture(scope='session')
def ragged_decode_task(ragged_merfish_data, ragged_preprocess_task,
                        ragged_optimize_task, ragged_global_align_task):
    from merlin.analysis import decode
    task = decode.Decode(
        ragged_merfish_data,
        parameters={'preprocess_task': 'raggedPreprocess',
                    'optimize_task': 'raggedOptimize',
                    'global_align_task': 'raggedGlobalAlign',
                    'single_fov_optimization': True},
        analysisName='raggedDecode')
    task.save()
    return task


@pytest.fixture(scope='session')
def ragged_segment_task(ragged_merfish_data, ragged_warp_task,
                         ragged_global_align_task):
    from merlin.analysis import segment
    task = segment.WatershedSegment(
        ragged_merfish_data,
        parameters={'warp_task': 'raggedWarp',
                    'global_align_task': 'raggedGlobalAlign'},
        analysisName='raggedSegment')
    task.save()
    return task


@pytest.fixture(scope='function')
def single_task(simple_data):
    task = testtask.SimpleAnalysisTask(
            simple_data, parameters={'a': 5, 'b': 'b_string'})
    yield task
    simple_data.delete_analysis(task)


@pytest.fixture(scope='function', params=[
    testtask.SimpleAnalysisTask, testtask.SimpleParallelAnalysisTask,
    testtask.SimpleInternallyParallelAnalysisTask])
def simple_task(simple_data, request):
    task = request.param(
            simple_data, parameters={'a': 5, 'b': 'b_string'})
    yield task
    simple_data.delete_analysis(task)


@pytest.fixture(scope='function', params=[
    testtask.SimpleAnalysisTask, testtask.SimpleParallelAnalysisTask,
    testtask.SimpleInternallyParallelAnalysisTask])
def simple_merfish_task(simple_merfish_data, request):
    task = request.param(
        simple_merfish_data, parameters={'a': 5, 'b': 'b_string'})
    yield task
    simple_merfish_data.delete_analysis(task)
