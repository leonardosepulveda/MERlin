import json
import os

import yaml

import merlin
from merlin import merlin as m
from merlin.util import snakewriter

root = os.path.join(os.path.dirname(merlin.__file__), '..', 'test')


def test_load_analysis_parameters_yaml_matches_json():
    jsonPath = os.sep.join(
        [root, 'auxiliary_files', 'test_analysis_parameters.json'])
    yamlPath = os.sep.join(
        [root, 'auxiliary_files', 'test_analysis_parameters.yaml'])

    with open(jsonPath, 'r') as f:
        jsonParameters = m._load_analysis_parameters(f)
    with open(yamlPath, 'r') as f:
        yamlParameters = m._load_analysis_parameters(f)

    assert jsonParameters == yamlParameters


def test_load_analysis_parameters_unrecognized_extension_defaults_to_json(
        tmp_path):
    # A recipe with no/unknown extension (e.g. an existing script that
    # doesn't set one) should still parse as JSON, matching the behavior
    # from before YAML support was added.
    jsonPath = os.sep.join(
        [root, 'auxiliary_files', 'test_analysis_parameters.json'])
    with open(jsonPath, 'r') as f:
        expectedParameters = json.load(f)

    noExtensionPath = tmp_path / 'recipe_without_extension'
    with open(jsonPath, 'r') as source:
        noExtensionPath.write_text(source.read())

    with open(noExtensionPath, 'r') as f:
        parameters = m._load_analysis_parameters(f)

    assert parameters == expectedParameters


def test_load_json_or_yaml_cluster_config_comment_is_a_no_op(tmp_path):
    # The actual point of adding YAML support for cluster_resource_
    # allocation files: a per-task calculated mem/time value can be left
    # as a commented-out reference line (ignored by the parser) next to
    # an active key, and uncommented later to override it.
    yamlText = """
    __default__:
      mem: 8000
      time: '3:00:00'
    FiducialCorrelationWarp:
      # calculated mem: 1 GB -- uncomment to override (value is in MB)
      # mem: 1000
      partition: zhuang
    """
    yamlPath = tmp_path / 'cluster_resource_allocation_test.yaml'
    yamlPath.write_text(yamlText)

    with open(yamlPath, 'r') as f:
        clusterConfig = m._load_json_or_yaml(f)

    assert clusterConfig['__default__'] == {'mem': 8000, 'time': '3:00:00'}
    # the commented-out mem line contributes nothing -- only the active
    # 'partition' key is present
    assert clusterConfig['FiducialCorrelationWarp'] == {'partition': 'zhuang'}


def test_snakefile_generator_accepts_yaml_recipe(simple_merfish_data):
    # Mirrors test_snakemake.py's test_snakemake_generator_task_chain, but
    # sources the same task-chain definition from a YAML string instead of a
    # Python dict literal, to confirm a YAML-sourced recipe flows through
    # SnakefileGenerator identically to a JSON/dict one. Only exercises
    # generate_workflow() (pure Python graph-building), not actual snakemake
    # execution, since the snakemake package itself is unrelated to this
    # loader change.
    recipeYaml = """
    analysis_tasks:
      - task: SimpleAnalysisTask
        module: merlin.analysis.testtask
        analysis_name: Task1
        parameters: {}
      - task: SimpleParallelAnalysisTask
        module: merlin.analysis.testtask
        analysis_name: Task2
        parameters:
          dependencies: [Task1]
      - task: SimpleParallelAnalysisTask
        module: merlin.analysis.testtask
        analysis_name: Task3
        parameters:
          dependencies: [Task2]
    """
    analysisParameters = yaml.safe_load(recipeYaml)

    generator = snakewriter.SnakefileGenerator(
        analysisParameters, simple_merfish_data)
    generator.generate_workflow()

    outputTask1 = simple_merfish_data.load_analysis_task('Task1')
    outputTask2 = simple_merfish_data.load_analysis_task('Task2')
    outputTask3 = simple_merfish_data.load_analysis_task('Task3')
    assert not outputTask1.is_complete()
    assert not outputTask2.is_complete()
    assert not outputTask3.is_complete()

    simple_merfish_data.delete_analysis(outputTask3)
    simple_merfish_data.delete_analysis(outputTask2)
    simple_merfish_data.delete_analysis(outputTask1)
