import json
import os

from merlin.core import dataset

microscopeJson = {'flip_horizontal': False, 'flip_vertical': True,
                  'transpose': True, 'microns_per_pixel': 0.108,
                  'image_dimensions': [128, 128], 'camera_rotation_deg': -0.95}

microscopeYaml = """\
# MF3, camera remounted 2026-09-01
flip_horizontal: false
flip_vertical: true
transpose: true
microns_per_pixel: 0.108   # ruler slide, 2026-09-02
image_dimensions: [128, 128]
# BC555_sample_05/epi beads -0.944, BC553_sample_02/epi -0.954 (2026-09-27)
camera_rotation_deg: -0.95
"""


def _image_dataset(tmp_path, name, microscopeParametersName):
    os.makedirs(tmp_path / 'data' / name)
    return dataset.ImageDataSet(
        name, dataHome=str(tmp_path / 'data'),
        analysisHome=str(tmp_path / 'analysis'),
        microscopeParametersName=microscopeParametersName)


def test_yaml_microscope_parameters_match_json_twin(tmp_path):
    jsonPath = tmp_path / 'mf3.json'
    jsonPath.write_text(json.dumps(microscopeJson))
    yamlPath = tmp_path / 'mf3.yaml'
    yamlPath.write_text(microscopeYaml)

    fromJson = _image_dataset(tmp_path, 'fromJson', str(jsonPath))
    fromYaml = _image_dataset(tmp_path, 'fromYaml', str(yamlPath))

    assert fromYaml.microscopeParameters == fromJson.microscopeParameters
    # readers of microscope_parameters.json see plain JSON; the source
    # keeps its comments next to it
    with open(os.path.join(fromYaml.analysisPath,
                           'microscope_parameters.json')) as f:
        assert json.load(f) == microscopeJson
    assert (open(os.path.join(fromYaml.analysisPath,
                              'microscope_parameters_source.yaml')).read()
            == microscopeYaml)


def test_json_microscope_parameters_copied_verbatim(tmp_path):
    jsonText = '{"microns_per_pixel": 0.108,\n "flip_vertical": true}\n'
    jsonPath = tmp_path / 'scope.json'
    jsonPath.write_text(jsonText)

    imageData = _image_dataset(tmp_path, 'fromJson', str(jsonPath))

    analysisFiles = os.listdir(imageData.analysisPath)
    assert 'microscope_parameters_source.yaml' not in analysisFiles
    assert open(os.path.join(imageData.analysisPath,
                             'microscope_parameters.json')).read() == jsonText
    assert imageData.get_microns_per_pixel() == 0.108
