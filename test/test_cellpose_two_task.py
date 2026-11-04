"""CellPoseEvalSAM (GPU, batched) + CellPoseSegmentSAM(eval_task=...) (CPU)
must give the same masks and features as the one-task CellPoseSegmentSAM.

cellpose itself is replaced by a deterministic fake model, so these run
without a GPU or model weights.
"""
import os

import numpy as np
import pytest
import tifffile

from merlin.analysis import segment

CELLPOSE = {'channel_1_name': 'bit1', 'warp_task': 'raggedWarp',
            'global_align_task': 'raggedGlobalAlign', 'expand_mask': 1,
            'downsample_factor': 2}


class FakeModel:
    """Returns two box-shaped cells per stack, like a cellpose eval. The
    synthetic test images are too flat to threshold into objects."""
    calls = []

    def eval(self, images, **kwargs):
        FakeModel.calls.append(images.shape)
        labels = np.zeros(images.shape[:3], dtype=np.uint16)
        z, rows, cols = labels.shape
        labels[:, rows // 8:rows // 3, cols // 8:cols // 3] = 1
        labels[:, rows // 2:rows - rows // 8, cols // 2:cols - cols // 8] = 2
        return (labels,)


@pytest.fixture
def fake_cellpose(monkeypatch):
    FakeModel.calls = []
    monkeypatch.setattr(segment.CellPoseSegmentSAM, '_build_model',
                        lambda self: FakeModel())
    return FakeModel


def _mask(task, fov):
    return tifffile.imread(task.dataSet._analysis_image_name(
        task, 'segmented_mask_', fov))


def _run(task, fragments):
    task.save()
    for i in fragments:
        task._run_analysis(i)


def test_two_task_matches_one_task(ragged_merfish_data, ragged_warp_task,
                                   ragged_global_align_task, fake_cellpose):
    fovs = sorted(ragged_merfish_data.get_fovs())
    one = segment.CellPoseSegmentSAM(ragged_merfish_data, dict(CELLPOSE),
                                     analysisName='cpOneTask')
    _run(one, fovs)

    evalTask = segment.CellPoseEvalSAM(
        ragged_merfish_data, {**CELLPOSE, 'fovs_per_job': 3},
        analysisName='cpEval')
    assert evalTask.fragment_count() == int(np.ceil(len(fovs) / 3))
    _run(evalTask, range(evalTask.fragment_count()))
    post = segment.CellPoseSegmentSAM(
        ragged_merfish_data, {**CELLPOSE, 'eval_task': 'cpEval'},
        analysisName='cpPost')
    assert 'cpEval' in post.get_dependencies()
    _run(post, fovs)

    cellCount = 0
    for fov in fovs:
        assert np.array_equal(_mask(one, fov), _mask(post, fov))
        oneCells = one.get_feature_database().read_features(fov)
        postCells = post.get_feature_database().read_features(fov)
        assert len(oneCells) == len(postCells)
        assert sorted(c.get_volume() for c in oneCells) \
            == sorted(c.get_volume() for c in postCells)
        cellCount += len(oneCells)
    assert cellCount > 0


def test_eval_batch_resumes_after_interruption(
        ragged_merfish_data, ragged_warp_task, ragged_global_align_task,
        fake_cellpose):
    evalTask = segment.CellPoseEvalSAM(
        ragged_merfish_data, {**CELLPOSE, 'fovs_per_job': 2},
        analysisName='cpEvalResume')
    _run(evalTask, [0])
    batch = evalTask._batch_fovs(0)
    assert len(fake_cellpose.calls) == len(batch)

    # as if the job was killed before the second fov was saved
    os.remove(evalTask._mask_path(batch[1]))
    fake_cellpose.calls = []
    evalTask._run_analysis(0)
    assert len(fake_cellpose.calls) == 1
    assert all(os.path.exists(evalTask._mask_path(f)) for f in batch)


def test_eval_task_rejects_eval_task_parameter(ragged_merfish_data):
    from merlin.core import analysistask
    with pytest.raises(analysistask.InvalidParameterException):
        segment.CellPoseEvalSAM(ragged_merfish_data,
                                {**CELLPOSE, 'eval_task': 'other'},
                                analysisName='cpEvalBad')


def test_build_model_fails_without_cuda(ragged_merfish_data, monkeypatch):
    torch = pytest.importorskip('torch')
    pytest.importorskip('cellpose.models')
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    task = segment.CellPoseEvalSAM(ragged_merfish_data,
                                   {**CELLPOSE, 'use_gpu': True},
                                   analysisName='cpEvalNoGpu')
    with pytest.raises(RuntimeError, match='CUDA'):
        task._build_model()


def test_use_ffc_requires_ffc_task(ragged_merfish_data):
    with pytest.raises(segment.analysistask.InvalidParameterException):
        segment.CellPoseSegmentSAM(ragged_merfish_data,
                                   {**CELLPOSE, 'use_ffc': True},
                                   analysisName='cpFfcMissing')


def test_ffc_task_is_a_dependency_only_when_used(ragged_merfish_data):
    for cls in (segment.CellPoseSegmentSAM, segment.CellPoseEvalSAM):
        off = cls(ragged_merfish_data, dict(CELLPOSE), analysisName='cpOff')
        on = cls(ragged_merfish_data,
                 {**CELLPOSE, 'use_ffc': True, 'ffc_task': 'ffc'},
                 analysisName='cpOn')
        assert 'ffc' not in off.get_dependencies()
        assert 'ffc' in on.get_dependencies()


def test_read_image_stack_applies_ffc_before_warp(
        ragged_merfish_data, ragged_warp_task, monkeypatch):
    from skimage import transform
    from merlin.analysis.createffc import CreateFfc
    fov = sorted(ragged_merfish_data.get_fovs())[0]
    channel = ragged_merfish_data.get_data_organization()\
        .get_data_channel_index('bit1')
    off = segment.CellPoseSegmentSAM(ragged_merfish_data, dict(CELLPOSE),
                                     analysisName='cpOff')
    on = segment.CellPoseSegmentSAM(
        ragged_merfish_data,
        {**CELLPOSE, 'use_ffc': True, 'ffc_task': 'ffc'},
        analysisName='cpOn')
    raw0 = ragged_merfish_data.get_raw_image(
        channel, fov, ragged_merfish_data.get_z_positions_segmentation(fov)[0])
    rows, cols = np.indices(raw0.shape)
    field = (0.5 + rows / rows.max()).astype(np.float32)
    dark = np.full(raw0.shape, 3, dtype=np.float32)
    monkeypatch.setattr(on, '_ffc_inputs', lambda ch: (field, dark))

    plain = off._read_image_stack(fov, channel)
    corrected = on._read_image_stack(fov, channel)
    assert plain.dtype == raw0.dtype
    assert corrected.dtype == np.float32
    assert corrected.shape == plain.shape

    tf = ragged_warp_task.get_transformation(fov, channel)
    expected = transform.warp(CreateFfc.apply_ffc(raw0, field, dark), tf,
                              preserve_range=True).astype(np.float32)
    assert np.allclose(corrected[0], expected)
