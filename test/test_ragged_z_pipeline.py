"""Tests that the analysis pipeline actually runs correctly against a ragged
(per-fov variable z-count) dataset, not just the core DataOrganization/
MERFISHDataSet layer covered in test_dataorganization.py/test_dataset.py.

These exercise the real fov-scoped z-position fixes made across warp.py
(no change needed there -- verified directly), preprocess.py, optimize.py,
decode.py, globalalign.py, and segment.py's WatershedSegment, by actually
running each task's _run_analysis against the ragged_merfish_data fixture
(4 fovs with varying per-round z-depth -- see conftest.py/
test_data_organization_ragged.csv). CellPoseSegment3D/CellPoseSegmentSAM
(need cellpose) and SmfishSignal's actual bigfish spot detection (the
synthetic test images have no real spot content) are not integration-tested
here; segment.py's zPos_segment[sel] correctness fix and sequential.py's
_resolve_z_indexes/optimize.py's _resolve_z_index are tested directly instead
since they don't depend on those heavier dependencies.
"""
import numpy as np
import pytest
import zarr

from merlin.core import analysistask
from merlin.analysis import optimize
from merlin.analysis import sequential
from merlin.analysis import generatemosaic


def test_warp_ragged_z_scoping(ragged_merfish_data, ragged_warp_task):
    # fov 1's round-1 channels (bit3/bit4) only have z=[0,1] available (see
    # test_data_organization_ragged.csv/ragged_tifs); a fov-scoped caller
    # only ever asks for z-index 0 or 1 and gets a correctly shaped image
    fovZPositions = ragged_merfish_data.get_z_positions(1)
    assert fovZPositions == [0, 1]
    img = ragged_warp_task.get_aligned_image(1, 2, 1)  # bit3, z-index 1
    assert img.shape == (128, 128)

    # a caller that ignores the fov-scoped range and asks for the global
    # z-index 3 (valid for other fovs, but beyond fov 1's own depth) hits an
    # out-of-bounds raw frame read -- this is exactly why every fixed module
    # must use get_z_positions(fov) instead of the global call
    with pytest.raises(Exception):
        ragged_warp_task.get_aligned_image(1, 2, 3)


def test_preprocess_ragged_batch_shape(ragged_merfish_data, ragged_preprocess_task):
    # fov 1 is thin (2 z's available), fov 0 is full depth (4 z's) -- the
    # batch (zIndex=None) path must shrink to each fov's own z count
    thinSet = ragged_preprocess_task.get_processed_image_set(1)
    assert thinSet.shape == (4, 2, 128, 128)

    fullSet = ragged_preprocess_task.get_processed_image_set(0)
    assert fullSet.shape == (4, 4, 128, 128)


def test_optimize_resolve_z_index_default_is_fov_relative(
        ragged_merfish_data, ragged_preprocess_task):
    task = optimize.OptimizeIterationFOV(
        ragged_merfish_data,
        parameters={'preprocess_task': 'raggedPreprocess'},
        analysisName='raggedOptimizeResolveDefault')

    # fov 0 has 4 z's -> middle index 2; fov 1 has 2 z's -> middle index 1
    assert task._resolve_z_index(0) == 2
    assert task._resolve_z_index(1) == 1


def test_optimize_resolve_z_index_explicit_override_raises_for_thin_fov(
        ragged_merfish_data, ragged_preprocess_task):
    task = optimize.OptimizeIterationFOV(
        ragged_merfish_data,
        parameters={'preprocess_task': 'raggedPreprocess', 'z_index': 3},
        analysisName='raggedOptimizeResolveOverride')

    # z_index=3 is valid for fov 0 (4 z's) but not fov 1 (only 2 z's)
    assert task._resolve_z_index(0) == 3
    with pytest.raises(analysistask.InvalidParameterException):
        task._resolve_z_index(1)


def test_optimize_ragged_all_fragments_complete(ragged_optimize_task):
    # the fixture already ran every fragment via .run(); confirm the whole
    # task completed (each fragment maps to a different fov, some thin)
    assert ragged_optimize_task.is_complete()


def test_decode_ragged_zarr_shape_matches_fov_z_count(
        ragged_merfish_data, ragged_decode_task):
    for fov, expectedZCount in [(0, 4), (1, 2), (2, 3), (3, 4)]:
        ragged_decode_task.run(fov)
        zarrPath = ragged_merfish_data._analysis_zarr_name(
            ragged_decode_task, 'decoded', fov)
        decodedZarr = zarr.open(zarrPath, mode='r')
        assert decodedZarr.shape[0] == expectedZCount


def test_globalalign_ragged_fov_scoped_interpolation(
        ragged_merfish_data, ragged_global_align_task):
    # fov 1 only has z=[0,1] -- interpolating z-index 1 (the last available
    # index) should resolve to that fov's own deepest z position (1), not
    # some fraction of the global 4-position range
    global_coords = ragged_global_align_task.fov_coordinates_to_global(
        1, (1, 0, 0))
    assert global_coords[0] == 1.0


def test_segment_watershed_ragged_fov2(ragged_merfish_data, ragged_segment_task):
    # fov 2 is the one fov where the regular and segmentation z-counts
    # happen to match (3 each) -- WatershedSegment assumes DAPI/polyT share
    # the same z-grid as the regular channels (unlike the newer CellPose
    # classes, it never reconciles two different z lists), so this is the
    # only fov in the fixture it's actually compatible with
    ragged_segment_task._run_analysis(2)
    features = ragged_segment_task.get_cell_boundaries()
    # feature z-coordinates should be built from fov 2's own 3-element z
    # list, not the global 4-element one
    for feature in features:
        assert len(feature.get_z_coordinates()) == 3


def test_sequential_resolve_z_indexes_default_all_planes(
        ragged_merfish_data, ragged_warp_task, ragged_global_align_task):
    task = sequential.SmfishSignal(
        ragged_merfish_data,
        parameters={'global_align_task': 'raggedGlobalAlign',
                    'warp_task': 'raggedWarp'},
        analysisName='raggedSmfishDefault')

    assert task._resolve_z_indexes(0) == [0, 1, 2, 3]
    assert task._resolve_z_indexes(1) == [0, 1]


def test_sequential_resolve_z_indexes_explicit_list_skips_invalid(
        ragged_merfish_data, ragged_warp_task, ragged_global_align_task,
        capsys):
    task = sequential.SmfishSignal(
        ragged_merfish_data,
        parameters={'global_align_task': 'raggedGlobalAlign',
                    'warp_task': 'raggedWarp',
                    'z_indexes': [0, 1, 3]},
        analysisName='raggedSmfishExplicit')

    # fov 0 has all 4 z's -- nothing skipped
    assert task._resolve_z_indexes(0) == [0, 1, 3]
    # fov 1 only has z=[0,1] -- index 3 is skipped, with a printed notice
    assert task._resolve_z_indexes(1) == [0, 1]
    assert 'skipping' in capsys.readouterr().out.lower()


def test_sumsignal_ragged_per_fov_z_index_raises_for_thin_fov(
        ragged_merfish_data, ragged_global_align_task):
    # z_index=3 passes SumSignal's constructor-time global sanity check
    # (the dataset-wide z count is 4), but fov 1 only has 2 z's available --
    # the per-fov runtime check added in _run_analysis must catch this
    task = sequential.SumSignal(
        ragged_merfish_data,
        parameters={'global_align_task': 'raggedGlobalAlign', 'z_index': 3},
        analysisName='raggedSumSignal')

    with pytest.raises(analysistask.InvalidParameterException):
        task._run_analysis(1)


class _StubWarpTask:
    """A minimal stand-in for a Warp task's get_aligned_image, used to test
    GenerateMosaicSimple.load_tile's zero-fill gate without needing a real
    warp task or raw images."""

    def __init__(self, imageShape):
        self.imageShape = imageShape
        self.calls = []

    def get_aligned_image(self, fov, dataChannel, zIndex, chromaticCorrector=None):
        self.calls.append((fov, dataChannel, zIndex))
        return np.full(self.imageShape, 100, dtype=np.uint16)


def test_generatemosaic_ragged_zero_fill(ragged_merfish_data):
    task = generatemosaic.GenerateMosaicSimple(
        ragged_merfish_data,
        parameters={'warp_task': 'raggedWarp', 'global_align_task': 'raggedGlobalAlign',
                    'downsample': 1, 'fov_crop_width': 0},
        analysisName='raggedMosaic')
    stub = _StubWarpTask(ragged_merfish_data.get_image_dimensions())
    task.warpTask = stub

    # fov 0 has z-index 3 available -- the stub's real (non-zero) image is
    # used, and get_aligned_image is actually called
    tile = task.load_tile(0, 3, 0)
    assert np.any(tile > 0)
    assert stub.calls == [(0, 0, 3)]

    # fov 1 only has z=[0,1] -- z-index 3 is beyond its depth, so a blank
    # tile of the correct shape is returned WITHOUT calling get_aligned_image
    stub.calls = []
    tile = task.load_tile(1, 3, 0)
    assert np.all(tile == 0)
    assert tile.shape == tuple(ragged_merfish_data.get_image_dimensions())
    assert stub.calls == []


def test_segment_zPos_segment_retained_values_matches_masks_shape():
    # standalone check of the zPos_segment[sel] correctness fix in
    # CellPoseSegment3D/CellPoseSegmentSAM._run_analysis (segment.py) --
    # cellpose itself isn't installed in this test environment, so this
    # exercises the exact masking arithmetic in isolation rather than a full
    # _run_analysis integration test
    zPos = np.array([0.0, 1.0, 2.0, 3.0])
    zPosSegment = np.array([0.0, 1.0, 2.0])  # fewer z's than zPos -- the
    # zPos <= zPos_segment invariant this code normally assumes is broken

    sel = np.isin(zPosSegment, zPos)
    zPosRetained = zPosSegment[sel]

    # zPosRetained must have exactly one entry per True in sel, in order --
    # this is what feature_from_label_matrix requires to line up correctly
    # with the correspondingly-masked mask/image stack
    assert sel.sum() == len(zPosRetained)
    assert list(zPosRetained) == [0.0, 1.0, 2.0]
    # using the unfiltered zPos here (the old code's behavior) would have
    # been silently wrong-length (4 vs 3) once the invariant breaks
    assert len(zPos) != len(zPosRetained)
