"""Tests for generatemosaic.py's core new placement geometry: _fov_placement,
the pixel-accurate overlap trim behind GenerateMosaicTile/CombineMosaicTiles
(see generatemosaic.py's module/class docstrings). These are tested directly
against a fake, minimal GlobalAlignment rather than through a full dataSet
fixture and real images, since placement is pure coordinate math independent
of any pixel content.
"""
import numpy as np
import pytest

from merlin.analysis import generatemosaic


def _mosaic_tile_parameters(**overrides):
    parameters = {
        'global_align_task': 'globalAlign', 'warp_task': 'warp',
        'preprocess_task': 'preprocess', 'ffc_task': 'ffc', 'downsample': 2,
        'z_index': 0, 'data_channels': ['bit1']}
    parameters.update(overrides)
    return parameters


def test_downsample_single_int_is_normalized_to_a_list(simple_merfish_data):
    task = generatemosaic.GenerateMosaicTile(
        simple_merfish_data, parameters=_mosaic_tile_parameters(downsample=2))
    assert task.parameters['downsample'] == [2]


def test_downsample_accepts_multiple_factors_including_one(simple_merfish_data):
    task = generatemosaic.GenerateMosaicTile(
        simple_merfish_data,
        parameters=_mosaic_tile_parameters(downsample=[1, 2]))
    assert task.parameters['downsample'] == [1, 2]

    width, height = simple_merfish_data.get_image_dimensions()
    assert task.get_tile_shape(1) == (height, width)
    assert task.get_tile_shape(2) == (height // 2, width // 2)


def test_downsample_must_evenly_divide_image_dimensions(simple_merfish_data):
    # the fixture's images are 128x128 -- 3 does not evenly divide 128
    with pytest.raises(ValueError):
        generatemosaic.GenerateMosaicTile(
            simple_merfish_data,
            parameters=_mosaic_tile_parameters(downsample=3))


def test_use_ffc_defaults_to_true(simple_merfish_data):
    task = generatemosaic.GenerateMosaicTile(
        simple_merfish_data, parameters=_mosaic_tile_parameters())
    assert task.parameters['use_ffc'] is True
    assert 'ffc' in task.get_dependencies()


class _FakeAlignTask:
    """A minimal GlobalAlignment stand-in: places each fov's pixel (0, 0) at
    a given global (x, y) micron offset, with no rotation/scale -- the same
    convention SimpleGlobalAlignment/LeastSquaresGlobalAlignment use.
    """

    def __init__(self, origins):
        self.origins = origins

    def fov_coordinates_to_global(self, fov, coordinates):
        x, y = self.origins[fov]
        return (x + coordinates[0], y + coordinates[1])

    def fov_to_global_transform(self, fov):
        x, y = self.origins[fov]
        return np.float32([[1, 0, x], [0, 1, y], [0, 0, 1]])


def test_fov_placement_2x2_grid_partitions_exactly():
    # a 2x2 grid of 100x100 tiles on a 90-pixel step (10px overlap on every
    # shared edge) -- every fov's trimmed region should exactly tile the
    # canvas, with adjoining fovs agreeing to the pixel on their shared
    # boundary (no gap, no overlap), and no trim at all on an outer edge
    # that has no neighbour.
    origins = {0: (0, 0), 1: (90, 0), 2: (0, 90), 3: (90, 90)}
    alignTask = _FakeAlignTask(origins)
    identity = np.eye(3, dtype=np.float32)
    tileShape = (100, 100)

    placements = {
        fov: generatemosaic._fov_placement(
            alignTask, fov, origins, identity, tileShape, 0.25)
        for fov in origins}

    # fov 0 (top-left corner): untrimmed on its outer top/left edges,
    # trimmed to the midpoint on its inner right/bottom edges
    x0, y0, top, bottom, left, right = placements[0]
    assert (x0, y0) == (0, 0)
    assert (top, left) == (0, 0)
    assert (bottom, right) == (95, 95)

    # every fov's absolute kept region, from its origin + its own crop
    def absolute_region(fov):
        x0, y0, top, bottom, left, right = placements[fov]
        return (x0 + left, y0 + top, x0 + right, y0 + bottom)

    regions = {fov: absolute_region(fov) for fov in origins}
    assert regions[0] == (0, 0, 95, 95)
    assert regions[1] == (95, 0, 190, 95)
    assert regions[2] == (0, 95, 95, 190)
    assert regions[3] == (95, 95, 190, 190)

    # adjoining fovs share their boundary exactly -- fov 0's right edge is
    # fov 1's left edge, fov 0's bottom edge is fov 2's top edge, etc.
    assert regions[0][2] == regions[1][0]  # fov0 right == fov1 left
    assert regions[0][3] == regions[2][1]  # fov0 bottom == fov2 top
    assert regions[2][2] == regions[3][0]  # fov2 right == fov3 left
    assert regions[1][3] == regions[3][1]  # fov1 bottom == fov3 top

    # every fov's kept region is exactly its own tileShape footprint at its
    # own origin, minus only the side(s) that border a real neighbour --
    # together the four regions tile the full 190x190 canvas with no gaps
    # or double coverage.
    canvas = np.zeros((190, 190), dtype=np.uint8)
    for fov in origins:
        x0, y0, x1, y1 = regions[fov]
        canvas[y0:y1, x0:x1] += 1
    assert np.all(canvas == 1)


def test_fov_placement_isolated_fov_has_no_true_neighbor():
    # two fovs much farther apart than the tile itself is wide -- they are
    # still one another's nearest (only) neighbour, but the computed trim
    # midpoint falls outside the tile entirely, so no pixels are actually
    # cropped (there's no real overlap to remove).
    origins = {0: (0, 0), 1: (0, 2000)}
    alignTask = _FakeAlignTask(origins)
    identity = np.eye(3, dtype=np.float32)
    tileShape = (100, 100)

    x0, y0, top, bottom, left, right = generatemosaic._fov_placement(
        alignTask, 0, origins, identity, tileShape, 0.25)
    assert (top, bottom, left, right) == (0, 100, 0, 100)
