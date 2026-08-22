import numpy as np
import pytest

from merlin.util import globalpositions


def test_find_grid_neighbor():
    positions = {
        0: (0.0, 0.0), 1: (100.0, 0.0), 2: (-100.0, 0.0),
        3: (0.0, 100.0), 4: (0.0, -100.0), 5: (300.0, 300.0),
    }
    assert globalpositions.find_grid_neighbor(0, positions, 1.0, 0.0, 0.25) == 1
    assert globalpositions.find_grid_neighbor(0, positions, -1.0, 0.0, 0.25) == 2
    assert globalpositions.find_grid_neighbor(0, positions, 0.0, 1.0, 0.25) == 3
    assert globalpositions.find_grid_neighbor(0, positions, 0.0, -1.0, 0.25) == 4
    # fov 5 is isolated (far from any grid step away) -- no match expected
    assert globalpositions.find_grid_neighbor(5, positions, 1.0, 0.0, 0.25) is None


def test_find_grid_neighbor_on_phase_shifted_bands():
    """Non-rectangular grid: two scan bands (columns) independently
    phase-shifted along y, as in MERci's irregular-grid layout. A single
    dataset-wide step + exact-offset match (the old algorithm) misses the
    true cross-band neighbour here: column A's own within-band step is 100,
    so the +x target from (0, 0) sits at (100, 0) -- 50um away from column
    B's actual closest fov at (100, 50), outside a 0.25 * 100 = 25um
    tolerance. The dominant-axis/local-step algorithm finds it anyway,
    because it never needs the two bands to share a common step or phase.
    """
    positions = {
        0: (0.0, 0.0), 1: (0.0, 100.0), 2: (0.0, 200.0), 3: (0.0, 300.0),
        10: (100.0, 50.0), 11: (100.0, 150.0), 12: (100.0, 250.0),
    }
    assert globalpositions.find_grid_neighbor(0, positions, 1.0, 0.0, 0.25) == 10
    assert globalpositions.find_grid_neighbor(1, positions, 1.0, 0.0, 0.25) == 10
    assert globalpositions.find_grid_neighbor(2, positions, 1.0, 0.0, 0.25) == 11
    assert globalpositions.find_grid_neighbor(10, positions, -1.0, 0.0, 0.25) == 0


def test_estimate_step_size_um():
    positions = {0: (0.0, 0.0), 1: (200.0, 0.0), 2: (0.0, 200.0), 3: (200.0, 200.0)}
    assert globalpositions.estimate_step_size_um(positions) == pytest.approx(200.0)


@pytest.mark.parametrize('dx,dy', [(1.0, 0.0), (0.0, 1.0)])
def test_register_neighbor_pair_sign_convention(dx, dy):
    """The exact class of bug this test guards against: a backwards
    anchor/neighbour crop selection would recover a shift with the WRONG
    SIGN, silently corrupting every correction in that axis (this is the
    real failure the sibling MERci project's own camera_rotation.py module
    documents having hit once for its own row-crop convention -- see
    globalpositions.py's own crop_overlap docstring).

    A known, injected sub-pixel-free shift is planted into a synthetic
    neighbour crop and must be recovered with the correct sign, not its
    negation.
    """
    rng = np.random.default_rng(0)
    pixelSizeUm = 0.1
    stepUm = 60.0
    frameWidth = 60
    overlapFraction = 0.5
    trueShiftPx = 3
    nOverlap = int(round(frameWidth * overlapFraction))
    nominalStartCol = frameWidth - nOverlap
    neighborStartCol = nominalStartCol + trueShiftPx

    if dx != 0:
        world = rng.random((frameWidth, frameWidth + frameWidth))
        anchorImg = world[:, :frameWidth]
        neighborImg = world[:, neighborStartCol:neighborStartCol + frameWidth]
    else:
        world = rng.random((frameWidth + frameWidth, frameWidth))
        anchorImg = world[:frameWidth, :]
        neighborImg = world[neighborStartCol:neighborStartCol + frameWidth, :]

    anchorXY = (0.0, 0.0)
    neighborXYNominal = (stepUm * dx, stepUm * dy)

    measuredXY, error = globalpositions.register_neighbor_pair(
        anchorImg, neighborImg, anchorXY, neighborXYNominal,
        dx=dx, dy=dy, overlap_fraction=overlapFraction, pixel_size_um=pixelSizeUm,
        upsample_factor=20)

    expected = (
        stepUm * dx + trueShiftPx * pixelSizeUm * dx,
        stepUm * dy + trueShiftPx * pixelSizeUm * dy,
    )
    assert measuredXY[0] == pytest.approx(expected[0], abs=0.02)
    assert measuredXY[1] == pytest.approx(expected[1], abs=0.02)


def test_filter_correspondence_outliers():
    good = [
        globalpositions.NeighborCorrespondence(0, 1, '+x', (10.0, 0.0), (10.1, 0.05), 0.01)
        for _ in range(8)
    ]
    bad = globalpositions.NeighborCorrespondence(0, 2, '+y', (0.0, 10.0), (0.0, 25.0), 0.9)
    kept, rejected = globalpositions.filter_correspondence_outliers(good + [bad], mad_threshold=5.0)
    assert rejected == [bad]
    assert len(kept) == len(good)


def test_filter_correspondence_outliers_too_few_to_filter():
    corr = [globalpositions.NeighborCorrespondence(0, 1, '+x', (0.0, 0.0), (5.0, 0.0), 0.1)]
    kept, rejected = globalpositions.filter_correspondence_outliers(corr)
    assert kept == corr
    assert rejected == []


def test_fit_global_positions_recovers_known_offsets():
    nominal = {0: (0.0, 0.0), 1: (100.0, 0.0), 2: (200.0, 0.0), 3: (0.0, 100.0)}
    # fov 1 and (transitively) fov 2 are really 3um further +x than nominal;
    # fov 3 doesn't move. fov 0 is the fixed reference.
    truePos = {0: (0.0, 0.0), 1: (103.0, 0.0), 2: (203.0, 0.0), 3: (0.0, 100.0)}

    def measured_from(anchor, neighbor, direction):
        # Mirrors register_neighbor_pair's own contract: measured_xy is the
        # anchor's NOMINAL position plus the real relative offset between
        # the two fovs' TRUE positions -- a real registration never knows
        # the anchor's own true position, only its nominal one.
        rel = (truePos[neighbor][0] - truePos[anchor][0],
               truePos[neighbor][1] - truePos[anchor][1])
        return globalpositions.NeighborCorrespondence(
            anchor, neighbor, direction, nominal[neighbor],
            (nominal[anchor][0] + rel[0], nominal[anchor][1] + rel[1]), 0.01)

    correspondences = [
        measured_from(0, 1, '+x'),
        measured_from(1, 2, '+x'),
        measured_from(0, 3, '+y'),
        measured_from(2, 1, '-x'),  # redundant 2nd measurement of fov 1, via fov 2
    ]

    correction = globalpositions.fit_global_positions(
        correspondences, nominal, lsqr_atol=1e-12, lsqr_btol=1e-12)

    for fov, expected in truePos.items():
        got = correction.positions[fov]
        assert got[0] == pytest.approx(expected[0], abs=1e-3)
        assert got[1] == pytest.approx(expected[1], abs=1e-3)
    assert correction.residual_rms_um < 1e-3
    assert correction.n_components == 1
    assert correction.n_fovs_solved == 4


def test_fit_global_positions_empty_input():
    correction = globalpositions.fit_global_positions([], {0: (0.0, 0.0)})
    assert correction.positions == {}
    assert correction.n_components == 0
    assert correction.residual_rms_um == 0.0


def test_compute_overlap_correlations_matches_at_correct_shift():
    """Correlation should be near-perfect once *positions* correctly
    accounts for the true relative shift between the two fovs, and
    strictly worse if *positions* is left at the (wrong) nominal offset
    instead -- mirrors `test_register_neighbor_pair_sign_convention`'s own
    synthetic-shift setup.
    """
    rng = np.random.default_rng(1)
    pixelSizeUm = 0.1
    stepUm = 6.0
    frameWidth = 60
    overlapFraction = 0.5
    trueShiftPx = 3
    nOverlap = int(round(frameWidth * overlapFraction))
    neighborStartCol = frameWidth - nOverlap + trueShiftPx

    world = rng.random((frameWidth, frameWidth + frameWidth))
    frames = {0: world[:, :frameWidth],
             1: world[:, neighborStartCol:neighborStartCol + frameWidth]}

    nominal = {0: (0.0, 0.0), 1: (stepUm, 0.0)}
    trueShiftUm = trueShiftPx * pixelSizeUm
    correctPositions = {0: (0.0, 0.0), 1: (stepUm + trueShiftUm, 0.0)}
    correspondence = globalpositions.NeighborCorrespondence(
        0, 1, '+x', nominal[1], (nominal[1][0] + trueShiftUm, 0.0), 0.01)

    correctCorrelations = globalpositions.compute_overlap_correlations(
        [correspondence], correctPositions, nominal, frames.__getitem__,
        pixel_size_um=pixelSizeUm, overlap_fraction=overlapFraction)
    wrongCorrelations = globalpositions.compute_overlap_correlations(
        [correspondence], nominal, nominal, frames.__getitem__,
        pixel_size_um=pixelSizeUm, overlap_fraction=overlapFraction)

    # Not exactly 1.0 even at the correct shift: the shift-compensated crop's
    # trailing edge (trueShiftPx of its nOverlap columns) has no real data to
    # interpolate from (`ndi_shift`'s `mode='nearest'` pads it instead) -- a
    # real, expected boundary effect, not a bug. The uncorrected (wrong)
    # position leaves two independent random crops, which correlate at ~0.
    assert correctCorrelations[(0, 1, '+x')] > 0.85
    assert wrongCorrelations[(0, 1, '+x')] == pytest.approx(0.0, abs=0.1)
    assert wrongCorrelations[(0, 1, '+x')] < correctCorrelations[(0, 1, '+x')]


def test_compute_overlap_correlations_degenerate_crop_returns_zero_not_nan():
    frames = {0: np.zeros((10, 10)), 1: np.zeros((10, 10))}
    nominal = {0: (0.0, 0.0), 1: (5.0, 0.0)}
    correspondence = globalpositions.NeighborCorrespondence(
        0, 1, '+x', nominal[1], nominal[1], 0.0)

    correlations = globalpositions.compute_overlap_correlations(
        [correspondence], nominal, nominal, frames.__getitem__,
        pixel_size_um=1.0, overlap_fraction=0.5)

    assert correlations[(0, 1, '+x')] == 0.0


def test_fit_global_positions_disconnected_components_solved_independently():
    nominal = {0: (0.0, 0.0), 1: (100.0, 0.0), 10: (500.0, 500.0), 11: (600.0, 500.0)}
    correspondences = [
        globalpositions.NeighborCorrespondence(0, 1, '+x', nominal[1], (102.0, 0.0), 0.01),
        globalpositions.NeighborCorrespondence(10, 11, '+x', nominal[11], (599.0, 500.0), 0.01),
    ]
    correction = globalpositions.fit_global_positions(correspondences, nominal)
    assert correction.n_components == 2
    # each component's own anchor (0 and 10) stays pinned at its nominal position
    assert correction.positions[0] == pytest.approx(nominal[0], abs=1e-3)
    assert correction.positions[10] == pytest.approx(nominal[10], abs=1e-3)
    assert correction.positions[1][0] == pytest.approx(102.0, abs=1e-3)
    assert correction.positions[11][0] == pytest.approx(599.0, abs=1e-3)
