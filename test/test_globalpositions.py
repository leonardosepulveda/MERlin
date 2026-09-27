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
    frameWidth = 60
    overlapFraction = 0.5
    trueShiftPx = 3
    nOverlap = int(round(frameWidth * overlapFraction))
    nominalStartCol = frameWidth - nOverlap
    stepUm = nominalStartCol * pixelSizeUm
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

    measuredXY, error = globalpositions.register_neighbor_pair(
        anchorImg, neighborImg, anchorXY,
        dx=dx, dy=dy, overlap_fraction=overlapFraction, pixel_size_um=pixelSizeUm,
        upsample_factor=20)

    expected = (
        stepUm * dx + trueShiftPx * pixelSizeUm * dx,
        stepUm * dy + trueShiftPx * pixelSizeUm * dy,
    )
    assert measuredXY[0] == pytest.approx(expected[0], abs=0.02)
    assert measuredXY[1] == pytest.approx(expected[1], abs=0.02)


def _bead_world(shape, count, seed):
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    world = np.zeros(shape)
    world[rng.integers(0, shape[0], count), rng.integers(0, shape[1], count)] = 1000
    return gaussian_filter(world, 1.5) + rng.normal(100, 2, shape)


@pytest.mark.parametrize('trueOffset', [
    (360, 0),    # regular grid
    (360, 3),    # small stage error across the axis
    (364, 0),    # small stage error along the axis
    (360, 30),   # offset scan band (non-rectangular grid)
    (350, 0),    # locally shorter step
])
def test_register_neighbor_pair_recovers_true_offset(trueOffset):
    """The measured position must equal the true one whatever the pair's
    nominal spacing. Adding the shift to the nominal offset instead of the
    offset the crops assume counted any difference between the two twice
    (e.g. a 30 um band offset measured as 60 um)."""
    world = _bead_world((1200, 1600), 4000, 0)
    frameSize, overlapFraction = 400, 0.1
    anchorXY = (100, 300)
    neighborXY = (anchorXY[0] + trueOffset[0], anchorXY[1] + trueOffset[1])

    def frame(xy):
        return world[xy[1]:xy[1] + frameSize, xy[0]:xy[0] + frameSize]

    measuredXY, _ = globalpositions.register_neighbor_pair(
        frame(anchorXY), frame(neighborXY), anchorXY, dx=1.0, dy=0.0,
        overlap_fraction=overlapFraction, pixel_size_um=1.0,
        upsample_factor=10)

    assert measuredXY[0] == pytest.approx(neighborXY[0], abs=0.3)
    assert measuredXY[1] == pytest.approx(neighborXY[1], abs=0.3)


def test_filter_correspondence_outliers_per_direction():
    """Each direction has its own residual (backlash), so outliers are
    judged against their own direction's median residual. The old pooled
    |measured - nominal| test would reject every good '+y' edge (larger
    residual than the majority) and keep the bad '+x' one (smaller)."""
    rng = np.random.default_rng(0)

    def corr(direction, residual):
        nominal = (200.0, 0.0) if direction == '+x' else (0.0, 200.0)
        noise = rng.normal(0, 0.01, 2)
        return globalpositions.NeighborCorrespondence(
            0, 1, direction, nominal,
            (nominal[0] + residual[0] + noise[0], nominal[1] + residual[1] + noise[1]),
            0.01)

    goodX = [corr('+x', (0.1, -3.3)) for _ in range(20)]
    goodY = [corr('+y', (4.5, 0.0)) for _ in range(8)]
    bad = corr('+x', (-0.9, -1.8))
    kept, rejected = globalpositions.filter_correspondence_outliers(
        goodX + goodY + [bad], mad_threshold=5.0)
    assert rejected == [bad]
    assert len(kept) == len(goodX) + len(goodY)


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

    correction = globalpositions.fit_global_positions(correspondences, nominal)

    # abs=0.01, not tighter: the weak affine prior pulls each edge by
    # ~prior_weight x its deviation from the affine (~2e-3 um here).
    for fov, expected in truePos.items():
        got = correction.positions[fov]
        assert got[0] == pytest.approx(expected[0], abs=0.01)
        assert got[1] == pytest.approx(expected[1], abs=0.01)
    assert correction.residual_rms_um < 0.01
    assert correction.n_components == 1
    assert correction.n_fovs_solved == 4


def test_fit_global_positions_empty_input():
    # No correspondences: the affine is the identity, so every fov stays at
    # its nominal position.
    nominal = {0: (0.0, 0.0), 1: (200.0, 0.0)}
    correction = globalpositions.fit_global_positions([], nominal)
    assert correction.positions == nominal
    assert correction.n_components == 1
    assert correction.residual_rms_um == 0.0
    np.testing.assert_allclose(correction.affine, np.eye(2), atol=1e-9)


def test_compute_overlap_correlations_matches_at_correct_shift():
    """Correlation should be near-perfect once *positions* correctly
    accounts for the true relative shift between the two fovs, and
    strictly worse if *positions* is left at the (wrong) nominal offset
    instead -- mirrors `test_register_neighbor_pair_sign_convention`'s own
    synthetic-shift setup.
    """
    rng = np.random.default_rng(1)
    pixelSizeUm = 0.1
    frameWidth = 60
    overlapFraction = 0.5
    trueShiftPx = 3
    nOverlap = int(round(frameWidth * overlapFraction))
    stepUm = (frameWidth - nOverlap) * pixelSizeUm
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
        [correspondence], correctPositions, frames.__getitem__,
        pixel_size_um=pixelSizeUm, overlap_fraction=overlapFraction)
    wrongCorrelations = globalpositions.compute_overlap_correlations(
        [correspondence], nominal, frames.__getitem__,
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
        [correspondence], nominal, frames.__getitem__,
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
    # Each component's anchor (0 and 10) is pinned at its affine position,
    # with fov 0 (the origin) at its nominal position.
    affinePositions = globalpositions.apply_affine(correction.affine, nominal, 0)
    assert correction.positions[0] == pytest.approx(nominal[0], abs=1e-3)
    assert correction.positions[10] == pytest.approx(affinePositions[10], abs=1e-3)
    assert correction.positions[1][0] - correction.positions[0][0] == pytest.approx(102.0, abs=0.01)
    assert correction.positions[11][0] - correction.positions[10][0] == pytest.approx(99.0, abs=0.01)


def test_register_fov_against_neighbors_skips_non_overlapping_neighbor():
    # Two 100 um frames 500 um apart: the neighbour is found on the grid but
    # the frames share no content, so no registration should be reported.
    frames = {0: np.random.default_rng(2).random((100, 100)),
              1: np.random.default_rng(3).random((100, 100))}
    positions = {0: (0.0, 0.0), 1: (500.0, 0.0)}
    correspondences = globalpositions.register_fov_against_neighbors(
        0, positions, frames.__getitem__, pixel_size_um=1.0,
        overlap_fraction=0.1)
    assert correspondences == []


def test_fit_global_positions_rotated_scaled_grid_with_backlash():
    """End to end on a synthetic grid: rotated and scaled camera-vs-stage
    map, direction-dependent backlash, a few corrupted edges, and one fov
    whose every edge is corrupted.

    Fovs are numbered in serpentine acquisition order (up one column, down
    the next), and each fov is offset +-0.36 um in y by the direction the
    stage arrived from. 9 columns, so each row's 8 column boundaries split
    the backlash evenly (see `filter_correspondence_outliers`' limitation).

    Every other fov must be recovered to < 0.01 um, the affine must
    recover the rotation, and the fov with no kept edge must follow the
    affine rather than stay at its (~3 um off) nominal position."""
    rng = np.random.default_rng(0)
    nCols, nRows, step = 9, 8, 200.0
    thetaDeg = -0.95
    theta = np.radians(thetaDeg)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    trueAffine = rotation @ np.diag([1.0145, 1.0120])

    nominal, backlash = {}, {}
    for c in range(nCols):
        movingUp = c % 2 == 0
        for r in range(nRows):
            fov = c * nRows + (r if movingUp else nRows - 1 - r)
            nominal[fov] = (c * step, r * step)
            backlash[fov] = (0.0, 0.36 if movingUp else -0.36)
    fovs = sorted(nominal)
    truePos = {f: tuple(trueAffine @ np.array(nominal[f]) + backlash[f]) for f in fovs}

    isolatedFov = 3 * nRows + 4
    pairs = globalpositions.grid_neighbor_pairs(nominal)
    corruptedPairs = {pairs[k] for k in (5, 40, 77)} | {
        p for p in pairs if isolatedFov in p}
    directionOf = {(1, 0): '+x', (-1, 0): '-x', (0, 1): '+y', (0, -1): '-y'}

    correspondences = []
    for a, b in pairs:
        rel = np.subtract(truePos[b], truePos[a]) + rng.normal(0, 0.002, 2)
        if (a, b) in corruptedPairs:
            rel = rel + rng.choice([-1, 1], 2) * rng.uniform(8, 12, 2)
        # both ends, as RegisterFovNeighbors measures them
        for anchor, neighbor, sign in ((a, b, 1), (b, a, -1)):
            unit = tuple(int(v) for v in np.sign(np.subtract(nominal[neighbor], nominal[anchor])))
            correspondences.append(globalpositions.NeighborCorrespondence(
                anchor, neighbor, directionOf[unit], nominal[neighbor],
                tuple(np.add(nominal[anchor], sign * rel)), 0.01))

    kept, rejected = globalpositions.filter_correspondence_outliers(correspondences)
    rejectedPairs = {(min(c.anchor_fov, c.neighbor_fov), max(c.anchor_fov, c.neighbor_fov))
                     for c in rejected}
    assert rejectedPairs == corruptedPairs
    correction = globalpositions.fit_global_positions(kept, nominal)
    assert correction.n_components == 1

    # Positions are defined up to one translation: compare after removing it.
    others = [f for f in fovs if f != isolatedFov]
    diff = {f: np.subtract(correction.positions[f], truePos[f]) for f in fovs}
    translation = np.mean([diff[f] for f in others], axis=0)
    assert max(np.hypot(*(diff[f] - translation)) for f in others) < 0.01

    rotationDeg, _ = globalpositions.rotation_and_scale(correction.affine)
    assert rotationDeg == pytest.approx(thetaDeg, abs=0.05)

    # The fov with no kept edge follows the affine: off by at most its
    # backlash, not by the several-micron nominal error.
    isolatedError = np.hypot(*(diff[isolatedFov] - translation))
    nominalError = np.hypot(*np.subtract(
        np.subtract(nominal[isolatedFov], nominal[0]),
        np.subtract(truePos[isolatedFov], truePos[0])))
    assert isolatedError < 0.5
    assert nominalError > 5 * isolatedError


def test_cross_validate_positions_beats_nominal():
    """Held-out error on a rotated grid: the full solve is far closer to
    the held-out measurements than the nominal positions."""
    rng = np.random.default_rng(1)
    theta = np.radians(-0.3)
    trueAffine = 0.984 * np.array([[np.cos(theta), -np.sin(theta)],
                                   [np.sin(theta), np.cos(theta)]])
    nominal = {c * 6 + r: (c * 200.0, r * 200.0) for c in range(6) for r in range(6)}
    truePos = {f: trueAffine @ np.array(xy) + rng.normal(0, 0.3, 2)
               for f, xy in nominal.items()}
    correspondences = []
    for a, b in globalpositions.grid_neighbor_pairs(nominal):
        rel = truePos[b] - truePos[a] + rng.normal(0, 0.01, 2)
        correspondences.append(globalpositions.NeighborCorrespondence(
            a, b, '+x' if nominal[b][1] == nominal[a][1] else '+y', nominal[b],
            tuple(np.add(nominal[a], rel)), 0.01))

    errors = globalpositions.cross_validate_positions(correspondences, nominal)
    assert len(errors['final']) == len(correspondences)
    assert np.median(errors['final']) < 0.1 * np.median(errors['nominal'])
    assert np.median(errors['final']) < np.median(errors['affine_only'])


def test_remove_hot_pixels_keeps_beads():
    img = _bead_world((64, 64), 20, 3).astype(np.uint16)
    hot = img.copy()
    hot[10, 10] = 60000
    cleaned = globalpositions.remove_hot_pixels(hot)
    assert cleaned[10, 10] < 200
    # only the spike changed: beads (several-pixel PSFs) are untouched
    changed = cleaned != hot
    assert changed.sum() == 1
    np.testing.assert_array_equal(globalpositions.remove_hot_pixels(img), img)
