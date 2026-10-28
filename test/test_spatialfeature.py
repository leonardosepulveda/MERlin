import pytest
import numpy as np
import json
import networkx as nx
from shapely import geometry

from merlin.util import spatialfeature


testCoords1 = [(1, 1), (1, 2), (2, 2), (2, 1), (1, 1)]
testCoords2 = [(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)]

feature1 = spatialfeature.SpatialFeature([[geometry.Polygon(testCoords1)]], 0)
feature2 = spatialfeature.SpatialFeature([[geometry.Polygon(testCoords2)]], 0)
feature3 = spatialfeature.SpatialFeature([[geometry.Polygon(testCoords1)],
                                          [geometry.Polygon(testCoords1)]],
                                         0, zCoordinates=np.array([0, 0.5]))
feature4 = spatialfeature.SpatialFeature([[geometry.Polygon(testCoords1)],
                                          [geometry.Polygon(testCoords2)]],
                                         0, zCoordinates=np.array([0, 0.5]))

p1 = spatialfeature.SpatialFeature(
    [[geometry.Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])]], 0)
p2 = spatialfeature.SpatialFeature(
    [[geometry.Polygon([(0, 0.5), (0, 1), (1, 1), (1, 0.5)])]], 0)
p3 = spatialfeature.SpatialFeature(
    [[geometry.Polygon([(0, 0.5), (0, 1.5), (1, 1.5), (1, 0.5)])]], 0)
p4 = spatialfeature.SpatialFeature(
    [[geometry.Polygon([(0, 1), (0, 2), (1, 2), (1, 1)])]], 0)
p5 = spatialfeature.SpatialFeature(
    [[geometry.Polygon([(4, 4), (4, 5), (5, 5), (5, 4)])]], 0)
allCells = [p1, p2, p3, p4, p5]


def test_feature_from_label_matrix():
    testLabels = np.zeros((1, 4, 4))
    testLabels[0, 1:3, 1:3] = 1

    feature = spatialfeature.SpatialFeature.feature_from_label_matrix(
        testLabels, 0)

    assert len(feature.get_boundaries()[0]) == 1
    assert feature.get_boundaries()[0][0].equals(geometry.Polygon(
        list(zip([2.1, 2.1, 2.0, 1.0, 0.9, 0.9, 1.0, 2.0, 2.1],
                 [2.0, 1.0, 0.9, 0.9, 1.0, 2.0, 2.1, 2.1, 2.0]))))


def test_feature_from_label_matrix_transform():
    testLabels = np.zeros((1, 4, 4))
    testLabels[0, 1:3, 1:3] = 1
    transformMatrix = np.array([[2.0, 0, 3],
                                [0, 2.5, 1],
                                [0, 0, 1]])

    feature = spatialfeature.SpatialFeature.feature_from_label_matrix(
        testLabels, 0, transformMatrix)

    assert len(feature.get_boundaries()[0]) == 1
    assert feature.get_boundaries()[0][0].equals(
        geometry.Polygon(list(zip(
            [7.2, 7.2, 7.0, 5.0, 4.8, 4.8, 5.0, 7.0, 7.2],
            [6.0, 3.5, 3.25, 3.25, 3.5, 6.0, 6.25, 6.25, 6.0]))))


@pytest.mark.parametrize('feature, volume',
                         [(feature1, 1), (feature2, 4), (feature3, 0.5)])
def test_feature_get_volume(feature, volume):
    assert feature.get_volume() == volume


@pytest.mark.parametrize('feature', [feature1, feature2, feature3])
def test_feature_equals(feature):
    assert feature.equals(feature)


@pytest.mark.parametrize('feature', [feature1, feature2, feature3])
def test_feature_serialization_to_json(feature):
    featureIn = spatialfeature.SpatialFeature.from_json_dict(
        json.loads(json.dumps(feature.to_json_dict())))

    assert featureIn.equals(feature)


def test_feature_hdf5_db_read_write_delete_one_fov(
        single_task, simple_merfish_data):
    featureDB = spatialfeature.HDF5SpatialFeatureDB(
        simple_merfish_data, single_task)
    featureDB.write_features([feature1, feature2], fov=0)
    readFeatures = featureDB.read_features(fov=0)
    featureDB.empty_database(0)
    readFeatures2 = featureDB.read_features(fov=0)

    assert len(readFeatures) == 2
    if readFeatures[0].get_feature_id() == feature1.get_feature_id():
        f1Index = 0
        f2Index = 1
    else:
        f1Index = 1
        f2Index = 0
    assert readFeatures[f1Index].equals(feature1)
    assert readFeatures[f2Index].equals(feature2)

    assert len(readFeatures2) == 0


def test_feature_hdf5_db_read_write_delete_multiple_fov(
        single_task, simple_merfish_data):
    tempFeature2 = spatialfeature.SpatialFeature(
        [[geometry.Polygon(testCoords2)]], 1)
    featureDB = spatialfeature.HDF5SpatialFeatureDB(
        simple_merfish_data, single_task)
    featureDB.write_features([feature1, tempFeature2])
    readFeatures = featureDB.read_features()
    readFeatures0 = featureDB.read_features(0)
    readFeatures1 = featureDB.read_features(1)
    featureMetadata = featureDB.read_feature_metadata()
    featureMetadata0 = featureDB.read_feature_metadata(0)
    featureMetadata1 = featureDB.read_feature_metadata(1)
    featureDB.empty_database()
    readFeaturesEmpty = featureDB.read_features()
    metaDataEmpty = featureDB.read_feature_metadata()

    assert len(readFeatures0) == 1
    assert len(featureMetadata0) == 1
    assert readFeatures0[0].equals(feature1)
    assert int(featureMetadata0.index[0]) == feature1.get_feature_id()
    assert np.allclose(featureMetadata0.iloc[0][
                           ['fov', 'volume', 'center_x', 'center_y',
                            'min_x', 'min_y', 'max_x', 'max_y']].values,
                       np.array([0, 1, 1.5, 1.5, 1, 1, 2, 2]))

    assert len(readFeatures1) == 1
    assert len(featureMetadata1) == 1
    assert readFeatures1[0].equals(tempFeature2)
    assert int(featureMetadata1.index[0]) == tempFeature2.get_feature_id()
    assert np.allclose(featureMetadata1.iloc[0][
        ['fov', 'volume', 'center_x', 'center_y',
         'min_x', 'min_y', 'max_x', 'max_y']].values,
           np.array([1, 4, 1, 1, 0, 0, 2, 2]))

    assert len(readFeatures) == 2
    assert len(featureMetadata) == 2
    if readFeatures[0].get_feature_id() == feature1.get_feature_id():
        f1Index = 0
        f2Index = 1
    else:
        f1Index = 1
        f2Index = 0
    assert readFeatures[f1Index].equals(feature1)
    assert readFeatures[f2Index].equals(tempFeature2)

    assert len(readFeaturesEmpty) == 0
    assert len(metaDataEmpty) == 0


def test_feature_hdf5_db_read_boundaries_at_z_matches_read_features(
        single_task, simple_merfish_data):
    """read_feature_boundaries_at_z() (used by SegmentationBoundaryPlot to
    avoid loading every z-plane) must return exactly the same polygons as
    slicing the equivalent z out of a full read_features() call.
    """
    featureDB = spatialfeature.HDF5SpatialFeatureDB(
        simple_merfish_data, single_task)
    featureDB.write_features([feature3, feature4], fov=0)

    fullFeatures = featureDB.read_features(fov=0)
    zCount = featureDB.get_feature_z_count(fov=0)
    assert zCount == 2

    for z in range(zCount):
        boundariesAtZ = featureDB.read_feature_boundaries_at_z(z, fov=0)
        expected = [f.get_boundaries()[z] for f in fullFeatures]
        assert len(boundariesAtZ) == len(expected)
        for actualPolys, expectedPolys in zip(boundariesAtZ, expected):
            assert len(actualPolys) == len(expectedPolys)
            for actualPoly, expectedPoly in zip(actualPolys, expectedPolys):
                assert actualPoly.equals(expectedPoly)

    featureDB.empty_database(0)
    assert featureDB.get_feature_z_count(fov=0) == 0
    assert featureDB.read_feature_boundaries_at_z(0, fov=0) == []


def test_feature_hdf5_db_read_ids_and_boundaries_at_z_matches_read_features(
        single_task, simple_merfish_data):
    """read_feature_ids_and_boundaries_at_z() (used by SmfishSignal to join
    detected spots against segmentation one z-plane at a time, instead of
    loading the whole 3D cell set up front) must pair each returned
    feature id with the same polygons a full read_features() call would
    report at that z.
    """
    featureDB = spatialfeature.HDF5SpatialFeatureDB(
        simple_merfish_data, single_task)
    featureDB.write_features([feature3, feature4], fov=0)

    fullFeatures = featureDB.read_features(fov=0)
    idToFeature = {str(f.get_feature_id()): f for f in fullFeatures}
    zCount = featureDB.get_feature_z_count(fov=0)
    assert zCount == 2

    for z in range(zCount):
        ids, boundariesAtZ = \
            featureDB.read_feature_ids_and_boundaries_at_z(z, fov=0)
        assert len(ids) == len(boundariesAtZ) == len(idToFeature)
        assert set(ids) == set(idToFeature.keys())
        for featureId, actualPolys in zip(ids, boundariesAtZ):
            expectedPolys = idToFeature[featureId].get_boundaries()[z]
            assert len(actualPolys) == len(expectedPolys)
            for actualPoly, expectedPoly in zip(actualPolys, expectedPolys):
                assert actualPoly.equals(expectedPoly)

    featureDB.empty_database(0)
    emptyIds, emptyBoundaries = \
        featureDB.read_feature_ids_and_boundaries_at_z(0, fov=0)
    assert emptyIds == []
    assert emptyBoundaries == []


def test_feature_hdf5_db_read_boundaries_at_own_z_picks_own_occupied_z(
        single_task, simple_merfish_data):
    """read_feature_boundaries_at_own_z() (used by SegmentationBoundaryPlot
    to stream one fov at a time) must pick each feature's own middle
    *occupied* z-plane, not a single z-index shared across every feature --
    a feature whose occupied z-planes don't include the dataset-wide middle
    index (the one read_feature_boundaries_at_z() would use) must still be
    found and read at its own z. A feature with no occupied z-plane at all
    must be skipped rather than erroring.
    """
    featureOwnZ = spatialfeature.SpatialFeature(
        [[], [geometry.Polygon(testCoords1)], [],
         [geometry.Polygon(testCoords2)], [geometry.Polygon(testCoords2)]],
        0, zCoordinates=np.array([0, 0.25, 0.5, 0.75, 1.0]))
    featureNoOccupiedZ = spatialfeature.SpatialFeature([[], []], 1)

    featureDB = spatialfeature.HDF5SpatialFeatureDB(
        simple_merfish_data, single_task)
    featureDB.write_features([featureOwnZ, featureNoOccupiedZ], fov=0)

    result = featureDB.read_feature_boundaries_at_own_z(fov=0)

    # featureOwnZ's occupied z-planes are [1, 3, 4] (0-indexed) -- the
    # middle of that list is z=3, not z=2 (the dataset-wide middle of all
    # 5 z-planes, which is empty for this feature and would have hidden it
    # entirely under the old shared-z approach).
    assert len(result) == 1
    assert len(result[0]) == 1
    assert result[0][0].equals(geometry.Polygon(testCoords2))

    featureDB.empty_database(0)
    assert featureDB.read_feature_boundaries_at_own_z(fov=0) == []


def test_feature_contained_within_boundary():
    interiorLabels = np.zeros((1, 8, 8))
    interiorLabels[0, 2:6, 2:6] = 1
    interiorFeature = spatialfeature.SpatialFeature.feature_from_label_matrix(
        interiorLabels, 0)

    exteriorLabels = np.zeros((1, 8, 8))
    exteriorLabels[0, 1:7, 1:7] = 1
    exteriorFeature = spatialfeature.SpatialFeature.feature_from_label_matrix(
        exteriorLabels, 0)

    overlappingLabels = np.zeros((1, 8, 8))
    overlappingLabels[0, 0:5, 0:5] = 1
    overlappingFeature = spatialfeature.SpatialFeature\
        .feature_from_label_matrix(overlappingLabels, 0)

    assert interiorFeature.is_contained_within_boundary(exteriorFeature)
    assert not exteriorFeature.is_contained_within_boundary(interiorFeature)

    assert interiorFeature.is_contained_within_boundary(overlappingFeature)
    assert overlappingFeature.is_contained_within_boundary(interiorFeature)

    assert exteriorFeature.is_contained_within_boundary(overlappingFeature)
    assert overlappingFeature.is_contained_within_boundary(exteriorFeature)


def test_feature_contains_point():
    point1 = geometry.Point(-0.1, -0.1)
    point2 = geometry.Point(0.9, 0.9)
    point3 = geometry.Point(1.5, 1.5)

    assert not feature1.contains_point(point1, 0)
    assert not feature1.contains_point(point2, 0)
    assert feature1.contains_point(point3, 0)
    assert not feature4.contains_point(point1, 0)
    assert not feature4.contains_point(point2, 0)
    assert feature4.contains_point(point3, 0)
    assert not feature4.contains_point(point1, 1)
    assert feature4.contains_point(point2, 1)
    assert feature4.contains_point(point3, 1)


def test_feature_contains_positions():
    positions1 = np.array([[0, 0, 0], [1.5, 1.5, 0]])
    positions2 = np.array([[-0.1, -0.1, 0], [0.9, 0.9, 0], [1.5, 1.5, 0],
                           [-0.1, -0.1, 1], [0.9, 0.9, 1], [1.5, 1.5, 1]])
    assert all([a == b for a, b in zip(feature1.contains_positions(positions1),
                                       [False, True])])
    assert all([a == b for a, b in zip(feature4.contains_positions(positions1),
                                       [False, True])])
    assert all([a == b for a, b in zip(feature4.contains_positions(positions2),
                                       [False, False, True,
                                        False, True, True])])


def test_find_overlapping_cells():
    t1 = p1.get_overlapping_features(allCells)
    t2 = p2.get_overlapping_features(allCells)
    t3 = p3.get_overlapping_features(allCells)
    t4 = p4.get_overlapping_features(allCells)
    t5 = p5.get_overlapping_features(allCells)

    assert ((p1 in t1) and (p3 in t1)
            and (p2 not in t1) and (p5 not in t1) and (p5 not in t1))
    assert len(t2) == 0
    assert ((p3 in t3) and (p1 in t3) and (p4 in t3)
            and (p2 not in t3) and (p5 not in t3))
    assert ((p4 in t4) and (p3 in t4) and
            (p1 not in t4) and (p2 not in t4) and (p5 not in t4))
    assert ((p5 in t5) and (p1 not in t5)
            and (p2 not in t5) and (p3 not in t5) and (p4 not in t5))


def _box_cell(fov, x, y, zs, halfWidth=0.6):
    """A square cell centred on (x, y), present in the z planes zs of a
    0..5 stack with 1 um spacing."""
    square = geometry.box(x - halfWidth, y - halfWidth,
                          x + halfWidth, y + halfWidth)
    return spatialfeature.SpatialFeature(
        [[square] if z in zs else [] for z in range(6)], fov,
        zCoordinates=np.arange(6, dtype=float))


def _two_fov_seam():
    """Fov 0's image spans x 0..10 and fov 1's x 8..18. Three cells in the
    band are seen by both, and fov 1 sees them 2 planes higher. A fourth
    cell of fov 0 grazes the first duplicate by 0.05 um."""
    extents = {0: (0, 0, 10, 10), 1: (8, 0, 18, 10)}
    copies0 = [_box_cell(0, x, y, {1, 2, 3})
               for x, y in ((8.75, 3), (8.75, 5), (9.4, 7))]
    copies1 = [_box_cell(1, x, y, {3, 4, 5})
               for x, y in ((8.75, 3), (8.75, 5), (9.4, 7))]
    graze = spatialfeature.SpatialFeature(
        [[geometry.box(5, 2.4, 8.2, 3.6)] if z in {1, 2, 3} else []
         for z in range(6)], 0, zCoordinates=np.arange(6, dtype=float))
    return extents, {0: copies0 + [graze], 1: copies1}, copies0, copies1, \
        graze


def _seam_graph():
    extents, cells, copies0, copies1, graze = _two_fov_seam()
    graph = nx.Graph()
    seams = []
    for fov in (0, 1):
        g, s = spatialfeature.construct_overlap_graph(
            fov, cells, extents, zStep=1.0, minSeamPairs=2)
        graph.update(g)
        seams.append(s)
    return graph, seams, copies0, copies1, graze


def test_plane_overlaps_with_z_shift():
    a = _box_cell(0, 0, 0, {1, 2, 3}, halfWidth=0.5)
    b = _box_cell(1, 0, 0, {3, 4, 5}, halfWidth=0.5)
    planes = [spatialfeature._cell_planes(c, 1.0) for c in (a, b)]
    pairs = np.array([[0, 1], [0, 1]])
    overlaps = spatialfeature._plane_overlaps(planes, pairs,
                                              np.array([0, 2]))
    assert overlaps == pytest.approx([1.0, 3.0])


def test_construct_overlap_graph_links_duplicates_after_z_shift():
    graph, seams, copies0, copies1, graze = _seam_graph()
    for a, b in zip(copies0, copies1):
        assert graph.has_edge(a.get_feature_id(), b.get_feature_id())
    # a 0.05 um sliver is not a conflict
    assert graph.degree(graze.get_feature_id()) == 0
    assert graph.number_of_edges() == 3

    assert seams[0].loc[0, 'neighbor_fov'] == 1
    assert seams[0].loc[0, 'n_pairs'] == 3
    assert seams[0].loc[0, 'z_offset_um'] == pytest.approx(2.0)
    assert seams[1].loc[0, 'z_offset_um'] == pytest.approx(-2.0)

    node = graph.nodes[copies0[0].get_feature_id()]
    assert node['originalFOV'] == 0
    assert node['edgeDistance'] == pytest.approx(1.25)
    assert graph.nodes[copies1[0].get_feature_id()]['edgeDistance'] == \
        pytest.approx(0.75)


def test_construct_overlap_graph_without_z_shift_misses_duplicates():
    extents, cells, _, _, _ = _two_fov_seam()
    graph, seams = spatialfeature.construct_overlap_graph(
        0, cells, extents, zStep=1.0, minSeamPairs=4)
    # 3 duplicates are too few to measure the seam, so it is not shifted
    # and the copies share only 1 of their 3 planes
    assert np.isnan(seams.loc[0, 'z_offset_um'])
    assert graph.number_of_edges() == 0


def test_construct_overlap_graph_adaptive_z_window_finds_shift_past_edge():
    extents, cells, _, _, _ = _two_fov_seam()
    # the copies are 2 planes apart; a window of +-1 plane finds its best
    # overlap on the edge (+1), and the adaptive window moves on to +2
    graph, seams = spatialfeature.construct_overlap_graph(
        0, cells, extents, zStep=1.0, minSeamPairs=2, maxZShift=1,
        adaptiveZWindow=True)
    assert seams.loc[0, 'z_offset_um'] == pytest.approx(2.0)
    assert graph.number_of_edges() == 3
    graph, seams = spatialfeature.construct_overlap_graph(
        1, cells, extents, zStep=1.0, minSeamPairs=2, maxZShift=1,
        adaptiveZWindow=True)
    assert seams.loc[0, 'z_offset_um'] == pytest.approx(-2.0)


def test_construct_overlap_graph_fixed_window_misses_shift_past_edge():
    extents, cells, _, _, _ = _two_fov_seam()
    # a fixed window of 0 planes sees 1 of 3 shared planes: no duplicates
    graph, seams = spatialfeature.construct_overlap_graph(
        0, cells, extents, zStep=1.0, minSeamPairs=2, maxZShift=0)
    assert np.isnan(seams.loc[0, 'z_offset_um'])


def test_remove_overlapping_cells_keeps_copy_away_from_edge():
    graph, _, copies0, copies1, graze = _seam_graph()
    kept = set(spatialfeature.remove_overlapping_cells(graph)['cell_id'])
    assert kept == {copies0[0].get_feature_id(), copies0[1].get_feature_id(),
                    copies1[2].get_feature_id(), graze.get_feature_id()}


def test_remove_overlapping_cells_rejects_old_graphs():
    graph = nx.Graph()
    graph.add_node(1, originalFOV=0, assignedFOV=0)
    with pytest.raises(ValueError, match='rerun'):
        spatialfeature.remove_overlapping_cells(graph)


def test_solve_fov_z_offsets():
    _, seams, _, _, _ = _seam_graph()
    import pandas
    seams = pandas.concat(seams + [pandas.DataFrame(
        {'fov': [1], 'neighbor_fov': [2], 'n_pairs': [5],
         'z_offset_um': [0.5]})], ignore_index=True)
    offsets = spatialfeature.solve_fov_z_offsets(seams, [0, 1, 2, 3])
    offsets = offsets.set_index('fov')['z_offset_um']
    # offset[fov] - offset[neighbor] = z_offset_um, mean 0 per group
    assert offsets[0] - offsets[1] == pytest.approx(2.0)
    assert offsets[1] - offsets[2] == pytest.approx(0.5)
    assert offsets[[0, 1, 2]].sum() == pytest.approx(0.0, abs=1e-9)
    assert offsets[3] == 0


def test_solve_fov_z_offsets_fills_unmeasured_seams_from_a_plane():
    import pandas
    # three fovs in a row; only the first seam has enough duplicates
    seams = pandas.DataFrame({'fov': [0, 1, 1], 'neighbor_fov': [1, 2, 0],
                              'n_pairs': [100, 3, 100],
                              'z_offset_um': [1.5, np.nan, -1.5]})
    centres = {0: (0, 0), 1: (10, 0), 2: (20, 0)}
    offsets = spatialfeature.solve_fov_z_offsets(
        seams, [0, 1, 2], centres).set_index('fov')['z_offset_um']
    assert offsets[0] - offsets[1] == pytest.approx(1.5)
    assert offsets[1] - offsets[2] == pytest.approx(1.5)
    # without centres fov 2 is on its own
    offsets = spatialfeature.solve_fov_z_offsets(
        seams, [0, 1, 2]).set_index('fov')['z_offset_um']
    assert offsets[2] == 0


def _synthetic_label_stack():
    labels = np.zeros((4, 60, 70), dtype=np.uint16)
    labels[0:3, 10:25, 12:30] = 1   # a box spanning three planes
    labels[1, 15:20, 18:24] = 0     # with a hole in the middle plane
    labels[1:4, 0:8, 40:52] = 2     # touching the top edge
    labels[2, 50:60, 60:70] = 3     # touching the bottom-right corner
    labels[3, 30:36, 5:11] = 4      # split into two pieces in one plane
    labels[3, 30:36, 20:26] = 4
    rr, cc = np.ogrid[:60, :70]
    labels[0][(rr - 40) ** 2 + (cc - 50) ** 2 < 49] = 5   # a disk
    return labels


@pytest.mark.parametrize('processes', [1, 2])
@pytest.mark.parametrize('transformationMatrix', [
    None, np.array([[0.1, 0, 25.0], [0, 0.1, -3.0], [0, 0, 1]])])
def test_features_from_label_matrix_stack_crop_matches_full_frame(
        processes, transformationMatrix):
    labels = _synthetic_label_stack()
    zCoordinates = np.array([0, 1.5, 3, 4.5])
    # 6 is absent from the stack, so it has no bounding box
    maskValues = np.array([1, 2, 3, 4, 5, 6])

    cropped = spatialfeature.SpatialFeature.features_from_label_matrix_stack(
        labels, maskValues, 0, transformationMatrix, zCoordinates,
        processes=processes)

    for value, feature in zip(maskValues, cropped):
        expected = spatialfeature.SpatialFeature.feature_from_label_matrix(
            labels == value, 0, transformationMatrix, zCoordinates)
        assert np.array_equal(feature.get_z_coordinates(),
                              expected.get_z_coordinates())
        for croppedPlane, expectedPlane in zip(
                feature.get_boundaries(), expected.get_boundaries()):
            assert len(croppedPlane) == len(expectedPlane)
            for p, q in zip(croppedPlane, expectedPlane):
                assert np.allclose(np.array(p.exterior.coords),
                                   np.array(q.exterior.coords))
                assert p.symmetric_difference(q).area < 1e-9
    assert all(len(b) == 0 for b in cropped[-1].get_boundaries())


def _tilted_grid_seams(tilt=(0.011, 0.004), outlier=True):
    """Seams of a 4x4 fov grid, 182 um apart, under a focal-plane tilt."""
    import pandas
    centres = {4 * r + c: (182.0 * c, 182.0 * r)
               for r in range(4) for c in range(4)}
    rows = []
    for f, (x, y) in centres.items():
        for g, (u, v) in centres.items():
            if f != g and max(abs(u - x), abs(v - y)) < 200:
                rows.append({'fov': f, 'neighbor_fov': g, 'n_pairs': 100,
                             'z_offset_um': tilt[0] * (u - x)
                             + tilt[1] * (v - y)})
    seams = pandas.DataFrame(rows)
    if outlier:
        seams.loc[0, 'z_offset_um'] += 10
    return seams, centres


def test_fit_focal_tilt_recovers_the_tilt_and_drops_outliers():
    seams, centres = _tilted_grid_seams()
    fit = spatialfeature.fit_focal_tilt(seams, centres)
    np.testing.assert_allclose(fit['tilt'], (0.011, 0.004), atol=1e-9)
    assert fit['n_seams'] == len(seams) - 1
    assert fit['residual_sd_um'] == pytest.approx(0, abs=1e-9)


def test_tissue_depth_agrees_across_a_seam_and_has_no_ramp():
    from unittest import mock
    import pandas
    from merlin.analysis import segment
    seams, centres = _tilted_grid_seams(outlier=False)
    fovs = list(centres)
    offsets = spatialfeature.solve_fov_z_offsets(seams, fovs, centres)
    fit = spatialfeature.fit_focal_tilt(seams, centres)
    c = np.array([centres[f] for f in fovs])
    task = segment.CombineCleanedBoundaries.__new__(
        segment.CombineCleanedBoundaries)
    task.get_fov_z_offsets = mock.Mock(
        return_value=offsets.set_index('fov')['z_offset_um'])
    task.get_focal_tilt = mock.Mock(return_value=pandas.Series({
        'gx_um_per_mm': fit['tilt'][0] * 1000,
        'gy_um_per_mm': fit['tilt'][1] * 1000,
        'depth_constant_um': np.mean(
            offsets.z_offset_um.values + c @ fit['tilt'])}))
    # the point midway between fov 0 and fov 1 is seen 2.0 um higher in
    # fov 1 (0.011 * 182); both copies must get one depth
    zA, zB = 10.0, 10.0 + 0.011 * 182
    depth = task.get_tissue_depth([0, 1], [91, 91], [0, 0], [zA, zB])
    assert depth[0] == pytest.approx(depth[1])
    # each fov centre at plane z is at depth z: no section-wide ramp
    centreDepth = task.get_tissue_depth(fovs, c[:, 0], c[:, 1],
                                        np.full(len(fovs), 5.0))
    np.testing.assert_allclose(centreDepth, 5.0, atol=1e-9)


def test_feature_hdf5_db_z_centres_use_only_filled_planes(
        single_task, simple_merfish_data):
    feature = spatialfeature.SpatialFeature(
        [[], [geometry.Polygon(testCoords1)], [],
         [geometry.Polygon(testCoords2)], [geometry.Polygon(testCoords2)]],
        0, zCoordinates=np.array([0, 0.25, 0.5, 0.75, 1.0]))
    empty = spatialfeature.SpatialFeature([[], []], 0,
                                          zCoordinates=np.array([0, 1.0]))
    featureDB = spatialfeature.HDF5SpatialFeatureDB(
        simple_merfish_data, single_task)
    featureDB.write_features([feature, empty], fov=0)

    centres = featureDB.read_feature_z_centres(fov=0)
    metadata = featureDB.read_feature_metadata(fov=0)
    featureDB.empty_database(0)

    assert sorted(centres.index) == sorted(metadata.index)
    assert centres[str(feature.get_feature_id())] == pytest.approx(2 / 3)
    assert np.isnan(centres[str(empty.get_feature_id())])


def _flat_depth_task(zPositions):
    """A CombineCleanedBoundaries with no tilt and no fov offsets, so a
    point's depth is its local z."""
    from unittest import mock
    import pandas
    from merlin.analysis import segment
    task = segment.CombineCleanedBoundaries.__new__(
        segment.CombineCleanedBoundaries)
    task.dataSet = mock.Mock()
    task.dataSet.get_z_positions.return_value = zPositions
    task.get_fov_z_offsets = mock.Mock(
        return_value=pandas.Series({0: 0.0, 1: 0.0}))
    task.get_focal_tilt = mock.Mock(return_value=pandas.Series({
        'gx_um_per_mm': 0.0, 'gy_um_per_mm': 0.0, 'depth_constant_um': 0.0}))
    return task


def test_barcode_depths_turn_plane_indexes_into_z_positions():
    import pandas
    task = _flat_depth_task([0.5, 1.0, 1.5, 2.0])
    barcodes = pandas.DataFrame({'fov': [0, 1, 1], 'global_x': [0.0] * 3,
                                 'global_y': [0.0] * 3,
                                 'z': [0.0, 2.0, 2.5]})
    np.testing.assert_allclose(task.get_barcode_depths(barcodes),
                               [0.5, 1.5, 1.75])


def test_cell_depths_are_indexed_by_cell_id():
    from unittest import mock
    import pandas
    task = _flat_depth_task([0.5, 1.0])
    metadata = pandas.DataFrame({'fov': [0, 1], 'center_x': [0.0, 0.0],
                                 'center_y': [0.0, 0.0]}, index=['7', '3'])
    featureDB = mock.Mock()
    featureDB.read_feature_z_centres.return_value = pandas.Series(
        {'3': 2.0, '7': 4.0})
    depths = task.get_cell_depths(featureDB, metadata)
    assert depths.to_dict() == {'7': 4.0, '3': 2.0}
