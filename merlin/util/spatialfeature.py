from abc import abstractmethod
import multiprocessing
import numpy as np
import uuid
import cv2
from skimage import measure
from typing import List
from typing import Tuple
from typing import Dict
import shapely
from shapely import geometry
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
import h5py
import merlin
import pandas
import networkx as nx
from scipy import ndimage


from merlin.core import dataset
from merlin.core import analysistask


# Module-level worker state for SpatialFeature.features_from_label_matrix_stack.
# Set once per worker process by _init_feature_worker (passed through the
# multiprocessing.Pool initializer, not pickled per task) so each worker
# holds its own copy of the label matrix stack instead of it being
# re-serialized for every mask value.
_workerLabelMatrixStack = None
_workerFov = None
_workerTransformationMatrix = None
_workerZCoordinates = None
_workerObjectSlices = None


def _init_feature_worker(labelMatrixStack, fov, transformationMatrix,
                          zCoordinates, objectSlices):
    global _workerLabelMatrixStack, _workerFov, \
        _workerTransformationMatrix, _workerZCoordinates, \
        _workerObjectSlices
    _workerLabelMatrixStack = labelMatrixStack
    _workerFov = fov
    _workerTransformationMatrix = transformationMatrix
    _workerZCoordinates = zCoordinates
    _workerObjectSlices = objectSlices


def _feature_for_mask_value(maskValue):
    return SpatialFeature._feature_from_cropped_label(
        _workerLabelMatrixStack, maskValue, _workerObjectSlices, _workerFov,
        _workerTransformationMatrix, _workerZCoordinates)


class SpatialFeature(object):

    """
    A spatial feature is a collection of contiguous voxels.
    """

    def __init__(self, boundaryList: List[List[geometry.Polygon]], fov: int,
                 zCoordinates: np.array = None, uniqueID: int = None,
                 label: int = -1) -> None:
        """Create a new feature specified by a list of pixels

        Args:
            boundaryList: a list of boundaries that define this feature.
                The first index of the list corresponds with the z index.
                The second index corresponds with the index of the shape since
                some regions might split in some z indexes.
            fov: the index of the field of view that this feature belongs to.
                The pixel list specifies pixel in the local fov reference
                frame.
            zCoordinates: the z position for each of the z indexes. If not
                specified, each z index is assumed to have unit height.
            uniqueID: the uuid of this feature. If no uuid is specified,
                a new uuid is randomly generated.
            label: unused
        """
        self._boundaryList = boundaryList
        self._fov = fov

        if uniqueID is None:
            self._uniqueID = uuid.uuid4().int
        else:
            self._uniqueID = uniqueID

        if zCoordinates is not None:
            self._zCoordinates = zCoordinates
        else:
            self._zCoordinates = np.arange(len(boundaryList))

    @staticmethod
    def feature_from_label_matrix(labelMatrix: np.ndarray, fov: int,
                                  transformationMatrix: np.ndarray = None,
                                  zCoordinates: np.ndarray = None,
                                  label: int = -1):
        """Generate a new feature from the specified label matrix.

        Args:
            labelMatrix: a 3d matrix indicating the z, x, y position
                of voxels that contain the feature. Voxels corresponding
                to the feature have a value of True while voxels outside of the
                feature should have a value of False.
            fov: the index of the field of view corresponding to the
                label matrix.
            transformationMatrix: a 3x3 numpy array specifying the
                transformation from fov to global coordinates. If None,
                the feature coordinates are not transformed.
            zCoordinates: the z position for each of the z indexes. If not
                specified, each z index is assumed to have unit height.
        Returns: the new feature
        """

        boundaries = [SpatialFeature._extract_boundaries(x)
                      for x in labelMatrix]

        if transformationMatrix is not None:
            boundaries = [SpatialFeature._transform_boundaries(
                x, transformationMatrix) for x in boundaries]

        return SpatialFeature([SpatialFeature._remove_invalid_boundaries(
            SpatialFeature._remove_interior_boundaries(
                [geometry.Polygon(x) for x in b if len(x) > 2]))
                               for b in boundaries], fov, zCoordinates)

    @staticmethod
    def features_from_label_matrix_stack(
            labelMatrixStack: np.ndarray, maskValues: np.ndarray, fov: int,
            transformationMatrix: np.ndarray = None,
            zCoordinates: np.ndarray = None,
            processes: int = 1) -> List['SpatialFeature']:
        """Generate one feature per mask value in a labeled image stack.

        Equivalent to calling
        feature_from_label_matrix(labelMatrixStack == value, ...) once per
        entry in maskValues, up to floating-point rounding of the polygon
        coordinates. Each object is contoured only inside its own bounding
        box (padded by 1 px) rather than on every full frame, which is
        what dominates the cost for large frames with many objects.
        Objects can also be split across worker processes, since each
        object's feature is independent of the others.

        Args:
            labelMatrixStack: a 3d matrix (z, x, y) where each segmented
                object is assigned its own integer label.
            maskValues: the label values to extract, one feature per value,
                in the returned order.
            fov: the index of the field of view corresponding to the
                label matrix.
            transformationMatrix: as in feature_from_label_matrix.
            zCoordinates: as in feature_from_label_matrix.
            processes: number of worker processes to use. 1 (the default)
                runs serially in the current process with no multiprocessing
                overhead.
        Returns: the list of features, one per entry in maskValues, in the
            same order.
        """
        if len(maskValues) == 0:
            return []
        objectSlices = ndimage.find_objects(labelMatrixStack)

        if processes <= 1:
            return [SpatialFeature._feature_from_cropped_label(
                        labelMatrixStack, v, objectSlices, fov,
                        transformationMatrix, zCoordinates)
                    for v in maskValues]

        with multiprocessing.Pool(
                processes=processes, initializer=_init_feature_worker,
                initargs=(labelMatrixStack, fov, transformationMatrix,
                          zCoordinates, objectSlices)) as pool:
            return pool.map(_feature_for_mask_value, maskValues)

    @staticmethod
    def _feature_from_cropped_label(
            labelMatrixStack: np.ndarray, maskValue: int, objectSlices,
            fov: int, transformationMatrix: np.ndarray = None,
            zCoordinates: np.ndarray = None) -> 'SpatialFeature':
        """feature_from_label_matrix(labelMatrixStack == maskValue, ...),
        contoured only inside the object's bounding box from objectSlices
        (the output of ndimage.find_objects on labelMatrixStack), padded
        by 1 px so the contour closes as it does on the full frame. All z
        planes are kept so the boundary list still has one entry per
        plane.
        """
        objectSlice = objectSlices[maskValue - 1] \
            if 0 < maskValue <= len(objectSlices) else None
        if objectSlice is None:
            return SpatialFeature.feature_from_label_matrix(
                labelMatrixStack == maskValue, fov, transformationMatrix,
                zCoordinates)

        _, rowSlice, colSlice = objectSlice
        row0 = max(rowSlice.start - 1, 0)
        col0 = max(colSlice.start - 1, 0)
        cropped = labelMatrixStack[
            :, row0:rowSlice.stop + 1, col0:colSlice.stop + 1] == maskValue
        # contours come back as (col, row) because _extract_boundaries
        # transposes each frame
        cropOffset = np.array([[1, 0, col0], [0, 1, row0], [0, 0, 1]],
                              dtype=float)
        if transformationMatrix is not None:
            cropOffset = transformationMatrix @ cropOffset
        return SpatialFeature.feature_from_label_matrix(
            cropped, fov, cropOffset, zCoordinates)

    @staticmethod
    def _extract_boundaries(labelMatrix: np.ndarray) -> List[np.ndarray]:
        """Determine the boundaries of the feature indicated in the
        label matrix.

        Args:
            labelMatrix: a 2 dimensional numpy array indicating the x, y
                position of pixels that contain the feature.
        Returns: a list of n x 2 numpy arrays indicating the x, y coordinates
            of the boundaries where n is the number of boundary coordinates
        """
        boundaries = measure.find_contours(np.transpose(labelMatrix), 0.9,
                                           fully_connected='high')
        return boundaries

    @staticmethod
    def _transform_boundaries(
            boundaries: List[np.ndarray],
            transformationMatrix: np.ndarray) -> List[np.ndarray]:

        transformedList = []
        for b in boundaries:
            reshapedBoundaries = np.reshape(
                b, (1, b.shape[0], 2)).astype(float) # np.float depreciation warning
            transformedBoundaries = cv2.transform(
                reshapedBoundaries, transformationMatrix)[0, :, :2]
            transformedList.append(transformedBoundaries)

        return transformedList

    @staticmethod
    def _remove_interior_boundaries(
            inPolygons: List[geometry.Polygon]) -> List[geometry.Polygon]:
        goodPolygons = []

        for p in inPolygons:
            if not any([pTest.contains(p)
                        for pTest in inPolygons if p != pTest]):
                goodPolygons.append(p)

        return goodPolygons

    @staticmethod
    def _remove_invalid_boundaries(
            inPolygons: List[geometry.Polygon]) -> List[geometry.Polygon]:
        return [p for p in inPolygons if p.is_valid]

    def make_a_buffered_copy(self, buffer_size: float):
        '''Make a buffered copy of the SpatialFeature where the
        boundaries in each Z plane is expanded (positive buffer_size)
        or shrinked (negative buffer_size) by the buffer_size.
        '''

        new_boundaryList = []
        for i in range(len(self._boundaryList)):
            new_boundaryList.append([])
            print(f'{i}')

            for pg in self._boundaryList[i]:

                p_b = pg.buffer(buffer_size)

                if p_b.geom_type == 'MultiPolygon':
                    for p in p_b.geoms:
                        new_boundaryList[i].append(p)
                else:
                    new_boundaryList[i].append(p_b)

        return SpatialFeature(new_boundaryList, self._fov, self._zCoordinates.copy(), self._uniqueID)

    def set_fov(self, newFOV: int) -> None:
        """Update the FOV for this spatial feature.

        Args:
            nowFOV: the new FOV index
        """
        self._fov = newFOV

    def get_fov(self) -> int:
        return self._fov

    def get_boundaries(self) -> List[List[geometry.Polygon]]:
        return self._boundaryList

    def get_feature_id(self) -> int:
        return self._uniqueID

    def get_z_coordinates(self) -> np.ndarray:
        return self._zCoordinates

    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Get the 2d box that contains all boundaries in all z plans of this
        feature.

        Returns:
            a tuple containing (x1, y1, x2, y2) coordinates of the bounding box
        """
        boundarySet = []
        for f in self.get_boundaries():
            for b in f:
                boundarySet.append(b)

        multiPolygon = geometry.MultiPolygon(boundarySet)
        return multiPolygon.bounds

    def get_volume(self) -> float:
        """Get the volume enclosed by this feature.

        Returns:
            the volume represented in global coordinates. If only one z
            slice is present for the feature, the z height is taken as 1.
        """
        boundaries = self.get_boundaries()

        zPos = np.array(self._zCoordinates)
        if len(zPos) > 1:
            zDiff = np.diff(zPos)
            zNum = np.array([[x, x + 1] for x in range(len(zPos) - 1)])
            areas = np.array([np.sum([y.area for y in x]) if len(x) > 0
                              else 0 for x in boundaries])
            totalVolume = np.sum([np.mean(areas[zNum[x]]) * zDiff[x]
                                  for x in range(zNum.shape[0])])
        else:
            totalVolume = np.sum([y.area for x in boundaries for y in x])

        return totalVolume

    def intersection(self, intersectFeature) -> float:

        intersectArea = 0
        for p1Set, p2Set in zip(self.get_boundaries(),
                                intersectFeature.get_boundaries()):
            for p1 in p1Set:
                for p2 in p2Set:
                    intersectArea += p1.intersection(p2).area

        return intersectArea

    def is_contained_within_boundary(self, inFeature) -> bool:
        """Determine if any part of this feature is contained within the
        boundary of the specified feature.

        Args:
            inFeature: the feature whose boundary should be checked whether
                it contains this feature
        Returns:
            True if inFeature contains pixels that are within inFeature,
                otherwise False. This returns false if inFeature only shares
                a boundary with this feature.
        """
        if all([b1.disjoint(b2) for b1List, b2List in zip(
                    self.get_boundaries(), inFeature.get_boundaries())
                for b1 in b1List for b2 in b2List]):
            return False

        for b1List, b2List in zip(
                self.get_boundaries(), inFeature.get_boundaries()):
            for b1 in b1List:
                for b2 in b2List:
                    x, y = b1.exterior.coords.xy
                    for p in zip(x, y):
                        if geometry.Point(p).within(b2):
                            return True

        return False

    def equals(self, testFeature) -> bool:
        """Determine if this feature is equivalent to testFeature

        Args:
            testFeature: the feature to test equivalency
        Returns:
            True if this feature and testFeature are equivalent, otherwise
                false
        """
        if self.get_fov() != testFeature.get_fov():
            return False
        if self.get_feature_id() != testFeature.get_feature_id():
            return False
        if not np.array_equal(self.get_z_coordinates(),
                              testFeature.get_z_coordinates()):
            return False

        if len(self.get_boundaries()) != len(testFeature.get_boundaries()):
            return False
        for b, bIn in zip(self.get_boundaries(), testFeature.get_boundaries()):
            if len(b) != len(bIn):
                return False
            for x, y in zip(b, bIn):
                if not x.equals(y):
                    return False

        return True

    def contains_point(self, point: geometry.Point, zIndex: int) -> bool:
        """Determine if this spatial feature contains the specified point.

        Args:
            point: the point to check
            zIndex: the z-index that the point corresponds to
        Returns:
            True if the boundaries of this spatial feature in the zIndex plane
                contain the given point.
        """
        for boundaryElement in self.get_boundaries()[zIndex]:
            if boundaryElement.contains(point):
                return True

        return False

    def contains_positions(self, positionList: np.ndarray) -> np.ndarray:
        """Determine if this spatial feature contains the specified positions

        Args:
            positionList: a N x 3 numpy array containing the (x, y, z)
                positions for N points where x and y are spatial coordinates
                and z is the z index. If z is not an integer it is rounded
                to the nearest integer.
        Returns:
            a numpy array of booleans containing true in the i'th index if
                the i'th point provided is in this spatial feature.
        """
        bounding_box = self.get_bounding_box()
        
        boundaries = self.get_boundaries()
        positionList[:, 2] = np.round(positionList[:, 2])

        containmentList = np.zeros(positionList.shape[0], dtype='bool')
        
        if len(bounding_box) != 4:
            return containmentList

        for zIndex in range(len(boundaries)):
            currentIndexes = np.where(np.all([positionList[:, 2] == zIndex,
                                              bounding_box[0] <= positionList[:, 0],
                                              bounding_box[1] <= positionList[:, 1],
                                              bounding_box[2] >= positionList[:, 0],
                                              bounding_box[3] >= positionList[:, 1]], axis=0))[0]
            
            currentContainment = [self.contains_point(
                geometry.Point(x[0], x[1]), zIndex)
                for x in positionList[currentIndexes]]
            containmentList[currentIndexes] = currentContainment

        return containmentList

    def get_overlapping_features(self, featuresToCheck: List['SpatialFeature']
                                 ) -> List['SpatialFeature']:
        """ Determine which features within the provided list overlap with this
        feature.

        Args:
            featuresToCheck: the list of features to check for overlap with
                this feature.
        Returns: the features that overlap with this feature
        """
        areas = [self.intersection(x) for x in featuresToCheck]
        overlapping = [featuresToCheck[i] for i, x in enumerate(areas) if x > 0]
        benchmark = self.intersection(self)
        contained = [x for x in overlapping if
                     x.intersection(self) == benchmark]
        if len(contained) > 1:
            overlapping = []
        else:
            toReturn = []
            for c in overlapping:
                if c.get_feature_id() == self.get_feature_id():
                    toReturn.append(c)
                else:
                    if c.intersection(self) != c.intersection(c):
                        toReturn.append(c)
            overlapping = toReturn

        return overlapping

    def to_json_dict(self) -> Dict:
        return {
            'fov': self._fov,
            'id': self._uniqueID,
            'z_coordinates': self._zCoordinates.tolist(),
            'boundaries': [[geometry.mapping(y) for y in x]
                           for x in self.get_boundaries()]
        }

    @staticmethod
    def from_json_dict(jsonIn: Dict):
        boundaries = [[geometry.shape(y) for y in x]
                      for x in jsonIn['boundaries']]

        return SpatialFeature(boundaries,
                              jsonIn['fov'],
                              np.array(jsonIn['z_coordinates']),
                              jsonIn['id'])


class SpatialFeatureDB(object):

    """A database for storing spatial features."""

    def __init__(self, dataSet, analysisTask):
        self._dataSet = dataSet
        self._analysisTask = analysisTask

    @abstractmethod
    def write_features(self, features: List[SpatialFeature], fov=None) -> None:
        """Write the features into this database.

        If features already exist in the database with feature IDs equal to
        those in the provided list, an exception is raised.

        Args:
            features: a list of features
            fov: the fov of the features if all feature correspond to the same
                fov. If the features correspond to different fovs, fov
                should be None
        """
        pass

    @abstractmethod
    def read_features(self, fov: int = None) -> List[SpatialFeature]:
        """Read the features in this database

        Args:
            fov: if not None, only the features associated with the specified
                fov are returned
        """
        pass

    @abstractmethod
    def empty_database(self, fov: int = None) -> None:
        """Remove all features from this database.

        Args:
            fov: index of the field of view. If specified, only features
                corresponding to the specified fov will be removed.
                Otherwise all barcodes will be removed.
        """
        pass


class HDF5SpatialFeatureDB(SpatialFeatureDB):

    """
    A data store for spatial features that uses a HDF5 file to store the feature
    information.
    """

    def __init__(self, dataSet: dataset.DataSet, analysisTask):
        super().__init__(dataSet, analysisTask)

    @staticmethod
    def _save_geometry_to_hdf5_group(h5Group: h5py.Group,
                                     polygon: geometry.Polygon) -> None:
        geometryDict = geometry.mapping(polygon)
        h5Group.attrs['type'] = np.string_(geometryDict['type'])
        h5Group['coordinates'] = np.array(geometryDict['coordinates'])

    @staticmethod
    def _save_feature_to_hdf5_group(h5Group: h5py.Group,
                                    feature: SpatialFeature,
                                    fov: int) -> None:
        featureKey = str(feature.get_feature_id())
        featureGroup = h5Group.create_group(featureKey)
        featureGroup.attrs['id'] = np.string_(feature.get_feature_id())
        featureGroup.attrs['fov'] = fov
        featureGroup.attrs['bounding_box'] = \
            np.array(feature.get_bounding_box())
        featureGroup.attrs['volume'] = feature.get_volume()
        featureGroup['z_coordinates'] = feature.get_z_coordinates()
        
        # add number of z planes with actual polygons
        featureGroup.attrs['num_z'] = len([element for element in feature.get_boundaries() if element])

        for i, bSet in enumerate(feature.get_boundaries()):
            zBoundaryGroup = featureGroup.create_group('zIndex_' + str(i))
            for j, b in enumerate(bSet):
                geometryGroup = zBoundaryGroup.create_group('p_' + str(j))
                HDF5SpatialFeatureDB._save_geometry_to_hdf5_group(
                    geometryGroup, b)

    @staticmethod
    def _load_geometry_from_hdf5_group(h5Group: h5py.Group):
        geometryDict = {'type': h5Group.attrs['type'].decode(),
                        'coordinates': np.array(h5Group['coordinates'])}

        return geometry.shape(geometryDict)

    @staticmethod
    def _load_feature_from_hdf5_group(h5Group):
        zCount = len([x for x in h5Group.keys() if x.startswith('zIndex_')])
        boundaryList = []
        for z in range(zCount):
            zBoundaryList = []
            zGroup = h5Group['zIndex_' + str(z)]
            pCount = len([x for x in zGroup.keys() if x[:2] == 'p_'])
            for p in range(pCount):
                zBoundaryList.append(
                    HDF5SpatialFeatureDB._load_geometry_from_hdf5_group(
                        zGroup['p_' + str(p)]))
            boundaryList.append(zBoundaryList)

        loadedFeature = SpatialFeature(
            boundaryList,
            h5Group.attrs['fov'],
            np.array(h5Group['z_coordinates']),
            int(h5Group.attrs['id']))

        return loadedFeature

    def write_features(self, features: List[SpatialFeature], fov=None) -> None:
        if fov is None:
            uniqueFOVs = np.unique([f.get_fov() for f in features])
            for currentFOV in uniqueFOVs:
                currentFeatures = [f for f in features
                                   if f.get_fov() == currentFOV]
                self.write_features(currentFeatures, currentFOV)

        else:
            with self._dataSet.open_hdf5_file(
                    'a', 'feature_data', self._analysisTask, fov, 'features') \
                    as f:
                featureGroup = f.require_group('featuredata')
                featureGroup.attrs['version'] = merlin.version()
                for currentFeature in features:
                    self._save_feature_to_hdf5_group(featureGroup,
                                                     currentFeature,
                                                     fov)

    def read_features(self, fov: int = None) -> List[SpatialFeature]:
        if fov is None:
            featureList = [f for x in self._dataSet.get_fovs()
                           for f in self.read_features(x)]
            return featureList

        featureList = []
        try:
            with self._dataSet.open_hdf5_file('r', 'feature_data',
                                              self._analysisTask, fov,
                                              'features') as f:
                featureGroup = f.require_group('featuredata')
                for k in featureGroup.keys():
                    featureList.append(
                        self._load_feature_from_hdf5_group(featureGroup[k]))
        except FileNotFoundError:
            pass

        return featureList

    def get_feature_z_count(self, fov: int = None) -> int:
        """Get the z-plane count of the first feature found, without
        loading any boundary geometry.

        Lets a caller (e.g. a plot that only needs one z-plane) pick a
        z index cheaply, instead of calling read_features() -- which
        would deserialize every z-plane's polygons for every feature just
        to check how many z-planes exist.

        Args:
            fov: restrict the search to a single fov. If None, fovs are
                checked in dataset order until one with a feature is found.
        Returns: the z count of the first feature found, or 0 if no
            features exist.
        """
        if fov is None:
            for x in self._dataSet.get_fovs():
                zCount = self.get_feature_z_count(x)
                if zCount > 0:
                    return zCount
            return 0

        try:
            with self._dataSet.open_hdf5_file('r', 'feature_data',
                                              self._analysisTask, fov,
                                              'features') as f:
                featureGroup = f.require_group('featuredata')
                keys = list(featureGroup.keys())
                if not keys:
                    return 0
                firstFeature = featureGroup[keys[0]]
                return len([k for k in firstFeature.keys()
                           if k.startswith('zIndex_')])
        except FileNotFoundError:
            return 0

    def read_feature_boundaries_at_z(
            self, zIndex: int, fov: int = None) -> List[List[geometry.Polygon]]:
        """Read only the boundary polygons at a single z index, for every
        feature, without loading any other z-plane's geometry.

        This is what a caller that only needs one z-plane (e.g.
        SegmentationBoundaryPlot) should use instead of read_features() --
        read_features() deserializes every z-plane of every feature, which
        for a whole dataset is ~20-25x more polygon data than a single
        z-plane needs.

        Args:
            zIndex: the z index to read boundaries for.
            fov: if not None, only the boundaries for the specified fov are
                returned.
        Returns: one list of polygons per feature (the z=zIndex entry of
            that feature's get_boundaries()), for features that have a
            zIndex_<zIndex> group. Features without one are skipped.
        """
        return self.read_feature_ids_and_boundaries_at_z(zIndex, fov)[1]

    def read_feature_ids_and_boundaries_at_z(
            self, zIndex: int, fov: int = None
            ) -> Tuple[List[str], List[List[geometry.Polygon]]]:
        """Read feature ids together with their boundary polygons at a
        single z index, for every feature, without loading any other
        z-plane's geometry.

        Companion to read_feature_boundaries_at_z() for a caller that also
        needs to know which feature each polygon set belongs to (e.g.
        SmfishSignal joining detected spots against segmentation
        boundaries one z-plane at a time instead of loading the whole 3D
        cell set with read_features() up front).

        Args:
            zIndex: the z index to read boundaries for.
            fov: if not None, only the boundaries for the specified fov are
                returned.
        Returns: a (featureIds, boundaries) pair of equal-length lists --
            one feature id (the same string read_features() would report
            via get_feature_id()) and one list of polygons (the z=zIndex
            entry of that feature's get_boundaries()) per feature that has
            a zIndex_<zIndex> group. Features without one are skipped.
        """
        if fov is None:
            featureIds: List[str] = []
            boundaryList: List[List[geometry.Polygon]] = []
            for x in self._dataSet.get_fovs():
                fIds, fBoundaries = \
                    self.read_feature_ids_and_boundaries_at_z(zIndex, x)
                featureIds.extend(fIds)
                boundaryList.extend(fBoundaries)
            return featureIds, boundaryList

        featureIds = []
        boundaryList = []
        zGroupName = 'zIndex_' + str(zIndex)
        try:
            with self._dataSet.open_hdf5_file('r', 'feature_data',
                                              self._analysisTask, fov,
                                              'features') as f:
                featureGroup = f.require_group('featuredata')
                for k in featureGroup.keys():
                    featG = featureGroup[k]
                    if zGroupName not in featG:
                        continue
                    zGroup = featG[zGroupName]
                    pCount = len([x for x in zGroup.keys()
                                 if x[:2] == 'p_'])
                    featureIds.append(k)
                    boundaryList.append([
                        self._load_geometry_from_hdf5_group(
                            zGroup['p_' + str(p)])
                        for p in range(pCount)])
        except FileNotFoundError:
            pass

        return featureIds, boundaryList

    def read_feature_boundaries_at_own_z(
            self, fov: int) -> List[List[geometry.Polygon]]:
        """Read each feature's boundary polygons at its own single most
        representative z-index -- the middle of that feature's own
        occupied (non-empty) z-planes -- rather than one fixed z-index
        shared by every feature in the fov.

        Needed because segmentation here is effectively per-z-plane: most
        cells occupy only a handful of a fov's z-planes, not clustered at
        the fov-wide geometric middle read_feature_boundaries_at_z() uses,
        so a single shared z-index can silently miss most cells (see
        FINDINGS.md's SegmentationBoundaryPlot coverage bug). Determining
        occupancy only checks each zIndex_<i> group's key count (cheap
        HDF5 metadata, same as num_z at write time), never loading a
        group's actual coordinates until its z is the chosen one -- so
        this costs about the same per feature as
        read_feature_boundaries_at_z(), not the ~num_z multiple
        read_features() would.

        Unlike read_feature_boundaries_at_z()/read_feature_ids_and_
        boundaries_at_z(), this always requires a specific fov (no
        fov=None dataset-wide convenience) -- the whole point is to let a
        caller stream one fov at a time instead of holding every fov's
        boundaries in memory together (see
        SegmentationBoundaryPlot._generate_plot()).

        Args:
            fov: the fov to read boundaries for.
        Returns: one list of polygons per feature (that feature's own
            chosen zIndex's boundaries). A feature with no occupied
            z-plane at all is skipped.
        """
        # A fov has ~100 zIndex groups per feature, so this is dominated
        # by per-group hdf5 metadata access: the whole file is read into
        # memory in one go (driver='core') instead of seeking for each
        # group, and occupancy uses h5py's low-level group API
        # (get_num_objs), skipping the high-level Group wrappers.
        boundaryList: List[List[geometry.Polygon]] = []
        try:
            with self._dataSet.open_hdf5_file('r', 'feature_data',
                                              self._analysisTask, fov,
                                              'features',
                                              driver='core') as f:
                featureGroup = f.require_group('featuredata')
                for k in featureGroup.keys():
                    featG = featureGroup[k]
                    occupied = []
                    i = 0
                    while featG.id.links.exists(b'zIndex_%d' % i):
                        zName = b'zIndex_%d' % i
                        if h5py.h5g.open(featG.id, zName).get_num_objs() > 0:
                            occupied.append(zName.decode())
                        i += 1
                    if not occupied:
                        continue
                    chosenGroup = featG[occupied[len(occupied) // 2]]
                    pCount = len([x for x in chosenGroup.keys()
                                 if x[:2] == 'p_'])
                    boundaryList.append([
                        self._load_geometry_from_hdf5_group(
                            chosenGroup['p_' + str(p)])
                        for p in range(pCount)])
        except FileNotFoundError:
            pass

        return boundaryList

    def empty_database(self, fov: int = None) -> None:
        if fov is None:
            for f in self._dataSet.get_fovs():
                self.empty_database(f)

        self._dataSet.delete_hdf5_file('feature_data', self._analysisTask,
                                       fov, 'features')

    def read_feature_metadata(self, fov: int = None) -> pandas.DataFrame:
        """ Get the metadata for the features stored within this feature
        database.

        Args:
            fov: an index of a fov to only get the features within the
                specified field of view. If not specified features
                within all fields of view are returned.
        Returns: a data frame containing the metadata, including:
            fov, volume, center_x, center_y, min_x, min_y, max_x, max_y.
            Coordinates are in microns.
        """
        if fov is None:
            finalDF = pandas.concat([self.read_feature_metadata(x)
                                     for x in self._dataSet.get_fovs()], axis = 0) # pandas > 2

        else:
            try:
                with self._dataSet.open_hdf5_file('r', 'feature_data',
                                                  self._analysisTask, fov,
                                                  'features') as f:
                    allAttrKeys = []
                    allAttrValues = []
                    for key in f['featuredata'].keys():
                        attrNames = list(f['featuredata'][key].attrs.keys())
                        attrValues = list(f['featuredata'][key].attrs.values())
                        allAttrKeys.append(attrNames)
                        allAttrValues.append(attrValues)

                    columns = list(np.unique(allAttrKeys))
                    df = pandas.DataFrame(data=allAttrValues, columns=columns)
                    finalDF = df.loc[:, ['fov', 'volume']].copy(deep=True)

                    # try to include num_z if it exists
                    if 'num_z' in columns:
                        finalDF['num_z'] = df['num_z']

                    finalDF.index = df['id'].str.decode(encoding='utf-8'
                                                        ).values.tolist()
                    boundingBoxDF = pandas.DataFrame(
                        df['bounding_box'].values.tolist(),
                        index=finalDF.index)
                    finalDF['center_x'] = \
                        (boundingBoxDF[0] + boundingBoxDF[2]) / 2
                    finalDF['center_y'] = \
                        (boundingBoxDF[1] + boundingBoxDF[3]) / 2
                    finalDF['min_x'] = boundingBoxDF[0]
                    finalDF['max_x'] = boundingBoxDF[2]
                    finalDF['min_y'] = boundingBoxDF[1]
                    finalDF['max_y'] = boundingBoxDF[3]
            except FileNotFoundError:
                return pandas.DataFrame()

        return finalDF


class JSONSpatialFeatureDB(SpatialFeatureDB):

    """
    A database for storing spatial features with json serialization.
    """

    def __init__(self, dataSet: dataset.DataSet, analysisTask):
        super().__init__(dataSet, analysisTask)

    def write_features(self, features: List[SpatialFeature], fov=None) -> None:
        if fov is None:
            raise NotImplementedError

        try:
            existingFeatures = [SpatialFeature.from_json_dict(x)
                                for x
                                in self._dataSet.load_json_analysis_result(
                    'feature_data', self._analysisTask, fov, 'features')]

            existingIDs = set([x.get_feature_id() for x in existingFeatures])

            for f in features:
                if f.get_feature_id() not in existingIDs:
                    existingFeatures.append(f)

            featuresAsJSON = [f.to_json_dict() for f in existingFeatures]

        except FileNotFoundError:
            featuresAsJSON = [f.to_json_dict() for f in features]

        self._dataSet.save_json_analysis_result(
            featuresAsJSON, 'feature_data', self._analysisTask,
            fov, 'features')

    def read_features(self, fov: int = None) -> List[SpatialFeature]:
        if fov is None:
            raise NotImplementedError

        features = [SpatialFeature.from_json_dict(x)
                    for x in self._dataSet.load_json_analysis_result(
                'feature_metadata', self._analysisTask, fov, 'features')]

        return features

    def empty_database(self, fov: int = None) -> None:
        pass

    @staticmethod
    def _extract_feature_metadata(feature: SpatialFeature) -> Dict:
        boundingBox = feature.get_bounding_box()
        return {'fov': feature.get_fov(),
                'featureID': feature.get_feature_id(),
                'bounds_x1': boundingBox[0],
                'bounds_y1': boundingBox[1],
                'bounds_x2': boundingBox[2],
                'bounds_y2': boundingBox[3],
                'volume': feature.get_volume()}


def simple_clean_cells(cells: List) -> List:
    """
    Removes cells that lack a bounding box or have a volume equal to 0

    Args:
        cells: List of spatial features

    Returns:
        List of spatial features

    """
    return [cell for cell in cells
            if len(cell.get_bounding_box()) == 4 and cell.get_volume() > 0]


def _cell_planes(cell: SpatialFeature) -> Dict[float, BaseGeometry]:
    """The cell's outline in each of its z planes, keyed by z position."""
    planes = {}
    for z, polygons in zip(cell.get_z_coordinates(), cell.get_boundaries()):
        if len(polygons) == 0:
            continue
        outline = polygons[0] if len(polygons) == 1 \
            else unary_union(polygons)
        if not outline.is_valid:
            outline = outline.buffer(0)
        planes[round(float(z), 6)] = outline
    return planes


def _plane_overlaps(planes: List[Dict[float, BaseGeometry]],
                    pairs: np.ndarray, zShifts: np.ndarray) -> np.ndarray:
    """Overlap area of each cell pair (i, j), summed over the planes where
    plane z of cell i meets plane z + zShift of cell j."""
    pairIndex, outlinesA, outlinesB = [], [], []
    for k, ((i, j), shift) in enumerate(zip(pairs, zShifts)):
        planesB = planes[j]
        for z, outline in planes[i].items():
            other = planesB.get(round(z + shift, 6))
            if other is not None:
                pairIndex.append(k)
                outlinesA.append(outline)
                outlinesB.append(other)
    if not pairIndex:
        return np.zeros(len(pairs))
    areas = shapely.area(shapely.intersection(
        np.array(outlinesA, dtype=object), np.array(outlinesB, dtype=object)))
    return np.bincount(pairIndex, weights=areas, minlength=len(pairs))


def construct_overlap_graph(
        currentFOV: int, cellsByFOV: Dict[int, List[SpatialFeature]],
        fovExtents: Dict[int, Tuple[float, float, float, float]],
        zStep: float, overlapThreshold: float = 0.5, maxZShift: int = 4,
        minSeamPairs: int = 10) -> Tuple[nx.Graph, pandas.DataFrame]:
    """Build the overlap graph for the cells of one fov.

    Two cells are linked when the volume they share is at least
    overlapThreshold of the smaller cell's volume (per-plane areas summed
    over z). Smaller overlaps are slivers where the outlines of
    neighbouring cells touch, and linking them chains whole seams into one
    component.

    Neighbouring fovs can disagree in z by a whole number of planes (a
    tilted coverslip under per-fov focus). Cells of another fov are
    therefore compared after shifting them by that fov's seam offset: the
    median z difference of the clear duplicates, the pairs whose z
    projections overlap by at least overlapThreshold and whose best overlap
    within +-maxZShift planes passes it too. With fewer than minSeamPairs
    duplicates the shift is 0.

    Args:
        currentFOV: the fov whose cells are added
        cellsByFOV: the cells of currentFOV and of every fov whose image
            overlaps it
        fovExtents: the global (xmin, ymin, xmax, ymax) of each fov's image
        zStep: the spacing of the z planes, in microns
    Returns:
        the graph, whose nodes carry originalFOV, assignedFOV (the fov
        whose image centre is nearest) and edgeDistance (distance from the
        cell's bounding-box centre to its own image edge), and a table of
        the seam offsets to each neighbouring fov (z_offset_um is the z of
        the neighbour's copy minus this fov's copy, NaN if not measured)
    """
    fovOfCell, cells = [], []
    for fov, fovCells in cellsByFOV.items():
        fovOfCell += [fov] * len(fovCells)
        cells += fovCells
    fovOfCell = np.array(fovOfCell)
    planes = [_cell_planes(c) for c in cells]
    areaSum = np.array([sum(o.area for o in p.values()) for p in planes])
    zCentre = np.array([sum(z * o.area for z, o in p.items()) / a
                        if a > 0 else np.nan
                        for p, a in zip(planes, areaSum)])
    bounds = np.array([c.get_bounding_box() for c in cells]).reshape(-1, 4)
    centres = (bounds[:, :2] + bounds[:, 2:]) / 2
    extents = np.array([fovExtents[f] for f in fovOfCell]).reshape(-1, 4)
    edgeDistance = np.min(np.hstack([centres - extents[:, :2],
                                     extents[:, 2:] - centres]), axis=1)
    fovList = list(cellsByFOV)
    fovCentres = np.array([[(fovExtents[f][0] + fovExtents[f][2]) / 2,
                            (fovExtents[f][1] + fovExtents[f][3]) / 2]
                           for f in fovList])
    assignedFOV = np.array(fovList)[np.argmin(
        ((centres[:, None, :] - fovCentres[None, :, :]) ** 2).sum(axis=2),
        axis=1)] if len(cells) else np.array([], dtype=int)

    own = np.nonzero(fovOfCell == currentFOV)[0]
    tree = shapely.STRtree(shapely.box(*bounds.T))
    qi, qj = tree.query(shapely.box(*bounds[own].T), predicate='intersects')
    i, j = own[qi], qj
    keep = (i != j) & ((fovOfCell[j] != currentFOV) | (i < j))
    pairs = np.stack([i[keep], j[keep]], axis=1)
    smaller = np.minimum(areaSum[pairs[:, 0]], areaSum[pairs[:, 1]])
    neighbourFOV = fovOfCell[pairs[:, 1]]

    # z projections, only for the cells in cross-fov pairs
    projections = np.full(len(cells), None, dtype=object)
    crossCells = np.unique(pairs[neighbourFOV != currentFOV])
    projections[crossCells] = [unary_union(list(planes[k].values()))
                               for k in crossCells]
    seamRows = []
    zShift = np.zeros(len(pairs))
    for fov in fovList:
        if fov == currentFOV:
            continue
        sel = np.nonzero(neighbourFOV == fov)[0]
        offset, nPairs = np.nan, 0
        if len(sel):
            a, b = pairs[sel, 0], pairs[sel, 1]
            projOverlap = shapely.area(shapely.intersection(
                projections[a], projections[b])) / np.minimum(
                shapely.area(projections[a]), shapely.area(projections[b]))
            candidates = sel[projOverlap >= overlapThreshold]
            shifts = np.arange(-maxZShift, maxZShift + 1)
            overlaps = np.stack([_plane_overlaps(
                planes, pairs[candidates],
                np.full(len(candidates), s * zStep)) for s in shifts], axis=1) \
                if len(candidates) else np.zeros((0, len(shifts)))
            duplicates = candidates[
                overlaps.max(axis=1, initial=0) / smaller[candidates]
                >= overlapThreshold] if len(candidates) else candidates
            nPairs = len(duplicates)
            if nPairs >= minSeamPairs:
                offset = float(np.median(zCentre[pairs[duplicates, 1]]
                                         - zCentre[pairs[duplicates, 0]]))
                zShift[sel] = np.round(offset / zStep) * zStep
        seamRows.append({'fov': currentFOV, 'neighbor_fov': fov,
                         'n_pairs': nPairs, 'z_offset_um': offset})

    overlapFraction = _plane_overlaps(planes, pairs, zShift) / smaller
    linked = overlapFraction >= overlapThreshold

    graph = nx.Graph()
    nodes = set(own) | set(pairs[linked].ravel())
    for k in sorted(nodes):
        graph.add_node(cells[k].get_feature_id(),
                       originalFOV=int(fovOfCell[k]),
                       assignedFOV=int(assignedFOV[k]),
                       edgeDistance=float(edgeDistance[k]))
    for (a, b), fraction in zip(pairs[linked], overlapFraction[linked]):
        graph.add_edge(cells[a].get_feature_id(), cells[b].get_feature_id(),
                       overlap=float(fraction))
    seams = pandas.DataFrame(seamRows, columns=[
        'fov', 'neighbor_fov', 'n_pairs', 'z_offset_um'])
    return graph, seams


def remove_overlapping_cells(graph: nx.Graph) -> pandas.DataFrame:
    """Keep one cell wherever cells overlap.

    Cells are visited from the farthest from their own image edge to the
    nearest, and a cell is kept unless it is linked to a cell already
    kept. Near an image edge a cell is often cut by the border, while the
    neighbouring fov sees it whole, so the copy away from its edge wins.
    Ties keep graph order.

    Args:
        graph: an undirected graph from construct_overlap_graph, with the
            originalFOV, assignedFOV and edgeDistance node attributes
    Returns:
        A pandas dataframe with the cell_id, originalFOV and assignedFOV of
        every kept cell, in graph order
    """
    nodes = list(graph.nodes())
    missing = [n for n in nodes if 'edgeDistance' not in graph.nodes[n]]
    if missing:
        raise ValueError(
            '%d cells have no edgeDistance; the graph was built by an older '
            'CleanCellBoundaries, rerun it' % len(missing))
    distance = np.array([graph.nodes[n]['edgeDistance'] for n in nodes])
    blocked = set()
    kept = set()
    for k in np.argsort(-distance, kind='stable'):
        node = nodes[k]
        if node in blocked:
            continue
        kept.add(node)
        blocked.update(graph.neighbors(node))
    return pandas.DataFrame(
        [[n, graph.nodes[n]['originalFOV'], graph.nodes[n]['assignedFOV']]
         for n in nodes if n in kept],
        columns=['cell_id', 'originalFOV', 'assignedFOV'])


def solve_fov_z_offsets(seams: pandas.DataFrame,
                        fovs: List[int]) -> pandas.DataFrame:
    """Per-fov z offsets that best explain the measured seam offsets.

    A seam row says that a cell seen by fov and neighbor_fov lies
    z_offset_um higher in neighbor_fov's planes. With the common z defined
    as local z + offset, offset[fov] - offset[neighbor_fov] = z_offset_um.
    Rows are weighted by the square root of their duplicate count, and the
    offsets of each connected group of fovs average to 0. A fov with no
    measured seam gets 0.

    Returns:
        a dataframe with the columns fov and z_offset_um
    """
    fovs = list(fovs)
    index = {f: k for k, f in enumerate(fovs)}
    measured = seams.dropna(subset=['z_offset_um'])
    measured = measured[measured.fov.isin(index)
                        & measured.neighbor_fov.isin(index)]
    graph = nx.Graph()
    graph.add_nodes_from(fovs)
    graph.add_edges_from(zip(measured.fov, measured.neighbor_fov))
    rows, values = [], []
    for f, g, offset, n in zip(measured.fov, measured.neighbor_fov,
                               measured.z_offset_um, measured.n_pairs):
        row = np.zeros(len(fovs))
        w = np.sqrt(n)
        row[index[f]], row[index[g]] = w, -w
        rows.append(row)
        values.append(w * offset)
    for component in nx.connected_components(graph):
        row = np.zeros(len(fovs))
        row[[index[f] for f in component]] = 1
        rows.append(row)
        values.append(0)
    solution = np.linalg.lstsq(np.array(rows), np.array(values), rcond=None)[0]
    return pandas.DataFrame({'fov': fovs, 'z_offset_um': solution})
