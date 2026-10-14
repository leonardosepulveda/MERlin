import warnings
import pandas
import numpy as np
import shapely

from merlin.core import analysistask
from merlin.util import spatialfeature


def contained_barcodes(cells, x, y, z):
    """The (barcode index, cell index) pairs of every barcode inside a cell.

    Same rule as SpatialFeature.contains_positions: a barcode is in a cell
    when one of the cell's polygons in plane round(z) contains (x, y), the
    boundary excluded. Barcodes are sorted by plane and then x once, so
    each polygon only tests the barcodes inside its bounding box.

    Returns:
        two int arrays, barcode indexes and cell indexes, ordered by cell
    """
    zRounded = np.round(z)
    order = np.lexsort((x, zRounded))
    zSorted, xSorted, ySorted = zRounded[order], x[order], y[order]
    planeCount = max((len(c.get_boundaries()) for c in cells), default=0)
    planes = np.arange(planeCount)
    planeStart = np.searchsorted(zSorted, planes, side='left')
    planeEnd = np.searchsorted(zSorted, planes, side='right')

    barcodeIndexes, cellIndexes = [], []
    for cellIndex, cell in enumerate(cells):
        inside = []
        for zIndex, polygons in enumerate(cell.get_boundaries()):
            lo, hi = planeStart[zIndex], planeEnd[zIndex]
            if lo == hi:
                continue
            xPlane = xSorted[lo:hi]
            for polygon in polygons:
                if polygon.is_empty:
                    continue
                minX, minY, maxX, maxY = polygon.bounds
                a = lo + np.searchsorted(xPlane, minX, side='left')
                b = lo + np.searchsorted(xPlane, maxX, side='right')
                yBox = ySorted[a:b]
                candidates = a + np.flatnonzero((yBox >= minY)
                                                & (yBox <= maxY))
                hits = shapely.contains_xy(polygon, xSorted[candidates],
                                           ySorted[candidates])
                inside.append(order[candidates[hits]])
        if inside:
            inside = np.unique(np.concatenate(inside))
            barcodeIndexes.append(inside)
            cellIndexes.append(np.full(len(inside), cellIndex))
    if not barcodeIndexes:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    return np.concatenate(barcodeIndexes), np.concatenate(cellIndexes)


# the only barcode columns partitioning reads
_BARCODE_COLUMNS = ['barcode_id', 'fov', 'global_x', 'global_y', 'z']


def _barcode_positions(barcodes):
    return (barcodes['global_x'].values, barcodes['global_y'].values,
            barcodes['z'].values)


def _counts_frame(counts, cells):
    return pandas.DataFrame(data=counts, columns=range(counts.shape[1]),
                            index=[c.get_feature_id() for c in cells])


def count_traditional(cells, barcodes, barcodeCount):
    """Counts of each barcode type inside each cell. A barcode inside
    several cells counts for each of them."""
    barcodeIds = barcodes['barcode_id'].values.astype(int)
    b, c = contained_barcodes(cells, *_barcode_positions(barcodes))
    counts = np.zeros((len(cells), barcodeCount))
    np.add.at(counts, (c, barcodeIds[b]), 1)
    return _counts_frame(counts, cells)


def count_with_boundary_correction(cells, barcodes, barcodeCount,
                                   bufferSize):
    """Counts of each barcode type per cell, each barcode given to at most
    one cell, using copies of the cells shrunk and expanded by bufferSize.

    A barcode outside every expanded cell is dropped. Otherwise, in order:
    - inside a shrunk cell: it goes to the first such cell;
    - inside a cell whose interior (shrunk copy) holds this barcode type,
      where that cell is the first cell containing it: it goes there;
    - otherwise it goes to a random one (np.random.choice) of the expanded
      cells whose interior holds this barcode type;
    - if there is none, it goes to the first cell containing it, if any.
    "First" is in the order of cells. The random choices are made in
    barcode order, so a seeded np.random gives the same result as the
    earlier per-barcode loop.
    """
    barcodeIds = barcodes['barcode_id'].values.astype(int)
    positions = _barcode_positions(barcodes)
    nBarcodes = len(barcodeIds)
    cellCount = len(cells)
    normal = contained_barcodes(cells, *positions)
    shrunk = contained_barcodes(
        [c.make_a_buffered_copy(-bufferSize) for c in cells], *positions)
    expanded = contained_barcodes(
        [c.make_a_buffered_copy(bufferSize) for c in cells], *positions)

    interiorCounts = np.zeros((cellCount, barcodeCount))
    np.add.at(interiorCounts, (shrunk[1], barcodeIds[shrunk[0]]), 1)

    def first_host(pairs):
        host = np.full(nBarcodes, cellCount)
        np.minimum.at(host, pairs[0], pairs[1])
        return host

    firstShrunk = first_host(shrunk)
    firstNormal = first_host(normal)
    inExpanded = np.zeros(nBarcodes, dtype=bool)
    inExpanded[expanded[0]] = True
    hasShrunk = inExpanded & (firstShrunk < cellCount)
    hasNormal = inExpanded & ~hasShrunk & (firstNormal < cellCount)

    # expanded hosts whose interior holds the barcode's type
    eB, eC = expanded
    keep = interiorCounts[eC, barcodeIds[eB]] > 0
    eB, eC = eB[keep], eC[keep]
    hasCandidate = np.zeros(nBarcodes, dtype=bool)
    hasCandidate[eB] = True

    normalHolds = np.zeros(nBarcodes, dtype=bool)
    normalHolds[hasNormal] = interiorCounts[
        firstNormal[hasNormal], barcodeIds[hasNormal]] > 0
    random = inExpanded & ~hasShrunk & hasCandidate \
        & ~(hasNormal & normalHolds)

    target = np.full(nBarcodes, -1)
    target[hasShrunk] = firstShrunk[hasShrunk]
    toNormal = hasNormal & ~random
    target[toNormal] = firstNormal[toNormal]

    order = np.lexsort((eC, eB))
    eB, eC = eB[order], eC[order]
    pick = random[eB]
    eB, eC = eB[pick], eC[pick]
    randomBarcodes, starts = np.unique(eB, return_index=True)
    for b, hosts in zip(randomBarcodes, np.split(eC, starts[1:])):
        target[b] = np.random.choice(hosts)

    assigned = target >= 0
    counts = np.zeros((cellCount, barcodeCount))
    np.add.at(counts, (target[assigned], barcodeIds[assigned]), 1)
    return _counts_frame(counts, cells)

class PartitionBarcodes(analysistask.ParallelAnalysisTask):

    """
    An analysis task that assigns RNAs and sequential signals to cells
    based on the boundaries determined during the segment task.

    A fov's cells are matched against the barcodes of every fov whose image
    overlaps it. With apply_fov_z_offsets (default True) and an assignment
    task that has a combine_cleaning_task, the z of a neighbouring fov's
    barcodes is first moved into this fov's planes using the per-fov z
    offsets of CombineCleanedBoundaries.
    """

    providesMemoryEstimate = True
    providesTimeEstimate = True

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)
        if 'boundary_correction_buffer_size' not in self.parameters:
            self.parameters['boundary_correction_buffer_size'] = 0.5
        if 'apply_fov_z_offsets' not in self.parameters:
            self.parameters['apply_fov_z_offsets'] = True

    def fragment_count(self):
        return len(self.dataSet.get_fovs())

    def get_estimated_memory(self):
        # LT066 fovs 265, 604 and 189 (7.7-13.8 M barcodes in the fov and
        # its neighbours, 1922-2584 cells): 3.3-5.0 GB
        return 6000

    def get_estimated_time(self):
        # the same fovs: 2.8-5.1 min
        return 10

    def get_dependencies(self):
        return [self.parameters['filter_task'],
                self.parameters['assignment_task'],
                self.parameters['alignment_task']]

    def get_partitioned_barcodes(self, fov: int = None) -> pandas.DataFrame:
        """Retrieve the cell by barcode matrixes calculated from this
        analysis task.

        Args:
            fov: the fov to get the barcode table for. If not specified, the
                combined table for all fovs are returned.

        Returns:
            A pandas data frame containing the parsed barcode information.
        """
        if fov is None:
            return pandas.concat(
                [self.get_partitioned_barcodes(fov)
                 for fov in self.dataSet.get_fovs()]
            )

        return self.dataSet.load_dataframe_from_csv(
            'counts_per_cell', self.get_analysis_name(), fov, index_col=0)

    def _z_shift_to_fov(self, assignmentTask, fov: int,
                        barcodeFOVs: np.ndarray) -> np.ndarray:
        """The shift, in planes, that moves each barcode's z into the planes
        of *fov*: (offset of the barcode's fov - offset of *fov*) divided by
        the plane spacing. Zeros when there are no offsets to apply."""
        combineName = assignmentTask.parameters.get('combine_cleaning_task')
        if not self.parameters['apply_fov_z_offsets'] or combineName is None:
            return np.zeros(len(barcodeFOVs))
        try:
            offsets = self.dataSet.load_analysis_task(
                combineName).get_fov_z_offsets()
        except FileNotFoundError:
            warnings.warn(
                '{0} has no fov_z_offsets (it ran before they existed); '
                'barcodes keep their own fov\'s z. Rerun CleanCellBoundaries '
                'and {0} to apply them.'.format(combineName))
            return np.zeros(len(barcodeFOVs))
        zPositions = np.unique(self.dataSet.get_z_positions())
        zStep = float(np.median(np.diff(zPositions))) \
            if len(zPositions) > 1 else 1.0
        return (offsets.reindex(barcodeFOVs).fillna(0).values
                - offsets.get(fov, 0.0)) / zStep

    def _run_analysis(self, fragmentIndex):
        if self.parameters['boundary_correction_buffer_size'] == 0:
            self._run_analysis_traditional(fragmentIndex)
        else:
            self._run_analysis_boundary_correction(fragmentIndex)

    def _run_analysis_boundary_correction(self, fragmentIndex):
        filterTask = self.dataSet.load_analysis_task(
            self.parameters['filter_task'])
        assignmentTask = self.dataSet.load_analysis_task(
            self.parameters['assignment_task'])
        alignTask = self.dataSet.load_analysis_task(
            self.parameters['alignment_task'])

        fovBoxes = alignTask.get_fov_boxes()
        fovIntersections = sorted([i for i, x in enumerate(fovBoxes) if
                                   fovBoxes[fragmentIndex].intersects(x)])

        codebook = filterTask.get_codebook()
        barcodeCount = codebook.get_barcode_count()

        bcDB = filterTask.get_barcode_database()
        for fi in fovIntersections:
            partialBC = bcDB.get_barcodes(fi, columnList=_BARCODE_COLUMNS)
            if fi == fovIntersections[0]:
                currentFOVBarcodes = partialBC.copy(deep=True)
            else:
                currentFOVBarcodes = pandas.concat(
                    [currentFOVBarcodes, partialBC], axis = 0) # pandas > 2

        currentFOVBarcodes = currentFOVBarcodes.reset_index().copy(deep=True)
        currentFOVBarcodes['z'] += self._z_shift_to_fov(
            assignmentTask, fragmentIndex, currentFOVBarcodes['fov'].values)

        sDB = assignmentTask.get_feature_database()
        currentCells = sDB.read_features(fragmentIndex)

        print(f'FOV {fragmentIndex} has {len(currentCells)} cells.')

        countsDF = count_with_boundary_correction(
            currentCells, currentFOVBarcodes, barcodeCount,
            self.parameters['boundary_correction_buffer_size'])

        barcodeNames = [codebook.get_name_for_barcode_index(x)
                        for x in countsDF.columns.values.tolist()]
        countsDF.columns = barcodeNames

        self.dataSet.save_dataframe_to_csv(
                countsDF, 'counts_per_cell', self.get_analysis_name(),
                fragmentIndex)
    
    def _run_analysis_traditional(self, fragmentIndex):
        filterTask = self.dataSet.load_analysis_task(
            self.parameters['filter_task'])
        assignmentTask = self.dataSet.load_analysis_task(
            self.parameters['assignment_task'])
        alignTask = self.dataSet.load_analysis_task(
            self.parameters['alignment_task'])

        fovBoxes = alignTask.get_fov_boxes()
        fovIntersections = sorted([i for i, x in enumerate(fovBoxes) if
                                   fovBoxes[fragmentIndex].intersects(x)])

        codebook = filterTask.get_codebook()
        barcodeCount = codebook.get_barcode_count()

        bcDB = filterTask.get_barcode_database()
        for fi in fovIntersections:
            partialBC = bcDB.get_barcodes(fi, columnList=_BARCODE_COLUMNS)
            if fi == fovIntersections[0]:
                currentFOVBarcodes = partialBC.copy(deep=True)
            else:
                currentFOVBarcodes = pandas.concat(
                    [currentFOVBarcodes, partialBC], axis = 0) # pandas > 2

        currentFOVBarcodes = currentFOVBarcodes.reset_index().copy(deep=True)
        currentFOVBarcodes['z'] += self._z_shift_to_fov(
            assignmentTask, fragmentIndex, currentFOVBarcodes['fov'].values)

        sDB = assignmentTask.get_feature_database()
        currentCells = sDB.read_features(fragmentIndex)

        countsDF = count_traditional(currentCells, currentFOVBarcodes,
                                     barcodeCount)

        barcodeNames = [codebook.get_name_for_barcode_index(x)
                        for x in countsDF.columns.values.tolist()]
        countsDF.columns = barcodeNames

        self.dataSet.save_dataframe_to_csv(
                countsDF, 'counts_per_cell', self.get_analysis_name(),
                fragmentIndex)


class ExportPartitionedBarcodes(analysistask.AnalysisTask):

    """
    An analysis task that combines counts per cells data from each
    field of view into a single output file.
    """

    def __init__(self, dataSet, parameters=None, analysisName=None):
        super().__init__(dataSet, parameters, analysisName)

    def get_estimated_memory(self):
        return 2048

    def get_estimated_time(self):
        return 5

    def get_dependencies(self):
        return [self.parameters['partition_task']]

    def _run_analysis(self):
        pTask = self.dataSet.load_analysis_task(
                    self.parameters['partition_task'])
        parsedBarcodes = pTask.get_partitioned_barcodes()

        self.dataSet.save_dataframe_to_csv(
                    parsedBarcodes, 'barcodes_per_feature',
                    self.get_analysis_name())
