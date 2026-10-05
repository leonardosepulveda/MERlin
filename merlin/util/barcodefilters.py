import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import pandas as pd
from typing import List


def remove_zplane_duplicates_all_barcodeids(barcodes: pd.DataFrame,
                                            zPlanes: int,
                                            maxDist: float,
                                            allZPos: List) -> pd.DataFrame:
    """ Depending on the separation between z planes, spots from a single
        molecule may be observed in more than one z plane. These putative
        duplicates are removed based on supplied distance and z plane
        constraints. In evaluating this method, when z planes are separated
        by 1.5 µm the likelihood of finding a putative duplicate above or below
        the selected plane is ~5-10%, whereas the false-positive rate is closer
        to 1%, as determined by checking two planes above or below, or comparing
        barcodes of different identities but similar abundance between
        adjacent z planes.

        Each barcode is linked to the nearest barcode of the same identity
        within maxDist on each plane up to zPlanes above and below. Linked
        barcodes form groups transitively, so with zPlanes = 1 a molecule
        seen on 4 consecutive planes is still one group. The brightest
        barcode (mean_intensity) of each group is kept.

    Args:
        barcodes: a pandas dataframe containing all the entries for a given
                  barcode identity
        zPlanes: number of planes above and below to consider when evaluating
                 potential duplicates
        maxDist: maximum euclidean distance allowed to separate centroids of
                 putative barcode duplicate, in pixels
        allZPos: the z positions of the fov; only barcodes on its first
                 len(allZPos) planes are compared
    Returns:
        keptBarcodes: pandas dataframe where barcodes of the same identity that
                      fall within parameters of z plane duplicates have
                      been removed.
    """
    if len(barcodes) == 0:
        return barcodes
    barcodes = barcodes.reset_index(drop=True)
    keep = _z_duplicate_keep_mask(barcodes, zPlanes, maxDist, len(allZPos))
    return barcodes[keep].sort_values(
        by=['barcode_id', 'z'], kind='stable').reset_index(drop=True)


def remove_zplane_duplicates_single_barcodeid(barcodes: pd.DataFrame,
                                              zPlanes: int,
                                              maxDist: float,
                                              allZPos: List) -> pd.DataFrame:
    """ Remove barcodes with a given barcode id that are putative z plane
        duplicates.

    Args:
        barcodes: a pandas dataframe containing all the entries for a given
                  barcode identity
        zPlanes: number of planes above and below to consider when evaluating
                 potential duplicates
        maxDist: maximum euclidean distance allowed to separate centroids of
                 putative barcode duplicate, in pixels
    Returns:
        keptBarcodes: pandas dataframe where barcodes of the same identity that
                      fall within parameters of z plane duplicates have
                      been removed.
    """
    barcodes.reset_index(drop=True, inplace=True)
    if not len(barcodes['barcode_id'].unique()) == 1:
        errorString = 'The method remove_zplane_duplicates_single_barcodeid ' +\
                      'should be given a dataframe containing molecules ' +\
                      'that all have the same barcode id. Please use ' +\
                      'remove_zplane_duplicates_all_barcodeids to handle ' +\
                      'dataframes containing multiple barcode ids'
        raise ValueError(errorString)
    return barcodes[_z_duplicate_keep_mask(barcodes, zPlanes, maxDist,
                                           len(allZPos))]


def _z_duplicate_keep_mask(barcodes: pd.DataFrame, zPlanes: int,
                           maxDist: float, planeCount: int) -> np.ndarray:
    """True for the brightest barcode of each z-duplicate group (see
    remove_zplane_duplicates_all_barcodeids). One kd-tree holds every
    barcode at (x, y, z * gap, barcode_id * gap), with gap far larger than
    maxDist, so a query at (x, y, (z + d) * gap, barcode_id * gap) can only
    find a barcode of the same identity on plane z + d."""
    n = len(barcodes)
    z = barcodes['z'].values.astype(float)
    compared = (z >= 0) & (z < planeCount)
    gap = 10 * (maxDist + 1)
    points = np.c_[barcodes['x'].values, barcodes['y'].values, z * gap,
                   barcodes['barcode_id'].values * gap]
    tree = cKDTree(points[compared])
    comparedIndex = np.flatnonzero(compared)
    src, dst = [], []
    for d in [d for d in range(-zPlanes, zPlanes + 1) if d != 0]:
        query = points[compared] + [0, 0, d * gap, 0]
        dist, idx = tree.query(query, k=1, distance_upper_bound=maxDist)
        found = np.isfinite(dist)
        src.append(comparedIndex[found])
        dst.append(comparedIndex[idx[found]])
    src = np.concatenate(src) if src else np.zeros(0, int)
    dst = np.concatenate(dst) if dst else np.zeros(0, int)
    graph = coo_matrix((np.ones(len(src), bool), (src, dst)), shape=(n, n))
    labels = connected_components(graph, directed=False)[1]

    # brightest member of each group; the lowest index among equals
    order = np.lexsort((np.arange(n), -barcodes['mean_intensity'].values,
                        labels))
    first = np.r_[True, np.diff(labels[order]) != 0]
    keep = np.zeros(n, bool)
    keep[order[first]] = True
    return keep


def keep_barcodes_of_nearest_fov(barcodes: pd.DataFrame, fov: int,
                                 fovIDs: List[int],
                                 fovCenters: np.ndarray) -> pd.DataFrame:
    """ Keep the barcodes that *fov* owns. Neighbouring fovs overlap, so a
        molecule in the overlap is decoded once in each fov. A barcode is
        owned by the fov whose image centre is nearest to it, over all
        fovs, so each overlap is counted once.

    Args:
        barcodes: barcodes decoded in *fov*, with global_x and global_y
        fov: the fov the barcodes were decoded in
        fovIDs: every fov of the dataset
        fovCenters: (len(fovIDs), 2) global x, y of each fov's image centre,
                    in the same units as global_x and global_y
    Returns:
        the barcodes owned by *fov*
    """
    if len(barcodes) == 0:
        return barcodes
    nearest = cKDTree(fovCenters).query(
        barcodes[['global_x', 'global_y']].values)[1]
    return barcodes[np.asarray(fovIDs)[nearest] == fov]
