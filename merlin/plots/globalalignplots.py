"""Verification figures for merlin.analysis.globalalign's
LeastSquaresGlobalAlignment, generated automatically once that task
completes (see AnalysisTask._generate_verification_figures's own
docstring for why these are plain functions here rather than
merlin.plots' AbstractPlot/PlotEngine classes -- they are a single
task's own quick self-check, not a plot built from several tasks
together). LeastSquaresGlobalAlignment._generate_verification_figures
just calls generate_all(self).
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt


def generate_all(alignTask) -> None:
    """Generate every LeastSquaresGlobalAlignment verification figure.

    Each plot is independently guarded -- e.g. an isolated fov with no
    surviving neighbour correspondence at all still gets a grid-overlay
    figure even though the direction-reliability one has nothing to show.
    """
    for plotFunction in (plot_direction_reliability, plot_grid_overlay,
                         plot_overlap_correlation_grid,
                         plot_overlap_correlation_histogram):
        try:
            plotFunction(alignTask)
        except Exception:
            logging.getLogger(alignTask.analysisName).exception(
                'Failed to generate a %s verification figure' % plotFunction.__name__)


def plot_direction_reliability(alignTask) -> None:
    """Per-direction deviation scatter -- ``measured_xy - nominal_xy``
    for every KEPT correspondence, colored by direction. A tight cluster
    for a direction means that direction's registration is reproducible;
    mirrors the real-data comparison in the sibling MERci project's
    `notebooks/tests/compare_stitching_correction_methods.ipynb` (section
    7), which is what first showed this dataset's error is a real,
    per-direction bias rather than noise.
    """
    correspondenceDF = alignTask.dataSet.load_dataframe_from_csv(
        'neighbor_correspondences', alignTask)
    kept = correspondenceDF[correspondenceDF['kept']]
    if kept.empty:
        return

    dx = kept['measured_x'] - kept['nominal_x']
    dy = kept['measured_y'] - kept['nominal_y']
    directionColors = {'+x': 'tab:orange', '-x': 'tab:green',
                       '+y': 'tab:blue', '-y': 'tab:red'}

    fig, ax = plt.subplots(figsize=(6, 6))
    for direction, color in directionColors.items():
        mask = (kept['direction'] == direction).to_numpy()
        if mask.any():
            ax.scatter(dx[mask], dy[mask], s=14, alpha=0.6, color=color,
                      label=direction)
    ax.axhline(0, color='0.85', lw=0.8, zorder=0)
    ax.axvline(0, color='0.85', lw=0.8, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlabel('dx = measured - nominal (microns)')
    ax.set_ylabel('dy = measured - nominal (microns)')
    ax.set_title('%s: per-direction deviation (%i kept correspondences)'
                 % (alignTask.analysisName, len(kept)))
    ax.legend()
    alignTask.dataSet.save_task_figure(alignTask, fig, 'direction_reliability')
    plt.close(fig)


def plot_grid_overlay(alignTask) -> None:
    """Overlay each fov's corrected boundary on its original (nominal)
    boundary, at true physical scale -- mirrors the real-data comparison
    in the sibling MERci project's `notebooks/tests/
    compare_stitching_correction_methods.ipynb` (section 11), where a
    visible gap between the gray (original) and colored (corrected)
    squares IS the real, physical size of the correction.
    """
    fovs = alignTask.dataSet.get_fovs()
    nominalPositions = {f: tuple(alignTask.dataSet.get_fov_offset(f)) for f in fovs}
    correctedPositions = alignTask._load_corrected_positions()
    micronsPerPixel = alignTask.dataSet.get_microns_per_pixel()
    frameWidthUm = alignTask.dataSet.get_image_dimensions()[0] * micronsPerPixel
    half = frameWidthUm / 2

    nominalXY = np.array([nominalPositions[f] for f in fovs])
    correctedXY = np.array([correctedPositions[f] for f in fovs])
    shiftUm = np.hypot(correctedXY[:, 0] - nominalXY[:, 0],
                       correctedXY[:, 1] - nominalXY[:, 1])
    margin = frameWidthUm * 2

    fig, ax = plt.subplots(figsize=(8, 8))
    for x, y in nominalXY:
        ax.add_patch(mpatches.Rectangle(
            (x - half, y - half), frameWidthUm, frameWidthUm,
            fill=False, edgecolor='0.75', linewidth=0.6, zorder=1))
    for x, y in correctedXY:
        ax.add_patch(mpatches.Rectangle(
            (x - half, y - half), frameWidthUm, frameWidthUm,
            fill=False, edgecolor='tab:green', linewidth=0.6, zorder=2))

    ax.set_xlim(nominalXY[:, 0].min() - margin, nominalXY[:, 0].max() + margin)
    # Inverted, matching the reference plot's image-like display
    # convention (row 0 / smallest y at the top) -- a display choice
    # only, not a correctness one (unlike crop_overlap's row convention).
    ax.set_ylim(nominalXY[:, 1].max() + margin, nominalXY[:, 1].min() - margin)
    ax.set_aspect('equal')
    ax.set_xlabel('x (microns)')
    ax.set_ylabel('y (microns)')
    ax.legend(handles=[
        mpatches.Patch(facecolor='none', edgecolor='0.75',
                      label='original (nominal) fov boundary'),
        mpatches.Patch(facecolor='none', edgecolor='tab:green',
                      label='corrected fov boundary'),
    ], loc='upper right')
    ax.set_title(
        '%s: corrected vs. nominal fov grid (mean shift=%.3f um, max=%.3f um)'
        % (alignTask.analysisName, shiftUm.mean(), shiftUm.max()))
    alignTask.dataSet.save_task_figure(alignTask, fig, 'grid_overlay')
    plt.close(fig)


def _load_overlap_correlation_edges(alignTask) -> pd.DataFrame:
    """Kept correspondences' `correlation` column, collapsed to one row
    per physical edge. An interior fov's edge is measured -- and
    correlated -- once from each side (see
    `globalpositions.sample_neighbor_correspondences`'s own docstring),
    so without this collapse the same edge would be drawn/counted
    twice; averaging the (usually 2, occasionally 1) directions'
    correlation for a shared edge keeps one value per physical overlap.
    """
    correspondenceDF = alignTask.dataSet.load_dataframe_from_csv(
        'neighbor_correspondences', alignTask)
    kept = correspondenceDF[correspondenceDF['kept']]
    if kept.empty:
        return kept
    fovA = np.minimum(kept['anchor_fov'], kept['neighbor_fov'])
    fovB = np.maximum(kept['anchor_fov'], kept['neighbor_fov'])
    edgeDF = kept.assign(fov_a=fovA, fov_b=fovB)
    return edgeDF.groupby(['fov_a', 'fov_b'], as_index=False)['correlation'].mean()


def plot_overlap_correlation_grid(alignTask) -> None:
    """The corrected grid only (no nominal comparison, unlike
    `grid_overlay`), with a line between every pair of neighbouring
    fovs colored by the Pearson correlation of their overlap region AT
    the corrected positions -- a low-correlation edge is a direct,
    visual flag that the correction (or the underlying registration)
    may be wrong there, independent of the shift-magnitude diagnostics
    the other two figures show.
    """
    edgeDF = _load_overlap_correlation_edges(alignTask)
    if edgeDF.empty:
        return

    correctedPositions = alignTask._load_corrected_positions()
    fovs = alignTask.dataSet.get_fovs()
    micronsPerPixel = alignTask.dataSet.get_microns_per_pixel()
    frameWidthUm = alignTask.dataSet.get_image_dimensions()[0] * micronsPerPixel
    half = frameWidthUm / 2
    correctedXY = np.array([correctedPositions[f] for f in fovs])
    margin = frameWidthUm * 2

    fig, ax = plt.subplots(figsize=(8, 8))
    for x, y in correctedXY:
        ax.add_patch(mpatches.Rectangle(
            (x - half, y - half), frameWidthUm, frameWidthUm,
            fill=False, edgecolor='0.75', linewidth=0.6, zorder=1))

    cmap = plt.get_cmap('RdYlGn')
    norm = plt.Normalize(vmin=min(0.0, edgeDF['correlation'].min()), vmax=1.0)
    for row in edgeDF.itertuples():
        xa, ya = correctedPositions[row.fov_a]
        xb, yb = correctedPositions[row.fov_b]
        ax.plot([xa, xb], [ya, yb], color=cmap(norm(row.correlation)),
                linewidth=1.5, zorder=2)

    ax.set_xlim(correctedXY[:, 0].min() - margin, correctedXY[:, 0].max() + margin)
    # Same inverted-y, image-like display convention as grid_overlay.
    ax.set_ylim(correctedXY[:, 1].max() + margin, correctedXY[:, 1].min() - margin)
    ax.set_aspect('equal')
    ax.set_xlabel('x (microns)')
    ax.set_ylabel('y (microns)')
    ax.set_title('%s: corrected grid, neighbor overlap correlation (%i edges)'
                 % (alignTask.analysisName, len(edgeDF)))
    scalarMappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalarMappable.set_array([])
    fig.colorbar(scalarMappable, ax=ax, label='overlap correlation')
    alignTask.dataSet.save_task_figure(alignTask, fig, 'overlap_correlation_grid')
    plt.close(fig)


def plot_overlap_correlation_histogram(alignTask) -> None:
    """Histogram of the same per-edge overlap correlations drawn as
    lines in `overlap_correlation_grid` -- summarizes the whole grid's
    overlap agreement into one distribution.
    """
    edgeDF = _load_overlap_correlation_edges(alignTask)
    if edgeDF.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(edgeDF['correlation'], bins=30, color='tab:blue', edgecolor='white')
    ax.set_xlabel('overlap correlation')
    ax.set_ylabel('number of neighbor edges')
    ax.set_title('%s: overlap correlation distribution (%i edges, mean=%.3f)'
                 % (alignTask.analysisName, len(edgeDF), edgeDF['correlation'].mean()))
    alignTask.dataSet.save_task_figure(alignTask, fig, 'overlap_correlation_histogram')
    plt.close(fig)
