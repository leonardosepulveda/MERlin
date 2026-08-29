"""Verification figures for merlin.analysis.warp's
FiducialCorrelationWarp, generated automatically once that task
completes (see AnalysisTask._generate_verification_figures's own
docstring for why this is a plain function here rather than
merlin.plots' AbstractPlot/PlotEngine classes -- it is a single task's
own quick self-check, not a plot built from several tasks together).
FiducialCorrelationWarp._generate_verification_figures just calls
generate_drift_qc(self).
"""

import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def _robust_max(values, pct: float = 99, floor: float = 10.0) -> float:
    """Outlier-resistant upper bound: the pct-th percentile of |values|,
    rounded up to a clean 1/2/5 x 10^k number. Adapted from the sibling
    fishtank project's `plot-drift` QC figure (`fishtank.pl.plot_drift`,
    PR jweissmanlab/fishtank#14), used there to keep a stray outlier from
    setting the axis scale for the whole figure.
    """
    v = np.abs(np.asarray(values, dtype=float))
    v = v[np.isfinite(v)]
    if v.size == 0:
        return floor
    hi = float(np.percentile(v, pct))
    if hi <= 0:
        return floor
    magnitude = 10 ** math.floor(math.log10(hi))
    for m in (1, 2, 5, 10):
        if hi <= m * magnitude:
            return float(m * magnitude)
    return float(10 * magnitude)


def _load_drift_dataframe(warpTask) -> pd.DataFrame:
    """Per-(fov, data channel) fiducial registration shift, in pixels,
    read back from the transformations warpTask already saved.
    """
    dataChannels = warpTask.dataSet.get_data_organization().get_data_channels()
    rows = [{'fov': fov, 'dataChannel': dc,
             'x_shift': t.translation[0], 'y_shift': t.translation[1],
             'distance': np.hypot(*t.translation)}
            for fov in warpTask.dataSet.get_fovs()
            for dc, t in zip(dataChannels, warpTask.get_transformation(fov))]
    return pd.DataFrame(rows)


def generate_drift_qc(warpTask) -> None:
    """Fiducial registration shift QC figure -- mirrors the sibling
    fishtank project's `plot-drift` QC figure (`fishtank.pl.plot_drift`,
    PR jweissmanlab/fishtank#14), built for the same kind of per-round
    fiducial registration.
    """
    driftDF = _load_drift_dataframe(warpTask)
    if driftDF.empty:
        return

    dataChannels = sorted(driftDF['dataChannel'].unique())
    channelIndex = {dc: i for i, dc in enumerate(dataChannels)}
    median = driftDF['distance'].median()

    scatterMax = _robust_max(np.concatenate(
        [driftDF['x_shift'].to_numpy(), driftDF['y_shift'].to_numpy()]))
    distanceMax = _robust_max(driftDF['distance'].to_numpy())

    fig, ax = plt.subplots(1, 3, figsize=(21, 6.5))
    fig.suptitle('%s: fiducial registration shift QC' % warpTask.analysisName)

    # (1) x vs y shift scatter, one point per (fov, data channel),
    # colored by data channel, on square symmetric axes so an outlier
    # does not set the scale for the whole panel.
    nOffScale = int(((driftDF['x_shift'].abs() > scatterMax)
                     | (driftDF['y_shift'].abs() > scatterMax)).sum())
    sc = ax[0].scatter(
        driftDF['x_shift'], driftDF['y_shift'],
        c=driftDF['dataChannel'].map(channelIndex), cmap='viridis',
        s=14, alpha=0.7, edgecolors='none')
    ax[0].axhline(0, color='k', lw=0.6, ls=':')
    ax[0].axvline(0, color='k', lw=0.6, ls=':')
    ax[0].set_xlim(-scatterMax, scatterMax)
    ax[0].set_ylim(-scatterMax, scatterMax)
    ax[0].set_aspect('equal')
    ax[0].set_xlabel('x shift (px)')
    ax[0].set_ylabel('y shift (px)')
    off = '  (%i outliers)' % nOffScale if nOffScale else ''
    ax[0].set_title('(1) registration shift vectors%s' % off)
    fig.colorbar(sc, ax=ax[0], label='data channel', fraction=0.046, pad=0.04)

    # (2) histogram of shift distance over [0, distanceMax], median marked.
    binSize = 5
    bins = np.arange(0, distanceMax + binSize, binSize)
    inRange = driftDF['distance'][driftDF['distance'] <= distanceMax]
    nOverMax = int((driftDF['distance'] > distanceMax).sum())
    ax[1].hist(inRange, bins=bins, color='slategray', edgecolor='white', lw=0.4)
    ax[1].axvline(median, color='crimson', lw=1.6, ls='--',
                  label='median = %.2f' % median)
    ax[1].set_xlim(0, distanceMax)
    ax[1].set_xlabel('shift distance (px)')
    ax[1].set_ylabel('count (fov x data channel)')
    over = '  (%i outliers)' % nOverMax if nOverMax else ''
    ax[1].set_title('(2) shift distance%s' % over)
    ax[1].legend(loc='best')

    # (3) heatmap of (shift distance - median): x = fov, y = data channel.
    pivot = driftDF.pivot_table(index='dataChannel', columns='fov',
                                values='distance', aggfunc='mean')
    pivot = pivot.reindex(sorted(pivot.columns), axis=1).sort_index()
    deviation = pivot.to_numpy() - median
    heatmapLimit = _robust_max(deviation, floor=1.0)
    im = ax[2].imshow(deviation, aspect='auto', origin='lower', cmap='coolwarm',
                      vmin=-heatmapLimit, vmax=heatmapLimit, interpolation='nearest')
    ax[2].set_xlabel('fov')
    ax[2].set_ylabel('data channel')
    ax[2].set_title('(3) shift distance - median (%.1f px)' % median)
    fovValues = pivot.columns.to_numpy()
    xTickIndex = np.linspace(0, len(fovValues) - 1, min(len(fovValues), 12)).astype(int)
    ax[2].set_xticks(xTickIndex)
    ax[2].set_xticklabels(fovValues[xTickIndex])
    dcTickIndex = np.linspace(0, len(dataChannels) - 1, min(len(dataChannels), 14)).astype(int)
    ax[2].set_yticks(dcTickIndex)
    ax[2].set_yticklabels(np.array(dataChannels)[dcTickIndex])
    fig.colorbar(im, ax=ax[2], label='distance - median (px)', fraction=0.046, pad=0.04)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    warpTask.dataSet.save_task_figure(warpTask, fig, 'drift_qc')
    plt.close(fig)
