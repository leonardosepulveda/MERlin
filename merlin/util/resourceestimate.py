"""Shared math for the AnalysisTask subclasses whose
get_estimated_memory()/get_estimated_time() give a real estimate
(providesMemoryEstimate/providesTimeEstimate = True) instead of an unused
placeholder constant. See merlin.core.analysistask.AnalysisTask
.providesMemoryEstimate for the opt-in contract, and
merlin.util.snakewriter for how the estimate is turned into a cluster
resource request (a safety margin is applied there, not here -- these
functions return the raw estimate).

Calibration (2026-10-13). The estimates marked "fit to sacct" at their
call sites were fit to every completed SLURM job of five runs from
2026-08-28 to 2026-10-13: LT066 (101 z, 2304x2304, 26 bits), BC553 and
BC555 disk (100 z, 2304x2304) and epi (25 z, 2048x2048, 16 bits). Only
jobs that ran the task's current code were used, unless a call site says
otherwise. Each estimate is set so that estimate x RESOURCE_ESTIMATE_MARGIN
plus the wrapper covers, with at least 5% to spare, the largest per-FOV
max RSS and elapsed time of every run. That is the slowest FOV, not the
typical one, because snakemake requests one size for all of a rule's
jobs. A rare FOV beyond it gets 2x on its retry (snakewriter
.AttemptScaled).

The predictor is chosen from what the task holds in memory: frame size
for one-frame-at-a-time tasks, z-plane count for tasks that hold a FOV's
cell boundaries or barcodes at every z. Where no known property ordered
the runs (e.g. Decode is heaviest on BC555 epi, which has the fewest z
planes and bits), the estimate is a constant at the max over runs.
"""

BYTES_PER_PIXEL = 2  # raw MERFISH frames are read as uint16 (imagereader.py)


def estimate_stack_memory_mb(
        dataSet, frameCount: float, downsampleFactor: float = 1,
        kTask: float = 3.0, baselineMb: float = 300) -> float:
    """A memory estimate (megabytes) for a task that holds up to
    `frameCount` same-size frames at once (each downsampled in x/y by
    `downsampleFactor`), plus a fixed per-process baseline.

    Args:
        dataSet: used only for get_image_dimensions().
        frameCount: how many raw-frame-sized buffers are held
            concurrently -- typically channelCount, or channelCount *
            zCount for a task that holds a full z-stack at once.
        downsampleFactor: side-length downsampling applied before
            processing (e.g. CellPoseSegmentSAM's downsample_factor);
            1 means no downsampling.
        kTask: multiplier over the raw uint16 frame bytes accounting for
            dtype promotion (float32/64), duplicate/working buffers, and
            library-internal overhead (FFT plans, filter kernels, model
            activations, etc.) -- task-specific, see each call site.
        baselineMb: fixed overhead independent of frame count (Python +
            imported libraries + any loaded model weights).
    """
    width, height = dataSet.get_image_dimensions()
    framePixels = (width / downsampleFactor) * (height / downsampleFactor)
    frameBytes = framePixels * BYTES_PER_PIXEL
    return baselineMb + (frameBytes * frameCount * kTask) / 1e6


def estimate_stack_time_minutes(
        frameCount: float, secondsPerFrame: float,
        baselineMinutes: float = 2.0) -> float:
    """A wall-clock estimate (minutes) for a task that processes
    `frameCount` frames sequentially at roughly `secondsPerFrame` seconds
    each, plus a fixed startup baseline (imports, model loading, etc.).

    No task's actual per-frame processing time has been measured yet --
    secondsPerFrame is a pure, task-specific guess at each call site.
    """
    return baselineMinutes + (frameCount * secondsPerFrame) / 60
