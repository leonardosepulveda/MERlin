"""Shared math for the small set of AnalysisTask subclasses whose
get_estimated_memory()/get_estimated_time() give a real, geometry-driven
estimate (providesResourceEstimate = True) instead of an unused
placeholder constant. See merlin.core.analysistask.AnalysisTask
.providesResourceEstimate for the opt-in contract, and
merlin.util.snakewriter for how the estimate is turned into a cluster
resource request (a safety margin is applied there, not here -- these
functions return the raw estimate).

Every constant below is a first-pass, deliberately conservative guess
calibrated against at most one real measurement (see each call site's own
docstring for which, if any) -- not a validated model. Tighten them
against real measured jobs as they become available rather than trusting
these numbers.
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
