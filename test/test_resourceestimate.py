from merlin.util import resourceestimate


class _FakeDataSet:
    """Stub exposing just what estimate_stack_memory_mb needs."""

    def __init__(self, width, height):
        self._dims = (width, height)

    def get_image_dimensions(self):
        return self._dims


def test_estimate_stack_memory_mb_scales_with_frame_count():
    dataSet = _FakeDataSet(2048, 2048)
    one = resourceestimate.estimate_stack_memory_mb(
        dataSet, frameCount=1, kTask=2, baselineMb=100)
    ten = resourceestimate.estimate_stack_memory_mb(
        dataSet, frameCount=10, kTask=2, baselineMb=100)
    # baseline stays fixed; only the frame-count-scaled part grows 10x
    assert one == 100 + (ten - 100) / 10
    assert ten > one


def test_estimate_stack_memory_mb_downsample_shrinks_estimate():
    dataSet = _FakeDataSet(2048, 2048)
    full = resourceestimate.estimate_stack_memory_mb(
        dataSet, frameCount=4, downsampleFactor=1, kTask=2, baselineMb=100)
    downsampled = resourceestimate.estimate_stack_memory_mb(
        dataSet, frameCount=4, downsampleFactor=4, kTask=2, baselineMb=100)
    # halving both x and y per downsampleFactor=4 -> 1/16th the pixels
    assert downsampled == 100 + (full - 100) / 16


def test_estimate_stack_memory_mb_matches_hand_calculation():
    dataSet = _FakeDataSet(100, 200)
    result = resourceestimate.estimate_stack_memory_mb(
        dataSet, frameCount=3, downsampleFactor=1, kTask=5, baselineMb=50)
    frameBytes = 100 * 200 * resourceestimate.BYTES_PER_PIXEL
    expected = 50 + (frameBytes * 3 * 5) / 1e6
    assert result == expected


def test_estimate_stack_time_minutes_scales_with_frame_count():
    one = resourceestimate.estimate_stack_time_minutes(
        frameCount=1, secondsPerFrame=60, baselineMinutes=2)
    ten = resourceestimate.estimate_stack_time_minutes(
        frameCount=10, secondsPerFrame=60, baselineMinutes=2)
    assert one == 3  # 2 + 60/60
    assert ten == 12  # 2 + 600/60
