"""The sacct-fitted estimates (merlin.util.resourceestimate) must keep
covering the per-FOV maxima they were fit to, at least 5% over, once
snakewriter adds its margin, wrapper and rounding."""
import math
from unittest import mock

import pytest

from merlin.analysis import decode
from merlin.analysis import filterbarcodes
from merlin.analysis import generatemosaic
from merlin.analysis import optimize
from merlin.analysis import preprocess
from merlin.analysis import segment
from merlin.analysis import sequential
from merlin.util import snakewriter

LT066 = dict(nz=101, image=(2304, 2304), bits=26)
DISK = dict(nz=100, image=(2304, 2304), bits=16)
EPI = dict(nz=25, image=(2048, 2048), bits=16)


def _task(cls, geometry, parameters=None):
    """A task with only the attributes its estimates read."""
    task = cls.__new__(cls)
    task.dataSet = mock.Mock()
    task.dataSet.get_z_positions.return_value = [0.0] * geometry['nz']
    task.dataSet.get_image_dimensions.return_value = geometry['image']
    task.parameters = parameters or {}
    codebook = mock.Mock()
    codebook.get_bit_count.return_value = geometry['bits']
    task.get_codebook = lambda: codebook
    return task


def _requested_mb(task):
    return snakewriter.MEM_ROUNDING_MB * math.ceil(
        (task.get_estimated_memory() * snakewriter.RESOURCE_ESTIMATE_MARGIN
         + snakewriter.SNAKEMAKE_WRAPPER_MB) / snakewriter.MEM_ROUNDING_MB)


def _requested_minutes(task):
    return math.ceil(
        task.get_estimated_time() * snakewriter.RESOURCE_ESTIMATE_MARGIN
        + snakewriter.SNAKEMAKE_LAUNCH_MINUTES)


LATER = {'previous_iteration': 'Optimize01'}

# class, parameters, geometry, observed max RSS (MB), max elapsed (min)
OBSERVED = [
    (decode.Decode, {'decode_3d': False}, LT066, 5820, 180),
    (decode.Decode, {'decode_3d': False}, EPI, 6935, 179),
    (optimize.OptimizeIteration, {}, LT066, 4107, 3.2),
    (optimize.OptimizeIteration, {}, EPI, 2243, 4.0),
    (optimize.OptimizeIteration, LATER, LT066, 6226, 139),
    (optimize.OptimizeIteration, LATER, EPI, 7038, 158),
    (sequential.SmfishSignal, {}, DISK, 1004, 22.6),
    (sequential.SmfishSignal, {}, EPI, 1582, 25.4),
    (sequential.SumSignal, {}, DISK, 1084, 13.8),
    (sequential.SumSignal, {}, EPI, 650, 11.3),
    (segment.CleanCellBoundaries, {}, LT066, 7985, 90),
    (segment.CleanCellBoundaries, {}, EPI, 1342, 8.1),
    (segment.RefineCellDatabases, {}, LT066, 1742, 13.2),
    (segment.RefineCellDatabases, {}, EPI, 623, 5.4),
    (filterbarcodes.AdaptiveFilterBarcodes, {}, LT066, 2322, 2.9),
    (filterbarcodes.AdaptiveFilterBarcodes, {}, EPI, 732, 3.0),
    (generatemosaic.GenerateMosaicTile, {}, LT066, 561, 3.1),
    (generatemosaic.GenerateMosaicTile, {}, DISK, 496, 42.3),
    (preprocess.DeconvolutionPreprocess, {'decon_iterations': 0}, LT066,
     912, 33.8),
    (preprocess.DeconvolutionPreprocess, {'decon_iterations': 0}, EPI,
     774, 13.9),
]


@pytest.mark.parametrize('cls,parameters,geometry,rssMb,minutes', OBSERVED)
def test_estimate_covers_observed_max(
        cls, parameters, geometry, rssMb, minutes):
    task = _task(cls, geometry, parameters)
    assert cls.providesMemoryEstimate and cls.providesTimeEstimate
    assert _requested_mb(task) >= 1.05 * rssMb
    assert _requested_minutes(task) >= 1.05 * minutes


def test_z_scaled_estimates_grow_with_z():
    for cls in [sequential.SumSignal, segment.CleanCellBoundaries,
                segment.RefineCellDatabases,
                filterbarcodes.AdaptiveFilterBarcodes]:
        assert _task(cls, LT066).get_estimated_memory() \
            > _task(cls, EPI).get_estimated_memory()


def test_decode_3d_keeps_stack_scaling():
    task2d = _task(decode.Decode, LT066, {'decode_3d': False})
    task3d = _task(decode.Decode, LT066, {'decode_3d': True})
    assert task3d.get_estimated_memory() > task2d.get_estimated_memory()


@pytest.mark.parametrize('cls', [
    optimize.OptimizeIterationFOV, sequential.SmfishColocalizationSignal,
    filterbarcodes.AdaptiveFilterBarcodesLocal])
def test_unmeasured_subclasses_opt_out(cls):
    assert not cls.providesMemoryEstimate
    assert not cls.providesTimeEstimate


def test_cellpose_time_request_is_at_least_one_hour():
    task = _task(segment.CellPoseSegmentSAM, EPI,
                 {'channel_2_name': None, 'do_3D': True})
    assert _requested_minutes(task) == 60
