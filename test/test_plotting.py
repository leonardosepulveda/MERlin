import numpy as np

from merlin.analysis import testtask
from merlin.plots import testplots
from merlin import plots


def test_metadata(simple_merfish_data):
    randomTask = testtask.RandomNumberParallelAnalysisTask(simple_merfish_data)
    randomMetadata = testplots.TestPlotMetadata(randomTask,
                                                {'test_task': randomTask})
    assert not randomTask.is_complete()
    assert not randomMetadata.is_complete()
    assert randomMetadata.metadata_name() == 'testplots/TestPlotMetadata'

    for i in range(randomTask.fragment_count()-1):
        randomTask.run(i)
        randomMetadata.update()
        assert not randomTask.is_complete()
        assert not randomMetadata.is_complete()

    randomTask.run(randomTask.fragment_count()-1)
    randomMetadata.update()
    assert np.isclose(
        randomMetadata.get_mean_values(),
        np.array([np.mean(randomTask.get_random_result(i))
                  for i in range(randomTask.fragment_count())])).all()
    assert randomTask.is_complete()
    assert randomMetadata.is_complete()
    simple_merfish_data.delete_analysis(randomTask)


def test_plotengine(simple_merfish_data):
    randomTask = testtask.RandomNumberParallelAnalysisTask(simple_merfish_data)
    assert not randomTask.is_complete()

    plotEngine = plots.PlotEngine(randomTask, {'test_task': randomTask})
    assert len(plotEngine.get_plots()) == 1
    assert not plotEngine.take_step()
    randomTask.run(0)
    assert not plotEngine.take_step()

    for i in range(1, randomTask.fragment_count()):
        randomTask.run(i)
    assert plotEngine.take_step()
    assert plotEngine.get_plots()[0].is_complete()

    simple_merfish_data.delete_analysis(randomTask)


def test_radial_distribution_matches_per_row_distances():
    """The vectorized _radial_distribution must bin exactly like the old
    per-row (iterrows) loop, on float32 x/y as stored in the barcode
    database."""
    import pandas
    from merlin.plots.filterplots import FOVSpatialDistributionMetadata

    m = FOVSpatialDistributionMetadata.__new__(FOVSpatialDistributionMetadata)
    m._width, m._height = 2048, 2048
    m.radialBins = np.arange(0, 1024, 1024 / 200)
    rng = np.random.default_rng(0)
    barcodes = pandas.DataFrame({
        'barcode_id': rng.integers(0, 10, 100000).astype(np.uint16),
        'x': rng.uniform(0, 2048, 100000).astype(np.float32),
        'y': rng.uniform(0, 2048, 100000).astype(np.float32)})
    ids = [1, 3, 5]

    selected = barcodes[barcodes['barcode_id'].isin(ids)]
    expected = np.histogram(
        [m._radial_distance(r['x'], r['y'])
         for _, r in selected.iterrows()], bins=m.radialBins)[0]

    assert np.array_equal(m._radial_distribution(barcodes, ids), expected)
