from matplotlib import pyplot as plt
import numpy as np

from merlin.plots._base import AbstractPlot


class SegmentationBoundaryPlot(AbstractPlot):

    def __init__(self, analysisTask):
        super().__init__(analysisTask)

    def get_required_tasks(self):
        return {'segment_task': 'all'}

    def get_required_metadata(self):
        return []

    def _generate_plot(self, inputTasks, inputMetadata):
        featureDB = inputTasks['segment_task'].get_feature_database()

        # Only read the single z-plane actually plotted below, instead of
        # every z-plane of every feature in the dataset (read_features()) --
        # for a whole-dataset plot like this one, that is ~20-25x less
        # polygon data to deserialize and hold in memory at once.
        zCount = featureDB.get_feature_z_count()
        zPosition = 0
        if zCount > 1:
            zPosition = int(zCount / 2)

        featuresSingleZ = featureDB.read_feature_boundaries_at_z(zPosition)
        featuresSingleZ = [x for y in featuresSingleZ for x in y]

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', 'datalim')

        if len(featuresSingleZ) == 0:
            return fig

        allCoords = [[feature.exterior.coords.xy[0].tolist(),
                      feature.exterior.coords.xy[1].tolist()]
                     for feature in featuresSingleZ]
        allCoords = [x for y in allCoords for x in y]
        plt.plot(*allCoords)

        plt.xlabel('X position (microns)')
        plt.ylabel('Y position (microns)')
        plt.title('Segmentation boundaries')
        return fig
