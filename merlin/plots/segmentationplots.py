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
        segmentTask = inputTasks['segment_task']
        featureDB = segmentTask.get_feature_database()

        fig = plt.figure(figsize=(15, 15))
        ax = fig.add_subplot(111)
        ax.set_aspect('equal', 'datalim')

        # Stream one fov at a time -- each fov's own boundaries (at that
        # fov's own per-cell occupied z, via read_feature_boundaries_at_
        # own_z(); see its docstring for why a single shared z-index isn't
        # enough) are plotted and discarded before the next fov is read,
        # so peak memory stays bounded by one fov's worth of polygons
        # regardless of how many fovs are in the dataset -- the whole
        # dataset's worth was previously held in memory at once (via
        # read_feature_boundaries_at_z()/read_features()), which OOM'd the
        # aggregate CellPoseSegmentSAMDone rule on large experiments (see
        # FINDINGS.md).
        anyFeatures = False
        for fov in segmentTask.dataSet.get_fovs():
            featuresForFov = featureDB.read_feature_boundaries_at_own_z(fov)
            featuresForFov = [x for y in featuresForFov for x in y]
            if len(featuresForFov) == 0:
                continue
            anyFeatures = True

            allCoords = [[feature.exterior.coords.xy[0].tolist(),
                          feature.exterior.coords.xy[1].tolist()]
                         for feature in featuresForFov]
            allCoords = [x for y in allCoords for x in y]
            plt.plot(*allCoords)

        if not anyFeatures:
            return fig

        plt.xlabel('X position (microns)')
        plt.ylabel('Y position (microns)')
        plt.title('Segmentation boundaries')
        return fig
