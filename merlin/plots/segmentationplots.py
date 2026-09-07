import os

from matplotlib import pyplot as plt

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
        dataSet = segmentTask.dataSet
        featureDB = segmentTask.get_feature_database()

        # Both of this method's callers (AbstractPlot.plot(), used by the
        # separate PlotPerformance task, and AnalysisTask.
        # _generate_plots_for_role(), which runs this inline once
        # segment_task itself completes -- the path that OOM'd
        # RefineCellDatabasesDone/CellPoseSegmentSAMDone on large
        # experiments, see FINDINGS.md's 2026-08-30/2026-09-06/
        # 2026-09-07 entries) save whatever plt.Figure they're handed
        # only once, at the very end. That means any matplotlib-artist
        # approach -- one Line2D per polygon (the original code here) or
        # a single LineCollection spanning every fov -- still needs
        # every fov's boundaries resident in the Figure simultaneously
        # right up to that final save, so peak memory scales with
        # dataset size either way. Writing each fov's polygons as plain
        # svg <polyline> elements straight to an open file handle and
        # discarding them before the next fov is read keeps peak memory
        # bounded to one fov's worth of polygons regardless of dataset
        # size, while keeping the output real vector paths (no
        # rasterization). The Figure actually returned here, to satisfy
        # the callers' contract, is just a small placeholder pointing at
        # the svg.
        svgPath = self._svg_path(dataSet, segmentTask)
        self._write_boundaries_svg(svgPath, dataSet, featureDB)

        fig = plt.figure(figsize=(6, 1.5))
        fig.text(0.02, 0.5,
                 'Segmentation boundaries written to\n%s' % svgPath,
                 va='center', fontsize=8)
        return fig

    def _svg_path(self, dataSet, segmentTask) -> str:
        subdirectory = type(self).__module__.split('.')[-1]
        return os.path.join(
            dataSet.get_analysis_subdirectory(segmentTask, subdirectory),
            self.figure_name() + '.svg')

    def _write_boundaries_svg(self, svgPath, dataSet, featureDB) -> None:
        fovs = dataSet.get_fovs()

        # The svg's viewBox comes from the fov grid itself (nominal
        # stage position + frame size), not from the feature boundaries
        # -- cheap (no feature reads needed to compute it) and a safe
        # superset, since a fov's own cells can't extend past that
        # fov's own image extent.
        micronsPerPixel = dataSet.get_microns_per_pixel()
        frameWidthUm, frameHeightUm = (
            d * micronsPerPixel for d in dataSet.get_image_dimensions())
        offsets = [dataSet.get_fov_offset(f) for f in fovs]
        minX = min(x for x, y in offsets) - frameWidthUm / 2
        maxX = max(x for x, y in offsets) + frameWidthUm / 2
        minY = min(y for x, y in offsets) - frameHeightUm / 2
        maxY = max(y for x, y in offsets) + frameHeightUm / 2
        margin = 0.02 * max(maxX - minX, maxY - minY)
        minX, maxX = minX - margin, maxX + margin
        minY, maxY = minY - margin, maxY + margin

        with open(svgPath, 'w') as svgFile:
            svgFile.write(
                '<svg xmlns="http://www.w3.org/2000/svg" '
                'viewBox="%.2f %.2f %.2f %.2f">\n'
                '<title>Segmentation boundaries</title>\n'
                % (minX, -maxY, maxX - minX, maxY - minY))

            # Stream one fov at a time -- each fov's own boundaries (at
            # that fov's own per-cell occupied z, via
            # read_feature_boundaries_at_own_z(); see its docstring for
            # why a single shared z-index isn't enough) are written out
            # and discarded before the next fov is read.
            for fov in fovs:
                featuresForFov = featureDB.read_feature_boundaries_at_own_z(fov)
                featuresForFov = [x for y in featuresForFov for x in y]
                for feature in featuresForFov:
                    xCoords, yCoords = feature.exterior.coords.xy
                    points = ' '.join(
                        '%.2f,%.2f' % (x, -y)
                        for x, y in zip(xCoords, yCoords))
                    svgFile.write(
                        '<polyline points="%s" fill="none" '
                        'stroke="#1f77b4" stroke-width="0.75" '
                        'vector-effect="non-scaling-stroke"/>\n' % points)

            svgFile.write('</svg>\n')
