Analysis pipeline
*******************

Each analysis task in an analysis-parameters file (see :doc:`usage`) names the other
tasks it needs through its own parameters (for example ``preprocess_task`` or
``segment_task``); a task's ``get_dependencies()`` method turns those parameter values
into the task names :class:`~merlin.util.snakewriter.SnakefileGenerator` needs to build
the Snakemake workflow. The diagram below groups the pipeline into two main columns --
decoding and segmentation -- with every node in each main column pointing back to only
its own single, immediate previous/producing task, not the full ``get_dependencies()``
set a task class actually declares (e.g. ``Decode`` also needs ``DeconvolutionPreprocess``
and ``SimpleGlobalAlignment`` directly, which are omitted from the main chain here). The
side branches around the two main columns (mosaic assembly, fixed-threshold barcode
filtering, cell-metadata export, sequential signal) still show each task's genuine
dependencies. ``CellPoseSegmentSAM`` is shown as the segmentation entry point rather
than ``WatershedSegment`` or ``CellPoseSegment3D`` -- see :doc:`tasks` for every
segmentation task's own parameters and alternatives.

.. image:: _static/merlin_pipeline_flow.svg
   :width: 100%
   :alt: Flow diagram of the MERlin analysis pipeline's two main columns (decoding and
         segmentation), each node linked only to its own previous task

A few things worth noting explicitly:

* The repeated ``OptimizeIteration`` rounds a real analysis-parameters file chains via
  ``previous_iteration`` (``Optimize1`` -> ``Optimize2`` -> ...) collapse into a single
  ``OptimizeIteration`` node with a self-loop, rather than one node per round.
* ``RefineCellDatabases`` (end of the segmentation column) feeds into
  ``PartitionBarcodes`` (``assignment_task``) alongside ``AdaptiveFilterBarcodes``
  (``filter_task``) from the decoding column -- this is where the two main columns
  converge.
* ``FilterBarcodes``/``ExportBarcodes`` (the alternative, fixed-threshold
  barcode-filtering strategy) and ``ExportCellMetadata``/``SumSignal``/
  ``ExportSumSignals`` branch off ``Decode`` and ``RefineCellDatabases`` respectively,
  and ``GenerateMosaicTile``/``CombineMosaicTiles`` (mosaic assembly) branch off
  ``DeconvolutionPreprocess`` and ``CreateFfc`` -- all side branches, not part of
  either main column.

The diagram's source (``merlin_pipeline_flow.dot``, next to the rendered SVG in
``docs/_static/``) can be regenerated with Graphviz after edits:
``dot -Tsvg merlin_pipeline_flow.dot -o merlin_pipeline_flow.svg``.
