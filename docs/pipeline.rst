Analysis pipeline
*******************

Each analysis task in an analysis-parameters file (see :doc:`usage`) names the other
tasks it needs through its own parameters (for example ``preprocess_task`` or
``segment_task``); a task's ``get_dependencies()`` method turns those parameter values
into the task names :class:`~merlin.util.snakewriter.SnakefileGenerator` needs to build
the Snakemake workflow. The diagram below groups the pipeline into two main columns --
decoding (column 2) and segmentation (column 3) -- with every node in each main column
pointing back to only its own single, immediate previous/producing task, not the full
``get_dependencies()`` set a task class actually declares (e.g. ``Decode`` also needs
``DeconvolutionPreprocess`` and ``SimpleGlobalAlignment`` directly, which are omitted
from the main chain here). The remaining tasks sit in a side column to the left
(column 1: mosaic assembly and fixed-threshold barcode filtering) and one to the right
(column 4: ``SimpleGlobalAlignment`` and the sequential-signal branch); these side
branches still show each task's genuine dependencies. ``CellPoseSegmentSAM`` is shown
as the segmentation entry point rather than ``WatershedSegment`` or
``CellPoseSegment3D`` -- see :doc:`tasks` for every segmentation task's own parameters
and alternatives.

.. image:: _static/merlin_pipeline_flow.svg
   :width: 100%
   :alt: Flow diagram of the MERlin analysis pipeline's two main columns (decoding and
         segmentation), each node linked only to its own previous task

A few things worth noting explicitly:

* The repeated ``OptimizeIteration`` rounds a real analysis-parameters file chains via
  ``previous_iteration`` (``Optimize1`` -> ``Optimize2`` -> ...) collapse into a single
  ``OptimizeIteration`` node with a self-loop, rather than one node per round.
* ``FiducialCorrelationWarp`` sits at the top of column 3 (the segmentation column,
  alongside ``CellPoseSegmentSAM``) rather than above column 2, since decoding's own
  chain starts one step later at ``DeconvolutionPreprocess`` -- the ``warp_task`` edge
  into ``DeconvolutionPreprocess`` then runs right-to-left, from column 3 into column 2.
  ``SimpleGlobalAlignment`` sits the same way one column further right (column 4),
  directly beside ``CellPoseSegmentSAM``, with its ``global_align_task`` edge also
  running right-to-left.
* ``RefineCellDatabases`` (end of the segmentation column) feeds into
  ``PartitionBarcodes`` (``assignment_task``) alongside ``AdaptiveFilterBarcodes``
  (``filter_task``) from the decoding column -- this is where the two main columns
  converge, and the two nodes are kept at the same level.
* ``FilterBarcodes``/``ExportBarcodes`` (the alternative, fixed-threshold
  barcode-filtering strategy) branches off ``Decode`` into column 1, alongside
  ``GenerateMosaicTile``/``CombineMosaicTiles`` (mosaic assembly, off
  ``DeconvolutionPreprocess`` and ``CreateFfc``). ``ExportCellMetadata`` branches off
  ``RefineCellDatabases`` and stays in column 3; ``SumSignal``/``ExportSumSignals``
  also branch off ``RefineCellDatabases`` but sit in column 4, under
  ``SimpleGlobalAlignment``.

The diagram's source (``merlin_pipeline_flow.dot``, next to the rendered SVG in
``docs/_static/``) can be regenerated with Graphviz after edits:
``dot -Tsvg merlin_pipeline_flow.dot -o merlin_pipeline_flow.svg``.
