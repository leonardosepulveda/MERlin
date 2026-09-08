Analysis pipeline
*******************

Each analysis task in an analysis-parameters file (see :doc:`usage`) names the other
tasks it needs through its own parameters (for example ``preprocess_task`` or
``segment_task``); a task's ``get_dependencies()`` method turns those parameter values
into the task names :class:`~merlin.util.snakewriter.SnakefileGenerator` needs to build
the Snakemake workflow. The diagram below renders that dependency graph for
``test/auxiliary_files/test_analysis_parameters.json`` -- the analysis-parameters file
the test suite itself runs the full pipeline against (see ``test/conftest.py`` and
``test/test_merfish.py``) -- as a default, always-up-to-date reference for how
MERlin's tasks fit together. Every edge below is read directly from the corresponding
task class's own ``get_dependencies()`` in ``merlin/analysis/*.py``, not just inferred
from parameter names. A real experiment's own analysis-parameters file will typically
use only part of this graph (e.g. one of the two barcode-filtering strategies shown,
or ``CellPoseSegmentSAM``/``CellPoseSegment3D`` in place of ``WatershedSegment`` -- see
:doc:`tasks` for every task's own parameters and alternatives).

.. image:: _static/merlin_pipeline_flow.svg
   :width: 100%
   :alt: Flow diagram of the MERlin default analysis pipeline's task dependency graph

Two branches worth noting explicitly:

* ``FilterBarcodes`` and ``GenerateAdaptiveThreshold``/``AdaptiveFilterBarcodes`` are
  two independent, alternative barcode-filtering strategies (fixed-threshold vs.
  adaptive-threshold) -- both depend only on ``Decode``, not on each other. The test
  fixture runs both to exercise each in CI; a real pipeline normally picks one.
* ``GenerateMosaicTile``/``CombineMosaicTiles`` (mosaic assembly) and the
  segmentation/partitioning/sequential-signal branches all depend only on
  ``FiducialCorrelationWarp`` and ``SimpleGlobalAlignment`` -- they run independently
  of the decoding branch and of each other.

The diagram's source (``merlin_pipeline_flow.dot``, next to the rendered SVG in
``docs/_static/``) can be regenerated with Graphviz after edits:
``dot -Tsvg merlin_pipeline_flow.dot -o merlin_pipeline_flow.svg``.
