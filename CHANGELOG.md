# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Versions are listed oldest first. Each feature/fix branch adds a line under
[Unreleased] at the end. A release moves those lines under a new version
heading, bumps `version` in `setup.py`, and tags the merge commit on master as
`vX.Y.Z`. Analyses whose saved `merlin_version` has a
different major number than the running MERlin fail to load
(`merlin.is_compatible`), so the major number only changes for a deliberate
break with existing analyses.

## [0.1.0] - 2019-09-30
### Added
- Iniatialization of this CHANGELOG file to track changes as the version increments

## [0.1.1] - 2019-10-03
### Fixed
- Fixed bug in sum signal

## [0.1.2] - 2019-10-16
### Added
- Exposed tolerance parameter in the adaptive filter barcodes method
- Added plot for scale factor magnitude vs bit index
- Fixed barcode partitioning to include cells from adjacent fields of view when a cell falls across fov boundaries

## [0.1.3] - 2019-12-04
### Fixed
- Addressed bugs present in cleaning overlapping cells and assigning them to a fov
### Added
- Added option to draw field of view labels overlaid on the mosaic

## [0.1.4] - 2019-12-05
### Added
- Added task to evaluate whether a parallel analysis task has completed
### Changed
- Changed the clean overlapping cells to run in parallel
- Snakemake job inputs were simplified using the ParallelCompleteTask to improve DAG construction speed and overall snakemake runtime performance

## [0.1.5] - 2020-01-22
### Changed
- Updated the filemap to only store the file name so that it can easily be pointed to new data home directories. This change maintains backward compatibility.
- Improved decoding speed
### Added
- Parameters to filter tasks that enable removing barcodes that were putatively duplicated across adjacent z planes. 

## [0.1.6] - 2020-04-17
### Fixed
- Fixed bug and edge cases in removal of barcodes duplicated across z planes. Moved to the decode step to prevent unintended conflict with misidentification rate determination.

### Added
- An alternative Lucy-Richardson deconvolution approach that requires ~10x fewer iterations.

## [0.2.0] - 2026-10-03
First release of the leonardosepulveda/MERlin fork since upstream 0.1.6. It
collects work by Xingjie Pan (2021-2022), aaron (2022-2026) and
leonardosepulveda (2023, 2026). Analyses saved by 0.1.6 stay loadable (same
major version).

### Added
- Cellpose segmentation: `CellPoseSegment3D` (one or two channels,
  downsampling, user-trained models) and `CellPoseSegmentSAM`. Spatial
  features store how many z planes carry a polygon.
- 3D acquisitions: `FiducialCorrelationWarp3D`, piezo correction, and
  `CAREPreprocess` denoising.
- Deconvolution with deconwolf (`DeconvolutionPreprocessDW`), resumable, plus
  an option not to save the pixel histogram.
- Decoding: GPU chunked decoding, masked decoding, resumable decoding,
  single-fov optimize/decode (`OptimizeIterationFOV`), chromatic correction
  from file, seeded or explicit fov/z choice and a distance threshold in
  Optimize, and local adaptive filtering (`GenerateAdaptiveThresholdLocal`,
  `AdaptiveFilterBarcodesLocal`).
- Zarr raw images for local files.
- smFISH with bigfish (`SmfishSignal`) and colocalization
  (`SmfishColocalizationSignal`). `SumSignal` takes `channel_names`.
- Mosaics as per-fov tiles combined into one image (`GenerateMosaicTile`,
  `CombineMosaicTiles`), with OME-TIFF output.
- Stitching: `RegisterFovNeighbors` and `LeastSquaresGlobalAlignment`, joint
  least-squares fov positions. `overlap_correlations: false` skips the QC
  correlation pass. With `max_projection_data_channel` set,
  RegisterFovNeighbors saves overlap edges for LSGA to reuse.
- `CreateFfc` flat-field correction and `RegistrationDiagnostics`.
- Fovs with different numbers of z planes (`--allow-ragged-z-stacks`) and
  missing channels (`--allow-missing-channels`).
- YAML (with comments) for analysis parameters (`-a`), microscope parameters
  (`-m`), cluster resource allocation and snakemake parameters (`-k`).
- Per-task memory and time estimates written into the Snakefile's
  `resources:`. A cluster-config entry still overrides them.
- Per-task verification figures (drift QC, overlap correlation, segmentation
  boundaries) and `--figures-path`.
- `--analysis-name`, `--parameters-home`, `--recalculate-filemap`, and
  absolute paths for `-m/-p/-a/-k`. Raw files are found in per-round
  subfolders.
- SLURM job names carry a per-experiment prefix (`job_name_prefix`).
- `python -m merlin.util.slurmlaunchdelay` measures the wait before each
  job's merlin step starts.

### Changed
- Snakemake 8+ API with snakemake-executor-plugin-slurm. Cluster settings
  become per-rule `resources:`, and the old `cluster` sbatch template is
  ignored. Snakemake keeps running unrelated jobs after a failure.
- Dependencies: setuptools 83 without pkg_resources, shapely 2, pandas 2,
  current numpy and tifffile.
- PartitionBarcodes is much faster and corrects molecules on cell boundaries.
- Spatial-feature extraction and global alignment run in parallel.
- FiducialCorrelationWarp upgraded, with fixes for edge artifacts and sparse
  beads.

### Removed
- `GenerateMosaic` (replaced by `GenerateMosaicTile` + `CombineMosaicTiles`).
- Unused segmentation classes, including the 2D `CellPoseSegment` variants.

### Fixed
- Many bugs found while running the lineage-tracing and breast-cancer
  experiments, including: filemap losing per-round subfolders, GenerateMosaic
  edge-tile overflow, global-alignment out-of-memory, non-rectangular fov
  grids in the neighbor search, smFISH crashes on empty boundaries,
  DeconvolutionPreprocess time estimates and TIFF writes, and SlurmReport
  without a task.json. See `git log --first-parent v0.1.6..v0.2.0`.

## [Unreleased]

### Fixed
- Verification figures no longer take hours on large experiments: the
  barcode radial-distribution metadata is vectorized (same bins, ~500x
  faster) and saved every 50 fovs, so a retry resumes. Segmentation-boundary
  reads are ~2x faster.
- CombineCleanedBoundaries no longer times out on large experiments. Three
  loops that grew with the square of the data size are now near-linear,
  with the same kept cells (ties are broken by node order). On LT066
  (1138 fovs, 2.05M cells), it takes 2 min and 2.5 GB, down from over 3 h
  at 100 GB.
- ExportBarcodes no longer runs out of memory: it reads and writes one
  fov at a time.
- Fov boxes (`GlobalAlignment.fov_global_extent`, `get_fov_boxes`) cover
  the whole image. They assumed 2048 px images, so on 2304 px images
  (LT066) they were 182.9 µm wide instead of 205.7 µm, anchored at pixel
  0, and missed most of the image-overlap band.
- Overlap removal no longer loses cells at fov seams (**breaking**: rerun
  CleanCellBoundaries before CombineCleanedBoundaries; graphs from older
  runs are rejected).
  - The old rule linked any two cells that overlapped at all, and slivers
    where outlines touch chained each seam into one component.
  - Cells are now linked only when they share at least
    `overlap_threshold` (0.5) of the smaller cell's volume.
  - Where cells conflict, the one farthest from its own image edge is
    kept, because a cell near the edge is often cut by the border.
  - On the 13 LT066 fovs whose 8 neighbours were all in a 33-fov test
    block, it keeps 14,510 cells instead of 12,756.
    - In the overlap bands it keeps 3,316 cells instead of 2,700.
    - Away from seams it keeps 99.8% of cells instead of 91.8%.
    - Cell density is now flat across the seams.
- Barcodes in the overlap of two fovs are no longer counted twice. Decode
  crops 100 px per edge, which on LT066 leaves 188 µm tiles on a 182 µm
  grid, so ~6 µm strips were decoded by both fovs, and PartitionBarcodes
  and ExportBarcodes counted them twice.
  - The filter tasks (FilterBarcodes, AdaptiveFilterBarcodes(Local)) now
    keep a barcode only in the fov whose image centre is nearest
    (`remove_overlap_duplicates`, default on). Fov positions come from the
    decode task's `global_align_task`.
  - On a 9-fov LT066 block this drops 5.5% of barcodes (4.7-6.0% per
    fov).
  - Takes effect when the filter task is rerun.
- z-duplicate removal (`remove_z_duplicated_barcodes`) is about 100x
  faster, with the same kept barcodes: an LT066 fov (2.4 M barcodes, 101
  planes) takes 8 s.
- AdaptiveFilterBarcodesLocal applies `remove_z_duplicated_barcodes`. It
  accepted the parameter but ignored it.

### Added
- Per-fov z offsets. Neighbouring fovs can disagree in z; on LT066 the
  same cell sits 1.57 µm higher in the right-hand fov and 0.54 µm higher
  in the upper fov.
  - CleanCellBoundaries measures the offset to each neighbour from
    duplicated cells (`max_z_shift_planes`, `min_seam_pairs`). It uses
    that offset when comparing their cells.
  - CombineCleanedBoundaries solves per-fov offsets into
    `fov_z_offsets.csv` (`get_fov_z_offsets()`): local z + offset is a
    common z.
    - Seams with too few cells to measure are filled from a plane fitted
      to the measured ones.
    - On the LT066 block, the solution fits the measured seams to
      0.24 µm rms.
  - PartitionBarcodes moves neighbouring fovs' barcodes into the current
    fov's planes before assigning them to cells (`apply_fov_z_offsets`,
    default on).
    - With a CombineCleanedBoundaries output from before this change, it
      warns and leaves z as it is.
  - Exported barcodes and cell boundaries still use each fov's own z.
- `z_duplicate_xy_distance_um` for AdaptiveFilterBarcodes: the
  z-duplicate radius in µm, used instead of
  `z_duplicate_xy_pixel_threshold` when set.
  - At 0.5 µm planes, one molecule is decoded on about 2.3 planes. On
    LT066 fov 786, 57% of barcodes have a same-gene barcode within
    0.27 µm on the next plane, against 0.014% for a different gene.
  - On spinning-disk data such as LT066, set
    `remove_z_duplicated_barcodes: true` and
    `z_duplicate_xy_distance_um: 0.27`. On a 9-fov block that leaves 1
    barcode per 2.28.

### Changed
- Verification figures are drawn by a separate `<Task>Figures` snakemake
  rule after each task is done (`merlin -t <Task> --figures-only`), not
  inside the task's run or its Done rule. Nothing depends on the Figures
  rule, so slow or failing figures no longer delay or stop the pipeline.
  Its cluster resources can be set per rule, e.g. `DecodeFigures:`.
  Running a task with `merlin -t` alone no longer draws its figures.
- `SegmentationBoundaryPlot.svg` boundaries are simplified to 0.5 µm and
  written with 0.1 µm precision, about 40x smaller.
- ExportBarcodes writes `barcodes.parquet` instead of `barcodes.csv` by
  default (**breaking** for anything that reads the csv). On LT066 (2.0
  billion barcodes), the csv would have been ~105 GB and taken ~7 h to
  write. The task now takes 29 min and writes a 35 GB parquet.
  - Set `"format": "csv"` in its parameters to keep writing
    `barcodes.csv`. That file is identical to before, and it is also
    written one fov at a time.
