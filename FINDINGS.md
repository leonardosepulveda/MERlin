# FINDINGS.md

Curated current-state summary. See `prompt_history/` for full provenance of each item
below; this file only tracks what's true *now* and the open next step.

## LeastSquaresGlobalAlignment OOM on BC553_sample_02/epi; stopgap applied, refactor in progress (2026-09-03)

`LeastSquaresGlobalAlignment` (`merlin/analysis/globalalign.py`) OOM-killed on every one
of 6 attempts today against real `BC553_sample_02/epi` (jobs 44117159, 44122230,
44129463, 44236639, 44239102, 44239650 -- `sacct` on the last: `ReqMem 8000M`, `MaxRSS`
~7710 MB at kill, step `.0` `OUT_OF_MEMORY`). Not a code-error crash.

**Root cause**: this task extends plain `AnalysisTask` (not `ParallelAnalysisTask`), so
it runs as a single job over the whole experiment's 1651 fovs at once. Its
`_run_analysis` calls `merlin.util.globalpositions.sample_neighbor_correspondences` and
`compute_overlap_correlations`, each of which caches every loaded fiducial frame in a
plain dict (`frameCache`) for the life of the call with no eviction -- by the end of the
anchor loop, nearly all 1651 fovs' frames (2048x2048 uint16, 8.389 MB each, confirmed
against a real raw `.zarr` frame) sit in memory simultaneously: ~13.8 GB, well past the
8000 MB `__default__` this task fell back to (it has no `providesMemoryEstimate` opt-in;
`get_estimated_memory()` is still `SimpleGlobalAlignment`'s hardcoded, unused `1000`
placeholder). Also affects the second `compute_overlap_correlations` pass identically.

**Stopgap applied** (unblocks this run only, no repo code touched): forced
`LeastSquaresGlobalAlignment: {mem: 17000}` in production's own
`cluster_resource_allocation_BC553_sample_02.yaml` (1651 * 8.389 MB + 300 MB baseline,
x1.2 margin, rounded up to the nearest GB), comment added in-file. Confirmed the live
`parameters_BC553_sample_02.json`'s `cluster_config` points at this `.yaml`, not the
sibling (unread, historical) `.json` of the same name. Resubmitted directly via `sbatch`
(job 44267625, `-t LeastSquaresGlobalAlignment` single-task invocation, `--mem 17000`)
rather than through the full snakemake pipeline, since a `PlotPerformance` job
(44232883) was already independently running against this experiment at investigation
time and the top-level pipeline driver's live state was unclear -- lets this task's
`.done` marker satisfy any concurrent/future automatic rerun without resubmitting it
again.

**Stopgap confirmed successful**: job 44267625 `COMPLETED` in 13m53s, `sacct` `MaxRSS`
14453812K (~14.45 GB) -- within the 4% of the 13.85 GB raw (pre-margin) estimate above,
confirming the frame-cache-size root-cause model directly. Real output: 6370
correspondences (6328 kept, 42 rejected), 1 connected component, `residual_rms_um`
0.168 -- a clean, healthy result, `LeastSquaresGlobalAlignment.done` written.

**Real fix landed** (branch `fix/globalalign-oom-and-parallelize`, not yet merged/deployed
to any production experiment -- BC553_sample_02/epi's own run above used only the
stopgap, not this): split into `RegisterFovNeighbors` (new `ParallelAnalysisTask`, one
fragment per fov, pairwise neighbor registration -- bounds each fragment to at most the
anchor + one neighbor frame, not the whole dataset) plus `LeastSquaresGlobalAlignment`
(kept as the reduce step -- same name/output paths as before, so nothing downstream that
already reads its outputs needs to change) doing only the joint least-squares fit, which
inherently cannot be parallelized further since it needs every fov's correspondences at
once -- mirrors the existing `CleanCellBoundaries` [per-fov] -> `CombineCleanedBoundaries`
[reduce] pattern in `segment.py`. `LeastSquaresGlobalAlignment` gained a
`neighbor_registration_task` parameter (defaults to `'RegisterFovNeighbors'`, so any
existing config with no parameters at all -- e.g. BC553_sample_02's own -- keeps working
once the new task is added to its `merlin_analysis_*.json`/`.yaml` task list). Also fixed
the same unbounded-cache bug at its root in `globalpositions.py` (both
`sample_neighbor_correspondences` and `compute_overlap_correlations` now share a bounded
`_BoundedFrameCache`, maxsize 8). Both new/changed tasks opted into real
`get_estimated_memory()`/`get_estimated_time()` formulas (`providesMemoryEstimate`/
`providesTimeEstimate = True`, via `merlin.util.resourceestimate`) instead of the old
placeholder constants -- explicitly marked uncalibrated (first-pass guesses) pending a
real measured job on the new task shape. `test_globalalign.py` updated and passing
(existing suite otherwise unaffected -- verified against a clean `master` checkout that
the handful of unrelated `test_core.py`/`test_dataset.py`/`test_snakemake.py` failures
seen in this session are pre-existing test-isolation flakiness, not a regression).

**Update (2026-09-04)**: this branch being checked out in `merlin_cc_env`'s editable
install turned out to deploy itself involuntarily -- any `merlin` invocation on the
cluster runs whatever's checked out here, config or no. A manual resubmit of
BC553_sample_02/epi (job 44403135, 12:07-12:11) crashed at Snakefile generation with
`FileNotFoundError` on `RegisterFovNeighbors/tasks/task.json`, since the live
`merlin_analysis_BC553_sample_02.yaml` (`merlin/analysis/`, not the stale
`parameters_home/analysis/` copy) had no entry for the new task yet. Fixed by adding
the `RegisterFovNeighbors` entry there, then reconciling the stopgap-run's
`LeastSquaresGlobalAlignment/tasks/task.json` (old pre-split parameter shape) to the
new one in place (backed up alongside it) so the already-completed, already-validated
result and everything downstream stayed intact rather than being invalidated for a
recompute. Full writeup: `prompt_history/
2026_09_04_1225_fix_bc553_registerfovneighbors_deploy_crash.md`. Still not done: the
now-oversized 17 GB forced `mem` override for `LeastSquaresGlobalAlignment` in that
experiment's `cluster_resource_allocation` yaml (harmless while its `.done` marker
stands, but worth relaxing eventually); deploying to any other production experiment;
merging the branch to `master`; the MERci handoff below.

**Not yet done (as of 2026-09-03)**: deploying this to BC553_sample_02/epi's own config
(today's run is already unblocked by the stopgap alone; the new task list entry can be
added on some future rerun rather than disrupting the currently-running production
pipeline) or to any other production experiment; merging the branch to `master`; the
MERci handoff below.

**Handoff to MERci**: needed because MERci's `create_cluster_resource_allocation()`
generates these per-experiment yaml configs for new experiments (same generator flagged
in the entry below) and its own analysis-parameter-recipe templates reference
`LeastSquaresGlobalAlignment` by name as `global_align_task` -- both need to learn about
the new `RegisterFovNeighbors` task (a new top-level entry in the generated
`merlin_analysis_*.json`/`.yaml` task list, plus its own `cluster_resource_allocation`
entry) now that its name is final. Logged in MERci's own `prompt_history/`, dated
2026-09-03 (status: pending).

## YAML support for cluster_resource_allocation configs; BC555_sample_05 converted (2026-08-31)

`merlin.merlin._load_json_or_yaml` (new, shared with the existing analysis-parameter
recipe loader) lets a `cluster_config` path be `.yaml`/`.yml`, parsed with native `#`
comments -- `.json` still parses exactly as before. This is what makes a "commented-out
calculated value, uncomment to override" workflow possible at all, since plain JSON has
no comment syntax.

Converted both real `BC555_sample_05` `cluster_resource_allocation_BC555_sample_05.json`
files (`epi`/`disk`, on shared lab storage, not in this repo) to `.yaml` in this format:
a task with a real, validated `get_estimated_memory()`/`get_estimated_time()` (currently
`FiducialCorrelationWarp` mem+time, `DeconvolutionPreprocess` mem+time,
`CellPoseSegmentSAM` mem only) has its active `mem`/`time` key commented out with a
`# calculated ...` reference line (rounded up to the nearest whole GB / 5 minutes, per
the user's convention); everything else keeps a plain active value unchanged.

**Found and fixed a real gap while verifying end-to-end, not just round-tripping the
YAML**: `Decode` on `epi` has `providesTimeEstimate = True` but was never validated (see
the per-fov-table entry above) and had no explicit JSON `time` override -- leaving it
untouched would have let the unvalidated formula silently activate on restart (confirmed
via a real generated Snakefile: `runtime=6` instead of the safe 180). A systematic check
across every task in both configs for "`providesEstimate=True`, no override, not
validated" found this as the only instance; pinned it to an explicit `time: '3:00:00'`
(its existing, safe `__default__`-inherited value) with an explanatory comment.

Both conversions verified twice: a round-trip parse check, and the actual end-to-end
`parameters_BC555_sample_05.json` -> `_load_json_or_yaml` -> Snakefile-generation path
run for real against both experiments' live datasets -- generated `resources:` blocks
match the intended rounded/margined values exactly. `parameters_BC555_sample_05.json`
(both runs) updated to point `cluster_config` at the new `.yaml`; the original `.json`
files were left in place, untouched, as an unreferenced historical backup.

**Handoff to MERci**: logged in MERci's own `prompt_history/`, dated 2026-08-31 15:31
(status: pending) -- asks `create_cluster_resource_allocation()` to emit this same format
for future experiments. Flags an open gap worth resolving on either side: there's no
code-level way today to distinguish "has a formula" from "formula is actually
calibrated" -- both are just `providesMemoryEstimate`/`providesTimeEstimate = True`.

**Not done**: the calibrated-vs-unvalidated code-level gap itself (flagged, not built);
MERci's generator changes (their side). Full detail in this project's own request-history
log, dated 2026-08-31 15:31.

## DeconvolutionPreprocess time estimate recalibrated before a real restart (2026-08-31)

`feature/estimated-cluster-resources` merged to `master`. Before recommending the user
restart `BC555_sample_05` jobs with the unchanged JSON config, checked what would actually
change: **memory is unaffected** for all 4 opted-in tasks (each already has its own
explicit `mem` entry in `cluster_resource_allocation_BC555_sample_05.json`, which still
overrides the computed estimate). **Time is not** -- `FiducialCorrelationWarp`,
`DeconvolutionPreprocess`, and `Decode` have no `time` entry of their own (only
`__default__`'s blanket `3:00:00`), so the computed estimate would take effect for them
on restart.

Checked each against the real per-fov time data before calling it safe.
`FiducialCorrelationWarp`'s margined estimate covers the real max on both runs -- safe,
unchanged. **`DeconvolutionPreprocess`'s did not** -- both real runs use
`decon_iterations: 0`, which zeroed out the formula's entire per-frame term, leaving only
a flat 2-minute baseline guess against a real median of 3.35 min (epi) -- would have
under-provisioned most fovs. Recalibrated the same way as the memory fix: two real data
points (epi 400 bit-z frames/4.65 min max, disk 1600/6.53 min max) solved jointly for a
real flat per-frame cost (0.094 sec/frame) and baseline (4 min); the existing
`decon_iterations * 0.2 sec` term stays as an uncalibrated additive extra for nonzero-
iteration configs (no real data exists there). New margined estimate now covers the real
max on both runs (5.55 min epi vs 4.65 min real max; 7.81 min disk vs 6.53 min real max).

**`Decode`'s time estimate is still entirely uncalibrated** (0/476 fragments complete on
`epi`, no `Decode` task on `disk` at all) -- flagged to the user as an open risk if they
restart before real `Decode` data exists; suggested an explicit JSON `time` override as a
stopgap.

**Not done**: `FiducialCorrelationWarp`'s time left un-tightened (safe, just generous);
`Decode`'s time uncalibrated. Full detail in this project's own request-history log,
dated 2026-08-31 15:05.

## DeconvolutionPreprocess/CellPoseSegmentSAM memory recalibrated from real data (2026-08-31)

Follow-up on the entry below (real per-fov calibration comparison): recalibrated both
constants that were found too low, fitting to each run's real **max** observed fov (not
median) -- a generated Snakefile rule applies one static `mem_mb` to every fragment of a
task, so fitting to the median would leave denser-than-typical fovs under-provisioned,
exactly the failure this recalibration fixes.

- `DeconvolutionPreprocess`: one clean real data point (`epi`, 476 fovs, real max
  695.91 MB). `baselineMb` fixed at 230 (the same measured "import merlin alone" baseline
  used for `FiducialCorrelationWarp` -- one data point can't independently solve two free
  parameters). Solved `kTask = 56` (was an unvalidated guess of 15). New raw estimate for
  `epi`: 700 MB (margined 840 MB) -- covers the real max, still well under the static
  JSON's 3000 MB.
- `CellPoseSegmentSAM`: two clean real data points at different z-depths (`epi`: 25 z,
  real max 3678.80 MB; `disk`: 100 z, `do_3D: true`, real max 9741.16 MB, only 71/603
  fovs completed so far -- provisional) -- enough to solve `baselineMb`/`kTask` jointly:
  `baselineMb = 2190`, `kTask = 114` (was an unvalidated guess of 3000/20). New raw
  estimates exactly reproduce both real maxes; margined, `epi` = 4421 MB, `disk` =
  11705 MB -- both far below the static JSON's blanket 32000 MB for this task, while now
  safely covering the worst fov actually observed (unlike the un-calibrated version,
  which fell short of even the *median*).

Overshoot vs. the typical (median) fov -- the real cost of fitting to the max: `epi`
`CellPoseSegmentSAM` 1.06x (tight), `disk` `CellPoseSegmentSAM` 1.82x (`disk`'s fovs vary
more in cell density). Verified via the actual `get_estimated_memory()` call against the
real dataset, not a hand recomputation.

**Not done**: `CAREPreprocess`/`Decode` remain uncalibrated (no real data exists for
either yet); `disk`'s `CellPoseSegmentSAM` fit is provisional (71/603 fovs); the
2-channel (`channel_2_name` set) case is unverified (both real runs use 1 channel); the
time-estimate wiring is still not recommended for merging (separate, still fully
uncalibrated). Full detail in this project's own request-history log, dated 2026-08-31
14:32.

## SlurmReport per-fov table: validated against real BC555_sample_05, formula gaps found (2026-08-31)

## SlurmReport per-fov table: validated against real BC555_sample_05, formula gaps found (2026-08-31)

Added `SlurmReport._save_per_fov_resource_table()` (parquet, one row per fov, two
columns per `ParallelAnalysisTask`) and ran it for real against `BC555_sample_05`
`epi`/`disk`. Found and fixed two real, pre-existing bugs in `_clean_slurm_dataframe`
along the way (both also affected the existing per-task CSV/plots, not just the new
table): `MaxRSS` aggregation took the first sacct job-step value instead of the max
across steps (a `python3.12` srun sub-step commonly uses far more memory than the outer
`batch` step sacct lists first -- confirmed 120.53M vs the real 3452.07M on one job); and
an all-NaN column for any single job raised `IndexError`, aborting the whole query
(silent at small scale, real once querying an experiment's full ~600-job count).

**A third issue, flagged not fixed**: a chunk of `disk`'s `FiducialCorrelationWarp`/
`DeconvolutionPreprocess` rows are contaminated by fragments that were run inside a
long-lived interactive SLURM session (confirmed: `sacct` shows `JobName=vscode.job` for
one, with the environment file's own `SLURM_JOB_START_TIME` matching sacct's `Start`
exactly -- not a stale job id, that fragment genuinely ran inside a VS Code tunnel job)
rather than merlin's own dedicated per-fragment submission -- `sacct` then reports that
whole session's near-idle usage instead. Median-level contamination on those two columns
for `disk`; only the upper tail is trustworthy. `epi` and both runs' `CellPoseSegmentSAM`
show no such contamination.

**Calibration comparison against the real, clean data** (raw `get_estimated_memory()`,
no margin applied): `FiducialCorrelationWarp` on `epi` and `disk` both check out well
(margined value covers the real max). `DeconvolutionPreprocess` on `epi`: predicted
426 MB, real median 605 MB, real max 696 MB -- **too low even after the 1.2x margin**.
`CellPoseSegmentSAM` on `disk` (`do_3D: true`, the deeper stack): predicted 4327 MB, real
median 5347 MB, real max 9741 MB (p90 8368 MB) -- **too low, seriously**, margined value
barely covers the median and falls far short of the tail. Concrete, data-backed grounds
to tighten these two constants specifically before trusting the memory-estimate wiring,
on top of the already-flagged zero-calibration risk on the time side (see the entry
above/below on the estimated-cluster-resources feature itself). Not yet recalibrated --
awaiting the user's direction.

**Not done**: no fix for the interactive-session-job contamination; no recalibration of
the constants found to be too low; merging/pushing `feature/estimated-cluster-resources`.
Full detail in this project's own request-history log, dated 2026-08-31 13:43 and 14:04.

## get_estimated_memory()/get_estimated_time() implemented for movie-size-scaling tasks (2026-08-31)

`AnalysisTask.providesMemoryEstimate`/`providesTimeEstimate` (new class attributes,
default `False`) are the opt-in contract: `SnakefileGenerator` (`snakewriter.py`) only
trusts `get_estimated_memory()`/`get_estimated_time()`'s return value for a task whose
concrete class sets the matching flag `True` -- every other task's existing hardcoded,
never-used constant is left exactly as harmless dead code, untouched. When trusted, the
generated Snakefile rule's `mem_mb`/`runtime` is the estimate times
`snakewriter.RESOURCE_ESTIMATE_MARGIN` (`1.2`) UNLESS that specific rule has its own
explicit entry in `cluster_resource_allocation_*.json` (an entry only inherited from
`__default__` does not count as an override) -- that stays a manual escape hatch. Applies
only to a task's own rule, never its 'Done' rule.

Opted in (real formulas, in new `merlin/util/resourceestimate.py`, shared math): `Warp`'s
`FiducialCorrelationWarp` (memory + time; memory's `kTask` is the only constant in this
whole feature calibrated against a real measurement, backed out from the 855 MB
`FiducialCorrelationWarp` peak documented below), `Decode` (memory + time, conditional on
`decode_3d`), `DeconvolutionPreprocess`/`CAREPreprocess` (+ `DeconvolutionPreprocessGuo`,
inherits both unchanged), `CellPoseSegmentSAM` (memory only -- its time is dominated by
per-cell post-processing, i.e. cell count, not frame geometry, per the entry below).
Every other constant is a documented, uncalibrated first-pass guess.

**Correction to last night's "confirmed movie-size-scaling" list**, found while actually
reading each task's code before implementing rather than trusting the earlier
investigation: `DeconvolutionPreprocess`/`CAREPreprocess`/`FiducialCorrelationWarp` do
NOT hold a whole z-stack in memory at any point -- each processes and writes one frame at
a time, so their memory is frame-size-driven only, not z/channel-count-driven.
`GenerateAdaptiveThreshold` isn't image/movie-driven at all (reads small column slices
from the decoded barcode database into small fixed-size histogram bins) and was dropped
from scope entirely. Only `CellPoseSegmentSAM` and `Decode` (and only when
`decode_3d: true`) actually hold a full z-stack in memory at once.

**Not done**: none of the `kTask`/`secondsPerFrame` constants besides
`FiducialCorrelationWarp`'s memory one are calibrated against a real job; merging/pushing
this branch (`feature/estimated-cluster-resources`, off `master`). Full detail in this
project's own request-history log, dated 2026-08-31 13:22.

**Follow-up, run against the real experiment (2026-08-31)**: loaded the actual, already-run
`FiducialCorrelationWarp`/`DeconvolutionPreprocess`/`Decode`/`CellPoseSegmentSAM` task
instances for both `BC555_sample_05` sub-runs (`epi`: 2048x2048, 25 z; `disk`: 2304x2304,
100 z -- the deep stack is `disk` specifically, correcting an ambiguity in this file's
earlier wording) and called the real `SnakemakeRule._cluster_resources_for_rule()` with
`useComputedEstimate` both ways. **Memory** is unchanged from today for all 4 (each
already has its own explicit JSON `mem` entry, which the override policy keeps
authoritative -- the estimates are informative-only unless that entry is removed).
**Time is live today and risky**: none of the three non-CellPose tasks has its own JSON
`time` entry (only `__default__`'s blanket `3:00:00` applies), so the computed time
estimate *would* immediately replace it if this branch were merged -- 180 min -> ~2-6 min
for each, a 30-40x cut, backed by zero real timing measurements (every `secondsPerFrame`
constant is a pure guess, unlike the one calibrated memory constant). **Recommendation
given to the user: do not merge this branch's time-estimate wiring without first
calibrating `secondsPerFrame` against at least one real timed job per task** -- if any
task's real per-fragment runtime exceeds a few minutes (including SLURM queue/cold-start
overhead), this would reproduce the exact TIMEOUT problem this feature exists to fix, at
scale. Not yet decided/actioned by the user.

## SmfishSignal/SmfishColocalizationSignal: bounded-memory streaming write (2026-08-31)

Follow-up on `docs/bc555-oom-findings-and-merci-handoff`'s cluster-mem-sizing
investigation (that branch's own FINDINGS.md section, not yet merged here), which
measured `SmfishSignal` climbing to 5.65 GB and still rising for one dense
`BC555_sample_05` `epi` FOV and concluded no movie-size formula could safely bound it,
since it accumulated every detected spot for the whole FOV (`sequential.py`, ~100
z-planes x 11 channels for that experiment) in a Python list before one final
`pandas.concat` + `to_parquet` write. `_load_feature_database` compounded this by
loading every z-plane's segmentation boundary for every cell up front
(`HDF5SpatialFeatureDB.read_features()`) just to filter to one z-plane at a time.

**Fix** (branch `fix/smfish-signal-streaming-memory`, off `master`): both
`SmfishSignal._run_analysis` and `SmfishColocalizationSignal._run_analysis` now loop z
as the outer loop, read that z's segmentation boundaries once via a new
`HDF5SpatialFeatureDB.read_feature_ids_and_boundaries_at_z(zIndex, fov)`, and stream
each z-plane's result to disk immediately via a new `DataSet.open_parquet_chunk_writer`
(`ParquetChunkWriter`, one parquet row group per chunk) instead of accumulating across
the whole FOV. Peak memory is now bounded by one z-plane's worth of spots + boundaries
instead of the whole FOV -- roughly two orders of magnitude less for a ~100-z-plane
stack. Not re-measured against a real job in this pass (structural fix only, not a
recalibration of the `mem: 32000` stopgap already applied to the live experiment's own
config).

**Found and fixed in passing**: `DataSet.load_dataframe_from_parquet` computed its save
path but had `return pandas.read_parquet(...)` commented out, so it always silently
returned `None`. Trivial one-line completion, needed to test the new writer.

**Not done**: the `get_estimated_memory()`/`get_estimated_time()` implementation this
was blocking (deferred to a follow-up turn, same session); re-measuring `SmfishSignal`
against a real job; `SmfishColocalizationSignal`'s pre-existing (unrelated,
unchanged) crash-on-all-empty-z-plane behavior; merging this branch. Full detail in this
project's own request-history log, dated 2026-08-31 12:45.

## CreateFfc task: flat-field correction, wired into GenerateMosaic (2026-08-30)

Implements the flat-field-correction (FFC) handoff written up on 2026-08-27
(originally requested from the sibling MERci session, which found MERlin's
mosaic task had no FFC support): a new standalone `createffc.CreateFfc`
analysis task estimates one flat-field-correction field per imaging *color*
(not per data channel, since vignetting is a fixed property of the
microscope/objective/color) from a small sample of fovs (the `fov_count`
farthest from the imaged footprint's centroid), reading raw frames directly
with no dependency on `Warp`/`GlobalAlign`. Fields are Gaussian-smoothed,
normalized to a percentile, and floor-clipped, mirroring the MERci reference
implementation the handoff pointed to.

Consumption is opt-in via a new `ffc_task` parameter on `GenerateMosaic`
(`merlin/analysis/generatemosaic.py`): when set, each fov's aligned image is
divided by that channel's cached field (`CreateFfc.apply_ffc`) before being
placed into the mosaic; when absent, behavior is unchanged. Only the
"minimum concrete deliverable" from the handoff is done — extending
`ffc_task` into `Warp`/`Preprocess` so decode/segment/smFISH also benefit is
the noted follow-on, not yet started.

**Status**: implemented and tested (`test/test_createffc.py`, 6/6 pass);
`GenerateMosaic` has no pre-existing test file to extend. Full non-slow suite
run on `feature/createffc-task`: 161 passed / 6 failed / 8 errored, same
pre-existing failure categories as documented elsewhere in this file
(`test_snakemake.py` `WorkflowError`, `test_core.py` "Directory not empty"
teardown races) — not a regression from this change. Not yet merged to
`master` or deployed to any experiment's own env.

## BC555_sample_05 epi/CellPoseSegmentSAMDone OOM in the aggregate "Done" figure step (2026-08-30)

Follow-up on the entry below: the per-fragment `CellPoseSegmentSAM` TIMEOUT problem is
confirmed **fixed** on this run — all 476/476 FOVs have `.done` markers and 0 `.error`
markers. A new, different failure showed up in the same task's aggregate
`CellPoseSegmentSAMDone` snakemake rule (`WorkflowError: At least one job did not
complete successfully`), consistently OOM-killed on all 3 automatic retries (SLURM logs
under `output/snakemake/slurm_logs/rule_CellPoseSegmentSAMDone/`).

Root cause: `ParallelAnalysisTask.is_complete()` (`merlin/core/analysistask.py`), the
first time every fragment is done, calls `_generate_figures_safely()` →
`FeatureSavingAnalysisTask._generate_verification_figures()` (`merlin/analysis/
segment.py`) → `SegmentationBoundaryPlot._generate_plot()` (`merlin/plots/
segmentationplots.py`), which calls `featureDB.read_features()` and loads **every**
cell boundary for **all 476 FOVs** into a single process at once just to draw one
boundary map. This experiment's on-disk feature database is 12 GB across 476 HDF5
files; unpacked into per-cell shapely `Polygon` objects in memory it exceeds the rule's
allocation, which is only 8000 MB because `cluster_resource_allocation_BC555_sample_05.json`
has no `CellPoseSegmentSAMDone` entry of its own and falls back to `__default__`'s
`mem: 8000`.

**Other error, same run**: `PlotPerformance` (which redundantly regenerates every
task's figures, including this same segmentation-boundary plot, at the end of the
pipeline) also failed twice, with `SLURM status: 'TIMEOUT'` instead of OOM — it already
has a `mem: 30000` override in the same config file but no `time` override, so it
inherits `__default__`'s `3:00:00` and exceeds it. Plausibly the same full-dataset
`read_features()` cost, just wall-clock-bound here instead of memory-bound.

**Fix implemented**: `HDF5SpatialFeatureDB` (`merlin/util/spatialfeature.py`) gained
`get_feature_z_count()` (z-plane count of the first feature, no geometry read) and
`read_feature_boundaries_at_z(zIndex)` (loads only one feature's `zIndex_<zIndex>`
group, skipping every other z-plane). `SegmentationBoundaryPlot._generate_plot`
(`merlin/plots/segmentationplots.py`) now uses these instead of `read_features()` —
same `zPosition` (middle-z) selection and same plot output, just without reading the
~24 other z-planes per cell it never plots. Cuts the ~9-13 GB estimate above to
~0.25-0.75 GB for this dataset, comfortably inside the existing 8 GB allocation (no
SLURM config change needed for this fix alone). Verified via a new equivalence test
(`test_feature_hdf5_db_read_boundaries_at_z_matches_read_features`) and the full
non-slow suite (same pre-existing order-dependent teardown flakiness as `master`, no
regression). Full detail: `prompt_history/
2026_08_30_1422_diagnose_cellposesegmentsamdone_oom.md`, `prompt_history/
2026_08_30_1512_read_single_z_for_segmentation_boundary_plot.md`.

**Not yet done**: this only reduces the memory `CellPoseSegmentSAMDone` needs — it
doesn't give a hard, dataset-size-independent bound (a subsample cap, "option D" from
the cost comparison, was discussed and declined for now). Not yet committed/branched,
merged, or deployed to the experiment's own env; the `PlotPerformance` TIMEOUT (a
separate resource-allocation gap, not a code bug) is also still open.

**Follow-up finding, same day**: benchmarked 4 candidate fixes for
`SegmentationBoundaryPlot` (`merlin/plots/segmentationplots.py`) against 9 real FOVs
from this experiment, in a throwaway comparison notebook (not committed — see the
prompt_history entry below for its scratchpad location). Turned up a second,
unrelated bug along the way: because Cellpose segments effectively per-z-plane here
(most cells occupy only 1-9 of ~25 z-planes, not clustered at the geometric middle),
the plot's existing single fixed z-index selection silently shows only 0-40% of cells
per FOV (0% on one sampled FOV) — independent of the OOM. A per-cell "own occupied
z" read strategy (`single_z_occupied` in the notebook) fixes both at once: ~100%
coverage at a fraction of `full`'s cost (~0.3 GB / ~8.5 min extrapolated
dataset-wide vs ~29 GB / ~44 min for today's `read_features()`-based path, if that
path were also restructured to stream one FOV at a time instead of holding all 476
simultaneously). Comparison only — no fix implemented yet, awaiting the user's
choice of option. Full detail: `prompt_history/
2026_08_30_1450_segmentation_boundary_plot_cost_notebook.md`.

## BC555_sample_05 epi/disk SLURM failures diagnosed; feature-extraction parallelized (2026-08-29)

Investigated the `epi` run's `CellPoseSegmentSAM` TIMEOUTs and the `disk`
run's `DeconvolutionPreprocess` FAILEDs (both on `BC555_sample_05`, the
`s5-BC555e_*`/`s5-BC555d_*` SLURM jobs). Neither was transient/flaky --
both fail identically on every attempt:

- **`disk`/`DeconvolutionPreprocess`**: 558/558 attempted fragments failed,
  0 succeeded, all with `ValueError: can only convert an array of size 1 to
  a Python scalar` from `dataorganization.py`'s `get_data_channel_for_bit`.
  Root cause is a config mismatch in the experiment's own run, not a MERlin
  bug: `save_pixel_histogram: true` on `DeconvolutionPreprocess` makes it
  loop over every codebook bit name and look each up in the data
  organization, but this run's codebook (`C3v1_codebook.csv`, full MERFISH
  barcode bits) doesn't match its data organization (`ST2`, 3 smFISH
  channels only -- correct for this decode-free pipeline). Worth noting for
  next time: `write_preprocessed_FOV` defaults to *every* FOV when unset, so
  `save_pixel_histogram: false` alone does not skip this code path -- both
  parameters must be set. Root cause traced to two MERci-side template
  defaults (`data/configs/merlin/analysis/tasks/deconvolution_preprocess.yaml`,
  `_PREPROCESS_DEFAULTS` in `merlin_config.py`) that always turn the
  histogram on regardless of whether the assembled pipeline decodes
  barcodes; handed off to MERci (its own `prompt_history/`, dated
  2026-08-29, "tumor_disk pixel histogram codebook crash"). The
  experiment's own generated yaml was hand-patched with both parameters as
  an immediate, non-durable stopgap.

- **`epi`/`CellPoseSegmentSAM`**: 64/476 FOVs stuck, 0 have ever succeeded
  even across snakemake's automatic retries. Cellpose GPU inference itself
  finishes in under a second; the actual bottleneck is the serial per-cell
  post-processing loop (`CellPoseSegmentSAM._run_analysis`, calling
  `SpatialFeature.feature_from_label_matrix` once per detected cell at
  ~0.9-1s each) against a fixed 10-minute SLURM time limit
  (`cluster_resource_allocation_*.json`) -- any FOV dense enough to exceed
  ~600 cells cannot finish in time regardless of retries.

  **Fix (branch `perf/parallel-spatialfeature-extraction`, commit
  `a542072`)**: added `SpatialFeature.features_from_label_matrix_stack`
  (`merlin/util/spatialfeature.py`) which spreads the independent per-cell
  feature extraction across a `multiprocessing.Pool` -- the label matrix
  stack is passed once per worker via the pool initializer (not once per
  cell) to avoid repeated pickling of a large array. `CellPoseSegmentSAM`
  gained a `feature_extraction_processes` parameter (default `1`, i.e. no
  behavior change unless configured) and now calls the new method instead
  of looping inline. Verified serial (`processes=1`) and parallel
  (`processes=2`) output match exactly on a synthetic mask; `test/
  test_spatialfeature.py` (18 tests) and the full non-slow suite still pass
  (pre-existing, order-dependent `test_analysis/` teardown flakiness in
  `test_core.py`/`test_snakemake.py` reproduces identically on `master`,
  unrelated to this change).

  **Not yet done**: this only helps once the per-rule SLURM allocation
  (`cluster_resource_allocation_BC555_sample_05.json`'s `CellPoseSegmentSAM`
  entry, currently `n: 1` CPU) is also raised to match whatever
  `feature_extraction_processes` is set to for the `epi` run, and the
  64 stuck FOVs still need their `.error`/`.start` markers cleared before
  resubmitting. Branch not yet merged to `master` or pushed.

## Hand-written verification figures moved into merlin/plots/ (2026-08-29)

`FiducialCorrelationWarp` (warp.py) and `LeastSquaresGlobalAlignment`
(globalalign.py) used to have their `_generate_verification_figures` plotting
code written directly inline in the task class. Moved to follow the same
one-file-per-analysis-module `merlin/plots/` paradigm as
decodeplots.py/filterplots.py/optimizationplots.py/segmentationplots.py: new
`merlin/plots/warpplots.py` (`generate_drift_qc(warpTask)`, plus the
`_robust_max`/`_load_drift_dataframe` helpers it needs) and `merlin/plots/
globalalignplots.py` (`generate_all(alignTask)` dispatching to
`plot_direction_reliability`/`plot_grid_overlay`/
`plot_overlap_correlation_grid`/`plot_overlap_correlation_histogram`). Each
task's `_generate_verification_figures` is now a one-line call into its new
module (`warpplots.generate_drift_qc(self)` /
`globalalignplots.generate_all(self)`); figure content, filenames, and save
location are unchanged. These stay plain functions, not `AbstractPlot`
subclasses -- `get_available_plots()` (used by `PlotPerformance`) picks up
nothing new from either file, confirmed directly.

Verified via the existing `test_globalalign.py` figure test (unchanged, still
green) and a scratch equivalent for warp.py's `drift_qc` (no dedicated test
existed for it; confirmed passing, not checked in). See `prompt_history/
2026_08_29_1629_move_verification_figures_to_plots_folder.md`.

**Update (2026-08-29): merged into `master` and pushed to `origin/master`**
(`--no-ff`, confirmed via `git rev-parse master origin/master` matching) --
this note's original "not yet merged" status is stale. Same branch as the
entry below (`feature/auto-figures-per-task`), a separate commit; that branch
was also pushed to `origin` and kept (not deleted), per repo convention.

## PlotPerformance figures now auto-generate per-task (2026-08-29)

The `merlin.plots` figures (decodeplots/filterplots/optimizationplots/
segmentationplots) no longer need the separate `PlotPerformance` task to be run --
each producing task now generates its own single-task-role figures right after it
completes, via a new shared helper `AnalysisTask._generate_plots_for_role(role)`
(`merlin/core/analysistask.py`) wired into `_generate_verification_figures` (the
existing "run automatically once this task is done" hook from the 2026-08-14
mechanism below) on `Decode` (`'decode_task'`), `AbstractFilterBarcodes` (shared
base of `FilterBarcodes`/`AdaptiveFilterBarcodes`(`Local`), `'filter_task'`),
`OptimizeIteration` (`'optimize_task'`), and `FeatureSavingAnalysisTask` (shared
base of `WatershedSegment`/`CellPoseSegment3D`/`CellPoseSegmentSAM`/
`RefineCellDatabases`, `'segment_task'`). Figures save via
`dataSet.save_task_figure` into the shared `figuresPath` folder (`merlin.
{taskName}.{figureName}.png`), not the old `PlotPerformance`-nested location.

Only plots whose *sole* declared requirement (`AbstractPlot.get_required_tasks`)
is that one task role are auto-generated this way. The two plots needing both
`filter_task` and `global_align_task` (`CodingBarcodeSpatialDistribution`,
`BlankBarcodeSpatialDistribution` in filterplots.py) are intentionally **not**
auto-generated -- per the user's explicit choice over walking
`filter_task -> decode_task -> global_align_task` to resolve the extra role --
and still require the standalone `PlotPerformance` task, which is otherwise
untouched and still works for anyone who keeps it in their pipeline.

Verified end-to-end against the real `test_analysis_parameters.json` fixture run
through `run_with_snakemake` with no `PlotPerformance` task in the pipeline at
all -- every wired task (`Decode`, `FilterBarcodes`, `AdaptiveFilterBarcodes`,
both `OptimizeIteration` instances, `WatershedSegment`, and the downstream
`RefineCellDatabases`) produced its figures automatically. `pytest -k "not
slowtest"` shows the same non-deterministic pre-existing failure/error classes on
this branch and on unmodified `master` (NFS teardown flakiness, `test_snakemake.py`
`WorkflowError`s, `test_plotting.py`'s two known-flaky tests -- see "Test
environment" note below); `test_analysistask_figures.py`/`test_globalalign.py`
unaffected. See `prompt_history/
2026_08_29_1621_auto_figure_generation_per_task.md` for full detail.

**Update (2026-08-29): merged into `master` and pushed to `origin/master`**
(`--no-ff`, confirmed via `git rev-parse master origin/master` matching) --
this note's original "not yet merged" status is stale. Committed on branch
`feature/auto-figures-per-task` off `master`, also pushed to `origin` and
kept (not deleted), per repo convention.

## Configurable verification-figures folder + `merlin.` filename prefix (2026-08-28)

`merlin`'s CLI gained `-f`/`--figures-path`, threaded through
`MERFISHDataSet`/`ImageDataSet`/`DataSet.__init__` as `figuresPath` and stored as
`dataSet.figuresPath`. It is used exactly as given (no dataset-name subfolder
appended, unlike `--analysis-home`/`--data-home`) -- meant for a layout like
`<experiment>/figures/` sitting alongside `<experiment>/analysis/`, `<experiment>/
merci/`, etc. Default unchanged: `{analysisPath}/figures`. `DataSet.save_task_figure`
(the per-task verification-figures sink, see "General per-task verification-figures
mechanism" below) now writes into `self.figuresPath` and prefixes every filename with
`merlin.`, i.e. `merlin.{taskName}.{figureName}.png` instead of the old
`{taskName}.{figureName}.png` -- makes files unambiguous when the figures folder is
shared with another tool's own outputs. `save_figure` (the separate
`merlin.plots`/`PlotPerformance` sink, nested under each task's own output folder) is
unaffected -- this only touches the shared/verification-figures path.

Verified via `test/test_analysistask_figures.py` (new
`test_custom_figures_path_is_used_as_is`, plus existing tests updated for the
`merlin.` prefix) and `test/test_globalalign.py`'s figure test, both green;
`test/conftest.py` gained a `custom_figures_merfish_data` fixture (mirrors the existing
`two_codebook_merfish_data` pattern) pointing `figuresPath` at a dedicated temp dir.
Pre-existing NFS teardown flakiness in `test_core.py`/`test_dataset.py`/
`test_snakemake.py` (`OSError: Directory not empty`, cascading into a few dependent
failures) reproduces identically on unmodified master via `git stash` -- confirmed
unrelated to this change.

Handoff written to MERci's own `prompt_history/` asking it to pass `-f
<experiment>/figures` when it invokes `merlin` from the notebooks that generate
merlin's input files. See `prompt_history/2026_08_28_1349_add_figures_path_cli_option.md`.

## GenerateMosaic / GenerateMosaicSimple merged (2026-08-27)

`merlin/analysis/generatemosaic.py`'s two mosaic-assembly classes are now one.
`GenerateMosaic` always uses `GenerateMosaicSimple`'s fast per-fov tile
placement (resample + slice, with a correct arbitrary-overlap-count average),
but -- unlike the old `GenerateMosaicSimple`, which read raw stage positions
straight from the dataset and never actually used the `global_align_task` it
declared a dependency on -- each fov's placement now comes from
`alignTask.fov_to_global_transform(fov)`, so a corrected, non-regular fov grid
(`LeastSquaresGlobalAlignment`, see below) is honored. This only works for a
translation/scale `fov_to_global_transform` (true of every alignment class
implemented today); a rotated/sheared one raises `NotImplementedError`
explicitly rather than silently placing tiles wrong -- the old
`GenerateMosaic`'s full-canvas-`cv2.warpAffine` path nominally supported
rotation, but nothing exercises that today (`CorrelationGlobalAlignment`,
the one sketched with rotation in mind, is unimplemented), so it wasn't
carried forward. Both classes' parameters are preserved: `microns_per_pixel`
or `downsample` (mutually exclusive) sizes the mosaic; `output_format`
(`"imagej"`/`"ome"`, the latter forcing `separate_files=True`) plus
`write_pyramidal_tiff`/`pyramidal_levels` picks the writer.
`GenerateMosaicSimple` is now a ~10-line subclass that only restores its old
defaults (`downsample=1`, `fov_crop_width=100`, `output_format="ome"`), kept
for existing analysis-parameters files that reference it by name.
`docs/tasks.rst`'s `GenerateMosaic` entry updated to match (it predated
`GenerateMosaicSimple` and didn't mention it). See `prompt_history/
2026_08_27_2159_merge_generatemosaic_and_generatemosaicsimple.md`.

This makes two references below stale, now corrected: the "ongoing ragged-Z"
section's `GenerateMosaic._prepare_mosaic_slice`/`GenerateMosaicSimple.load_tile`
(that method is gone; both classes now share one `load_tile`/`_build_mosaic`
path), and "LeastSquaresGlobalAlignment"'s 2026-08-22 update's
"`generatemosaic.py`'s absolute-offset tile placement" (that was
specifically `GenerateMosaicSimple`'s old raw-stage-position placement, which
this merge replaced with `alignTask`-sourced placement -- so
`LeastSquaresGlobalAlignment`'s corrected positions are now actually honored
by `GenerateMosaic`, which wasn't true before this merge).

**Not yet committed** -- working tree change only, on `master`; needs a
feature branch before committing per the git-workflow rule.

## FiducialCorrelationWarp: drift_qc verification figure (2026-08-23)

Added a `drift_qc` verification figure to `FiducialCorrelationWarp`
(`merlin/analysis/warp.py`), using the same `_generate_verification_figures`
hook (see "General per-task verification-figures mechanism" below) that
`LeastSquaresGlobalAlignment` already uses. One 3-panel figure per data set,
built from `get_transformation(fov)` across every fov and data channel: (1)
x/y shift scatter colored by data channel on square symmetric axes, (2)
shift-distance histogram with median marked, (3) a fov x data-channel
heatmap of distance minus median -- both axis limits outlier-resistant via a
new `_robust_max` helper. `FiducialCorrelationWarp3D` inherits this
unchanged (same inherited `get_transformation`/`_save_transformations` path
for its 2D registration).

Design mirrors the sibling `260614_LT058_fishtank_debug` project's `fishtank
plot-drift` QC feature (`plot_drift.py`, submitted as `jweissmanlab/fishtank`
PR #14 `add-drift-qc`, still awaiting maintainer review there) -- same
3-panel layout, adapted from fishtank's "round" to MERlin's "data channel"
and labeled *shift* rather than *drift* to sidestep that project's raw
phase-correlation sign-convention ambiguity.

Verified against a synthetic fake dataSet/task (15 fovs x 6 data channels,
no real image data needed). `pytest -k "not slowtest"` before/after on a
clean fixture state showed the same pre-existing flaky failure class
(`test_analysis/` teardown `OSError`s, `test_snakemake.py` `WorkflowError`s
-- both already documented elsewhere in this file); none touch `warp.py`.
Committed on new branch `feature/fiducial-drift-qc-figure`; not yet
merged/pushed.

## SumSignal channel_names merged; sequential.py's missing deps fixed (2026-08-22)

`feature/sumsignal-channel-names` merged `--no-ff` into `master` (pushed).
`archive/yaml-and-snakemake-mixed` (superseded by the three-branch split
below) deleted, local-only, never pushed.

Verifying the merge surfaced that `merlin/analysis/sequential.py` has two
module-level imports never added to `requirements.txt` -- `geopandas`
(colocalization spatial joins) and `bigfish.detection` (PyPI package
`big-fish`, smFISH spot detection) -- both from commit `46bb2431`
(2025-11-06). This blocks importing the *whole* module, not just the
colocalization/smFISH code, in any env missing either package (e.g.
`merlin_cc_env`, which lacked both). Fixed on new branch
`fix/geopandas-missing-dependency` (both added to `requirements.txt`;
branch created, verified). **Update (2026-08-23): merged into `master` and
pushed to `origin/master`** (confirmed via `git merge-base --is-ancestor
fix/geopandas-missing-dependency origin/master`) -- this note's original
"not yet merged" status was stale. Installed both
into `merlin_cc_env` directly (`geopandas==1.1.4`, `big-fish==0.6.2`);
`test_sequential.py` now passes (was failing at collection). Full fast
suite from a clean fixture state: 148 passed; remaining 8
failed/10 errors are pre-existing and unrelated (`test_snakemake.py`
local-execution `WorkflowError`s, `test_core.py`/`test_dataset.py`
`OSError: Directory not empty` teardown flakiness -- the Linux analog of
this file's documented Windows teardown issue; none touch `sequential.py`
or either new package).

**Correction to the "Pending -x/--analysis-name flag" note below**:
`merlin_cc_env`'s editable install does *not* resolve to
`~/Software/merlin_cc/MERlin` -- confirmed by reading the env's own
`__editable___merlin_0_1_6_finder.py`, whose `MAPPING` points `merlin` at
this repo (`260715_LT059_merlin_update/MERlin`) directly. The
`~/Software/merlin_cc/MERlin` clone is real and still holds that
uncommitted `-x` WIP (untouched, on `master`@`1fae07e`, 17 commits behind
but cleanly fast-forwardable), but it is not what `merlin_cc_env` actually
runs -- don't assume package installs into `merlin_cc_env` need a
corresponding change there.

## Three independent branches: YAML recipes, snakemake>=8, setuptools 83 (2026-08-16)

Per explicit "each must be written as a separate branch" instruction, split what had
previously been implemented as two commits on one mixed branch, plus a new dependabot
fix, into three clean branches -- all off `master`, none merged or pushed:

- `feature/yaml-analysis-recipes` -- YAML analysis-recipe support (`.yaml`/`.yml`
  alongside existing `.json`), single clean cherry-pick of the prior investigation's
  implementation commit. 3/3 targeted tests pass; 121/121 broader non-slow suite passes
  (excludes `test_snakemake.py` -- untouched on this branch, needs its own migration --
  and the pre-existing geopandas-gap/slowtest exclusions below).
- `feature/snakemake-v8-migration` -- `merlin.py`'s `run_with_snakemake` ported to the
  `snakemake>=8` builder API (`SnakemakeApi`/`workflow()`/`dag()`/`execute_workflow()`),
  `requirements.txt` pinned to `snakemake>=8.0`. Reconstructed as an independent diff
  (not a cherry-pick, since the original commit's test-file portion depended on the YAML
  branch's fixture file) via `git apply --3way`, one manual conflict resolved (an import
  line that assumed YAML's `import yaml` was already present). Verified against a real,
  freshly-installed snakemake 9.25.1 in an isolated venv: all 5 `test_snakemake.py`
  tests pass (real subprocess-spawned tasks actually completing), 123/123 broader suite.
- `fix/setuptools-upgrade` -- `requirements.txt` pinned `setuptools==83.0.0` (dependabot
  suggestion); `merlin.version()` switched from `pkg_resources.get_distribution(...)` to
  `importlib.metadata.version(...)`, since setuptools 83 no longer bundles
  `pkg_resources` (confirmed: a venv with only 83.0.0 installed has no `pkg_resources`
  of its own). Verified `merlin.version()` returns the correct `0.1.6` under a real
  setuptools-83.0.0-only venv; 118/118 broader suite plus all 5 `test_snakemake.py`
  tests (old pre-migration API, against this branch's own untouched `snakemake==7.32.4`
  base) pass.

The old mixed branch (YAML commit + snakemake-migration commit together) was renamed to
`archive/yaml-and-snakemake-mixed` rather than deleted, in case any of its history is
wanted later.

**Verification environment**: RC cluster conda env `merlin_cc_env`
(`/n/holylabs/zhuang_lab/Lab/lsepulvedaduran/conda/envs/merlin_cc_env`, Python 3.12,
pinned to `snakemake==7.32.4`/`setuptools==73.0.1` per its own prior real-run-tested
state) was never modified directly. Instead, two isolated venvs were built on top of it
with `python -m venv --system-site-packages` (inheriting its heavy deps -- tensorflow,
cellpose, etc. -- for free) and only `snakemake>=8`/`setuptools==83.0.0` overridden
locally per venv, then `pip install -e . --no-deps` per branch. Both venvs left in the
Claude session scratchpad, not `merlin_cc_env` itself.

**Known pre-existing gaps, unrelated to this work**: `merlin_cc_env` has no `geopandas`
(`test_ragged_z_pipeline.py` excluded from every run above for this reason). This repo's
working tree also had unrelated uncommitted changes (absolute-path support for
`-m`/`-p` in `dataset.py`, a `snakemake<8.0` pin in `requirements.txt`) predating this
request -- left untouched, restored via `git stash`/`git stash pop` around the branch
work rather than discarded.

**Update (2026-08-23): all three merged into `master`/`origin/master`.**
`feature/yaml-analysis-recipes` via an explicit merge commit (`b69ad4d`);
`fix/setuptools-upgrade`'s commit (`99d0a5f`) landed directly on master's
mainline (fast-forward, no separate merge commit); `feature/snakemake-v8-
migration`'s core commit (`bedd65f`) is in master's history, subsumed into
the later `feature/slurm-job-naming-verbosity` work that built on top of it
and was merged (`cf99f63`/`036188d`). Confirmed directly via
`git merge-base --is-ancestor <branch> origin/master` for all three, not
assumed from this note.

## SumSignal channel_names parameter (2026-08-16)

Added an optional `channel_names` parameter to `merlin.analysis.sequential.
SumSignal` (branch `feature/sumsignal-channel-names`, off `master`), mirroring
`SmfishSignal`'s existing `channel_names` pattern: when set, restricts which
sequential (non-barcode) channels get summed to that subset; when absent
(default), behavior is unchanged (all sequential channels, as before). This
lets an experiment with multiple sequential channels (e.g. γ-H2AX, CENPA)
assign `SumSignal`/`SmfishSignal` independently per channel -- the motivating
case is in `prompt_history/2026_08_16_1433.txt`. Validated fail-fast in
`__init__` against `get_sequential_rounds()`'s actual gene names, raising
`analysistask.InvalidParameterException` naming any unrecognized entries (no
silent empty/wrong output). New `SumSignal._select_channels()` helper does the
actual filtering, called from `_run_analysis` right after
`get_sequential_rounds()`; nothing else in the task changed. `SmfishSignal`
and `get_sequential_rounds()` itself were deliberately left untouched, per
explicit instruction. A MERci-side follow-up (threading a
`sum_signal_channel_names` option into `MerlinAnalysisSpec`) is noted as
out-of-scope future work, not done this session.

Verified via new `test/test_sequential.py` (3 tests: default selects
everything unchanged, an explicit subset filters correctly against the
`simple_merfish_data` fixture's real `DAPI`/`polyT` sequential channels, an
invalid entry raises at construction). Full fast suite
(`pytest -k "not slowtest"`) re-run afterward: 134 passed; every
failure/error traces to already-documented pre-existing issues below
(`test_snakemake.py`, Windows-teardown `PermissionError`s, one intermittent
zarr-write flake that passed cleanly on rerun) -- none related to this
change. See `prompt_history/2026_08_16_1505_add_sumsignal_channel_names.md`
for full detail.

## LeastSquaresGlobalAlignment: camera/stage position correction (2026-08-13)

New `merlin/util/globalpositions.py` + `merlin.analysis.globalalign.
LeastSquaresGlobalAlignment` (branch `feature/least-squares-alignment`, stacked on
`feature/verification-figures`). Ports the "global_lsq" method from the sibling
`MERci` project (`251225_LT027_saving_time/MERci/src/MERci/acquisition/
camera_rotation.py`), which a real-data comparison there (`notebooks/tests/
compare_stitching_correction_methods.ipynb`) found to be the best of several
candidate corrections for a real, small, highly direction-dependent nominal-vs-true
fov position error (most likely stage backlash, not a fixed camera rotation -- a
single global affine transform provably cannot correct it, see
`fit_global_positions`'s own docstring). Registers every fov against its real
4-connected neighbours via the fiducial channel, filters outlier registrations (MAD
threshold), then jointly solves a sparse least-squares system for every fov's
corrected position (`scipy.sparse.linalg.lsqr`, `atol=btol=1e-12` -- looser default
tolerances silently under-converge on a dense system, per MERci's own prior finding).
`CorrelationGlobalAlignment`'s existing stub was left untouched, per explicit
instruction -- this is a new, separate class, not a rewrite of it. Also implements two
verification figures (`direction_reliability`, `grid_overlay`, ported from the same
MERci notebook's sections 7 and 11) via the sibling branch's new per-task figures hook.

Verified: phase-correlation sign convention confirmed via a synthetic known-shift
test (both x and y independently) before trusting it -- this is the same class of
bug (a backwards row-crop convention) the sibling MERci project already hit once for
real. `test/test_globalpositions.py` (9 tests, pure algorithm) and
`test/test_globalalign.py` (4 tests, including a real `.run()` against the
`simple_merfish_data` fixture, and confirming both verification-figure PNGs appear)
all pass. Full suite (`pytest -k "not slowtest"`) repeatedly: 123-130 passed across
runs (same Windows teardown non-determinism noted below); every failure traces to the
same two already-documented pre-existing issues, confirmed unrelated to this change.
See `prompt_history/2026_08_13_1927_implement_least_squares_global_alignment.md` for
full detail.

**Update (2026-08-22): non-rectangular-grid neighbor-search fix.** A cross-repo
handoff from MERci (`251225_LT027_saving_time/MERci/cache/
prompt_merlin_verify_nonrectangular_grid.md`) flagged that MERci's own new
"irregular" (non-rectangular, independently-phase-shifted-band) FOV grid API
broke any tool assuming one global grid step + fixed cardinal-direction
neighbor offsets (measured on real data: cross-band neighbors found 12-491/572
times vs. 424-476/572 within-band, across a swept tolerance) -- and that
`globalpositions.py` implements exactly that pattern (it was ported from this
same area of MERci's code). Verified by reading the code directly: confirmed
`find_grid_neighbor` did match a candidate against an exact target point
(`anchor + (dx,dy) * one dataset-wide median step`, `tolerance_fraction=0.25`),
which would systematically miss/mistolerance a cross-band neighbor the same
way. Independently re-checked the handoff's other claims
(`generatemosaic.py`'s absolute-offset tile placement, `dataset.py`'s
FOV-id-keyed `_load_positions`, `spatialfeature.py`'s rtree/shapely cell-overlap
partitioning) -- all unaffected, confirmed rather than taken on faith. A grep
sweep found no other rectangular-grid assumption elsewhere in MERlin.

Fixed: `find_grid_neighbor` now classifies each candidate by which axis/sign of
its displacement from the anchor dominates, then accepts the closest candidate
in that direction if it's within `tolerance_fraction` of the anchor's own local
step (its own nearest-neighbor distance), not a dataset-wide one -- this needs
no shared step or phase between regions. `sample_neighbor_correspondences` no
longer takes a `step_size_um` argument (derived locally per anchor now); its
one call site in `globalalign.py` was updated accordingly (the module's
separate, unrelated `stepSizeUm` global-median use for `overlap_fraction`
estimation was untouched). Added
`test_find_grid_neighbor_on_phase_shifted_bands` (two independently
phase-shifted scan-band columns, mirroring MERci's layout) demonstrating the
old algorithm would reject the true cross-band neighbor while the new one
finds it. Full targeted suite (`test_globalpositions.py`,
`test_globalalign.py`, including `LeastSquaresGlobalAlignment`'s own
integration tests) passes; unrelated `test_dataset.py`/`test_snakemake.py`/
`test_core.py` failures confirmed pre-existing (identical on unmodified
master). Committed on branch `fix/nonrectangular-grid-neighbor-search`.
**Update (2026-08-23): merged into `master` and pushed to `origin/master`**
(confirmed via `git merge-base --is-ancestor
fix/nonrectangular-grid-neighbor-search origin/master`) -- this note's
original "not yet merged" status was stale. See
`prompt_history/2026_08_22_1906_verify_nonrectangular_grid_neighbor_search.md`.

**Update (2026-08-23): overlap-correlation verification figures merged.**
`feature/overlap-correlation-figures` (two new `LeastSquaresGlobalAlignment`
verification figures -- `overlap_correlation_grid`,
`overlap_correlation_histogram` -- see `prompt_history/
2026_08_22_1928_add_overlap_correlation_figures.md`) pushed to `origin`,
merged `--no-ff` into `master` (commit `c242d6c`), master pushed. 16/16
targeted tests re-verified passing on master post-merge before pushing.

## General per-task verification-figures mechanism (2026-08-14)

Added a new, general hook -- `AnalysisTask._generate_verification_figures()` (no-op
default, overridable), called automatically and safely (exceptions logged, never
raised) exactly once a task completes: from `AnalysisTask.run()` for a plain task, and
from `ParallelAnalysisTask.is_complete()` (reusing its existing "first caller after the
last fragment finishes" idiom) for a parallel one. Figures are saved via the new
`dataset.save_task_figure` into one shared `{analysisPath}/figures/` folder (sibling to
every task's own output folder), named `{taskName}.{figureName}.png`. This is
deliberately separate from the existing `merlin.plots`/`PlotEngine` framework
(`merlin/plots/_base.py`, driven by the standalone `PlotPerformance` task) -- that one
builds heavier cross-task summary plots nested under `PlotPerformance`'s own folder and
needs explicit pipeline wiring; this new hook is for a single task's own quick,
automatic self-check, no wiring needed beyond overriding the method.

Verified via `test/test_analysistask_figures.py` (4 tests, all passing, built on
`merlin.analysis.testtask`'s existing dummy task classes): default no-op, a figure
actually appearing after `.run()` completes, a deliberately-raising figure method not
breaking the task, and a parallel task's figure appearing exactly once, only after its
LAST fragment (not per-fragment). Full fast suite (`pytest -k "not slowtest"`) re-run
repeatedly afterward, since this touches the core `AnalysisTask`/`ParallelAnalysisTask`
base classes every task in the pipeline extends: 114-117 passed across runs (this
Windows environment's known teardown non-determinism, below, shifts the exact count
run to run); every failure traces to the same two already-documented pre-existing
issues (confirmed by re-running one flagged failure, `test_get_analysis_tasks`, alone
with a clean start -- it passed). See `prompt_history/
2026_08_14_1218_add_verification_figures_mechanism.md` for full detail.

**Update (2026-08-28):** the shared figures folder and filename pattern described
above are stale -- see "Configurable verification-figures folder + `merlin.` filename
prefix" above. The folder is now `dataSet.figuresPath` (configurable via
`--figures-path`, defaults unchanged to `{analysisPath}/figures`), and filenames are
`merlin.{taskName}.{figureName}.png`.

## Test environment (merlin_env, built 2026-08-14)

A new `merlin_env` (`C:\Users\las262\AppData\Local\miniforge3\envs\merlin_env`) was
built for running the real test suite on this machine (which had no Python installed
at all beforehand): minimal `environment.yml` (python=3.12, ipykernel, pip only --
conda solving the full requirements.txt directly failed on tensorflow/snakemake,
resolving to ancient python-3.6-only builds) plus pip installs mirroring a real,
working Harvard-cluster install recipe (torch/torchvision CPU, cellpose, big-fish,
then `pip install -e .`, with `setuptools==73.0.1` re-pinned after `pip install -e .`
transitively upgraded it and broke `pkg_resources`). Could not pin
`snakemake==7.32.4` as that recipe does -- its `datrie` dependency needs MSVC Build
Tools, not installed on this machine -- so `test_snakemake.py` remains broken here
(pre-existing, unrelated to this session's changes). `geopandas` also needed (not in
requirements.txt; matches the older `merlin_test` env's own history below).

## tifffile TiffWriter.write() migration (2026-08-01)

New branch `fix/tifffile-writer-api` (off `feature/ragged-z-stacks`, kept deliberately
independent of the sibling `feature/yaml-analysis-recipes` branch's YAML/snakemake
work). tifffile removed `TiffWriter.save()` (a deprecated alias for `.write()`)
between 2025.1.10 and 2025.2.18 -- confirmed empirically by bisecting installed
versions, same approach as the snakemake investigation. All 12 `outputTif.save(...)`
call sites across `warp.py`/`decode.py`/`generatemosaic.py`/`optimize.py`/
`preprocess.py`/`segment.py` renamed to `.write(...)` (identical signature for every
parameter used: `photometric`, `contiguous`, `metadata`, positional `data` --
confirmed via `inspect.signature` first). `requirements.txt` pinned to
`tifffile>=2023.1.23`. Verified with a real write+readback (bypassing snakemake,
since this branch doesn't include that fix): `FiducialCorrelationWarp` with
`write_aligned_images=True` produced a real, correctly-shaped, non-empty tiff. See
`prompt_history/2026_08_01_2248_fix_tifffile_writer_api.md` for full detail.

## snakemake >=8 migration: cluster-execution port (feature/snakemake-v8-migration)

**Implemented 2026-08-17, awaiting the user's own real-SLURM verification
before commit/merge/push to master.** Branch merged up to date with master
first (brought in the yaml-analysis-recipes, setuptools, and filemap-fix
commits cleanly — only `merlin/merlin.py`/`requirements.txt` conflicted, both
trivial import/pin conflicts).

- `-k`/cluster execution now works via `snakemake-executor-plugin-slurm`
  (`executor='slurm'` in `merlin.merlin.run_with_snakemake`) instead of
  raising `NotImplementedError`. `requirements.txt` pins
  `snakemake>=8.0` + adds `snakemake-executor-plugin-slurm` (installed here:
  snakemake 9.25.1 / plugin 2.8.0).
- Design intent from the prior scoping turn was followed as planned: the
  existing `cluster_resource_allocation_*.json`/`-k` JSON **shapes are
  unchanged** (no MERci-side changes needed) — translation to the
  executor-plugin's per-rule `resources:` model happens internally in
  `merlin/util/snakewriter.py` (`_translate_cluster_resources`,
  `_parse_slurm_time_to_minutes`), applied per-rule (including the
  `...Done` check rules) via `SnakemakeRule._cluster_resources_for_rule`,
  which merges `__default__` with any rule-name-keyed override exactly like
  old snakemake's `--cluster-config` did (confirmed: a `...Done` rule with
  no JSON entry of its own correctly falls back to `__default__`, not to its
  parent task's override).
- Key mapping implemented (verified against the actually-installed
  `snakemake_executor_plugin_slurm` 2.8.0 source, not just docs — one detail
  differed from the prior turn's scoping notes): `mem`→`mem_mb`,
  `partition`→`slurm_partition`, `account`→`slurm_account`, `time`
  (`[D-]H:MM:SS`)→`runtime` (minutes, rounded up), `constraint`/`gres` same
  key (omitted when empty/falsy), `exclude`→`slurm_extra="--exclude=..."`.
  **`requeue` is not a per-rule resource in this plugin version** (no
  `slurm_requeue` key exists) — it's the executor-wide
  `ExecutorSettings.requeue` flag, so `run_with_snakemake` reads
  `clusterConfig['__default__']['requeue']` once and passes it globally,
  not per-rule. Top-level `-k` JSON's `nodes`→`ResourceSettings(nodes=...)`
  (max concurrent SLURM jobs; `coreCount`/`-n` stays on `ResourceSettings
  .cores`, the local-rules budget — confirmed via `snakemake.cli` source
  that `--jobs`/`nodes` and `--cores` are genuinely distinct in
  cluster/cloud mode, unlike local mode where `--jobs` aliases `--cores`).
  `restart_times`→`ExecutionSettings(retries=...)`. `out`/`err` are gone,
  replaced by `SlurmExecutorSettings(logdir=...)` under the dataset's
  snakemake path.
- `merlin.py`'s `-k`/cluster_config JSON is now loaded **before** snakefile
  generation (was only loaded at run time) so `SnakefileGenerator` can bake
  per-task resources into the generated rules at generation time.
- Tests: 6 new tests added to `test/test_snakemake.py` (pure resource/time
  translation, a `SnakemakeRule.as_string()` resources-block check with
  `__default__`-fallback, and a `SnakefileGenerator` integration test that
  embeds resources *and* still executes locally) — all pass, plus the
  original 5 local-execution tests unchanged. Additionally ran the real CLI
  end-to-end with `--generate-only` against the *actual* production
  `parameters_BC553_sample_02.json` / `cluster_resource_allocation_
  BC553_sample_02.json` / 19-task `merlin_analysis_BC553_sample_02.json`
  against the 20-FOV test experiment below — confirmed by inspecting the
  generated Snakefile that real per-task resources (e.g.
  `FiducialCorrelationWarp`: `mem_mb=1000, ..., gres='gpu:0'`) and the
  `__default__`-fallback for `...Done` rules came out correctly, and that
  all 20 FOVs were correctly detected (`expand(..., g=list(range(20)))`).
- **Real SLURM submission (2026-08-17, user-run) found and fixed a real
  bug**: every job crashed with `sbatch: error: gres_job_state_validate:
  --ntasks-per-tres needs either a GRES GPU specification or a node/ntask
  specification`. Root cause: `snakemake_executor_plugin_slurm` treats any
  `gres` resource containing the substring `"gpu"` as a GPU job
  *regardless of count* and unconditionally adds `--ntasks-per-gpu=1`,
  which `sbatch` rejects when paired with the real `cluster_resource_
  allocation` JSON's `__default__.gres: "gpu:0"` (a harmless no-op under
  the old sbatch-template mechanism, applied to every task including
  non-GPU ones). Fixed: `_translate_cluster_resources` now omits `gres`
  entirely when its count is zero, still passing through real requests
  (e.g. `CellPoseSegmentSAM`'s `gpu:1`) unchanged. New regression test
  `test_translate_cluster_resources_omits_zero_gres`. Full detail:
  `prompt_history/2026_08_17_1753_fix_zero_gres_slurm_crash.md`.
- Near-miss during testing, worth flagging: attempting a supposedly-inert
  snakemake dry-run (`OutputSettings(dryrun=True)`) via the builder API
  against the real 19-task pipeline actually **executed for real** (local
  executor, no confirmation) before being caught and killed ~17/600 steps
  in (only cheap `FiducialCorrelationWarp` alignment had run; no
  decode/segmentation reached). `dryrun` on `OutputSettings` alone does not
  suppress execution in this API version — the CLI's `--dry-run` apparently
  wires further plumbing this ad hoc script didn't reproduce. The ~3.6GB of
  accidental output was deleted and the 20-FOV test experiment restored to
  its documented clean state (no `merlin/` subfolder). If a real dry-run
  check via the builder API is needed again, don't trust `OutputSettings
  (dryrun=True)` alone without first confirming what CLI's `--dry-run`
  actually sets.
- An external fork's fix (BrewerLabSDU/MERlin_epigen-UCSD) was evaluated and
  rejected: it repurposes `ConfigSettings.config` (a `--config`-equivalent
  no-op for MERlin's Snakefiles, verified via grep) and silently falls back
  to running everything locally instead of submitting real distributed SLURM
  jobs — worse than the current loud error.
- Full key-mapping table and MERci handoff:
  `prompt_history/2026_08_17_1618_scope_slurm_port_yaml_question_merci_handoff_test_experiment.md`.
  MERci-side prep requested via a handoff prompt there (cross-project
  changes are never made directly from this repo — see `CLAUDE.md`'s
  "Cross-project boundary"); no MERci JSON-shape changes are needed by this
  implementation, confirming that handoff's scope was correctly limited to
  investigation.
- The 20-FOV test experiment for the real SLURM smoke test:
  `/n/holylfs05/LABS/zhuang_lab/Lab/shared/projects/breast_cancer/experiments/BC553_sample_02_test/epi/`
  (symlinked raw data, new positions file; a submit-ready `merlin/` config
  tree — real per-task resource values, `nodes` scaled to 20, `-x` dropped,
  see its own `README.md` — was added 2026-08-17).
- **Second real submission (job 39874098, gres fix applied) got past
  `FiducialCorrelationWarp` cleanly, then hit a second real failure --
  unrelated to the snakemake port** -- in `DeconvolutionPreprocess`:
  `ValueError: can only convert an array of size 1 to a Python scalar` from
  `DataOrganization.get_data_channel_for_bit` (`merlin/data/
  dataorganization.py:178`). Root cause: this experiment's
  `data_organization_MF3_BC553_sample_02.csv` names bits `b1-RS0015`,
  `b2-RS0083`, ... (`readoutName` column) while its codebook
  (`C3v1_codebook.csv`) names the same bits `RS0015`, `RS0083`, ... (no
  `bN-` prefix, from `codebook.get_bit_names()` = column headers) --
  `get_data_channel_for_bit` does an exact-string match, so it always finds
  zero rows and `.item()` on the empty result raises. This code path only
  runs when `save_pixel_histogram: true` (set in this experiment's real
  19-task analysis-parameters JSON, copied unchanged from production).
  Confirmed **not test-experiment-specific**: production's own
  `BC553_sample_02/epi/merlin/dataorganization/data_organization_MF3_
  BC553_sample_02.csv` has the identical `bN-`-prefixed `readoutName`
  convention, and production's analysis JSON sets the same
  `save_pixel_histogram: true`. Production's own
  `data/DeconvolutionPreprocess/` has no output and no matching error in
  its logs -- this parameter combination appears never to have actually
  been exercised there either (task graph likely never reached that far),
  so this is a real, pre-existing bit-naming mismatch, not something
  introduced by the snakemake migration or by hand-assembling the test
  experiment. Job cancelled (`scancel 39874098`, deterministic failure --
  every one of the 20 FOVs hit it identically, retries can't fix it). Full
  detail:
  `prompt_history/2026_08_17_1807_investigate_deconvolutionpreprocess_bitname_mismatch.md`.
  **Resolved (2026-08-18, user call): the dataorg file's `bN-` prefix is
  wrong** -- it was created by MERci and doesn't match the bare bit names
  MERlin's codebooks use. Fixed by stripping `^b\d+-` from the test
  experiment's `readoutName` column (backup kept as `.bak_bN_prefix`
  alongside it); verified all 16 of the real codebook's runtime bit names
  (from `C3v1_codebook.csv`'s `bit_names` line) are now present. Production's
  own copy of this dataorg CSV has the identical bug (unfixed -- out of
  scope for "this time"). A handoff was written directly into MERci's own
  `prompt_history/` (`251225_LT027_saving_time/MERci/prompt_history/
  2026_08_18_1359_fix_dataorg_readoutname_bn_prefix.md`) so the generator
  itself (`src/MERci/acquisition/data_organization.py`, `readoutName`
  copied verbatim from the `readouts` table's `"Name"` column) stops
  emitting the `bN-` prefix going forward -- see
  `prompt_history/2026_08_18_1400_resubmit_job_direct_merci_handoff_global_rule.md`
  (supersedes the in-this-repo-only handoff approach from
  `2026_08_18_1328_fix_test_dataorg_bN_prefix_and_merci_handoff.md`; a new
  global rule, `~/.claude/rules/cross-repo-handoff.md`, now governs this).
  **Resubmitted with the fix: job `40079595`** (2026-08-18) -- confirmed
  the fix works: 19/20 `DeconvolutionPreprocess` fragments completed before
  the driver hit `NODE_FAIL` (node `holy8a24201`, a cluster hardware/
  scheduler failure, not a code bug). MERci's own session independently
  confirmed+refined the fix on their side (`readoutName` now built from
  `readouts["Probe name"]` instead of `readouts["Name"]` -- see their
  `prompt_history/2026_08_18_1359_...md`, `status: done`); production's own
  dataorg CSV is still unfixed/out of scope pending user go-ahead.
  **Second finding from that same resubmission**: fragment 17 had a stale
  `DeconvolutionPreprocess_17.error` marker left over from the *old*,
  pre-fix, cancelled job (`39874098`) -- `.error` markers are sticky
  (`analysistask.py`/`dataset.py`'s `is_error()` is a plain file-existence
  check, no staleness logic), so it would never have retried on its own.
  Backed up (`.bak_stale_prefix_bug`, not deleted) and resubmitted:
  job **40082610**. Full detail:
  `prompt_history/2026_08_18_1550_node_fail_resubmit_stale_error_marker.md`.
  **That job was user-cancelled (2026-08-18 15:50)** after hitting a third,
  unrelated real bug -- see the "Optimize02 chromatic-correction" entry
  below. Not yet resubmitted again; pending user confirmation of that fix.
- **Production's own copy of this bug finally fixed and resolved
  (2026-09-03)**. A real production run
  (`BC553_sample_02/epi/merlin`) had every `DeconvolutionPreprocess`
  fragment crash with this identical `bN-`-prefix error -- confirming the
  "pending user go-ahead" fix noted above was still outstanding in
  production 2.5 weeks later. Applied the same fix directly to production's
  `dataorganization/data_organization_MF3_BC553_sample_02.csv` (backup kept
  as `.bak_bN_prefix`): stripped `^b\d+-` from all 27 `readoutName` values,
  verified against the real `C3v1_codebook.csv` (all 16 runtime bit names
  now resolve, no duplicates), verified via pandas that no other column or
  row order changed. Resubmitted (job 44215423) -- this exposed a **second,
  previously-masked bug**: with the naming lookup fixed, `DeconvolutionPreprocess`
  fragments finally did real work, and 268 of ~298 attempted fragments were
  silently OOM-killed by the SLURM cgroup killer at the calculated 840MB
  limit (`cluster_resource_allocation_BC553_sample_02.yaml`'s
  `DeconvolutionPreprocess: {}` entry, 1.2x-margin default) -- this never
  raises a Python exception (cgroup kills are invisible to the app), so
  scanning task logs for `ERROR` lines completely misses it; only `sacct`
  (`State=OUT_OF_MEMORY` on the `.0` job step) shows it. sacct's own
  periodic `MaxRSS` sampling only ever caught ~690MB between polls, well
  under the 840MB limit, meaning the real trigger is a brief spike between
  samples (plausibly the FFT-based high-pass filter + iterative
  Richardson-Lucy deconvolution, run across all 16 bits x 25 z per fragment)
  -- the exact same failure shape already documented above for
  `FiducialCorrelationWarp` in this same yaml. Cancelled the run (driver +
  all in-flight child jobs), forced `DeconvolutionPreprocess: {mem: 3000}`
  in production's yaml (comment added in-file), resubmitted (job 44222104):
  confirmed clean -- 15+ fragments completed with 0 failures/0 OOM before
  the run was left to continue on its own. No MERlin repo files were
  touched; both fixes are entirely in the production experiment's own
  config files (outside this git repo).

## Optimize02 chromatic-correction KeyError bug (2026-08-18, feature/slurm-job-naming-verbosity)

- Job 40082610 (see above) got past `DeconvolutionPreprocess`/
  `FiducialCorrelationWarp`/`Optimize01` and then most `Optimize02`
  fragments crashed with `KeyError: '560'` in
  `_get_chromatic_transformations` (`merlin/analysis/optimize.py`).
- Root cause: that function keys `colorPairDisplacements` by data-channel
  *color* (from `_get_used_colors()`, which correctly maps bit position ->
  bit name -> data channel -> color), but then computed each barcode's
  actual color pair via `dataOrganization.get_data_channel_color(onBit)`
  -- passing the raw bit *position* (0..N-1, an index into
  `codebook.get_bit_names()`) directly into a function that expects a
  *data channel* index. Since the data-organization table has more rows
  than there are bits (DAPI/fiducial/etc. channels interleaved), this
  silently read an unrelated row's color instead of raising immediately,
  and only failed once that wrong color wasn't a key in
  `colorPairDisplacements`.
- Pre-existing bug in the inherited (colleague's) branch, unrelated to the
  snakemake-v8 migration or this session's other fixes; only triggers when
  `optimize_chromatic_correction: true` (set for every `Optimize0N` round
  in this experiment's analysis-parameters JSON).
- **Fixed**: map bit position -> bit name (`codebook.get_bit_names()`) ->
  data channel (`get_data_channel_for_bit`) -> color, matching the pattern
  `_get_used_colors()` already uses correctly.
- **Committed, pushed, and merged to master** (branch
  `fix/optimize02-chromatic-color-keyerror`, commit `d0a07d1`, merged
  `f65e268`; kept on origin per this repo's existing fix/* convention).
  `feature/slurm-job-naming-verbosity` merged master back in to pick it up.
  Full detail:
  `prompt_history/2026_08_18_1633_diagnose_optimize02_crash_slurm_naming_verbosity_branch.md`,
  `prompt_history/2026_08_18_1646_commit_push_merge_color_fix_branch.md`.
  **Resubmitted with the fix, after clearing stale `.error` markers left
  over from the earlier cancelled/crashed runs: job `40113193`
  (2026-08-18 16:57).** **Completed successfully** -- `COMPLETED`,
  `ExitCode 0:0`, 506/506 steps (100%), ran 16:57:36-18:29:11 (1h31m), no
  errors in its own log. Confirms the `Optimize02` chromatic-correction fix
  works end-to-end on the real 20-FOV test experiment. Note: the driver's
  cumulative `.err`/`.out` files (one file per script, appended across every
  resubmission) still contain `Error in rule .../WorkflowError` text from
  the earlier failed attempts -- a naive `grep`/tail of the whole file will
  match those stale lines; only content after the run's own
  `Snakefile generated at .../<timestamp>.Snakefile` marker (or the matching
  `Complete log(s): .../<timestamp>.snakemake.log` line) belongs to that
  submission.

## SLURM job naming / driver-log verbosity (feature/slurm-job-naming-verbosity, 2026-08-18)

User request: per-rule/per-fragment SLURM job names
(`{experiment prefix}-{task initials}-{fov}`), a `task_initials()`
abbreviation function, and a terser driver-log job-submission message.

- **Per-rule/per-fragment `-J` job names are not possible**:
  `snakemake_executor_plugin_slurm` hardcodes the SLURM job name to one
  shared per-run UUID (needed for its own `sacct --name=<uuid>`/
  `squeue --name` status polling) and explicitly rejects `--job-name`/`-J`
  via `slurm_extra` (`validation.py::get_forbidden_slurm_options`).
  **Already available today, zero code changes**: the plugin unconditionally
  sets `sbatch --comment rule_<RuleName>[_wildcards_<fragment>]` -- confirmed
  live via `sacct -o JobID,Comment` against a real fragment. Add `Comment`
  to a `sacct -o` format string to get exactly this per-rule/per-fragment
  info.
- Implemented what's achievable: `job_name_prefix` (default `'merlin'`), a
  new key in the `-k`/snakemake-parameters JSON, wired to
  `SlurmExecutorSettings(jobname_prefix=...)` in `merlin.py`'s
  `run_with_snakemake` -- this is the one per-run (not per-rule) knob the
  plugin actually exposes.
- `task_initials(taskName, length=3)` implemented and tested
  (`merlin/util/naming.py`, `test/test_naming.py`) per the given
  algorithm/examples. **User call (2026-08-18): rely on `sacct -o Comment`
  as-is rather than renaming Snakemake rules** -- so `task_initials()` is
  built and available but has no call site in this codebase yet.
  - Also flagged: the worked example `SimpleGlobalAlignment -> SinGloAli`
    doesn't match the stated algorithm (would be `SimGloAli`) -- implemented
    per the stated algorithm, not silently matched to the example.
- Driver-log verbosity: added `_SlurmSubmissionLogFilter` (`merlin.py`),
  attached only on the SLURM execution path, reformatting the submission
  message into the requested indented 3-line block and substituting the
  run's snakemake-path prefix with `$OUTPUT_DIR` (printed once at the start
  of the run). Verified against the real message format.
- **Expanded (2026-08-18) into `_SlurmDriverLogFilter`**, prompted by real
  driver-log confusion: snakemake's submission message ("Job N has been
  submitted with SLURM jobid J") and its later completion message
  ("Finished jobid: N (Rule: R)") both call the number `jobid`, but N is
  snakemake's own internal per-DAG-node counter (`job.jobid`, confirmed at
  `snakemake/scheduling/job_scheduler.py:483/488` and
  `snakemake_executor_plugin_slurm/__init__.py:1189` -- both read the same
  attribute) while J is the real SLURM id, appearing nowhere in the finish
  message. Fixed by having the filter remember `slurm_id`/`Rule`/`Fragment`
  (the latter two recovered from the submission log path, which
  `snakewriter` already lays out as `slurm_logs/rule_<Rule>/[<Fragment>/]
  <slurm_id>.log`) keyed by N at submission time, then reusing that on the
  matching finish line:
  `Submitted jobid: N (slurm_id: J) (Rule: R) (Fragment: F)` /
  `Finished jobid: N (slurm_id: J) (Rule: R) (Fragment: F)` (Fragment
  omitted for non-per-fragment rules; a finish line with no prior
  submission record, e.g. the local `all` rule, is left unchanged rather
  than guessed at). Also reformats the startup `Command: ...` line to one
  flag per line (split on ` -X`/` --X` token boundaries), by overwriting
  the `cmd` field snakemake's `workflow_started` event already carries as a
  separate record attribute (`workflow.py:297`) -- not string-matched from
  message text, so it can't misfire on unrelated log lines.
  **Deliberately NOT done**: putting the finish line's timestamp on the
  same line as the message text (`[ts] Finished jobid: ...` all on one
  line) -- unlike the submission message (a plain untagged log call, so the
  filter fully controls its rendering), the finish message's timestamp is
  hardcoded by snakemake's own formatter (`format_job_finished`) as a
  separate line *before* whatever text the filter supplies, and merging
  the two would require overriding the record's internal `event` tag to
  reroute it through a different, undocumented formatter path -- more
  fragile than anything else built here, and would incidentally bypass
  snakemake's own `--quiet` filtering for that message. **User call:
  keep the timestamp on its own line for both messages** (matches today's
  behavior; content enrichment is all that changed). Verified against
  the real message formats and with new unit tests, `test/test_merlin.py`
  (5 tests, all passing) -- log-path parsing (with/without fragment),
  submission/finish correlation, the no-prior-submission fallback, and the
  command reformat.
- MERci handoff written directly into MERci's own `prompt_history/
  2026_08_18_1632_add_short_experiment_name_for_slurm_job_prefix.md` (short
  experiment-name function for `job_name_prefix`, from 2 example mappings).
- **Committed, pushed, and merged to master** (commit `ab30e51`, branch
  `feature/slurm-job-naming-verbosity` pushed to origin; merged `--no-ff`
  into `master` as `cf99f63`, pushed). Full suite re-verified on the merge
  commit before pushing: 134 passed, same 6 pre-existing
  `test_snakemake.py` local-execution failures (confirmed via `git stash`
  against the prior commit -- unchanged by this change). `cache/` (local
  reference material, untracked) deliberately left out of the commit.

## `-x`/`--analysis-name` flag ported into this repo (2026-08-28)

`BC553_sample_02`'s real slurm script uses `-x "output"` (decouples analysis
output location from the raw-data path), which this repo's `merlin.py`
previously did not implement — confirmed via grep, no `-x`/`analysisName`
anywhere here. The real implementation existed but was **uncommitted, never
pushed**, in a different local clone of the same origin:
`~/Software/merlin_cc/MERlin` (checked out at `1fae07e`), added directly
from a MERci session on 2026-08-14 (MERci `prompt_history/
2026_08_14_1815_fix_merlin_data_organization_and_output_path.md`) — exactly
the untracked-cross-project-edit pattern `CLAUDE.md`'s "Cross-project
boundary" section now exists to prevent.
Full detail: `prompt_history/2026_08_17_1741_investigate_x_flag_origin.md`.

**Confirmed real-world manifestation (2026-08-28)**: `BC555_sample_05`'s
slurm job crashed with `DataFormatException: No image files found at
.../lineage_tracing/experiments/output.` — reproduced directly that
`parse_known_args()` silently mis-parses `-x "output" <dataset>` when `-x`
is unimplemented, matching the bare `"output"` token to the required
`dataset` positional and discarding the real dataset path. The sibling
`BC553_sample_02` script has the identical bug but is masked by an
already-cached `filemap.csv` from a prior run, so it doesn't crash.
`BC555_sample_05`'s script also had an unrelated second bug: `-e` pointed
at the wrong project (`lineage_tracing` instead of `breast_cancer`). Full
detail: `prompt_history/2026_08_28_1755_diagnose_bc555_sample_05_dataset_arg_bug.md`.

**Ported** the ~25-line diff (`merlin.py` + `dataset.py`) from
`~/Software/merlin_cc/MERlin`'s working tree into this repo, on
`feature/analysis-name-flag`: new `-x`/`--analysis-name` CLI flag, and an
`analysisName` parameter threaded through `DataSet`/`ImageDataSet`/
`MERFISHDataSet.__init__` (defaults to `dataDirectoryName`, so omitting the
flag is unchanged behavior). The `dataorganization.py` piece from that same
working tree was **not** ported — already superseded by this repo's own
`adc0482` ("Fix filemap losing per-round subfolder paths"). Verified the
mis-parse is fixed directly (`build_parser().parse_known_args()` on the
real `BC555_sample_05` argv now yields `dataset='BC555_sample_05/epi/data'`,
`analysis_name='output'`, empty `unknown`). Full fast suite before/after
(`git stash` diff against this branch): 153 passed both times, same
pre-existing `test_snakemake.py`/`test_core.py` teardown flakiness on
both — no regression. Merged `--no-ff` into `master` and pushed
(commit `bafc897`).

**Follow-up bug found and fixed (2026-08-28, same day)**: the port above
only covered the top-level `merlin` CLI entry point. It missed
`SnakemakeRule._base_shell_command()` in `merlin/util/snakewriter.py` —
the function that builds the actual `shell:` command embedded in every
generated per-task/per-fragment snakemake rule. That command already
re-passes `-e`/`-s` so each fragment's subprocess can reconstruct an
equivalent `DataSet`, but never re-passed `-x`, so every fragment task
silently fell back to `analysisName=dataDirectoryName` instead of the real
value — mismatching the `analysisPath` the top-level run actually wrote
`dataorganization.csv` to. Surfaced as a real failure:
`BC555_sample_05`'s `FiducialCorrelationWarp` fragment 49 crashed with
`FileNotFoundError: .../merlin/BC555_sample_05/epi/data/dataorganization.csv`
(should have been `.../merlin/output/dataorganization.csv`). Fixed by
storing `self.analysisName` on `DataSet` (mirroring `dataHome`/
`analysisHome`) and having `_base_shell_command` emit `-x` the same way it
already emits `-e`/`-s`. Verified directly (rebuilt the generated shell
string before/after) and via the full fast suite (154 passed, same
pre-existing `test_snakemake.py` failures — confirmed one is an unrelated
subprocess/env mismatch, not caused by this change).
Full detail: `prompt_history/
2026_08_28_1835_fix_analysis_name_snakemake_propagation.md`.

**Also fixed, same commit set**: `-f`/`--figures-path` had the identical
gap — also missing from `_base_shell_command`, so a custom `-f` path was
silently lost for every snakemake-invoked fragment task (falling back to
the default `analysisPath/figures`). Added `-f "<dataSet.figuresPath>"`
to `_base_shell_command` the same way, per user go-ahead. Verified: full
fast suite, 153 passed, same pre-existing failures as above.

## Repo / branch layout

- `origin` = `leonardosepulveda/MERlin` (this fork), `upstream` = `emanuega/MERlin`,
  `aaron` = colleague `aaronhalpern/MERlin`.
- `variable_z_per_fov` tracks `aaron/segmentation_oversample_v2` (no commits of its own;
  kept as a clean base branch).
- Active implementation branch: `feature/ragged-z-stacks` (branched off
  `variable_z_per_fov`). Not yet pushed to `origin`.

## Ongoing: per-FOV variable Z-count support ("ragged Z")

**Motivation**: a separate tissue-thickness analysis (in the sibling `MERci` project,
`251225_LT027_saving_time/MERci`) found many FOVs have far less signal depth than the
fixed acquisition Z-range. Trimming acquisition per-FOV to each FOV's own depth (same Z
start/step everywhere, only the max depth differs) saves real acquisition time/disk
space, but produces a dataset MERlin can't process today.

**Design decisions (confirmed with user, 2026-07-15)**:
1. Per-FOV Z-extent is inferred from each FOV's own raw file's frame count (not a
   separate sidecar table).
2. Thin FOVs contribute blank/zero pixels at mosaic depths beyond their own signal.
3. Support is opt-in/backward-compatible (`allowRaggedZStacks` flag, default off).

**Status — done, Stages 1-5** (Stage 1: commit `60bba1d`; Stages 2-5: next commit on
`feature/ragged-z-stacks`, both 2026-07-16). The user confirmed this experiment does not
use `FiducialCorrelationWarp3D` (the colleague's 3D-registration path), and directly
re-verifying `warp.py` showed `Warp.get_aligned_image` needs **no changes at all** for
the plain-2D case: a thin fov's z-range is always a literal prefix of the global sorted
list (same start/step, only depth differs), so any consumer that simply loops over
`get_z_positions(fov)` instead of the global list never asks for an out-of-range z in
the first place. This made Stages 2-5 simpler than originally sketched — no central
warp.py zero-fill was needed after all. What actually changed:
- `decode.py`, `filterbarcodes.py`: per-fov z-position calls (2 call sites each).
- `preprocess.py` (`CAREPreprocess`/`DeconvolutionPreprocess`/`DeconvolutionPreprocessDW`):
  per-fov batch-shape and pixel-histogram loops.
- `globalalign.py`: `SimpleGlobalAlignment.fov_coordinates_to_global` fov-scoped.
- `segment.py`: `WatershedSegment`, `CellPoseSegment3D`/`CellPoseSegmentSAM` per-fov
  z-position calls, plus a genuine correctness fix (not just raggedness-related): feature
  z-coordinates now come from the actually-retained `zPos_segment[sel]` values instead of
  the unfiltered `zPos`, with a warning (not a raise) when the assumed
  `zPos(fov) ⊆ zPos_segment(fov)` invariant breaks for a given fov.
- `optimize.py`: `OptimizeIterationFOV`'s single global `z_index` default became
  fov-relative (middle of that fov's own depth), computed in `_run_analysis` instead of
  `__init__` (no fov known there); an explicit user-supplied override invalid for a
  specific fov raises `analysistask.InvalidParameterException` naming that fov (user's
  choice: no silent clipping). `OptimizeIteration`'s random per-iteration z sampling now
  samples per-fov instead of from one shared global range.
- `sequential.py`: `SumSignal` keeps its constructor-time global sanity check, plus a new
  per-fov runtime check that raises for a fov where the configured `z_index` is invalid.
  `SmfishSignal`/`SmfishColocalizationSignal` resolve `z_indexes` per-fov via a shared
  `_resolve_z_indexes` helper — default is "all of this fov's own planes"; an explicit
  user list has invalid-for-this-fov entries skipped with a printed notice (user's
  choice, since this is naturally a "process what's there" loop, unlike SumSignal's
  single required value). Also fixed a pre-existing latent bug: `z_indexes` isinstance
  check ran before the "key not present" check, which would `KeyError` if absent.
- `generatemosaic.py`: (both classes' mosaic-building methods at the time; since merged
  into one `GenerateMosaic.load_tile`, see "GenerateMosaic / GenerateMosaicSimple merged"
  above) gate the `get_aligned_image` call on whether the requested z is within that
  fov's own depth; when not, substitute a same-shape zero array. The existing
  `pixel > 0`-based coverage/division-mask accounting already correctly treats an
  all-zero contribution as "not imaged here" — no separate explicit mask was needed.

**Verified end-to-end**, not just unit-tested: built a real `warp → globalalign →
preprocess → optimize → decode` chain against the ragged fixture (`test/
test_ragged_z_pipeline.py`) and confirmed correct, per-fov-scoped shapes throughout
(e.g. decode's per-fov zarr shape matches that fov's own z-count: 4/2/3/4 for fovs
0/1/2/3). Also confirmed `WatershedSegment` works for the one fov where its (pre-existing,
raggedness-independent) assumption that segmentation and regular channels share the same
z-grid happens to hold. `CellPoseSegment3D`/`CellPoseSegmentSAM` (need cellpose, not
installed) and `SmfishSignal`'s actual bigfish spot detection (synthetic images have no
real spot content) are not integration-tested — their fixed z-logic was verified directly
via `_resolve_z_indexes`/the `zPos_segment[sel]` masking arithmetic instead.

**Known test flakiness discovered this round**: `test_decode_ragged_zarr_shape_matches_fov_z_count`
intermittently fails with `PermissionError: WinError 5` *during* a zarr write (inside
zarr's own atomic tmp-file-rename), not on the shape assertion itself — confirmed
non-deterministic by rerunning identical code repeatedly (pass/fail alternated). This is
the same class of pre-existing Windows file-lock flakiness already documented below, now
also manifesting inside zarr's write path specifically; not a decode.py logic bug.

## Test environment

No local machine environment had MERlin's dependencies. A dedicated conda env
`merlin_test` (Python 3.11, `C:\Users\Leonardo\anaconda3\envs\merlin_test`) was built
with a lightweight subset of `requirements.txt`, later extended with scikit-image, rtree,
scikit-learn, pyclustering, geopandas, and bigfish (`big-fish` on pip) once it became
clear those were needed to import/exercise `segment.py`/`optimize.py`/`decode.py`/
`sequential.py` for real. This env now covers every test file except
`test_decon.py`/`test_snakemake.py`/`test_merfish.py` (need cv2-heavy/snakemake/
tensorflow) and anything needing cellpose, csbdeep/CARE, or the external `dw` binary
(not installed — these remain untested at the integration level). Reuse this env for
future test runs. A venv under the Claude session's Temp scratchpad fails with a
Windows long-path DLL error on `pytables` — use a conda env under `anaconda3/envs`
instead (short, fixed path).

**Known pre-existing flakiness** (not a regression from any of the above):
`test_core.py`'s multiprocessing-executor tests intermittently fail teardown with
`PermissionError: WinError 32` on Windows, non-deterministically. Confirmed by running
unmodified code twice and getting different failing tests each time.
