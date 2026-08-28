# FINDINGS.md

Curated current-state summary. See `prompt_history/` for full provenance of each item
below; this file only tracks what's true *now* and the open next step.

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

## Pending: port the `-x`/`--analysis-name` flag into this repo

`BC553_sample_02`'s real slurm script uses `-x "output"` (decouples analysis
output location from the raw-data path), which this repo's `merlin.py` does
not implement — confirmed via grep, no `-x`/`analysisName` anywhere here.
The real implementation exists but is **uncommitted, never pushed**, in a
different local clone of the same origin: `~/Software/merlin_cc/MERlin`
(checked out at `1fae07e`), added directly from a MERci session on
2026-08-14 (MERci `prompt_history/2026_08_14_1815_fix_merlin_data_organization_and_output_path.md`)
— exactly the untracked-cross-project-edit pattern `CLAUDE.md`'s
"Cross-project boundary" section now exists to prevent. Small diff (~25
lines, `merlin.py` + `dataset.py`; the dataorganization.py piece in that
same working tree is already superseded by this repo's own `adc0482`).
Full detail: `prompt_history/2026_08_17_1741_investigate_x_flag_origin.md`.
Not yet ported — deferred at the user's request ("we can work on that after
this is done").

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
