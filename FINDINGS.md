# FINDINGS.md

Curated current-state summary. See `prompt_history/` for full provenance of each item
below; this file only tracks what's true *now* and the open next step.

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

**Next step**: awaiting user inspection of the three branches before merging any of them
into `master` (not done automatically, per the request's own "commit and merge with main
after inspection" phrasing) or pushing to `origin` (standing "confirm before push"
agreement).

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
- `generatemosaic.py`: `GenerateMosaic._prepare_mosaic_slice`/`GenerateMosaicSimple
  .load_tile` gate the `get_aligned_image` call on whether the requested z is within that
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
