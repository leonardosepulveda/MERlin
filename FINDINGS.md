# FINDINGS.md

Curated current-state summary. See `prompt_history/` for full provenance of each item
below; this file only tracks what's true *now* and the open next step.

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

**Status — Stage 1 of 5 done** (commit `60bba1d` on `feature/ragged-z-stacks`,
2026-07-16): `DataOrganization`/`MERFISHDataSet` gained a per-fov `get_z_positions(fov)`
API and an opt-in `allowRaggedZStacks` flag; `merlin.py` gained a matching CLI flag; new
ragged-Z test fixtures and tests were added (all passing).

**Open next step — Stage 2**: centralize "return zeros for a fov/z beyond that fov's
depth" in `Warp.get_aligned_image` / `FiducialCorrelationWarp3D.get_aligned_image`
(`warp.py`) — this is the real choke point nearly every downstream consumer (segment,
decode, preprocess, sequential, generatemosaic) reads through, and it currently *raises*
rather than returning zeros for an out-of-range z. `Warp3D`'s own z-interpolation
(picks neighbors from the *global* z array today) needs particular care. After that:
Stage 3 (mechanical per-module swaps in segment.py/decode.py/preprocess.py/
globalalign.py), Stage 4 (optimize.py/sequential.py constructor-time global z_index
validation → per-fov runtime checks), Stage 5 (generatemosaic.py's coverage/
division-mask needs to distinguish "not imaged" from "imaged but dark", not just Stage
2's zero-fill).

## Test environment

No local machine environment had MERlin's dependencies. A dedicated conda env
`merlin_test` (Python 3.11, `C:\Users\Leonardo\anaconda3\envs\merlin_test`) was built
with a lightweight subset of `requirements.txt` sufficient for every test file except
`test_decon.py`/`test_snakemake.py`/`test_merfish.py` (need cv2-heavy/snakemake/
tensorflow) and `test_spatialfeature.py` (needs scikit-image, not installed). Reuse this
env for future test runs. A venv under the Claude session's Temp scratchpad fails with a
Windows long-path DLL error on `pytables` — use a conda env under `anaconda3/envs`
instead (short, fixed path).

**Known pre-existing flakiness** (not a regression from any of the above):
`test_core.py`'s multiprocessing-executor tests intermittently fail teardown with
`PermissionError: WinError 32` on Windows, non-deterministically. Confirmed by running
unmodified code twice and getting different failing tests each time.
