# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

MERlin is an extensible pipeline for MERFISH (spatially resolved RNA profiling) image
analysis: decoding barcodes from raw microscope images and segmenting cells. Workflows
run as a graph of analysis tasks, each splittable into per-FOV (field of view) subtasks
for parallel execution locally, on a cluster, or in the cloud. This fork (`origin` =
`leonardosepulveda/MERlin`) descends from `emanuega/MERlin` (`upstream`) and currently
sits on a colleague's extended branch (cellpose 3D segmentation, deconwolf, CARE
preprocessing, zarr-based resumable decoding, smFISH/colocalization analysis, OME-tiff
mosaic writing) rather than plain upstream master.

## Commands

Environment setup (one-time, per machine): `merlin --configure` prompts for and writes
`~/.merlinenv` with three paths that every other command depends on:
- `DATA_HOME` — raw data, one subfolder per experiment
- `ANALYSIS_HOME` — where analysis results are written, mirrored per experiment
- `PARAMETERS_HOME` — codebooks, data-organization files, microscope/position/snakemake
  configs (subfolders: `analysis/`, `codebooks/`, `dataorganization/`, `microscope/`,
  `positions/`, `snakemake/`)

Install (editable, so code edits take effect immediately):
```
pip install -e .
```

Run the full pipeline against an experiment (reads `PARAMETERS_HOME/analysis/<name>.json`,
generates a Snakefile, executes it):
```
merlin <experiment-dir> -a <analysis_parameters.json> -o <data_organization.csv> -c <codebook.csv> -n <core-count>
```
Run/inspect a single analysis task directly (bypassing snakemake), optionally one
fragment (FOV) at a time:
```
merlin <experiment-dir> -t <TaskClassName> -i <fragment-index>
merlin <experiment-dir> -t <TaskClassName> --check-done
```
`--generate-only` builds the task graph/Snakefile without running anything —
useful for validating a new analysis-parameters JSON.

Tests (from the repo root; `test/conftest.py` builds synthetic fixture datasets under
`test_data/`/`test_analysis/` on first use):
```
pytest                                    # full suite
pytest test/test_dataorganization.py      # single file
pytest test/test_dataorganization.py::test_dataorganization_get_z_positions   # single test
pytest -k "not slowtest"                  # skip slow/full-pipeline tests (see test/pytest.ini markers)
pytest --cov --cov-report=xml             # matches CI (.circleci/config.yml)
```
Note: on Windows, some `ParallelAnalysisTask`/multiprocessing-executor tests in
`test_core.py` intermittently fail teardown with `PermissionError: WinError 32` (a file
briefly still held by another process) — this is pre-existing flakiness in the test
harness's cleanup, not a code regression; rerunning usually passes.

No configured linter/formatter is run locally or in CI beyond pytest itself
(`.pep8speaks.yml` only drives a GitHub PR-comment bot, not a local command).

## Architecture

**Two core objects everything else is built on**, both in `merlin/core/dataset.py` and
`merlin/data/dataorganization.py`:
- `MERFISHDataSet` (extends `ImageDataSet`/`DataSet`) — one instance per experiment;
  owns raw-data/analysis paths, codebooks, FOV positions, and z-position queries. Most
  analysis code reaches everything else through a `dataSet` reference.
- `DataOrganization` — parses `dataorganization.csv` (one row per data channel/bit,
  columns documented in `docs/usage.rst`), maps `(dataChannel, fov, round)` to raw image
  file paths and frame indices via regex-matched filenames, and answers "what are the z
  positions for this experiment" (`get_z_positions()`) and the DAPI/polyT-channel
  equivalent (`get_z_positions_segmentation()`). Both now optionally take a `fov`
  argument to get that FOV's own available z range (inferred from its raw file's actual
  frame count) rather than the dataset-wide list — see `_get_available_z_positions`/
  `_get_fov_frame_count` and the `allowRaggedZStacks` flag for datasets where FOVs were
  acquired with different numbers of z-planes.

**Analysis tasks** (`merlin/core/analysistask.py`) are the unit of work; concrete tasks
live in `merlin/analysis/*.py` (one file per pipeline stage: `warp` → `preprocess` →
`optimize`/`decode` → `filterbarcodes` → `segment` → `partition`, plus
`generatemosaic`, `sequential`/smFISH, `globalalign`, `slurmreport`,
`plotperformance`, `exportbarcodes`). Three base classes:
- `AnalysisTask` — declares `get_dependencies()` on other tasks, has parameters (a
  dict from the analysis-parameters JSON), tracks a `.done` file per task/fragment.
- `ParallelAnalysisTask` — the common case; `fragment_count()` is almost always
  `len(dataSet.get_fovs())`, and `_run_analysis(fragmentIndex)` processes exactly one
  FOV per fragment (`fragmentIndex` *is* the fov index directly, no indirection).
- `InternallyParallelAnalysisTask` — parallelizes inside a single task invocation
  instead of via fragments.

`merlin/util/snakewriter.py` (`SnakefileGenerator`) turns an analysis-parameters JSON
(a list of `{analysisModule.ClassName, parameters, ...}` entries) plus each task's
declared dependencies into a Snakemake workflow (one rule per task, `expand()`-ed over
fragments for `ParallelAnalysisTask`s); `merlin/core/executor.py` (`LocalExecutor`) or
snakemake itself then runs it. `merlin/merlin.py` is the CLI entry point tying
argument parsing, dataset construction, Snakefile generation, and execution together.

**Raw image reading** (`merlin/util/imagereader.py`) dispatches by file extension to
`DaxReader`/`TifReader`/`ZarrReader`, each of which derives its own frame count from
that specific file's header/shape — this per-file frame count is what makes per-FOV
z-range inference (above) possible without any extra metadata. `merlin/util/dataportal.py`
abstracts local filesystem vs. S3 vs. Google Cloud Storage raw-data access behind one
interface used throughout.

**Segmentation output** (`merlin/util/spatialfeature.py`, `SpatialFeature`) stores each
detected cell's boundary/volume with its own per-feature z-coordinate array, so it isn't
inherently tied to a dataset-wide z list; `segment.py`'s `CleanCellBoundaries`/
`CombineCleanedBoundaries` merge cells that straddle FOV borders using 2D projections.

**Barcode storage/filtering**: decoded barcodes are written per-FOV via
`merlin/util/barcodedb.py` and post-processed by `merlin/util/barcodefilters.py`
(z-plane duplicate removal among other filters) before `filterbarcodes.py` and
`partition.py` assign them to segmented cells.

## History records (two tiers)

This project keeps two complementary, local-only histories. Both live *inside*
the project, and both are gitignored.

1. **`verbatim_history/`** — *uncompressed*. The exact text Claude writes each turn,
   appended automatically by a `Stop` hook (`.claude/hooks/save_verbatim.ps1`). No
   action needed from Claude — the harness captures it. One file per day,
   `{YYYY-MM-DD}_verbatim.md`.
2. **`prompt_history/`** — *compressed summary*. One file per request (format below).
   The append-only source of truth: records the verbatim prompt, plan, what was
   done, and the dead-ends. **Never edit past entries** — their value is provenance.

**Maintenance habit:** for **every user question/request**, write a tier-2
`prompt_history` entry (format below). Keep `prompt_history` append-only.

### `prompt_history/` entry format

- One Markdown file per request, named `{YYYY_MM_DD_HH_MM}_{short_description}.md`
  (e.g. `2026_06_04_1432_add_prompt_history_convention.md`).
- **Get the actual date/time from the system** (`date "+%Y-%m-%d %H:%M:%S"` on
  bash, `Get-Date` on PowerShell) for both the filename and the `date:` field.
  The environment context provides only the date, never the time of day — never
  fabricate the `HH_MM` (see the "No fabrication" rule above).
- **This depends on non-repo-tracked global config** — `~/.claude/CLAUDE.md`'s
  "No fabrication" section and the `UserPromptSubmit` hook in `~/.claude/settings.json`
  that injects `Current local date/time: … (epoch N)` every turn. That global config
  has already been silently lost once (machine migration), causing fabricated,
  suspiciously-regular timestamps in a sibling project — see
  `251225_LT027_saving_time/MERci/prompt_history/
  2026_08_12_1916_fix_timestamp_fabrication_and_backup_hooks.md` — before a
  Dropbox-synced backup (`Dropbox/claude-global-backup/`) was set up to catch it
  going forward. If the hook's injected line is ever missing from context here too,
  query the clock directly rather than estimating — never reuse a "plausible" value
  or space entries at a suspiciously round interval.
- **`elapsed`** (just before `status`): wall-clock from prompt submission to task
  completion. Start = the `epoch N` injected by the `UserPromptSubmit` date/time
  hook for the message that began this request (for a request that spans several
  turns, the FIRST turn's epoch); end = `date +%s` (bash) /
  `[DateTimeOffset]::Now.ToUnixTimeSeconds()` (PowerShell) run at finish. Write a
  human-readable duration (e.g. `12m 30s`). Omit the field if no submit epoch is
  available — do not guess.
- YAML frontmatter for queryable metadata, then prose sections. Template:

```markdown
---
date: YYYY-MM-DD HH:MM
title: <short description>
files_modified:
  - path/relative/to/repo
elapsed: <e.g. 12m 30s — submit→completion wall-clock>
status: completed | in-progress | abandoned
---

## Prompt
<verbatim copy of the user's request>

## Plan
<Claude's plan of action before executing>

## Summary
<what was actually done, including any deviations from the plan>
```

Format rationale: Markdown + YAML frontmatter is Claude-native, human-readable,
and lets all entries be scanned/grepped by metadata without reading every body.
