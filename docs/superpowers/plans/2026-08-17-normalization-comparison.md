# Normalization Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run one reproducible Python command that trains `none` and `gravity` normalization variants from scratch, evaluates both on official val/test, and stores separated artifacts plus comparison summaries.

**Architecture:** A standalone script deep-copies the base YAML into two variant configs, invokes the existing single-run train/evaluate CLIs sequentially with `sys.executable`, validates required artifacts at every boundary, and atomically summarizes metrics. The script accepts an injectable command runner so orchestration can be tested without GPU training.

**Tech Stack:** Python 3.10, argparse, copy, csv, json, pathlib, subprocess, NumPy/PyTorch project CLIs, PyYAML, pytest

## Global Constraints

- Remove only the user-approved legacy directory `/home/xreal/egocharm/artifacts/runs/baseline` before real execution.
- Do not modify or regenerate any NPZ cache.
- Train both variants for exactly 30 epochs from scratch in the real comparison.
- Keep seed, model, official split, optimizer, scheduler, batch size, workers, device, and every non-normalization setting identical.
- Run variants sequentially in order `none`, then `gravity` on the single GPU.
- Evaluate each variant's `best.pt` on official `val` and `test`.
- Store every artifact beneath `artifacts/comparisons/normalization/<variant>`.
- Refuse accidental weight overwrite without `--resume`; require `last.pt` when `--resume` is used.
- Stop on the first nonzero child command or missing required artifact.
- Do not change existing model attributes, checkpoint format, train/evaluate CLI behavior, or Dataset normalization behavior.
- This directory is not a Git repository, so verification checkpoints replace commit steps.

---

### Task 1: Variant Configuration and Safety Guards

**Files:**
- Create: `scripts/run_normalization_comparison.py`
- Create: `tests/test_normalization_comparison.py`

**Interfaces:**
- Produces: `VARIANT_NAMES = ("none", "gravity")`
- Produces: `build_variant_config(base_config, normalization, output_root, epochs)`
- Produces: `validate_variant_state(variant_directory, resume)`
- Produces: `write_yaml_atomic(path, payload)`

- [x] **Step 1: Write failing tests for isolated configs**

Create a literal base config fixture. Assert two calls return independent deep
copies with only these intended changes:

```python
assert none_config["data"]["normalization"] == "none"
assert gravity_config["data"]["normalization"] == "gravity"
assert none_config["training"]["number_of_epochs"] == 30
assert gravity_config["training"]["output_root"] == str(output_root.resolve())
assert base_config["data"]["normalization"] == "gravity"
```

Mutate one returned nested value and assert the other result and base fixture do
not change.

- [x] **Step 2: Write failing tests for overwrite and resume guards**

Assert a fresh variant passes with `resume=False`; an existing `best.pt` or
`last.pt` rejects `resume=False`; `resume=True` rejects a missing `last.pt`; and
`resume=True` accepts an existing `last.pt`.

- [x] **Step 3: Run focused tests and verify RED**

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
/home/xreal/miniconda3/envs/egocharm/bin/python -m pytest \
  -p no:cacheprovider tests/test_normalization_comparison.py -q
```

Expected: collection fails because the new script module does not exist.

- [x] **Step 4: Implement minimal config and guard functions**

Use `copy.deepcopy`, set `data.normalization`,
`training.number_of_epochs`, and absolute `training.output_root`, and validate
`epochs > 0`. Implement exact checkpoint guards. Write YAML through a temporary
file in the destination directory followed by `os.replace`.

- [x] **Step 5: Run focused tests and verify GREEN**

Run the command from Step 3. Expected: all Task 1 tests pass.

---

### Task 2: Sequential Command Orchestration and Logging

**Files:**
- Modify: `scripts/run_normalization_comparison.py`
- Modify: `tests/test_normalization_comparison.py`

**Interfaces:**
- Produces: `run_command(command, log_path)`
- Produces: `train_command(config_path, variant_name, resume)`
- Produces: `evaluate_command(config_path, checkpoint_path, split, output_path)`
- Produces: `execute_variant(variant_name, config_path, output_root, resume, command_runner)`

- [x] **Step 1: Write failing tests for exact command order and separation**

Use a fake runner that records `(command, log_path)` and creates the required
sentinel outputs after each call. Assert one variant executes:

```text
python -m egocharm.train ... --run-name <variant>
python -m egocharm.evaluate ... --split val --output <variant>/val
python -m egocharm.evaluate ... --split test --output <variant>/test
```

Assert logs are `train.log`, `val/evaluate.log`, and `test/evaluate.log`, and
that evaluation uses the same variant's `best.pt`.

- [x] **Step 2: Write failing tests for failure propagation**

Make the fake runner return nonzero on val evaluation and assert test evaluation
is never called. Make training return success without creating `best.pt` and
assert a clear `FileNotFoundError` before evaluation.

- [x] **Step 3: Run focused tests and verify RED**

Run the Task 1 focused command. Expected: new orchestration tests fail because
the command builders and executor do not exist.

- [x] **Step 4: Implement command builders and streaming runner**

Build all commands from `sys.executable` and absolute paths. `run_command` uses
`subprocess.Popen(..., stdout=PIPE, stderr=STDOUT, text=True)`; each output line
is printed and appended to the dedicated log. Raise `subprocess.CalledProcessError`
on nonzero exit. `execute_variant` checks `best.pt`, `last.pt`, and each
`metrics.json` immediately after the producing command.

- [x] **Step 5: Run focused tests and verify GREEN**

Run the Task 1 focused command. Expected: all Task 1 and Task 2 tests pass.

---

### Task 3: Dry Run and Atomic Comparison Summaries

**Files:**
- Modify: `scripts/run_normalization_comparison.py`
- Modify: `tests/test_normalization_comparison.py`

**Interfaces:**
- Produces: `collect_comparison_rows(output_root)`
- Produces: `write_comparison_summaries(output_root, base_config_path, rows)`
- Produces: `main(arguments=None, command_runner=run_command)`

- [x] **Step 1: Write failing summary tests**

Create literal val/test `metrics.json` files and minimal checkpoints containing:

```python
{
    "training_state": {"epoch": 8, "best_macro_f1": 0.75},
}
```

Assert four rows ordered by variant then split and exact fields:
`normalization`, `split`, `accuracy`, `macro_f1`, `best_epoch`,
`best_val_macro_f1`, and absolute `checkpoint`.

- [x] **Step 2: Write failing dry-run and full-main tests**

For `--dry-run`, assert both YAML files are written, no fake runner call occurs,
and reported commands contain distinct variant paths. For normal execution, let
the fake runner create all expected artifacts and assert `comparison.csv` and
`comparison.json` contain four separated rows.

- [x] **Step 3: Run focused tests and verify RED**

Run the focused test command. Expected: summary and main tests fail because the
functions are missing.

- [x] **Step 4: Implement summary and CLI orchestration**

Load checkpoints with `torch.load(..., map_location="cpu", weights_only=False)`.
Validate numeric metric fields and training-state fields. Atomically write JSON
and CSV only after all four metrics files exist. `main` writes configs first,
prints a JSON dry-run plan when requested, otherwise validates state and executes
`none` then `gravity` before summarizing.

- [x] **Step 5: Run focused tests and verify GREEN**

Run the focused command. Expected: all comparison-script tests pass.

---

### Task 4: Documentation and Project Verification

**Files:**
- Modify: `README.md`
- Test: `tests/test_normalization_comparison.py`

**Interfaces:**
- Documents: command, output layout, fair-variable constraint, resume behavior,
  and summary interpretation

- [x] **Step 1: Document the comparison workflow**

Add the normal, dry-run, and resume commands; state that only normalization
differs; show the output tree; and explain that `comparison.csv/json` contain
val/test accuracy and macro-F1 for both variants.

- [x] **Step 2: Run the full automated suite**

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
MPLCONFIGDIR=/tmp/egocharm-matplotlib \
/home/xreal/miniconda3/envs/egocharm/bin/python -m pytest \
  -p no:cacheprovider -q
```

Expected: all tests pass.

- [x] **Step 3: Run compilation verification**

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
/home/xreal/miniconda3/envs/egocharm/bin/python -m compileall -q scripts src tests
```

Expected: exit code `0`; remove generated `__pycache__` directories afterward.

- [x] **Step 4: Run a real dry run**

```bash
python scripts/run_normalization_comparison.py --dry-run
```

Expected: two generated configs and six planned child commands, with no training
process and no checkpoint created.

---

### Task 5: Approved Cleanup and Real Two-Variant Experiment

**Files:**
- Delete: `artifacts/runs/baseline/`
- Create: `artifacts/comparisons/normalization/**`

**Interfaces:**
- Consumes: 1159 existing NPZ caches and the verified comparison script
- Produces: two 30-epoch checkpoint sets, four evaluation directories, and two summaries

- [x] **Step 1: Record safety snapshots and delete only the approved run**

Record NPZ count/modification times and list every file beneath
`artifacts/runs/baseline`. Delete that exact directory, then confirm it is absent
and unrelated artifacts remain.

- [x] **Step 2: Execute both experiments**

```bash
env PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
MPLCONFIGDIR=/tmp/egocharm-matplotlib \
/home/xreal/miniconda3/envs/egocharm/bin/python \
  scripts/run_normalization_comparison.py
```

Expected: `none` completes 30 epochs and both evaluations before `gravity`
starts; then `gravity` completes the same sequence.

- [x] **Step 3: Verify comparison artifacts and data safety**

Require both `best.pt`/`last.pt`, four `metrics.json` files, two logs per
evaluation split, `comparison.csv`, and `comparison.json`. Compare NPZ snapshots
byte-for-byte, confirm both resolved configs differ only in normalization and
variant output paths, and confirm no train/evaluate process remains.

- [x] **Step 4: Report measured results**

Report best epochs, best validation macro-F1, val/test accuracy, val/test
macro-F1, output paths, and which normalization performed better on test
macro-F1. Do not infer a paper-level conclusion from a single seed.
