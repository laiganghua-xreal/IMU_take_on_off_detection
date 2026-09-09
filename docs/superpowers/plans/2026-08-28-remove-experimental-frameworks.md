# Remove Experimental Frameworks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the full-grid, dual-accelerometer, and impact comparison implementations and all of their generated experiment data while keeping the original EgoCHARM and ordinary Xreal wear workflows operational.

**Architecture:** Retain the shared original modules and the five `wear_*` modules. Remove experiment-only packages, entry points, tests, documentation, caches, checkpoints, and reports, then remove the one optional impact-model dispatch from `wear_model.py`. Verify the two retained CLI families through imports, dry runs, focused tests, and the complete remaining test suite.

**Tech Stack:** Python 3.10, PyTorch, pytest, YAML configuration, NumPy caches.

## Global Constraints

- Keep the original EgoCHARM seven-class reproduction.
- Keep the ordinary Xreal wear-state and wear-event classification workflow.
- Delete all contents of the four approved later experiment directories, approximately 1.56 GB in total.
- Do not delete ordinary EgoCHARM or Xreal configurations, caches, checkpoints, or reports.
- The directory is not a Git repository, so commit steps are not applicable and deleted experiment results are not recoverable through Git.

---

### Task 1: Establish the retained-workflow baseline

**Files:**
- Test: `tests/test_config.py`
- Test: `tests/test_dataset.py`
- Test: `tests/test_model.py`
- Test: `tests/test_train.py`
- Test: `tests/test_evaluate.py`
- Test: `tests/test_wear_dataset.py`
- Test: `tests/test_wear_model.py`
- Test: `tests/test_train_wear.py`
- Test: `tests/test_evaluate_wear.py`
- Test: `tests/test_wear_evaluation.py`

**Interfaces:**
- Consumes: the current original and ordinary Xreal module APIs.
- Produces: a pre-deletion pass/fail baseline for the two retained workflows.

- [ ] **Step 1: Run the retained focused test set**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider \
  tests/test_config.py tests/test_dataset.py tests/test_model.py \
  tests/test_train.py tests/test_evaluate.py \
  tests/test_wear_dataset.py tests/test_wear_model.py \
  tests/test_train_wear.py tests/test_evaluate_wear.py \
  tests/test_wear_evaluation.py -q
```

Expected: all selected tests pass before deletion.

- [ ] **Step 2: Record the exact approved deletion targets**

Run `du -sh` and `find` against only the paths listed in Task 2. Confirm that no
ordinary configuration, artifact directory, or retained source module appears.

---

### Task 2: Remove the experimental implementations and generated data

**Files:**
- Modify: `src/egocharm/wear_model.py:108-113`
- Delete: `src/egocharm/full_grid/`
- Delete: `src/egocharm/dual_acc/`
- Delete: `src/egocharm/impact_model.py`
- Delete: `src/egocharm/impact_experiment.py`
- Delete: `experiments/imu_wear_full_grid_v1/`
- Delete: `experiments/xreal_dual_acc_v1/`
- Delete: `experiments/xreal_dual_acc_four_class_v1/`
- Delete: `experiments/xreal_dual_acc_impact_500hz_v1/`
- Delete: `tests/test_dual_acc_experiment.py`
- Delete: `tests/test_dual_acc_four_class_entrypoints.py`
- Delete: `tests/test_dual_acc_preprocess.py`
- Delete: `tests/test_full_grid_cache.py`
- Delete: `tests/test_full_grid_dataset.py`
- Delete: `tests/test_full_grid_evaluation.py`
- Delete: `tests/test_full_grid_preprocess.py`
- Delete: `tests/test_full_grid_report.py`
- Delete: `tests/test_full_grid_runner.py`
- Delete: `tests/test_full_grid_specs.py`
- Delete: `tests/test_full_grid_training.py`
- Delete: `tests/test_impact_experiment.py`
- Delete: `tests/test_impact_experiment_entrypoints.py`
- Delete: `tests/test_impact_model.py`
- Delete: `docs/superpowers/specs/2026-08-20-imu-wear-full-grid-design.md`
- Delete: `docs/superpowers/specs/2026-08-21-xreal-dual-accelerometer-design.md`
- Delete: `docs/superpowers/specs/2026-08-21-xreal-dual-acc-four-class-design.md`
- Delete: `docs/superpowers/specs/2026-08-21-dual-acc-impact-comparison-design.md`
- Delete: `docs/superpowers/plans/2026-08-20-imu-wear-full-grid.md`
- Delete: `docs/superpowers/plans/2026-08-21-xreal-dual-accelerometer.md`
- Delete: `docs/superpowers/plans/2026-08-21-xreal-dual-acc-four-class.md`
- Delete: `docs/superpowers/plans/2026-08-21-dual-acc-impact-comparison.md`

**Interfaces:**
- Consumes: `build_random_wear_model(model_config, label_mode)`.
- Produces: the same function supporting the ordinary `EgoCHARM` model only, without the removed `dual_acc_impact` architecture.

- [ ] **Step 1: Remove the impact architecture dispatch**

Delete this block from `build_random_wear_model`:

```python
if model_config.get("architecture") == "dual_acc_impact":
    from egocharm.impact_model import build_impact_aware_wear_model

    return build_impact_aware_wear_model(model_config, label_mode)
```

Do not change the ordinary LLE, HLE, frozen-LLE, or fine-tuned-LLE construction paths.

- [ ] **Step 2: Delete the approved experiment source and result directories**

Delete only the four source targets and four experiment directories listed in
this task. Use literal absolute or workspace-relative paths; do not use globs,
environment variables, or a recursive workspace-root target.

- [ ] **Step 3: Delete experiment-only tests and historical experiment plans**

Delete exactly the fourteen test files and eight historical documentation files
listed in this task. Preserve the cleanup design and cleanup implementation plan.

- [ ] **Step 4: Check for dangling references**

Run:

```bash
rg -n 'egocharm\.(full_grid|dual_acc|impact_model|impact_experiment)|dual_acc_impact' \
  src scripts configs tests README.md
```

Expected: no matches.

---

### Task 3: Document and verify the reduced project

**Files:**
- Modify: `README.md:11-27`
- Preserve: `docs/superpowers/specs/2026-08-28-remove-experimental-frameworks-design.md`
- Preserve: `docs/superpowers/plans/2026-08-28-remove-experimental-frameworks.md`

**Interfaces:**
- Consumes: retained CLI modules `egocharm.train`, `egocharm.evaluate`, `egocharm.train_wear`, and `egocharm.evaluate_wear`.
- Produces: a README source tree describing both retained workflows and verification evidence that they still operate.

- [ ] **Step 1: Update the README source tree**

Make the source tree show the original modules plus:

```text
wear_dataset.py         Xreal labels, selection and Dataset
wear_model.py           random, frozen-LLE and fine-tuned-LLE models
wear_evaluation.py      wear-event metrics and threshold selection
train_wear.py           Xreal training entry point
evaluate_wear.py        Xreal evaluation entry point
```

Do not mention the deleted experiment frameworks.

- [ ] **Step 2: Import every retained CLI module without bytecode output**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -c \
  'import egocharm.train, egocharm.evaluate, egocharm.train_wear, egocharm.evaluate_wear'
```

Expected: exit code 0 with no exception.

- [ ] **Step 3: Run CLI dry-run checks**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m egocharm.train --config configs/egocharm.yaml --run-name cleanup-check --dry-run
PYTHONDONTWRITEBYTECODE=1 python -m egocharm.train_wear --config configs/xreal_end_to_end_four_class.yaml --run-name cleanup-check --dry-run
```

Expected: both commands exit 0 and report `training_started: false`. They must
not create run directories.

- [ ] **Step 4: Run the complete remaining test suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -p no:cacheprovider -q
```

Expected: every remaining test passes.

- [ ] **Step 5: Remove generated caches and audit the final tree**

Delete `.pytest_cache` and every `__pycache__` directory under the project. Then
confirm the deleted source, experiment, test, and historical documentation paths
do not exist, and report the disk space removed.
