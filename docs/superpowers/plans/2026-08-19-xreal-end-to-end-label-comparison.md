# Xreal End-to-End Label Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge both Xreal recording days into one reproducible 7:1:2 dataset, then train and evaluate randomly initialized end-to-end EgoCHARM models for four-, three-, and two-class label modes with a validation-only HLE capacity search.

**Architecture:** Keep preprocessing, task label mapping, model construction, metric calculation, and experiment orchestration as separate units. The shared four-class selection is the only data index; Dataset maps it to each task. A runner first chooses one HLE size on three-class validation data, then runs three seeds for every task and aggregates common event metrics.

**Tech Stack:** Python 3.10, PyTorch 2.5, NumPy, SciPy, scikit-learn, PyYAML, pytest.

## Global Constraints

- Source roots are the 2026-08-18 and 2026-08-19 `free_record_only_imu` directories.
- Raw label priority is `WALK_ON/WALK_OFF` before plain `ON/OFF`.
- Cache format stays 50 Hz, five seconds, `(250, 6)`, CPF left/up/forward.
- Stable windows start every one second and never cross split boundaries.
- Split ratio is train/val/test = 7:1:2; event files are stratified by date and original label.
- Existing Xreal cache and selection are replaced; all historical run/checkpoint directories are preserved.
- HLE search candidates are 16, 32, and 64; only validation metrics may select the size.
- Formal comparison uses seeds 42, 43, and 44 for all three label modes.
- The workspace is not a Git repository, so commit steps are intentionally omitted.

---

### Task 1: Merge and reproducibly split both Xreal recording days

**Files:**
- Modify: `Xreal_datasets/preprocess_xreal_imu.py`
- Modify: `tests/test_preprocess_xreal_imu.py`

**Interfaces:**
- Produces: `infer_label(path) -> str` with WALK priority.
- Produces: `allocate_split_counts(number_of_items, split_ratios) -> tuple[int, int, int]`.
- Produces: `assign_event_splits(paths, seed, split_ratios=(7, 1, 2)) -> dict[Path, str]`.
- Produces: `build_stable_window_assignments(..., split_ratios=(7, 1, 2)) -> list[tuple[int, str]]`.
- Produces: repeatable `--source-root` CLI input and a combined four-class selection.

- [ ] **Step 1: Add failing label-priority and multi-root tests**

```python
def test_walk_recordings_are_stable_labels():
    converter = load_converter()
    assert converter.infer_label(Path("L551_WALK_ON_fixture/imu_0.csv")) == "WORN"
    assert converter.infer_label(Path("L551_WALK_OFF_fixture/imu_0.csv")) == "NOT_WORN"


def test_repeated_source_roots_are_all_discovered(tmp_path):
    converter = load_converter()
    roots = [tmp_path / "20260818", tmp_path / "20260819"]
    # Create one ON file below each root, invoke main with --source-root twice,
    # and assert both source paths appear in the resulting selection.
```

- [ ] **Step 2: Add failing 7:1:2 allocation and isolation tests**

```python
def test_event_files_use_deterministic_seven_one_two_split():
    converter = load_converter()
    paths = [Path(f"event-{index}.csv") for index in range(20)]
    first = converter.assign_event_splits(paths, seed=42, split_ratios=(7, 1, 2))
    second = converter.assign_event_splits(paths, seed=42, split_ratios=(7, 1, 2))
    assert first == second
    assert list(first.values()).count("train") == 14
    assert list(first.values()).count("val") == 2
    assert list(first.values()).count("test") == 4
```

Extend the stable assignment fixture to assert 70%/10%/20% chronological regions and one skipped five-second base slot at both boundaries.

- [ ] **Step 3: Run the new tests and confirm RED**

Run:

```bash
MPLCONFIGDIR=/tmp/mpl-egocharm \
  /home/xreal/miniconda3/envs/egocharm/bin/python -m pytest \
  tests/test_preprocess_xreal_imu.py -q
```

Expected: failures for WALK misclassification, missing repeated roots, and old 60/20/20 defaults.

- [ ] **Step 4: Implement deterministic split allocation and source discovery**

Use a largest-remainder allocator with deterministic split-order tie breaking:

```python
def allocate_split_counts(number_of_items, split_ratios=(7, 1, 2)):
    total = sum(split_ratios)
    exact = [number_of_items * ratio / total for ratio in split_ratios]
    counts = [int(value) for value in exact]
    remainder = number_of_items - sum(counts)
    order = sorted(
        range(len(counts)),
        key=lambda index: (-(exact[index] - counts[index]), index),
    )
    for index in order[:remainder]:
        counts[index] += 1
    return tuple(counts)
```

Match `WALK_ON`, `WALK_OFF`, `WORKING_WEAR`, and `NOWEAR` before plain ON/OFF. Change `--source-root` to `action="append"`; when absent, use both approved roots. For event assignment, call `assign_event_splits` separately for every `(date, original_label)` stratum and merge the returned dictionaries.

- [ ] **Step 5: Pass ratios through stable-region assignment**

`assign_stable_window_splits` subtracts two guard slots, calls the new allocator, and returns train slots + guard + val slots + guard + test slots. `build_stable_window_assignments` accepts and forwards `split_ratios` while preserving its existing non-crossing region logic.

- [ ] **Step 6: Run focused and full tests**

Run the focused command from Step 3, then:

```bash
MPLCONFIGDIR=/tmp/mpl-egocharm \
  /home/xreal/miniconda3/envs/egocharm/bin/python -m pytest -q
```

Expected: all tests pass.

---

### Task 2: Add the three-class task and temporal record metadata

**Files:**
- Modify: `src/egocharm/wear_dataset.py`
- Modify: `tests/test_wear_dataset.py`
- Modify: `tests/test_wear_model.py`

**Interfaces:**
- Produces: `THREE_CLASS_MODE = "three_class"`.
- Produces: `THREE_CLASS_NAMES = ("OTHERS", "PUT_ON", "TAKE_OFF")`.
- Extends: `WearRecord` with `source_path: Path | None` and `start_sample: int`.
- Preserves: existing four-class and binary mappings.

- [ ] **Step 1: Write failing three-class mapping tests**

```python
@pytest.mark.parametrize(
    ("source_label", "expected"),
    [
        ("WORN", "OTHERS"),
        ("NOT_WORN", "OTHERS"),
        ("PUT_ON", "PUT_ON"),
        ("TAKE_OFF", "TAKE_OFF"),
    ],
)
def test_three_class_mode_maps_original_labels(source_label, expected):
    module = load_module()
    assert module.map_wear_label(source_label, "three_class") == expected
```

Also assert `get_task_class_names("three_class")` has exact order and a three-class Dataset returns targets 0, 1, and 2.

- [ ] **Step 2: Write a failing selection metadata test**

Add `source_path` and `start_sample` to a fixture entry and assert `load_wear_selection` preserves both. Existing four-positional-argument `WearRecord` construction must remain valid through defaults.

- [ ] **Step 3: Run tests and confirm RED**

Run:

```bash
/home/xreal/miniconda3/envs/egocharm/bin/python -m pytest \
  tests/test_wear_dataset.py tests/test_wear_model.py -q
```

- [ ] **Step 4: Implement constants, mappings, and metadata fields**

Add:

```python
THREE_CLASS_NAMES = ("OTHERS", "PUT_ON", "TAKE_OFF")
THREE_CLASS_MODE = "three_class"

_THREE_CLASS_LABELS = {
    "WORN": "OTHERS",
    "NOT_WORN": "OTHERS",
    "PUT_ON": "PUT_ON",
    "TAKE_OFF": "TAKE_OFF",
}
```

Extend `_TASK_CLASS_NAMES`, `map_wear_label`, and `WearRecord`. Selection loading reads absolute source paths and integer starts, using `None` and zero only for legacy fixtures that omit them.

- [ ] **Step 5: Update configuration validation tests and run GREEN**

Assert `validate_wear_model_config` accepts exactly three outputs in the declared order and rejects reordered labels. Run the focused command and the full suite.

---

### Task 3: Support random end-to-end Xreal model training

**Files:**
- Modify: `src/egocharm/wear_model.py`
- Modify: `src/egocharm/train_wear.py`
- Modify: `tests/test_wear_model.py`
- Modify: `tests/test_train_wear.py`

**Interfaces:**
- Produces: `build_random_wear_model(model_config, label_mode) -> EgoCHARM`.
- Produces: `build_wear_model(model_config, label_mode) -> EgoCHARM`, routing on `model.initialization`.
- Adds CLI overrides: `--seed` and `--high-level-hidden-size`.
- Preserves existing default frozen-LLE behavior for old configs.

- [ ] **Step 1: Write failing random-model tests**

```python
def test_random_end_to_end_model_trains_both_levels():
    module = load_module()
    config = three_class_model_config()
    config["initialization"] = "random_end_to_end"
    model = module.build_wear_model(config, "three_class")
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert model(torch.randn(2, 5, 6, 50)).shape == (2, 3)
```

Patch `torch.load` to fail if called, proving random initialization never opens the Ego checkpoint.

- [ ] **Step 2: Write failing CLI override tests**

Dry-run a minimal random config with `--seed 44 --high-level-hidden-size 16`; assert JSON reports seed 44, HLE size 16, initialization strategy, task classes, output directory, and that training did not start.

- [ ] **Step 3: Run focused tests and confirm RED**

```bash
/home/xreal/miniconda3/envs/egocharm/bin/python -m pytest \
  tests/test_wear_model.py tests/test_train_wear.py -q
```

- [ ] **Step 4: Implement model routing**

Build random LLE through the existing `_build_low_level_encoder`, construct `HighLevelClassifier` with `len(get_task_class_names(label_mode))`, and return ordinary `EgoCHARM`. For frozen mode, require `pretrained_lle_checkpoint` and call the existing builder. Reject unknown initialization strings.

- [ ] **Step 5: Implement training overrides and metadata**

Resolve seed and HLE hidden size before validation and model construction. Record initialization, effective seed, effective HLE size, number of trainable parameters, and best checkpoint metadata. Preserve `create_optimizer` filtering by `requires_grad` so both strategies use the same optimizer code.

- [ ] **Step 6: Run focused and full tests**

Expected: random models train all parameters; legacy frozen tests still pass.

---

### Task 4: Implement task-independent event and direction evaluation

**Files:**
- Create: `src/egocharm/wear_evaluation.py`
- Modify: `src/egocharm/evaluate_wear.py`
- Create: `tests/test_wear_evaluation.py`
- Modify: `tests/test_evaluate_wear.py`

**Interfaces:**
- Produces: `event_probabilities(logits, class_names) -> np.ndarray`.
- Produces: `select_event_threshold(labels, probabilities) -> float`.
- Produces: `calculate_common_metrics(records, probabilities, class_names, threshold, sampling_rate_hz, window_seconds) -> dict`.
- Adds evaluator flags: `--event-threshold` and `--select-event-threshold`.
- Writes: `predictions.csv`, `metrics.json`, and `event-threshold.json` for validation selection.

- [ ] **Step 1: Write failing probability and threshold tests**

For four/three classes, assert event probability is the sum of PUT_ON and TAKE_OFF softmax probabilities. For binary, assert it is `P(EVENT)`. Provide a small fixed label/probability vector and assert threshold selection maximizes event F1 with deterministic lower-threshold tie breaking.

- [ ] **Step 2: Write failing direction-metric tests**

Create true PUT_ON/TAKE_OFF records and fixed class probabilities. Assert direction Accuracy and Macro-F1 are computed only on true event samples and are absent for binary mode.

- [ ] **Step 3: Write failing interval-merging tests**

```python
def test_overlapping_positive_windows_form_one_false_alarm():
    # Starts 0, 50, and 100 at 50 Hz with five-second windows overlap.
    # They must count as one alarm interval, not three alarms.
```

Assert stable exposure uses the union of `[start, start + 250)` intervals per source. Verify that two disjoint positive groups yield exactly two false alarms and scale by union duration hours.

- [ ] **Step 4: Run the new module tests and confirm RED**

```bash
/home/xreal/miniconda3/envs/egocharm/bin/python -m pytest \
  tests/test_wear_evaluation.py tests/test_evaluate_wear.py -q
```

- [ ] **Step 5: Implement pure metric functions**

Keep the new module free of model loading and filesystem side effects. Convert original labels to event truth, compute precision/recall/F1 with zero-division handling, calculate direction metrics for non-binary tasks, merge intervals per source, and report both window FPR and merged false alarms/hour.

- [ ] **Step 6: Integrate inference and persisted threshold flow**

Evaluator saves one row per record with UID, original label, source path, start sample, predicted task label, event probability, and event prediction. `--select-event-threshold` is valid only for val and writes the chosen value. Test evaluation requires the numeric `--event-threshold` supplied by the runner and never searches test labels.

- [ ] **Step 7: Run focused and full tests**

Expected: old metric files remain available, while corrected overlap-aware metrics are added.

---

### Task 5: Add configs and an HLE-search/comparison runner

**Files:**
- Create: `configs/xreal_end_to_end_four_class.yaml`
- Create: `configs/xreal_end_to_end_three_class.yaml`
- Create: `configs/xreal_end_to_end_binary_event.yaml`
- Create: `scripts/run_xreal_end_to_end_comparison.py`
- Create: `tests/test_xreal_end_to_end_comparison.py`
- Modify: `Xreal_datasets/README.md`

**Interfaces:**
- Runner stages: `hle-search`, `formal`, `all`.
- Writes: `artifacts/xreal_wear_detection/runs/xreal_end_to_end/comparison/selected-hle.json`.
- Writes: per-seed metrics and aggregate `summary.json` plus `summary.csv`.

- [ ] **Step 1: Add failing configuration tests**

Load all three YAML files. Assert random initialization, exact task class order, common data selection, 50 Hz, five seconds, gravity normalization, equal training hyperparameters, and common output root.

- [ ] **Step 2: Add failing HLE-selection tests**

Feed synthetic validation records for hidden sizes 16, 32, and 64. Assert the runner maximizes
`(event_f1 + direction_macro_f1) / 2`, applies the 0.005 smaller-model tie rule, and never accepts a test metrics path for selection.

- [ ] **Step 3: Add failing orchestration and aggregation tests**

Mock subprocess execution and assert HLE search launches only three-class seed 42 without test evaluation. After selection, assert formal commands cover all Cartesian pairs of three modes and seeds 42/43/44, pass the selected hidden size, evaluate val with threshold selection, then test with that stored threshold. Verify mean and population standard deviation output fields.

- [ ] **Step 4: Run tests and confirm RED**

```bash
/home/xreal/miniconda3/envs/egocharm/bin/python -m pytest \
  tests/test_xreal_end_to_end_comparison.py -q
```

- [ ] **Step 5: Create the three configs**

Use the same LLE parameters and training hyperparameters as the current Xreal baseline, but set `model.initialization: random_end_to_end`, remove the pretrained checkpoint, declare the correct classes, and set output root to `artifacts/xreal_wear_detection/runs/xreal_end_to_end`.

- [ ] **Step 6: Implement runner stages**

Use `subprocess.run(..., check=True)` with argument lists. Never invoke shell strings. HLE search trains 16/32/64, evaluates val only, computes the specified score, and persists the choice. Formal mode requires that choice, runs nine isolated directories, evaluates val, reads each validation threshold, evaluates test, and aggregates requested metrics.

- [ ] **Step 7: Document commands and run full tests**

Document preprocessing, search-only, formal-only, and all-stage commands. Run all tests and compile all Python sources.

---

### Task 6: Regenerate and verify the real merged cache

**Files:**
- Replace: `Xreal_datasets/metadata/xreal_wear_selection.json`
- Replace/add: `artifacts/xreal_wear_detection/cache/xreal_cpf/*.npz`

**Interfaces:**
- Consumes the completed converter from Task 1.
- Produces the single shared selection used by all experiments.

- [ ] **Step 1: Record protected run-directory inventory**

Hash or list all files beneath existing run directories outside `xreal_end_to_end`; retain this output for the final unchanged check.

- [ ] **Step 2: Run a dry run over both roots**

```bash
python Xreal_datasets/preprocess_xreal_imu.py --dry-run
```

Expected source counts include 62 PUT_ON, 55 TAKE_OFF, two WORN long recordings, and two NOT_WORN long recordings.

- [ ] **Step 3: Regenerate cache and selection**

```bash
python Xreal_datasets/preprocess_xreal_imu.py \
  --stable-stride-seconds 1
```

Do not pass `--reuse-existing-events`; all event metadata must receive the new stratified split.

- [ ] **Step 4: Verify data integrity**

Check every selected path exists, every NPZ has values `(250, 6)` and timestamps `(250,)`, UIDs are unique, label/split counts match source-level assignment, stable starts differ by 50 samples inside each region, and no source file appears in multiple event splits.

- [ ] **Step 5: Verify historical runs are unchanged**

Compare the inventory from Step 1. Only cache, selection, new configs/docs/code, and the new `xreal_end_to_end` output root may differ.

---

### Task 7: Run HLE search, nine formal trainings, and final evaluation

**Files:**
- Create outputs beneath: `artifacts/xreal_wear_detection/runs/xreal_end_to_end/`

**Interfaces:**
- Consumes the shared merged selection and runner.
- Produces the requested comparison results.

- [ ] **Step 1: Run complete experiment orchestration on CUDA**

```bash
python scripts/run_xreal_end_to_end_comparison.py --stage all
```

Monitor every process. Stop on the first failure; do not silently skip a seed or mode.

- [ ] **Step 2: Verify HLE search isolation**

Confirm only validation files exist in `hle_search`, `selected-hle.json` contains 16, 32, or 64 plus all three candidate scores, and no HLE-search test directory exists.

- [ ] **Step 3: Verify all formal outputs**

For each of nine mode/seed directories, require `best.pt`, `last.pt`, resolved config, val metrics/predictions/threshold, and test metrics/predictions/confusion matrices.

- [ ] **Step 4: Review the aggregate comparison**

Validate aggregate sample counts against selection, ensure test thresholds equal their corresponding val-selected thresholds, and inspect per-seed variance before drawing conclusions.

- [ ] **Step 5: Run final verification**

```bash
MPLCONFIGDIR=/tmp/mpl-egocharm \
  /home/xreal/miniconda3/envs/egocharm/bin/python -m pytest -q
/home/xreal/miniconda3/envs/egocharm/bin/python -m compileall -q \
  src scripts Xreal_datasets
```

Report selected HLE size, data distribution, nine-run mean/std table, event/direction trade-offs, corrected false alarms/hour, test limitations, and clickable output paths.
