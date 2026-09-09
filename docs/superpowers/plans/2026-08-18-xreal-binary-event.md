# Xreal Binary Wear-Event Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional `NO_EVENT` versus `EVENT` training and evaluation mode while preserving the existing Xreal four-class experiment.

**Architecture:** Keep original four-class labels in the selection and map them inside `WearDataset` according to `task.label_mode`. Reuse the frozen CPF LLE, create a mode-sized HLE, and make training/checkpoint/evaluation class names mode-aware. Add a separate binary configuration, run directory, and original-label event breakdown.

**Tech Stack:** Python 3.10, NumPy, PyTorch, scikit-learn, PyYAML, pytest.

## Global Constraints

- `four_class` remains the default when `task.label_mode` is absent.
- `binary_event` maps `WORN` and `NOT_WORN` to `NO_EVENT`, and `PUT_ON` and `TAKE_OFF` to `EVENT`.
- Input remains 50 Hz, five seconds, six channels, CPF directions, and gravity normalization.
- The Ego CPF-pretrained LLE remains frozen and in evaluation mode.
- Binary artifacts use `artifacts/xreal_wear_detection/runs/binary_event` and never overwrite `frozen_lle`.
- No state machine, direction head, additional features, or cache regeneration is included.
- The workspace is not a valid Git repository, so commit steps are omitted.

---

### Task 1: Mode-aware Dataset labels

**Files:**
- Modify: `src/egocharm/wear_dataset.py`
- Test: `tests/test_wear_dataset.py`

**Interfaces:**
- Produces: `FOUR_CLASS_MODE`, `BINARY_EVENT_MODE`, `EVENT_CLASS_NAMES`, `get_task_class_names(label_mode)`, `map_wear_label(label, label_mode)`, and `WearDataset(..., label_mode="four_class")`.
- Preserves: `WEAR_CLASS_NAMES`, `WearRecord`, and four-class behavior.

- [ ] **Step 1: Write failing mapping and Dataset tests**

Add table-driven literal expectations:

```python
@pytest.mark.parametrize(
    ("source_label", "expected_target"),
    [
        ("WORN", "NO_EVENT"),
        ("NOT_WORN", "NO_EVENT"),
        ("PUT_ON", "EVENT"),
        ("TAKE_OFF", "EVENT"),
    ],
)
def test_binary_event_mode_maps_original_labels(source_label, expected_target):
    module = load_module()
    assert module.map_wear_label(source_label, "binary_event") == expected_target
```

Add a real cache-backed Dataset test asserting that `PUT_ON` returns target
index `1`, `WORN` returns `0`, and `class_labels` contains mapped indices.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest tests/test_wear_dataset.py -q`

Expected: fail because binary label-mode interfaces do not exist.

- [ ] **Step 3: Implement minimal mode-aware mapping**

Use fixed mappings, validate label modes explicitly, keep the original label
on each record, and derive Dataset targets at `__getitem__` and
`class_labels` time.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `pytest tests/test_wear_dataset.py -q`

Expected: all tests pass.

---

### Task 2: Mode-sized frozen transfer model

**Files:**
- Modify: `src/egocharm/wear_model.py`
- Test: `tests/test_wear_model.py`
- Test: `tests/test_evaluate_wear.py`

**Interfaces:**
- Consumes: `get_task_class_names(label_mode)`.
- Produces: `validate_wear_model_config(model_config, label_mode="four_class")` and `build_frozen_wear_model(model_config, checkpoint_path, label_mode="four_class")`.

- [ ] **Step 1: Write failing binary model tests**

Create a binary model config with class names `NO_EVENT`, `EVENT` and two
classes. Assert the real model emits `(2, 2)`, all LLE parameters remain
frozen, and an incompatible class order raises `ValueError`.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `pytest tests/test_wear_model.py tests/test_evaluate_wear.py -q`

Expected: fail because validation and construction are fixed to four classes.

- [ ] **Step 3: Implement mode-aware validation and head construction**

Validate exact class order against `get_task_class_names(label_mode)`. Build
the HLE output with `len(class_names)` while retaining the existing LLE load,
freeze, and forced-eval behavior.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `pytest tests/test_wear_model.py tests/test_evaluate_wear.py -q`

Expected: all tests pass.

---

### Task 3: Mode-aware training and binary evaluation breakdown

**Files:**
- Modify: `src/egocharm/train_wear.py`
- Modify: `src/egocharm/evaluate_wear.py`
- Test: `tests/test_train_wear.py`
- Test: `tests/test_evaluate_wear.py`

**Interfaces:**
- Produces: `get_label_mode(config)`, `_build_dataset(config, split, label_mode)`, and `calculate_original_label_breakdown(records, predictions, event_class_index, window_seconds)`.
- Checkpoint metadata adds: `label_mode`.

- [ ] **Step 1: Write failing training-mode tests**

Assert that a dry-run binary config validates `NO_EVENT`, `EVENT`, resolves an
isolated run directory, and that a checkpoint mismatch is protected by
mode-specific class names.

- [ ] **Step 2: Write failing event-breakdown test**

Use literal original labels and predictions:

```python
records = [
    WearRecord("w1", Path("w1.npz"), "WORN", "test"),
    WearRecord("n1", Path("n1.npz"), "NOT_WORN", "test"),
    WearRecord("p1", Path("p1.npz"), "PUT_ON", "test"),
    WearRecord("t1", Path("t1.npz"), "TAKE_OFF", "test"),
]
predictions = [1, 0, 1, 0]
```

Assert `WORN` false-event rate `1.0`, `NOT_WORN` false-event rate `0.0`,
`PUT_ON` event recall `1.0`, `TAKE_OFF` event recall `0.0`, and `WORN`
false-events-per-hour `720.0` for one five-second window.

- [ ] **Step 3: Run focused tests and verify RED**

Run: `pytest tests/test_train_wear.py tests/test_evaluate_wear.py -q`

Expected: fail because training mode propagation and breakdown do not exist.

- [ ] **Step 4: Implement training mode propagation**

Default absent mode to `four_class`, select mode-specific class names, pass
the mode into Dataset/model construction, compute class weights from mapped
targets, size loss metrics by mode, and save `label_mode` in metadata.

- [ ] **Step 5: Implement evaluation mode and breakdown**

Load the mode-specific model/checkpoint, calculate ordinary task metrics, and
for binary mode append `original_label_breakdown` without changing the files
written by `save_metrics`.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run: `pytest tests/test_train_wear.py tests/test_evaluate_wear.py -q`

Expected: all tests pass.

---

### Task 4: Binary configuration and documentation

**Files:**
- Create: `configs/xreal_wear_binary_event.yaml`
- Modify: `Xreal_datasets/README.md`

**Interfaces:**
- Training command: `python -m egocharm.train_wear --config configs/xreal_wear_binary_event.yaml --run-name binary_event`.
- Evaluation commands use the same config and `runs/binary_event/best.pt`.

- [ ] **Step 1: Add the isolated binary configuration**

Copy the proven five-second frozen-LLE settings, add:

```yaml
task:
  label_mode: binary_event
model:
  class_names: [NO_EVENT, EVENT]
  number_of_classes: 2
```

Keep the existing CPF checkpoint and output root.

- [ ] **Step 2: Document training, evaluation, mapping, and output paths**

State that no preprocessing is needed, direction is not predicted, and this
experiment uses only the existing 50 Hz six-axis cache.

- [ ] **Step 3: Run dry-run configuration validation**

Run: `python -m egocharm.train_wear --config configs/xreal_wear_binary_event.yaml --run-name binary_event --dry-run`

Expected: JSON reports the binary run directory and no training start.

---

### Task 5: Regression verification, training, evaluation, and analysis

**Files:**
- Produce: `artifacts/xreal_wear_detection/runs/binary_event/best.pt`
- Produce: `artifacts/xreal_wear_detection/runs/binary_event/last.pt`
- Produce: `artifacts/xreal_wear_detection/runs/binary_event/{val,test}`

**Interfaces:**
- Consumes the completed binary implementation and existing Xreal cache.
- Produces verified binary metrics and an evidence-based comparison with the four-class baseline.

- [ ] **Step 1: Run full tests and compilation**

Run: `pytest -q`

Run: `python -m compileall -q src scripts Xreal_datasets`

Expected: zero failures and zero compilation errors.

- [ ] **Step 2: Train the isolated binary run**

Run: `python -m egocharm.train_wear --config configs/xreal_wear_binary_event.yaml --run-name binary_event`

Expected: 30 epochs and mode-compatible `best.pt`/`last.pt`.

- [ ] **Step 3: Evaluate validation and test splits**

Run both `python -m egocharm.evaluate_wear` commands with binary config,
binary checkpoint, and `--split val` / `--split test`.

- [ ] **Step 4: Inspect fresh artifacts and analyze**

Read the checkpoint epoch, validation/test binary confusion matrices, event
precision/recall/F1, per-original-label detection, and stable false events per
hour. Compare against the collapsed behavior of the prior four-class model,
and distinguish conclusions supported by the small test set from hypotheses.
