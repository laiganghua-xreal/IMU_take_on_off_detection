# Acceleration Gravity Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional Dataset-time `gravity` normalization mode that divides all three accelerometer channels by `9.8` and leaves gyroscope channels unchanged.

**Architecture:** Keep NPZ caches raw and apply a named window transformation inside `EgoCharmDataset.__getitem__` before reshaping. A small dispatcher provides `none` and `gravity` methods, while train and evaluation resolve the method from `data.normalization` with a backward-compatible `none` default.

**Tech Stack:** Python 3.10, NumPy, PyTorch Dataset/DataLoader, PyYAML, pytest

## Global Constraints

- Do not modify or regenerate any NPZ cache.
- Preserve existing model attributes, state-dict keys, CLI commands, and checkpoint loading behavior.
- `gravity` divides channels `0:3` by exactly `9.8`; channels `3:6` remain unchanged.
- Missing `data.normalization` means `none`.
- The checked-in `configs/egocharm.yaml` selects `gravity`.
- Unknown normalization names raise `ValueError` before a training or evaluation loop begins.
- Normalized samples remain finite, contiguous `float32` tensors with shape `(S, 6, T)`.
- The existing `baseline` checkpoint was trained with raw input and must not be resumed for normalized training.
- This directory is not a Git repository, so verification checkpoints replace commit steps.

---

### Task 1: Dataset Window Normalization Strategy

**Files:**
- Modify: `tests/test_dataset.py`
- Modify: `src/egocharm/dataset.py`

**Interfaces:**
- Produces: `normalize_window(values: np.ndarray, method: str) -> np.ndarray`
- Produces: `EgoCharmDataset(..., normalization="none")`
- Consumes: cached window values with shape `(S * T, 6)` and dtype `float32`

- [x] **Step 1: Write failing tests for gravity, none, and invalid methods**

Add tests that construct channel-distinct values, request `gravity`, and assert:

```python
expected_acceleration = original_values[:, :3] / 9.8
expected_gyroscope = original_values[:, 3:]
np.testing.assert_allclose(actual_values[:, :3], expected_acceleration)
np.testing.assert_array_equal(actual_values[:, 3:], expected_gyroscope)
```

Keep the existing raw-value test as proof that the default `none` mode remains unchanged. Add:

```python
with pytest.raises(ValueError, match="normalization"):
    EgoCharmDataset(window_index, 50, 30, normalization="unknown")
```

- [x] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
/home/xreal/miniconda3/envs/egocharm/bin/python -m pytest \
  -p no:cacheprovider tests/test_dataset.py -q
```

Expected: FAIL because `EgoCharmDataset` does not accept `normalization` and no gravity transformation exists.

- [x] **Step 3: Implement the minimal normalization dispatcher**

In `dataset.py`, define focused functions and a dispatcher table:

```python
GRAVITY_METERS_PER_SECOND_SQUARED = 9.8


def _keep_raw_window(values):
    return values


def _divide_acceleration_by_gravity(values):
    normalized = values.copy()
    normalized[:, :3] /= GRAVITY_METERS_PER_SECOND_SQUARED
    return normalized


WINDOW_NORMALIZERS = {
    "none": _keep_raw_window,
    "gravity": _divide_acceleration_by_gravity,
}


def normalize_window(values, method):
    try:
        normalizer = WINDOW_NORMALIZERS[method]
    except KeyError as error:
        raise ValueError(f"不支持的 normalization: {method}") from error
    return normalizer(values)
```

Validate the method in `EgoCharmDataset.__init__`, store it as
`self.normalization`, and call `normalize_window(values, self.normalization)`
after finite-value validation and before `(S * T, C) -> (S, C, T)` reshaping.

- [x] **Step 4: Run Dataset tests and verify GREEN**

Run the focused command from Step 2. Expected: all Dataset tests pass.

- [x] **Step 5: Verification checkpoint**

Inspect the diff-equivalent changed sections and confirm that NPZ writing,
window indexing, labels, and DataLoader behavior were not changed.

---

### Task 2: Training and Evaluation Configuration Wiring

**Files:**
- Modify: `tests/test_train.py`
- Modify: `tests/test_evaluate.py`
- Modify: `src/egocharm/train.py`
- Modify: `src/egocharm/evaluate.py`

**Interfaces:**
- Consumes: `data_config.get("normalization", "none")`
- Consumes: `EgoCharmDataset(..., normalization=<resolved method>)`
- Produces: identical normalization selection in training, validation, and test evaluation

- [x] **Step 1: Write failing orchestration tests**

Monkeypatch `EgoCharmDataset` with a capturing callable. For train, capture both
Dataset constructions and assert their fourth argument is `gravity`. For
evaluation, capture the single Dataset construction and assert `gravity`.
Add a second assertion path with no `normalization` key and expect `none`.
Stop orchestration immediately after Dataset construction using a local sentinel
exception so no training or evaluation loop runs.

- [x] **Step 2: Run focused CLI tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
/home/xreal/miniconda3/envs/egocharm/bin/python -m pytest \
  -p no:cacheprovider tests/test_train.py tests/test_evaluate.py -q
```

Expected: FAIL because train and evaluation currently construct each Dataset
with only three arguments.

- [x] **Step 3: Pass the resolved method into every Dataset**

In both CLI modules resolve once:

```python
normalization = str(data_config.get("normalization", "none"))
```

Pass `normalization=normalization` to train, validation, and evaluation Dataset
instances. Do not change CLI arguments or checkpoint structures.

- [x] **Step 4: Run focused CLI tests and verify GREEN**

Run the command from Step 2. Expected: all train and evaluation tests pass.

- [x] **Step 5: Verification checkpoint**

Confirm train and validation use the same resolved method and evaluation does
not silently choose a different default.

---

### Task 3: Default Configuration, Documentation, and End-to-End Verification

**Files:**
- Modify: `configs/egocharm.yaml`
- Modify: `README.md`
- Test: `tests/test_dataset.py`
- Test: `tests/test_train.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Produces: checked-in `data.normalization: gravity`
- Documents: raw NPZ values versus Dataset-time normalized model input

- [x] **Step 1: Set and document the checked-in mode**

Add below `window_seconds`:

```yaml
normalization: gravity
```

Update README data-flow text to state that NPZ retains physical units, while
Dataset `gravity` mode divides acceleration channels by `9.8` immediately after
loading each complete window and leaves gyro values unchanged. Document `none`
as the compatibility option and warn that the raw-input `baseline` checkpoint
must not be resumed for a normalized experiment.

- [x] **Step 2: Run the complete automated suite**

Run:

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
MPLCONFIGDIR=/tmp/egocharm-matplotlib \
/home/xreal/miniconda3/envs/egocharm/bin/python -m pytest \
  -p no:cacheprovider -q
```

Expected: all tests pass with no new warning or error.

- [x] **Step 3: Run compilation verification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
/home/xreal/miniconda3/envs/egocharm/bin/python -m compileall -q scripts src tests
```

Expected: exit code `0`.

- [x] **Step 4: Verify a real cached window without training**

Load one real train window twice, once with `none` and once with `gravity`, and
assert:

```python
torch.testing.assert_close(gravity[:, :3], raw[:, :3] / 9.8)
torch.testing.assert_close(gravity[:, 3:], raw[:, 3:])
assert gravity.shape == (30, 6, 50)
assert gravity.dtype == torch.float32
```

Do not start or resume training and do not rewrite any NPZ.

- [x] **Step 5: Confirm process and artifact safety**

Verify no `egocharm.train` process is running, no NPZ modification times changed,
and the existing `artifacts/runs/baseline` files remain untouched.
