# Remove Input Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove train mean/standard-deviation normalization so Dataset returns the resampled IMU values stored in NPZ without scaling.

**Architecture:** Delete normalization production and configuration from preprocessing, then simplify Dataset construction to require only window geometry. Training and evaluation construct the simplified Dataset directly. Model-internal `BatchNorm1d` layers remain unchanged.

**Tech Stack:** Python 3.10, NumPy 1.26, PyTorch 2.5, pytest 8.

## Global Constraints

- Remove only input data normalization.
- Keep all model `BatchNorm1d` layers and exact parameter counts.
- Preserve NPZ values, signal preprocessing, official splits, window geometry, and weighted loss.
- Do not read real VRS or run real training during verification.
- Use explicit beginner-readable code and Chinese comments.
- The workspace is not a Git repository; use test checkpoints instead of commits.

---

### Task 1: Dataset Returns Unscaled NPZ Values

**Files:**
- Modify: `tests/test_dataset.py`
- Modify: `src/egocharm/dataset.py`

**Interfaces:**
- Produces: `EgoCharmDataset(window_references, sampling_rate_hz, window_seconds)`
- Removes: `NormalizationStatistics` and `load_normalization`

- [ ] **Step 1: Write a failing unscaled-value test**

```python
def test_dataset_returns_unscaled_cache_values(tmp_path):
    cache_root = tmp_path / "cache"
    write_constant_cache(cache_root, value=12.5)
    index = build_window_index(
        [make_record()], cache_root, "train", 50, 30, 10
    )
    dataset = EgoCharmDataset(index, 50, 30)
    features, _ = dataset[0]
    assert torch.all(features == 12.5)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest tests/test_dataset.py -q`

Expected: FAIL because the current constructor requires normalization statistics.

- [ ] **Step 3: Remove normalization from Dataset**

Delete `NormalizationStatistics`, `load_normalization`, the constructor parameter and stored attribute. In `__getitem__`, validate `values`, reshape `(1500,6)` directly to `(30,50,6)`, transpose to `(30,6,50)`, make float32 contiguous, and return it without subtraction or division.

- [ ] **Step 4: Run Dataset tests and verify GREEN**

Run: `python -m pytest tests/test_dataset.py -q`

Expected: all Dataset tests pass and the literal 12.5 value is unchanged.

### Task 2: Remove Normalization Production and Consumers

**Files:**
- Modify: `tests/test_preprocess_data.py`
- Modify: `tests/test_integration.py`
- Modify: `scripts/preprocess_data.py`
- Modify: `src/egocharm/train.py`
- Modify: `src/egocharm/evaluate.py`
- Modify: `configs/egocharm.yaml`

**Interfaces:**
- Removes: `fit_training_normalization`
- Removes: `--fit-normalization`
- Removes: `data.normalization_path`

- [ ] **Step 1: Change tests to the desired no-normalization workflow**

Delete normalization statistic tests and imports. Add:

```python
def test_parser_rejects_removed_fit_normalization_option():
    with pytest.raises(SystemExit):
        main(["--fit-normalization"])
```

Update integration construction to `EgoCharmDataset(index, 50, 30)` and remove all normalization JSON creation.

- [ ] **Step 2: Run preprocessing and integration tests and verify RED**

Run: `python -m pytest tests/test_preprocess_data.py tests/test_integration.py -q`

Expected: FAIL because the parser still accepts `--fit-normalization` and consumers still use normalization.

- [ ] **Step 3: Remove normalization production and consumers**

Delete Welford accumulation, normalization JSON writing, parser flag, and post-preprocessing normalization branch. Remove `normalization_path` from YAML. Remove `load_normalization` imports and calls from train/evaluate, then use `EgoCharmDataset(index, rate, window)`.

- [ ] **Step 4: Run affected tests and verify GREEN**

Run: `python -m pytest tests/test_preprocess_data.py tests/test_integration.py tests/test_train.py tests/test_evaluate.py -q`

Expected: all affected tests pass.

### Task 3: Documentation and Final Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/simplified-beginner-workflow-design.md`

**Interfaces:**
- Documents raw resampled sensor units entering Dataset.

- [ ] **Step 1: Remove normalization instructions from documentation**

Delete the `--fit-normalization` command, normalization artifact, train-only statistics description, and normalized shape wording. State that Dataset directly returns resampled acceleration in `m/s²` and angular velocity in `rad/s`, while model `BatchNorm1d` remains.

- [ ] **Step 2: Run complete verification**

Run: `python -m compileall -q scripts src tests`

Run: `python -m pytest -p no:cacheprovider -q`

Run: `python scripts/preprocess_data.py --config configs/egocharm.yaml --split train --max-takes 1 --dry-run`

Run: `python -m egocharm.train --config configs/egocharm.yaml --run-name smoke --dry-run`

Expected: compilation and all tests pass; both dry-runs report that neither VRS reading nor training started.

- [ ] **Step 3: Confirm model parameter counts remain exact**

Run a Python check constructing default low-level and seven/nine-class high-level modules.

Expected: 21,863; 63,111; and 63,369 respectively.
