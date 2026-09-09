# Simplified Beginner EgoCHARM Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current layered CLI with one standalone preprocessing script and a small, beginner-readable PyTorch package for cached data loading, training, and evaluation.

**Architecture:** `scripts/preprocess_data.py` is the only module that imports Project Aria Tools or opens VRS files. `src/egocharm` starts at the published NPZ cache boundary and contains one file each for configuration, Dataset/DataLoader, model, training, and evaluation. New code and tests are established first; old modules and disposable caches are deleted only after the replacement passes complete verification.

**Tech Stack:** Python 3.10, NumPy 1.26, PyTorch 2.5, scikit-learn 1.7, PyYAML 6, Matplotlib 3.10, Project Aria Tools 2.1, pytest 8.

## Global Constraints

- Write explicit beginner-readable control flow and meaningful full variable names.
- Add Chinese comments at data-shape changes, algorithm decisions, and PyTorch state transitions.
- Do not use complex nested comprehensions, nested ternary expressions, or dynamic dispatch tables in the new learning path.
- Preserve the official Ego-Exo4D train/val/test split and fixed seven-class order.
- Only `scripts/preprocess_data.py` may import Project Aria Tools or open VRS files.
- Never write to, rename, move, or delete anything below `ego-exoD4_dataset/takes/`.
- Do not read a real VRS during implementation verification.
- Fit normalization from train caches only.
- Default input geometry is `(batch, 30, 6, 50)` using 30-second windows and 10-second stride.
- Preserve model parameter counts: low-level 21,863; seven-class high-level 63,111; nine-class high-level 63,369.
- The directory is not a valid Git repository, so skip commit commands and use test checkpoints instead.

---

### Task 1: Single Configuration and Package Contract

**Files:**
- Create: `configs/egocharm.yaml`
- Rewrite: `src/egocharm/config.py`
- Rewrite: `tests/test_config.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `load_config(path: Path) -> dict[str, Any]`
- Produces: `project_path(value: str | Path) -> Path`
- Produces: `set_random_seed(seed: int) -> None`

- [ ] **Step 1: Write failing tests for the single configuration**

```python
def test_load_config_returns_all_beginner_workflow_sections(tmp_path):
    path = write_config(tmp_path)
    config = load_config(path)
    assert set(config) == {"data", "model", "training"}
    assert config["data"]["sampling_rate_hz"] == 50

def test_project_path_keeps_absolute_paths():
    path = Path("/tmp/example.npz")
    assert project_path(path) == path
```

- [ ] **Step 2: Run the configuration tests and verify RED**

Run: `python -m pytest tests/test_config.py -q`

Expected: FAIL because the simplified `load_config` contract does not exist.

- [ ] **Step 3: Create one commented YAML configuration**

Define `data.selection`, `data.dataset_root`, `data.cache_root`, `data.normalization_path`, `sampling_rate_hz=50`, `max_gap_s=0.1`, `minimum_segment_seconds=30`, `window_seconds=30`, and `stride_seconds=10`. Define the seven classes and exact model dimensions. Define seed 42, 30 epochs, batch size 32, four workers, Adam learning rate 0.001, StepLR 10/0.1, AMP enabled, automatic device, and `artifacts/runs`.

- [ ] **Step 4: Implement explicit configuration helpers**

`load_config` must reject missing top-level sections with a message naming the section. `project_path` must resolve relative paths against `/home/xreal/egocharm`. `set_random_seed` must seed `random`, NumPy, PyTorch CPU, and all available CUDA devices.

- [ ] **Step 5: Remove the console entry point from package metadata and run tests**

Remove `egocharm = "egocharm.cli.main:main"`. Keep editable package discovery under `src`.

Run: `python -m pytest tests/test_config.py -q`

Expected: all configuration tests pass.

### Task 2: Standalone VRS Preprocessing Script

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/preprocess_data.py`
- Create: `tests/test_preprocess_data.py`

**Interfaces:**
- Consumes: selection JSON entries and `configs/egocharm.yaml`
- Produces: `RawImu(timestamps_seconds, values)`
- Produces: `clean_samples(timestamps, values) -> tuple[np.ndarray, np.ndarray]`
- Produces: `find_continuous_segments(timestamps, maximum_gap_seconds) -> list[slice]`
- Produces: `resample_segment(timestamps, values, sampling_rate_hz) -> tuple[np.ndarray, np.ndarray]`
- Produces: `preprocess_take(record, settings, reader=read_left_imu) -> Path`
- Produces: `fit_training_normalization(records, cache_root, output_path) -> None`
- Produces: command `python scripts/preprocess_data.py --config configs/egocharm.yaml --split train`

- [ ] **Step 1: Write failing pure signal tests**

```python
def test_clean_samples_sorts_and_keeps_last_duplicate():
    timestamps = np.array([0.02, 0.00, 0.01, 0.01])
    values = np.array([[2], [0], [1], [9]], dtype=np.float32)
    clean_timestamps, clean_values = clean_samples(timestamps, values)
    np.testing.assert_allclose(clean_timestamps, [0.00, 0.01, 0.02])
    np.testing.assert_allclose(clean_values[:, 0], [0, 9, 2])

def test_resample_segment_produces_regular_linear_values():
    timestamps = np.array([0.0, 0.5, 1.0])
    values = np.array([[0.0], [1.0], [2.0]])
    output_timestamps, output_values = resample_segment(timestamps, values, 4)
    np.testing.assert_allclose(output_timestamps, [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(output_values[:, 0], [0.0, 0.5, 1.0, 1.5, 2.0])
```

- [ ] **Step 2: Run the signal tests and verify RED**

Run: `python -m pytest tests/test_preprocess_data.py -q`

Expected: FAIL because `scripts.preprocess_data` does not exist.

- [ ] **Step 3: Implement the explicit VRS adapter and signal functions**

Place the lazy `projectaria_tools` import inside `read_left_imu`. Reject paths not ending exactly in `_noimagestreams.vrs`. Read only the `imu-left` label, skip rows without valid accelerometer and gyroscope values, convert nanoseconds to float64 seconds, and construct float32 values in `[accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z]` order. Add Chinese comments before each conversion.

- [ ] **Step 4: Write failing atomic cache tests with an injected reader**

```python
def test_preprocess_take_writes_complete_cache_without_real_vrs(tmp_path):
    source, record, settings = fixture_take(tmp_path)
    output = preprocess_take(record, settings, reader=fake_imu_reader)
    with np.load(output, allow_pickle=False) as cache:
        assert cache["values"].shape[1] == 6
        assert cache["segment_bounds"].shape[1] == 2
    assert source.read_bytes() == b"vrs!"
    assert not list(output.parent.glob("*.tmp"))
```

- [ ] **Step 5: Implement readable per-take preprocessing and atomic publication**

Check source existence and exact size before calling the reader. Clean samples, split where gaps exceed 0.1 seconds, discard segments under 30 seconds, resample each surviving segment separately, concatenate them while preserving half-open segment bounds, then write a same-directory temporary NPZ. Re-open and validate the temporary file before publishing it with `os.replace`. Cache metadata must include take UID, activity, official split, participant UID, source size/mtime, six-channel order, and processing settings.

- [ ] **Step 6: Write and implement train-only normalization tests**

Test that a val cache raises `ValueError`, two incremental batches match a direct population mean/std calculation, and zero-variance channels are rejected. Implement float64 Welford accumulation and atomically write `artifacts/normalization/train.json`.

- [ ] **Step 7: Implement the beginner-facing parser and dry-run**

Support `--config`, `--split {train,val,test,all}`, repeatable `--take-uid`, `--max-takes`, `--dry-run`, `--fit-normalization`, and `--fail-fast`. Use explicit `if/elif` command flow. Dry-run may read selection metadata but must not call `Path.stat` or open a VRS.

- [ ] **Step 8: Run standalone preprocessing tests**

Run: `python -m pytest tests/test_preprocess_data.py -q`

Expected: all tests pass without accessing `ego-exoD4_dataset/takes/`.

### Task 3: Cache-Only Dataset and DataLoader

**Files:**
- Create: `src/egocharm/dataset.py`
- Create: `tests/test_dataset.py`

**Interfaces:**
- Produces: immutable `TakeRecord` and `WindowReference`
- Produces: `load_selection(path: Path) -> list[TakeRecord]`
- Produces: `load_normalization(path: Path) -> NormalizationStatistics`
- Produces: `build_window_index(records, cache_root, split, rate, window, stride) -> list[WindowReference]`
- Produces: `EgoCharmDataset`
- Produces: `create_data_loader(dataset, batch_size, shuffle, workers) -> DataLoader`

- [ ] **Step 1: Write failing Dataset geometry tests**

```python
def test_dataset_returns_one_hierarchical_window(tmp_path):
    cache_path = write_cache(tmp_path, seconds=31, rate=50)
    dataset = build_dataset_for_cache(cache_path)
    features, label = dataset[0]
    assert features.shape == (30, 6, 50)
    assert features.dtype == torch.float32
    assert label.dtype == torch.long

def test_window_index_never_crosses_a_signal_gap(tmp_path):
    cache_path = write_two_segment_cache(tmp_path)
    record = fixture_record(take_uid="fixture", split="train")
    index = build_window_index(
        [record], cache_path.parent, "train", 50, 30, 10
    )
    assert all(window_is_inside_one_segment(item, cache_path) for item in index)
```

- [ ] **Step 2: Run Dataset tests and verify RED**

Run: `python -m pytest tests/test_dataset.py -q`

Expected: FAIL because the flat beginner Dataset module does not exist.

- [ ] **Step 3: Implement metadata and cache loading explicitly**

Reject unknown splits, unsafe relative paths, duplicate take UIDs, invalid cache shapes, channel-order mismatches, non-finite normalization, and non-positive standard deviation. Project Aria must not be imported anywhere in this file.

- [ ] **Step 4: Implement window index and Dataset**

Create window starts in samples using `window_samples = sampling_rate_hz * window_seconds` and `stride_samples = sampling_rate_hz * stride_seconds`. In `__getitem__`, load `(1500, 6)`, normalize channel-wise, reshape to `(30, 50, 6)`, transpose to `(30, 6, 50)`, make the array contiguous, and return float32 features plus a long label. Add a Chinese shape comment at every transformation.

- [ ] **Step 5: Implement a small DataLoader factory and run tests**

Use `shuffle=True` only when requested, never infer it from the split. Set `pin_memory=torch.cuda.is_available()` and `persistent_workers=workers > 0` using named intermediate variables.

Run: `python -m pytest tests/test_dataset.py -q`

Expected: all Dataset/DataLoader tests pass.

### Task 4: Flat Beginner-Readable Hierarchical Model

**Files:**
- Create: `src/egocharm/model.py`
- Create: `tests/test_model.py`

**Interfaces:**
- Produces: `LowLevelEncoder`, `HighLevelClassifier`, `EgoCHARM`
- Produces: `build_model(config: Mapping[str, Any]) -> EgoCHARM`
- Produces: `count_trainable_parameters(model: nn.Module) -> int`

- [ ] **Step 1: Write failing shape and exact parameter tests**

```python
def test_complete_model_maps_windows_to_seven_logits():
    model = build_default_model(num_classes=7)
    inputs = torch.randn(2, 30, 6, 50)
    assert model(inputs).shape == (2, 7)

def test_paper_inferred_parameter_counts_are_preserved():
    assert count_trainable_parameters(LowLevelEncoder()) == 21863
    assert count_trainable_parameters(HighLevelClassifier(num_classes=7)) == 63111
    assert count_trainable_parameters(HighLevelClassifier(num_classes=9)) == 63369
```

- [ ] **Step 2: Run model tests and verify RED**

Run: `python -m pytest tests/test_model.py -q`

Expected: FAIL because `egocharm.model` does not exist.

- [ ] **Step 3: Implement the three model classes in reading order**

Use explicit asymmetric same padding for five dilation branches `[1,2,4,8,16]`, 11 output channels per branch, kernel size 2, BatchNorm, MaxPool, LeakyReLU, dropout, three blocks, and a 32-hidden GRU. Vectorize `(B,30,6,50)` into `(B*30,6,50)`, restore `(B,30,32)`, then apply the 128-hidden high-level GRU and `Linear(128, classes)`. Do not apply softmax.

- [ ] **Step 4: Run model tests**

Run: `python -m pytest tests/test_model.py -q`

Expected: all model tests pass with exact counts.

### Task 5: Explicit PyTorch Training Module

**Files:**
- Create: `src/egocharm/train.py`
- Create: `tests/test_train.py`

**Interfaces:**
- Produces: `choose_device(name: str) -> torch.device`
- Produces: `calculate_class_weights(labels, num_classes) -> Tensor`
- Produces: `EpochMetrics(loss: float, accuracy: float, macro_f1: float, number_of_samples: int)`
- Produces: `TrainingState(epoch: int, best_macro_f1: float)`
- Produces: `train_one_epoch(model, loader, optimizer, loss_function, device, scaler, maximum_gradient_norm, number_of_classes) -> EpochMetrics`
- Produces: `validate_one_epoch(model, loader, loss_function, device, number_of_classes) -> EpochMetrics`
- Produces: `save_checkpoint(path, model, optimizer, scheduler, scaler, state, metadata) -> None`
- Produces: `load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None) -> TrainingState`
- Produces: command `python -m egocharm.train --config configs/egocharm.yaml --run-name baseline [--resume] [--dry-run]`

- [ ] **Step 1: Write failing training-loop and class-weight tests**

```python
def test_class_weights_are_inverse_to_window_frequency():
    weights = calculate_class_weights([0, 0, 0, 1], 2)
    torch.testing.assert_close(weights, torch.tensor([2 / 3, 2.0]))

def test_train_one_epoch_changes_model_parameters():
    model, loader = tiny_training_objects()
    before = clone_parameters(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    result = train_one_epoch(
        model,
        loader,
        optimizer,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        None,
        None,
        2,
    )
    assert result.number_of_samples == 4
    assert parameters_changed(before, model)
```

- [ ] **Step 2: Run training tests and verify RED**

Run: `python -m pytest tests/test_train.py -q`

Expected: FAIL because the simplified training module does not exist.

- [ ] **Step 3: Implement beginner-style train and validation loops**

Write `model.train()`, device transfer, `optimizer.zero_grad`, optional autocast, forward, loss, backward, optional gradient clipping, optimizer step, and metric collection as separate statements with Chinese comments. Validation must use `model.eval()` and `torch.no_grad()`. Macro-F1 must always use the configured full class ID list.

- [ ] **Step 4: Write failing complete checkpoint round-trip test**

Save a model, Adam optimizer, StepLR scheduler, epoch 3, best macro-F1 0.7, AMP state, and Python/NumPy/PyTorch random states. Mutate the model, load the checkpoint, and assert every model tensor plus epoch and best metric are restored. Assert no `.tmp` remains.

- [ ] **Step 5: Implement atomic checkpoint and resume**

Save to a same-directory `.tmp`, then publish with `os.replace`. `last.pt` is written every epoch; `best.pt` is written only when val macro-F1 improves or ties. Resume only from `last.pt`, restoring optimizer, scheduler, scaler, epoch, best metric, and random state.

- [ ] **Step 6: Implement the readable training entry point**

Parse only `--config`, `--run-name`, `--resume`, and `--dry-run`. Dry-run prints the resolved run directory without building a Dataset or starting training. Otherwise construct train and val datasets explicitly, make train shuffle and val deterministic, calculate weights from train window labels, create Adam/StepLR/AMP, and print one concise row per epoch. Never construct a test dataset.

- [ ] **Step 7: Run training tests**

Run: `python -m pytest tests/test_train.py -q`

Expected: all CPU fixture training and checkpoint tests pass.

### Task 6: Independent Evaluation Module

**Files:**
- Create: `src/egocharm/evaluate.py`
- Create: `tests/test_evaluate.py`

**Interfaces:**
- Produces: `calculate_metrics(labels, predictions, class_names) -> dict[str, Any]`
- Produces: `save_metrics(metrics, output_directory) -> None`
- Produces: command `python -m egocharm.evaluate --config configs/egocharm.yaml --checkpoint artifacts/runs/baseline/best.pt --split {val,test}`

- [ ] **Step 1: Write failing literal metric tests**

```python
def test_metrics_use_accuracy_and_full_class_macro_f1():
    metrics = calculate_metrics([0, 0, 1, 1], [0, 1, 1, 1], ["a", "b"])
    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["macro_f1"] == pytest.approx(((2 / 3) + 0.8) / 2)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
```

- [ ] **Step 2: Run evaluation tests and verify RED**

Run: `python -m pytest tests/test_evaluate.py -q`

Expected: FAIL because the simplified evaluation module does not exist.

- [ ] **Step 3: Implement metrics and atomic JSON/CSV output**

Use explicit class IDs `range(len(class_names))`, zero division value 0, and class insertion order. Save `metrics.json`, `per-class.csv`, raw confusion matrix PNG, and row-normalized confusion matrix PNG. Use the non-interactive Matplotlib backend and close figures.

- [ ] **Step 4: Implement explicit checkpoint evaluation entry point**

Require `--checkpoint`; default `--split val` and permit explicit `test`. Load the model configuration, checkpoint weights, cached Dataset, and DataLoader; collect predictions under `torch.no_grad`; then save outputs below the checkpoint run directory.

- [ ] **Step 5: Run evaluation tests**

Run: `python -m pytest tests/test_evaluate.py -q`

Expected: all metric and artifact tests pass.

### Task 7: Documentation and Fixture-Only Integration

**Files:**
- Rewrite: `README.md`
- Rewrite: `tests/conftest.py`
- Create: `tests/test_integration.py`
- Modify: `environment.yml`

**Interfaces:**
- Documents the exact two-stage workflow and output schemas.
- Verifies fake raw IMU through NPZ, Dataset, one optimizer step, checkpoint, and evaluation.

- [ ] **Step 1: Write the failing integration test**

Use an injected deterministic six-channel sine-wave reader and a four-byte source fixture. Preprocess 31 seconds at 50 Hz, fit train normalization, construct one `(30,6,50)` window, train a small hierarchical model for one step, save/load a checkpoint, and calculate evaluation metrics. Assert no Project Aria import and no path under the real `takes/` directory is accessed.

- [ ] **Step 2: Run integration test and verify RED**

Run: `python -m pytest tests/test_integration.py -q`

Expected: FAIL until all new public interfaces are connected.

- [ ] **Step 3: Complete fixture helpers and README**

Document Conda activation, `python scripts/preprocess_data.py`, train normalization, `python -m egocharm.train`, resume, val evaluation, explicit test evaluation, cache schema, tensor shapes, seven-versus-nine-class limitation, and the fact that fixture data never contributes to results.

- [ ] **Step 4: Run the complete replacement suite**

Run: `python -m compileall -q scripts src tests`

Run: `python -m pytest tests/test_config.py tests/test_preprocess_data.py tests/test_dataset.py tests/test_model.py tests/test_train.py tests/test_evaluate.py tests/test_integration.py -q`

Expected: all new tests pass without network, VRS reads, or long training.

### Task 8: Remove Replaced Code and Disposable Directories

**Files:**
- Delete after Task 7 passes: `src/egocharm/cli/`
- Delete after Task 7 passes: `src/egocharm/data/`
- Delete after Task 7 passes: `src/egocharm/models/`
- Delete after Task 7 passes: `src/egocharm/training/`
- Delete after Task 7 passes: `src/egocharm/evaluation/`
- Delete after Task 7 passes: old nested test directories and `configs/data.yaml`, `configs/model.yaml`, `configs/train.yaml`
- Delete: `.bootstrap/`, `.venv/`, `.pytest_cache/`, empty `.git/`, all `__pycache__/`, all `.pyc`
- Modify: `.gitignore`

**Interfaces:**
- Produces the exact clean directory described in the approved design.

- [ ] **Step 1: Re-resolve every deletion target**

Use `find` and `du` on the exact listed paths. Confirm `ego-exoD4_dataset/takes/`, `.vscode/`, the Conda environment, new source files, and new tests are not targets.

- [ ] **Step 2: Remove old source only after replacement verification**

Use patch-based file deletion for source, tests, and configurations. Use explicit directory removal only for the already inspected disposable environment/cache directories. Do not use globs targeting the workspace root or dataset.

- [ ] **Step 3: Update `.gitignore`**

Keep `.venv/`, `.pytest_cache/`, `__pycache__/`, `*.pyc`, artifacts, and temporary cache patterns ignored even though current copies are removed.

- [ ] **Step 4: Run fresh final verification after deletion**

Run: `python -m compileall -q scripts src tests`

Run: `python -m pytest -q`

Run: `python scripts/preprocess_data.py --config configs/egocharm.yaml --split train --max-takes 1 --dry-run`

Run: `python -m egocharm.train --config configs/egocharm.yaml --run-name smoke --dry-run`

Expected: compilation succeeds, only the new tests are collected and pass, both dry-runs report intended work, and neither command opens a real VRS or starts training.

- [ ] **Step 5: Inspect the final top-level and source trees**

Run: `find . -maxdepth 2 -mindepth 1 -not -path './ego-exoD4_dataset/takes*' -print | sort`

Run: `find src/egocharm scripts tests configs -maxdepth 2 -type f -print | sort`

Expected: the approved simplified files remain, disposable directories and replaced modules are absent, and the dataset directory is unchanged.
