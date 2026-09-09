# Xreal Wear-State Frozen-LLE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated CPF-coordinate transfer-learning pipeline that pretrains EgoCHARM on Ego-Exo4D and trains a frozen LLE with a new four-class Xreal HLE.

**Architecture:** Extend the existing VRS preprocessing with an opt-in CPF rotation and a separate cache/config, leaving the reproduction defaults unchanged. Convert Xreal right-up-back IMU CSV files into five-second CPF NPZ examples, then use focused wear dataset/model/train/evaluate modules that load only the CPF-pretrained LLE and optimize a new small HLE.

**Tech Stack:** Python 3.10, NumPy, SciPy, PyTorch, Project Aria Tools, PyYAML, Matplotlib, pytest.

## Global Constraints

- Common coordinate order is CPF left-up-forward.
- Ego-Exo4D uses each VRS's `T_Cpf_ImuLeft` rotation only.
- Xreal right-up-back uses `diag(-1, 1, -1)` for acceleration and gyroscope.
- Neither source applies translation or lever-arm compensation.
- Sampling rate is 50 Hz; Xreal windows are five seconds or 250 samples.
- Existing `artifacts/cache/imu-left`, `artifacts/runs`, and `artifacts/comparisons/normalization` remain untouched.
- New output stays under `artifacts/xreal_wear_detection`.
- No long-running preprocessing or training starts as part of code verification.

---

### Task 1: CPF-aware Ego-Exo4D preprocessing

**Files:**
- Modify: `scripts/preprocess_data.py`
- Modify: `tests/test_preprocess_data.py`
- Create: `configs/xreal_wear_ego_pretrain_cpf.yaml`

**Interfaces:**
- Consumes: `data.coordinate_frame`, either `sensor` or `cpf`.
- Produces: `RawImu.coordinate_metadata`, `rotate_imu_vectors(values, rotation)`, and CPF NPZ metadata.

- [ ] **Step 1: Add failing tests for rotation and CPF metadata**

```python
def test_rotate_imu_vectors_rotates_acceleration_and_gyroscope():
    values = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float32)
    rotation = np.diag([-1, 1, -1])
    transformed = rotate_imu_vectors(values, rotation)
    np.testing.assert_allclose(transformed, [[-1, 2, -3, -4, 5, -6]])

def test_preprocess_take_writes_coordinate_metadata(tmp_path):
    settings = make_settings(tmp_path, coordinate_frame="cpf")
    output_path = preprocess_take(make_take(), settings, reader=fake_cpf_reader)
    with np.load(output_path, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata_json"].item()))
    assert metadata["target_coordinate_frame"] == "cpf"
```

- [ ] **Step 2: Run the focused tests and verify missing interfaces fail**

Run: `pytest tests/test_preprocess_data.py -q`

Expected: FAIL because `rotate_imu_vectors` and coordinate-frame settings do not exist.

- [ ] **Step 3: Implement opt-in per-VRS CPF rotation**

Add `coordinate_frame` to `PreprocessSettings`; read `T_Cpf_ImuLeft` with `get_transform_cpf_sensor("imu-left")`; apply only its `3 x 3` rotation to both vector triplets; store device serial, full transform, and rotation in metadata. Preserve `sensor` as the default behavior.

- [ ] **Step 4: Add the isolated CPF pretraining configuration**

Create `configs/xreal_wear_ego_pretrain_cpf.yaml` with the existing official selection, cache root `artifacts/xreal_wear_detection/cache/ego_exo_cpf`, 30-second windows, gravity normalization, seven activity classes, and output root `artifacts/xreal_wear_detection/runs`.

- [ ] **Step 5: Run focused and existing preprocessing tests**

Run: `pytest tests/test_preprocess_data.py ego-exoD4_dataset/tests -q`

Expected: all pass.

### Task 2: Xreal CSV-to-CPF conversion

**Files:**
- Create: `Xreal_datasets/preprocess_xreal_imu.py`
- Create: `tests/test_preprocess_xreal_imu.py`

**Interfaces:**
- Produces: `read_xreal_imu(path: Path)`, `rotate_xreal_to_cpf(values: np.ndarray)`, `resample_xreal_imu(raw: RawXrealImu, sampling_rate_hz: int)`, `assign_event_splits(paths: Sequence[Path], seed: int)`, and `main(arguments=None)`.
- Writes: one `(250, 6)` NPZ per example and `Xreal_datasets/metadata/xreal_wear_selection.json`.

- [ ] **Step 1: Add failing tests for CSV filtering, axis conversion, split isolation, and cache shape**

```python
def test_rotate_xreal_right_up_back_to_cpf():
    values = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float32)
    np.testing.assert_allclose(
        rotate_xreal_to_cpf(values),
        [[-1, 2, -3, -4, 5, -6]],
    )

def test_preprocess_writes_five_second_example(tmp_path):
    source_path = make_1000_hz_fixture(tmp_path, label="PUT_ON")
    result = preprocess_recording(
        source_path,
        label="PUT_ON",
        split="train",
        cache_root=tmp_path / "cache",
        sampling_rate_hz=50,
        window_seconds=5,
    )
    with np.load(result[0].cache_path) as cache:
        assert cache["values"].shape == (250, 6)
```

- [ ] **Step 2: Run and verify the new test module fails**

Run: `pytest tests/test_preprocess_xreal_imu.py -q`

Expected: FAIL because the Xreal converter does not exist.

- [ ] **Step 3: Implement the minimal converter**

Filter types 1 and 2, read three vector columns, align their shared timestamps, rotate right-up-back to CPF, use `scipy.signal.resample_poly` to 50 Hz, split stable recordings before creating non-overlapping windows, center-crop one window from ON/OFF files, and atomically write NPZ/selection outputs.

- [ ] **Step 4: Add optional plots**

When `--plot` is passed, save one accelerometer and gyroscope figure per source recording under `artifacts/xreal_wear_detection/plots`; do not show a GUI window.

- [ ] **Step 5: Run the converter tests**

Run: `pytest tests/test_preprocess_xreal_imu.py -q`

Expected: all pass.

### Task 3: Wear dataset and frozen transfer model

**Files:**
- Create: `src/egocharm/wear_dataset.py`
- Create: `src/egocharm/wear_model.py`
- Create: `tests/test_wear_dataset.py`
- Create: `tests/test_wear_model.py`

**Interfaces:**
- Produces: `WEAR_CLASS_NAMES`, `WearDataset`, `load_wear_selection`, `build_frozen_wear_model`, and `load_pretrained_low_level_encoder`.

- [ ] **Step 1: Add failing dataset tests**

Test that `(250, 6)` cache values become `(5, 6, 50)`, gravity normalization affects only acceleration, and class order is `WORN`, `NOT_WORN`, `PUT_ON`, `TAKE_OFF`.

- [ ] **Step 2: Run dataset tests and verify failure**

Run: `pytest tests/test_wear_dataset.py -q`

Expected: FAIL because `egocharm.wear_dataset` does not exist.

- [ ] **Step 3: Implement the wear dataset**

Read only the selection entries for the requested split and lazily load each fixed-size NPZ example. Return `(S, C, T)` float tensors and scalar long labels.

- [ ] **Step 4: Add failing frozen-model tests**

Test that only HLE parameters require gradients, the LLE stays in evaluation mode after `model.train()`, the new head emits `(B, 4)`, and only `low_level_encoder.*` weights load from an Ego checkpoint.

- [ ] **Step 5: Run model tests and verify failure**

Run: `pytest tests/test_wear_model.py -q`

Expected: FAIL because frozen transfer model interfaces do not exist.

- [ ] **Step 6: Implement the frozen transfer model**

Reuse `LowLevelEncoder` and `HighLevelClassifier`, initialize an HLE with hidden size 32 and four outputs, load the LLE state by prefix, freeze its parameters, and override training-mode behavior so LLE BatchNorm and Dropout stay in evaluation mode.

- [ ] **Step 7: Run wear dataset and model tests**

Run: `pytest tests/test_wear_dataset.py tests/test_wear_model.py -q`

Expected: all pass.

### Task 4: Isolated wear training and evaluation

**Files:**
- Create: `src/egocharm/train_wear.py`
- Create: `src/egocharm/evaluate_wear.py`
- Create: `tests/test_train_wear.py`
- Create: `tests/test_evaluate_wear.py`
- Create: `configs/xreal_wear_frozen_lle.yaml`

**Interfaces:**
- Commands: `python -m egocharm.train_wear --config configs/xreal_wear_frozen_lle.yaml --run-name frozen_lle` and `python -m egocharm.evaluate_wear --config configs/xreal_wear_frozen_lle.yaml --checkpoint artifacts/xreal_wear_detection/runs/frozen_lle/best.pt --split test`.

- [ ] **Step 1: Add failing CLI wiring tests**

Test that training loads only Xreal train/val splits, passes only trainable HLE parameters to Adam, and writes beneath the configured isolated output root. Test that evaluation accepts val/test only and writes the existing metric artifacts.

- [ ] **Step 2: Run the new CLI tests and verify failure**

Run: `pytest tests/test_train_wear.py tests/test_evaluate_wear.py -q`

Expected: FAIL because the CLI modules do not exist.

- [ ] **Step 3: Implement training and evaluation commands**

Reuse engine, checkpoint, metrics, device, and seed helpers. Select `best.pt` by val macro-F1; include source checkpoint, coordinate frame, normalization, window size, and class names in metadata.

- [ ] **Step 4: Add the downstream configuration**

Set selection/cache paths to Xreal artifacts, five-second windows, gravity normalization, four classes, hidden size 32, source checkpoint `artifacts/xreal_wear_detection/runs/ego_pretrain_cpf/best.pt`, and output root `artifacts/xreal_wear_detection/runs`.

- [ ] **Step 5: Run CLI tests**

Run: `pytest tests/test_train_wear.py tests/test_evaluate_wear.py -q`

Expected: all pass.

### Task 5: Documentation and full verification

**Files:**
- Create: `Xreal_datasets/README.md`
- Modify: `README.md`

**Interfaces:**
- Documents the four commands for CPF preprocessing, CPF pretraining, Xreal preprocessing, frozen-LLE training, and evaluation without starting them automatically.

- [ ] **Step 1: Document commands and output separation**

Include exact config paths, output trees, CPF axis definitions, current split limitation, and checkpoint dependency.

- [ ] **Step 2: Run dry-run and tiny-fixture checks**

Run the Xreal converter with `--dry-run`, Ego preprocessing with the CPF config and `--dry-run --max-takes 1`, and wear train command with `--dry-run`.

- [ ] **Step 3: Run the complete test suite**

Run: `pytest -q`

Expected: all tests pass with no failures.

- [ ] **Step 4: Confirm old artifact paths were not modified**

Inspect created files and confirm no files were written under the previous cache, run, or normalization-comparison directories.
