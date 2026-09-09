# Ego–Xreal Joint Nine-Class Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate a nine-class 5-second, 200 Hz EgoCHARM model using the seven Ego-Exo4D activities plus Xreal PUT_ON and TAKE_OFF events.

**Architecture:** Store each continuous recording as a memory-mappable float32 NPY and describe it in a compact JSON manifest. Build a balanced stochastic training dataset that samples virtual 25 ms-stride windows without materializing them, applies train-only IMU augmentation, and feeds the existing hierarchical EgoCHARM model.

**Tech Stack:** Python 3.10, NumPy, SciPy, PyTorch 2.5, Project Aria Tools, pytest, YAML.

## Global Constraints

- Window length is exactly 5 seconds at 200 Hz: input shape `(B, 5, 6, 200)`.
- Training candidate start step is exactly 25 ms: 5 samples at 200 Hz.
- Channel order is acceleration XYZ followed by gyroscope XYZ.
- Do not divide acceleration by 9.8 or standardize input values.
- Downsample both devices to 200 Hz with the same polyphase anti-aliasing filter.
- Rotate both devices to CPF direction; never apply sensor translation.
- Keep existing EgoCHARM and Xreal experiment outputs unchanged.
- Train classes are the seven original Ego labels followed by PUT_ON and TAKE_OFF.
- Use only Xreal PUT_ON and TAKE_OFF for training.

---

### Task 1: Memory-mapped joint cache format

**Files:**
- Create: `src/egocharm/joint_cache.py`
- Create: `tests/test_joint_cache.py`
- Create: `scripts/preprocess_joint_data.py`

**Interfaces:**
- Produces: `JointRecording`, `load_joint_manifest(path)`, `write_recording_cache(...)`.
- Produces: a manifest with `recording_uid`, `source`, `split`, `label`, `values_path`, `segment_bounds`, `sampling_rate_hz`, `coordinate_frame`, and `channel_names`.

- [ ] **Step 1: Write failing manifest and NPY round-trip tests**

Create small arrays and assert that `write_recording_cache` writes float32 `(N, 6)` NPY data, that `np.load(..., mmap_mode="r")` works, and that manifest loading preserves labels and segment bounds.

- [ ] **Step 2: Run the focused test and verify the missing module failure**

Run: `pytest tests/test_joint_cache.py -v`

Expected: FAIL because `egocharm.joint_cache` does not exist.

- [ ] **Step 3: Implement the cache data model and writer**

Implement atomic NPY/JSON output with explicit six-channel validation. Reuse `read_left_imu`, `clean_samples`, `find_continuous_segments`, and `resample_segment` for Ego. Reuse Xreal reading, CPF rotation, polyphase resampling, and event-file split assignment, while filtering out WORN and NOT_WORN.

- [ ] **Step 4: Verify the cache tests**

Run: `pytest tests/test_joint_cache.py -v`

Expected: PASS.

### Task 2: Compact virtual window sampling and augmentation

**Files:**
- Create: `src/egocharm/joint_dataset.py`
- Create: `tests/test_joint_dataset.py`

**Interfaces:**
- Produces: `ImuAugmentation`, `BalancedJointDataset`, `EvaluationJointDataset`, and `create_joint_data_loader(...)`.
- Consumes: `JointRecording` entries and NPY memory maps from Task 1.

- [ ] **Step 1: Write failing shape, label, sampling, and augmentation tests**

Assert output shape `(5, 6, 200)`, fixed nine-class ordering, no normalization, five-sample candidate stride, equal per-class epoch labels, deterministic evaluation references, constant bias semantics, shared 3-D rotation for accel/gyro, and independent Gaussian-noise channel groups.

- [ ] **Step 2: Run the focused test and verify expected failures**

Run: `pytest tests/test_joint_dataset.py -v`

Expected: FAIL because the dataset module does not exist.

- [ ] **Step 3: Implement compact cumulative-count sampling**

Represent each Ego segment by its count of legal window starts. Select a virtual start with cumulative counts instead of constructing `WindowReference` objects. Open NPY files with `mmap_mode="r"`, copy only the selected `(1000, 6)` slice, augment it, then reshape `(1000, 6) -> (5, 200, 6) -> (5, 6, 200)`.

- [ ] **Step 4: Verify the dataset tests**

Run: `pytest tests/test_joint_dataset.py -v`

Expected: PASS.

### Task 3: Nine-class training and evaluation commands

**Files:**
- Create: `src/egocharm/train_joint.py`
- Create: `src/egocharm/evaluate_joint.py`
- Create: `tests/test_joint_commands.py`
- Create: `configs/ego_xreal_joint_9class.yaml`

**Interfaces:**
- Produces: `python -m egocharm.train_joint --config ...` and `python -m egocharm.evaluate_joint --config ... --checkpoint ...`.
- Consumes: joint manifests and datasets from Tasks 1–2 and the existing `build_model`, epoch engine, checkpoint, metrics, and plotting functions.

- [ ] **Step 1: Write failing dry-run and model-shape tests**

Assert the resolved run directory is isolated, the model emits `(B, 9)`, config values resolve to 5 seconds/200 Hz/25 ms, and no training begins under `--dry-run`.

- [ ] **Step 2: Run the focused command tests and verify expected failures**

Run: `pytest tests/test_joint_commands.py -v`

Expected: FAIL because the new commands and configuration do not exist.

- [ ] **Step 3: Implement training, checkpoint metadata, and evaluation outputs**

Train 30 epochs with balanced 32400-sample epochs, batch size 32, AMP, Adam, and StepLR. Save `best.pt`, `last.pt`, resolved config, per-epoch JSONL history, classification metrics, confusion matrices, and stable-Xreal PUT_ON/TAKE_OFF false-positive statistics under `artifacts/ego_xreal_joint_9class`.

- [ ] **Step 4: Verify command tests and all regression tests**

Run: `pytest tests/test_joint_commands.py -v`

Run: `pytest -q`

Expected: all tests PASS.

### Task 4: Produce caches and run the experiment

**Files:**
- Create: `artifacts/ego_xreal_joint_9class/cache/ego/manifest.json`
- Create: `artifacts/ego_xreal_joint_9class/cache/xreal_events/manifest.json`
- Create: `artifacts/ego_xreal_joint_9class/runs/joint_9class/*`
- Create: `artifacts/ego_xreal_joint_9class/results/joint_9class/*`

**Interfaces:**
- Consumes: preprocessing, training, and evaluation commands from Tasks 1–3.

- [ ] **Step 1: Run a two-recording preprocessing smoke test**

Run: `python scripts/preprocess_joint_data.py --config configs/ego_xreal_joint_9class.yaml --max-ego-takes 2`

Expected: two Ego NPY caches plus Xreal event caches and readable manifests.

- [ ] **Step 2: Run a one-batch CUDA smoke test**

Run: `python -m egocharm.train_joint --config configs/ego_xreal_joint_9class.yaml --smoke-test`

Expected: CUDA is selected, one forward/backward update succeeds, and tensor shape is `(B, 5, 6, 200)`.

- [ ] **Step 3: Generate all 200 Hz caches**

Run: `python scripts/preprocess_joint_data.py --config configs/ego_xreal_joint_9class.yaml`

Expected: all available Ego takes and all Xreal event recordings are listed in their manifests; stable Xreal examples are absent.

- [ ] **Step 4: Train and evaluate**

Run: `python -m egocharm.train_joint --config configs/ego_xreal_joint_9class.yaml --run-name joint_9class`

Run: `python -m egocharm.evaluate_joint --config configs/ego_xreal_joint_9class.yaml --checkpoint artifacts/ego_xreal_joint_9class/runs/joint_9class/best.pt --split test`

Expected: checkpoints, history, per-class precision/recall/F1, raw and normalized confusion-matrix PNG files, and stable-Xreal event false-positive diagnostics are written without modifying previous experiments.
