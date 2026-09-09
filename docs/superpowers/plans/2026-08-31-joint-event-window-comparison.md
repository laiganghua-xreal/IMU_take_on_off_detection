# Joint Event Window Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the joint nine-class test predictions, locate all true- and false-positive PUT_ON/TAKE_OFF windows, and save comparable six-axis IMU plots and an index below the existing test directory.

**Architecture:** Reuse the resolved config, combined manifests, `EvaluationJointDataset`, model builder, and checkpoint loader used by `evaluate_joint.py`. A temporary Python program performs deterministic inference, obtains each selected window from its manifest-backed NPY recording, computes common sensor limits, and writes diagnostic artifacts without changing project code or existing evaluation files.

**Tech Stack:** Python 3.10, PyTorch 2.5, NumPy, Matplotlib, CSV/JSON standard libraries.

## Global Constraints

- Use `artifacts/ego_xreal_joint_9class/runs/joint_9class/best.pt`.
- Use the resolved 200 Hz, 5-second, no-augmentation test pipeline.
- Select false-positive and true-positive `PUT_ON` and `TAKE_OFF` windows.
- Use common acceleration and gyroscope y-axis limits across all figures.
- Save outputs only under `artifacts/ego_xreal_joint_9class/runs/joint_9class/test/event-window-comparison/`.
- Do not modify existing metrics, confusion matrices, caches, checkpoints, or source code.

---

### Task 1: Reproduce predictions and select event windows

**Files:**
- Create temporarily: `/tmp/plot_joint_event_windows.py`
- Read: `configs/ego_xreal_joint_9class.yaml`
- Read: `artifacts/ego_xreal_joint_9class/runs/joint_9class/best.pt`
- Read: `artifacts/ego_xreal_joint_9class/cache/ego/manifest.json`
- Read: `artifacts/ego_xreal_joint_9class/cache/xreal_events/manifest.json`
- Read: `artifacts/ego_xreal_joint_9class/runs/joint_9class/test/metrics.json`

**Interfaces:**
- Consumes: `EvaluationJointDataset.references`, where every reference exposes `recording`, `start_sample`, and `class_index`.
- Produces: a list of records containing `group`, `true_label`, `predicted_label`, `recording_uid`, `values_path`, `start_sample`, `start_seconds`, `confidence`, and a float32 `(1000, 6)` IMU window.

- [ ] **Step 1: Hash existing evaluation outputs**

Run:

```bash
sha256sum artifacts/ego_xreal_joint_9class/runs/joint_9class/test/metrics.json artifacts/ego_xreal_joint_9class/runs/joint_9class/test/confusion-matrix.png artifacts/ego_xreal_joint_9class/runs/joint_9class/test/confusion-matrix-normalized.png
```

Save the three hashes in `/tmp/joint-test-existing.sha256` for final comparison.

- [ ] **Step 2: Recreate deterministic test inference**

The temporary program must:

```python
recordings = load_joint_manifest(ego_manifest) + load_joint_manifest(xreal_manifest)
dataset = EvaluationJointDataset(
    recordings,
    split="test",
    sampling_rate_hz=200,
    window_seconds=5,
    evaluation_step_seconds=5,
)
model = build_model(config["model"])
load_checkpoint(checkpoint, model, expected_class_names=JOINT_CLASS_NAMES)
model.eval()
```

Run inference in dataset order with `torch.inference_mode()` and `shuffle=False`.

- [ ] **Step 3: Select the four required groups**

Use these exact predicates:

```python
is_true_positive = true_label == predicted_label and predicted_label in event_labels
is_false_positive = true_label != predicted_label and predicted_label in event_labels
```

Assert exact group counts before plotting:

```python
assert counts == {
    ("false_positive", "PUT_ON"): 2,
    ("false_positive", "TAKE_OFF"): 5,
    ("true_positive", "PUT_ON"): 10,
    ("true_positive", "TAKE_OFF"): 10,
}
```

Expected: 27 selected windows and no assertion failure.

### Task 2: Render and verify diagnostic artifacts

**Files:**
- Create: `artifacts/ego_xreal_joint_9class/runs/joint_9class/test/event-window-comparison/window-index.csv`
- Create: `artifacts/ego_xreal_joint_9class/runs/joint_9class/test/event-window-comparison/summary.json`
- Create: `artifacts/ego_xreal_joint_9class/runs/joint_9class/test/event-window-comparison/put-on-comparison.png`
- Create: `artifacts/ego_xreal_joint_9class/runs/joint_9class/test/event-window-comparison/take-off-comparison.png`
- Create: 27 individual PNG files below the four group/class directories.
- Remove: `/tmp/plot_joint_event_windows.py`

**Interfaces:**
- Consumes: the 27 selected records from Task 1.
- Produces: individual figures, two event-level overview figures, a machine-readable index, and summary metadata.

- [ ] **Step 1: Calculate shared sensor limits**

Concatenate the 27 `(1000, 6)` windows. Calculate acceleration limits from channels `0:3` and gyroscope limits from channels `3:6` using global minima and maxima, adding five percent padding to each range. Apply these same two limits to every plot.

- [ ] **Step 2: Write individual figures**

For each window, create two synchronized panels over `0 <= t < 5` seconds:

```python
axes[0].plot(time_seconds, values[:, :3])
axes[1].plot(time_seconds, values[:, 3:])
axes[0].set_ylabel("Acceleration (m/s²)")
axes[1].set_ylabel("Angular velocity (rad/s)")
axes[1].set_xlabel("Time (s)")
```

The title must contain TP/FP group, true label, predicted label, recording UID, start time, and softmax confidence. Use filenames with group, predicted label, recording UID, and zero-padded start sample.

- [ ] **Step 3: Write event overview figures**

Create one overview per predicted event. Use one row per comparison position and four columns: false-positive acceleration, false-positive gyroscope, true-positive acceleration, and true-positive gyroscope. Empty cells are hidden when one group has fewer windows. Use the shared sensor limits from Step 1.

- [ ] **Step 4: Write the index and summary**

`window-index.csv` must have exactly these columns:

```text
group,true_label,predicted_label,recording_uid,start_sample,start_seconds,confidence,image_path
```

`summary.json` must record checkpoint, sampling rate, window size, coordinate frame, augmentation state, shared limits, total count, and the four group counts.

- [ ] **Step 5: Verify outputs**

Run a Python verification that asserts:

```python
assert len(csv_rows) == 27
assert all(Path(row["image_path"]).is_file() for row in csv_rows)
assert png_counts == {
    ("false_positive", "PUT_ON"): 2,
    ("false_positive", "TAKE_OFF"): 5,
    ("true_positive", "PUT_ON"): 10,
    ("true_positive", "TAKE_OFF"): 10,
}
```

Run `sha256sum -c /tmp/joint-test-existing.sha256` from the project root. Expected: all three existing files report `OK`.

- [ ] **Step 6: Remove temporary files**

Delete `/tmp/plot_joint_event_windows.py` and `/tmp/joint-test-existing.sha256`. This project directory is not a Git repository, so no commit step is available.
