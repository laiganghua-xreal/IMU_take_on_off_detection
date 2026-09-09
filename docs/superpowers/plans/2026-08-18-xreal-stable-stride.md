# Xreal Stable-Window One-Second Stride Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Xreal stable-state caches with five-second windows starting every second while preserving event examples byte-for-byte.

**Architecture:** Add a tested helper that derives overlapping starts inside the existing chronological split regions. Add explicit CLI options for stable stride and event reuse, stage new stable files before replacing the old stable subset, and atomically rewrite the selection.

**Tech Stack:** Python 3.10, NumPy, SciPy, pytest.

## Global Constraints

- Window length remains five seconds at 50 Hz.
- Stable stride is one second for this regeneration.
- Split guards remain five seconds and no window crosses them.
- Existing `PUT_ON` and `TAKE_OFF` NPZ files and assignments remain unchanged.
- Existing four-class and binary-event training code is not modified.
- The workspace is not a Git repository, so commit steps are omitted.

---

### Task 1: Stable overlapping-window assignments

**Files:**
- Modify: `Xreal_datasets/preprocess_xreal_imu.py`
- Test: `tests/test_preprocess_xreal_imu.py`

- [x] Add a failing test for five-second windows starting every one second.
- [x] Assert that no start crosses the five-second train/val/test guards.
- [x] Run the focused test and confirm it fails because the helper is absent.
- [x] Implement `build_stable_window_assignments` using existing split slots.
- [x] Run the focused tests and confirm they pass.

### Task 2: Preserve events and safely replace stable caches

**Files:**
- Modify: `Xreal_datasets/preprocess_xreal_imu.py`
- Test: `tests/test_preprocess_xreal_imu.py`

- [x] Add failing tests that reuse event entries without calling event conversion.
- [x] Add `--stable-stride-seconds` and `--reuse-existing-events`.
- [x] Stage new stable caches, replace stable outputs, remove stale stable files,
      and atomically write the combined selection.
- [x] Run focused tests and then the complete suite.

### Task 3: Regenerate and verify real data

**Files:**
- Replace: `artifacts/xreal_wear_detection/cache/xreal_cpf/<stable examples>.npz`
- Preserve: `artifacts/xreal_wear_detection/cache/xreal_cpf/<event examples>.npz`
- Replace: `Xreal_datasets/metadata/xreal_wear_selection.json`

- [x] Record event checksums and selection entries before regeneration.
- [x] Run the converter with one-second stride and event reuse.
- [x] Verify stable starts advance by 50 samples inside each split region.
- [x] Verify all event checksums and selection entries are unchanged.
- [x] Report the new class/split distribution and disk usage.
