# Xreal Binary Wear-Event Design

## Goal

Add an optional binary wear-event task to the existing Xreal transfer-learning
pipeline. The model detects whether a five-second IMU window contains a wear
transition, without attempting to infer `WORN` versus `NOT_WORN` from a stable
window or to classify the transition direction.

The existing four-class task and its checkpoints remain compatible and
unchanged.

## Scope

The first binary-event experiment deliberately keeps the current input and
transfer setup:

- six channels: accelerometer XYZ followed by gyroscope XYZ;
- 50 Hz sampling rate;
- five-second windows with shape `(5, 6, 50)`;
- acceleration divided by 9.8 and gyroscope values unchanged;
- CPF left-up-forward coordinate directions;
- Ego-Exo4D CPF-pretrained LLE frozen in evaluation mode;
- a newly initialized GRU-based HLE.

This version does not add vector magnitudes, temporal derivatives, a 200 Hz
cache, a direction head, endpoint-state heads, or an online state machine.

## Configuration

Task behavior is selected with `task.label_mode`:

```yaml
task:
  label_mode: binary_event
```

Supported values are:

- `four_class`: existing `WORN`, `NOT_WORN`, `PUT_ON`, `TAKE_OFF` behavior;
- `binary_event`: new `NO_EVENT`, `EVENT` behavior.

When `task.label_mode` is absent, the code defaults to `four_class` so the
existing configuration and checkpoint remain backward compatible.

A separate configuration, `configs/xreal_wear_binary_event.yaml`, selects
`binary_event`. Its intended run name is `binary_event`, producing artifacts
under `artifacts/xreal_wear_detection/runs/binary_event` and never overwriting
`artifacts/xreal_wear_detection/runs/frozen_lle`.

## Label Mapping

The selection JSON continues to store the original four recording labels. The
Dataset maps them at load time:

| Original label | Binary target |
|---|---|
| `WORN` | `NO_EVENT` |
| `NOT_WORN` | `NO_EVENT` |
| `PUT_ON` | `EVENT` |
| `TAKE_OFF` | `EVENT` |

No cache regeneration is required. Retaining original labels makes it
possible to evaluate `PUT_ON` and `TAKE_OFF` detection separately even though
they share the training target.

## Model and Training

The frozen LLE produces five 32-dimensional one-second embeddings with shape
`(B, 5, 32)`. In `binary_event` mode, the new HLE produces two logits ordered
as `NO_EVENT`, `EVENT`.

Weighted cross-entropy is computed from the mapped binary labels in the
training split. Checkpoints record `label_mode` and the mode-specific class
names. The optimizer still receives only trainable HLE parameters. Best-model
selection continues to use validation macro-F1, now calculated across the two
binary classes.

Four-class checkpoints remain loadable only with four-class configuration;
binary checkpoints remain loadable only with binary-event configuration. A
class-name mismatch fails explicitly instead of partially loading a head with
the wrong output size.

## Evaluation

Binary-event evaluation writes the existing artifacts in a separate run
directory:

- `metrics.json`;
- `per-class.csv`;
- raw and normalized confusion-matrix images.

The standard metrics use the class order `NO_EVENT`, `EVENT`. The JSON also
contains an `original_label_breakdown` calculated from the untouched source
labels:

- `WORN`: number of windows and false-event rate;
- `NOT_WORN`: number of windows and false-event rate;
- `PUT_ON`: number of windows and event recall;
- `TAKE_OFF`: number of windows and event recall.

For `WORN` and `NOT_WORN`, evaluation additionally reports false events per
hour as:

```text
false_event_count / (number_of_windows * 5 / 3600)
```

This number describes the current non-overlapping evaluation windows. It is
not yet an online sliding-window false-alarm rate.

## Components

`wear_dataset.py` owns label modes, class-name selection, and original-to-task
label mapping. Each `WearRecord` retains its original four-class label.

`wear_model.py` validates the mode-specific model configuration and builds an
HLE whose output size matches the selected task classes, while preserving the
strictly frozen LLE behavior.

`train_wear.py` reads the selected label mode, constructs mapped datasets,
uses mapped class weights, and writes mode metadata into checkpoints.

`evaluate_wear.py` evaluates the selected task classes and adds the
original-label event breakdown for binary mode.

The existing four-class configuration needs no behavior change. The new
binary configuration and documentation provide explicit commands and output
paths.

## Validation and Compatibility

Focused tests cover:

- all four original-to-binary mappings;
- binary Dataset targets and class labels;
- two-logit model output and frozen LLE parameters;
- configuration validation for both modes;
- checkpoint class-name isolation;
- binary evaluation breakdown and false-events-per-hour calculation;
- unchanged four-class Dataset, model, training, and evaluation behavior.

The complete test suite must pass before the binary mode is handed off.
