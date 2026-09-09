# Xreal Wear-State Transfer Learning Design

## Goal

Build an experiment that detects four Xreal wear states from five-second IMU
windows:

- `WORN`
- `NOT_WORN`
- `PUT_ON`
- `TAKE_OFF`

The experiment first retrains a Low-Level Encoder (LLE) on Ego-Exo4D activity
labels after expressing the left IMU vectors in Central Pupil Frame (CPF)
directions. It then freezes that encoder and trains a new four-class High-Level
Classifier on Xreal data expressed in the same CPF directions. Its
configuration, cache, checkpoints, logs, and evaluation results remain
separate from the existing EgoCHARM reproduction.

## Current Data

The source directory is:

`/home/xreal/record_data_tools/data/20260818/free_record_only_imu`

It contains:

| Label | Source recordings | Approximate five-second examples |
|---|---:|---:|
| `WORN` | one 600.65-second recording | 120 |
| `NOT_WORN` | one 485.92-second recording | 97 |
| `PUT_ON` | twenty 5.45-5.89-second recordings | 20 |
| `TAKE_OFF` | twenty-four 5.45-5.85-second recordings | 24 |

Each CSV uses nanosecond timestamps. Type `1` is gyroscope and type `2` is
accelerometer. Types `3` and `12` are outside this experiment and are ignored.
Gyroscope and accelerometer samples are recorded at approximately 1000 Hz.

## Project Separation

The wear-state experiment uses these paths:

- conversion entry point: `Xreal_datasets/preprocess_xreal_imu.py`
- metadata: `Xreal_datasets/metadata/xreal_wear_selection.json`
- Ego-Exo4D CPF pretraining configuration:
  `configs/xreal_wear_ego_pretrain_cpf.yaml`
- Xreal transfer configuration: `configs/xreal_wear_frozen_lle.yaml`
- Ego-Exo4D CPF cache: `artifacts/xreal_wear_detection/cache/ego_exo_cpf`
- Xreal CPF cache: `artifacts/xreal_wear_detection/cache/xreal_cpf`
- CPF pretraining output: `artifacts/xreal_wear_detection/runs/ego_pretrain_cpf`
- training output: `artifacts/xreal_wear_detection/runs/frozen_lle`
- evaluation output: `artifacts/xreal_wear_detection/runs/frozen_lle/{val,test}`

The existing paths under `artifacts/cache/imu-left`, `artifacts/runs`, and
`artifacts/comparisons/normalization` are read-only dependencies or remain
untouched.

## Common Coordinate Convention

Both datasets store vectors in Aria CPF axis order:

- channel axis `x`: left;
- channel axis `y`: up;
- channel axis `z`: forward.

For every Ego-Exo4D VRS, preprocessing reads the per-recording factory
calibration with `get_transform_cpf_sensor("imu-left")`. It takes only the
upper-left `3 x 3` rotation `R_Cpf_ImuLeft` and applies it to both
accelerometer and gyroscope vectors. The translation is recorded as metadata
but is not applied.

Xreal vectors use right-up-back axis order. Their fixed direction conversion
to CPF is:

`(right, up, back) -> (-right, up, -back)`

or `R_Cpf_Xreal = diag(-1, 1, -1)`. The same rotation is applied to Xreal
accelerometer and gyroscope vectors.

This experiment changes only the coordinate directions. It does not attempt
to move linear acceleration from either physical IMU origin to the CPF
origin, because doing so would require angular-acceleration and lever-arm
compensation.

## Ego-Exo4D CPF Preprocessing and Pretraining

The existing VRS preprocessing entry point gains a configuration-controlled
CPF mode while retaining its current sensor-frame behavior as the default.
CPF mode performs the following operations for each VRS:

1. Read valid `imu-left` acceleration and gyroscope samples.
2. Read that VRS's `T_Cpf_ImuLeft` calibration.
3. Rotate both vectors with `R_Cpf_ImuLeft`.
4. Resample and store them at 50 Hz in the dedicated Ego-Exo4D CPF cache.
5. Store the source frame, target frame, full transform, and applied rotation
   in NPZ metadata.

The new Ego-Exo4D pretraining configuration uses the same official splits,
seven activity labels, 30-second windows, and gravity normalization as the
existing experiment, but reads only the dedicated CPF cache and writes only
under `artifacts/xreal_wear_detection/runs/ego_pretrain_cpf`.

The existing sensor-frame cache and checkpoints are not reused as CPF
pretrained weights and are not overwritten.

## Xreal Preprocessing

The converter performs the following operations:

1. Discover `imu_0.csv` files one directory below the configured source root.
2. Infer the recording label from the directory token: `WORKING_WEAR`,
   `NOWEAR`, `ON`, or `OFF`.
3. Read only type `1` and type `2`, retaining `data0`, `data1`, and `data2`.
4. Align accelerometer and gyroscope by timestamp.
5. Convert both vectors from Xreal right-up-back to CPF left-up-forward:
   `(x, y, z) -> (-x, y, -z)`.
6. Resample both signals from approximately 1000 Hz to 50 Hz with anti-alias
   filtering.
7. Store channels in EgoCHARM order:
   `accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z`.
8. Split stable long recordings into non-overlapping five-second windows.
9. Use one centered five-second window from each `ON` and `OFF` recording.
10. Optionally save accelerometer and gyroscope plots when `--plot` is passed.

Each processed example is a separate NPZ file containing `values` with shape
`(250, 6)` and JSON metadata. The metadata declares the source frame as Xreal
right-up-back, the target as CPF, and records `diag(-1, 1, -1)` as the applied
rotation. The converter also writes a selection JSON that maps every example
to its label and split.

## Development Split

The available data does not support a participant- or session-independent
test. The initial split is explicitly a development split:

- stable long recordings are divided chronologically into 60% train, 20% val,
  and 20% test before windows are generated;
- a five-second guard interval separates adjacent portions;
- `PUT_ON` and `TAKE_OFF` recordings are split at file level with a fixed seed
  in the same 60/20/20 proportions;
- windows never cross split boundaries;
- Ego-Exo4D samples never enter Xreal validation or test data.

Reported results must state that train, val, and test currently come from the
same device, person, day, and recording session. A later trustworthy test set
requires independent Xreal sessions.

## Model

The input tensor has shape `(B, 5, 6, 50)`:

- `B`: batch size;
- `5`: five one-second windows;
- `6`: three accelerometer plus three gyroscope channels;
- `50`: samples per second.

The existing EgoCHARM LLE encodes each second into a 32-dimensional embedding,
producing `(B, 5, 32)`. A newly initialized GRU-based classifier consumes the
five embeddings and produces four logits.

The initial classifier hidden size is 32 to limit overfitting on the small
Xreal dataset. The pretrained LLE comes from the new gravity-normalized,
CPF-direction Ego-Exo4D checkpoint. All LLE parameters are frozen and the LLE
remains in evaluation mode during head training so its Dropout and BatchNorm
state do not change.

Both datasets are expressed in CPF directions before training. Both
pretraining and downstream inputs use acceleration divided by 9.8 while
gyroscope values stay in radians per second.

## Training

Only the new High-Level Classifier is optimized. Weighted cross-entropy uses
weights computed from the Xreal training split to compensate for the smaller
`PUT_ON` and `TAKE_OFF` classes.

The best checkpoint is selected by validation macro-F1. Checkpoints include
the four class names, CPF-pretrained LLE source checkpoint, coordinate
conversion, normalization method, and window configuration.

This first experiment uses Xreal data only in downstream training. Adding a
capped Ego-Exo4D `WORN` subset is deliberately deferred to a separate
comparison so it cannot obscure whether strict transfer works.

## Evaluation

Validation and test report:

- accuracy;
- macro-F1;
- per-class precision, recall, and F1;
- confusion matrix;
- number of evaluated windows.

All evaluation data is Xreal. The initial metrics measure window
classification, not millisecond-accurate transition localization.

## Failure Handling and Scope

The converter fails with a concise error when an expected label token is
missing or a recording cannot produce one complete five-second window. It does
not attempt to repair arbitrary malformed CSV files or infer transition
boundaries.

The first implementation does not include partial LLE fine-tuning, mixed
Ego-Exo4D `WORN` downstream data, streaming state smoothing, or real-time event
debouncing. Those are later comparison stages after the frozen-LLE baseline is
measured.
