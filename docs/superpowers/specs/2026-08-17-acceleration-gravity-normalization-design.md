# Acceleration Gravity Normalization Design

## Goal

Add an optional Dataset-time normalization mode that divides the three
accelerometer channels by `9.8` while leaving the three gyroscope channels
unchanged. Keep the stored NPZ caches untouched and leave a small, explicit
extension point for future normalization methods.

## Configuration

Add one setting under `data`:

```yaml
data:
  normalization: gravity
```

Supported values initially are:

- `none`: return all six cached channels without scaling;
- `gravity`: divide `accel_x`, `accel_y`, and `accel_z` by `9.8`, while
  preserving `gyro_x`, `gyro_y`, and `gyro_z` exactly.

If `normalization` is absent, use `none` so older configuration files retain
their original behavior. Unknown method names fail with a clear `ValueError`.

## Data Flow and Extension Point

Normalization happens in `EgoCharmDataset.__getitem__` after a complete
`window_seconds` slice has been loaded and validated, but before it is reshaped
from `(S * T, C)` into `(S, C, T)`.

The Dataset receives a normalization method name. A standalone window
normalization dispatcher maps that name to a focused implementation. Adding a
future method therefore requires adding a new normalization function and one
dispatcher entry, without changing Dataset indexing, DataLoader, training, or
evaluation loops.

The initial transformation is:

```text
input window:       (S * T, 6)
acc channels 0:3:  values[:, 0:3] / 9.8
gyro channels 3:6: unchanged
model sample:       (S, 6, T)
```

The returned array remains finite, contiguous, and `float32`.

## Training and Evaluation Integration

Both `egocharm.train` and `egocharm.evaluate` pass
`data.normalization` into their Dataset instances. The checked-in configuration
selects `gravity`, so new training and evaluation runs use the same input
transformation.

Existing model attributes and checkpoint state-dict keys are unchanged. A
checkpoint trained with `none` is structurally loadable, but its learned input
distribution does not match `gravity`; new normalized experiments must use a
new run name instead of resuming the existing `baseline` run.

## Error Handling

- Reject unsupported normalization names when constructing the Dataset.
- Reject non-finite input windows before normalization, as today.
- Keep the existing incomplete-window and shape checks unchanged.

## Tests

Tests will establish that:

1. `gravity` divides all three accelerometer channels by exactly `9.8`;
2. `gravity` leaves all three gyroscope channels unchanged;
3. `none` preserves the existing raw-value behavior;
4. an omitted config value resolves to `none` in train and evaluation wiring;
5. an unknown method raises a clear error;
6. the output remains `float32` with shape `(S, 6, T)`;
7. the full existing test suite still passes.

No NPZ file is regenerated or modified by this feature.
