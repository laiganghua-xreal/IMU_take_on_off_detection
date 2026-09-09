# Xreal Stable-Window One-Second Stride Design

## Goal

Regenerate only the Xreal `WORN` and `NOT_WORN` examples as five-second
windows whose start times are one second apart. Preserve all existing
`PUT_ON` and `TAKE_OFF` examples, cache files, labels, and split assignments.

## Window and Split Rules

Stable windows use 50 Hz data, a five-second length (250 samples), and a
one-second stride (50 samples). Consecutive windows overlap by four seconds.

The chronological train/validation/test time regions remain identical to the
previous non-overlapping conversion. One complete five-second source interval
is excluded between train and validation, and another is excluded between
validation and test. Overlapping windows are generated independently inside
each region and may not cross a split or guard boundary.

## Cache Replacement

The converter accepts `--stable-stride-seconds`. It first writes all new
stable examples to a temporary cache directory. After successful conversion,
it moves them into the configured Xreal cache, overwriting matching old stable
files, removes stable files referenced by the previous selection but absent
from the new stable set, and atomically replaces the selection JSON.

With `--reuse-existing-events`, event entries are loaded from the existing
selection and reused without opening, rewriting, renaming, or reassigning
their NPZ files. The command fails if an existing selection is unavailable,
because preserving event identities cannot otherwise be guaranteed.

## Output

The selection continues to contain the original four labels. Only the number
and start offsets of `WORN` and `NOT_WORN` entries change. Each stable entry
retains a `(250, 6)` cache tensor and records its one-second-aligned
`start_sample`.

The current command is:

```bash
python Xreal_datasets/preprocess_xreal_imu.py \
  --stable-stride-seconds 1 \
  --reuse-existing-events
```

## Validation

Tests verify one-second stable starts, four-second overlap, unchanged
five-second split guards, and event-entry reuse. After regeneration, the old
and new event file checksums and selection fields must match exactly. Stable
selection entries must stay within their split regions and advance by 50
samples inside each region.
