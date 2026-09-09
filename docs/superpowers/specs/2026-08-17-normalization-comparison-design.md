# Normalization Comparison Design

## Goal

Provide one reproducible command that trains EgoCHARM twice from scratch under
identical settings, changing only Dataset normalization between `none` and
`gravity`. Evaluate each run on official `val` and `test`, store every artifact
under an unambiguous variant directory, and write machine-readable comparison
summaries.

Before executing the new comparison, remove the user-approved legacy
`artifacts/runs/baseline` directory in full. Do not remove NPZ caches,
preprocessing logs, source files, or unrelated runs.

## Approaches Considered

An independent Python orchestration script is preferred over a shell script
because it can safely generate YAML, stream subprocess output to the terminal
and log files, validate artifacts, and write summaries without shell quoting or
pipeline-status ambiguity. Extending `egocharm.train` to own multiple variants
was rejected because the existing entry point should remain responsible for one
training run.

## Command Interface

Create `scripts/run_normalization_comparison.py` with:

```text
--config           base configuration; default configs/egocharm.yaml
--output           comparison root; default artifacts/comparisons/normalization
--epochs           override both variants' epoch count; default 30
--resume           pass --resume to both training runs
--dry-run          generate and report the plan without training or evaluation
```

The normal command is:

```bash
python scripts/run_normalization_comparison.py
```

Only one variant runs at a time on the single GPU. Each child command uses
`sys.executable`, ensuring the same active Python/Conda environment as the
comparison script.

## Variant Configuration

Load the base YAML twice with independent deep copies. Both variants receive
the same seed, model, official selection, batch size, optimizer, scheduler,
worker count, device, and epoch count. The only experimental difference is:

```text
none:     data.normalization = none
gravity:  data.normalization = gravity
```

Set `training.output_root` to the absolute comparison root and invoke training
with `--run-name none` or `--run-name gravity`. Save each generated YAML inside
its variant directory before starting the child process. The generated config
therefore documents the intended setup, while training's `resolved-config.json`
documents what the training entry point consumed.

## Execution and Failure Behavior

For each variant, in order `none` then `gravity`:

1. run `python -m egocharm.train` with the variant config;
2. require `best.pt` and `last.pt` to exist after training;
3. run `python -m egocharm.evaluate --split val` using `best.pt`;
4. run `python -m egocharm.evaluate --split test` using `best.pt`;
5. require each split's `metrics.json` to exist.

Stream child output to the current terminal and the variant's `train.log`,
`val/evaluate.log`, or `test/evaluate.log`. If any command returns nonzero,
stop immediately and preserve all completed artifacts for diagnosis or resume.
Never continue to the next stage with missing or stale output.

`--resume` is valid only when the corresponding `last.pt` exists; otherwise the
script fails clearly rather than silently starting a fresh run. Without
`--resume`, refuse to overwrite an existing `best.pt` or `last.pt` so a repeated
command cannot accidentally mix experiments. `--dry-run` may write generated
configs and a plan description but must not invoke training or evaluation.

## Output Layout

```text
artifacts/comparisons/normalization/
├── none/
│   ├── config.yaml
│   ├── resolved-config.json
│   ├── best.pt
│   ├── last.pt
│   ├── train.log
│   ├── val/
│   │   ├── evaluate.log
│   │   ├── metrics.json
│   │   ├── per-class.csv
│   │   └── confusion-matrix*.png
│   └── test/
│       └── same evaluation artifacts
├── gravity/
│   └── same structure
├── comparison.csv
└── comparison.json
```

The variant directory is simultaneously the `egocharm.train` run directory, so
checkpoints, resolved config, logs, and evaluations stay together.

## Comparison Summaries

After both variants finish, read their val/test `metrics.json` files and the
metadata in `best.pt`. Write one row per `(normalization, split)` with:

```text
normalization
split
accuracy
macro_f1
best_epoch
best_val_macro_f1
checkpoint
```

`comparison.json` stores the same rows plus the base config path and output
root. `comparison.csv` provides a compact table for spreadsheet inspection.
Summary files are written atomically only after all four evaluations succeed.

## Testing and Safety

Unit tests use a temporary directory and a fake command runner. They verify
variant config isolation, command order, distinct output paths, dry-run behavior,
overwrite/resume guards, failure propagation, and summary contents without
starting real training.

Before deleting the legacy baseline, list the exact directory contents. Delete
only `/home/xreal/egocharm/artifacts/runs/baseline` after the user's explicit
approval already recorded in this conversation. Verify NPZ count and modification
times remain unchanged. Real execution then trains both variants for 30 epochs,
evaluates official val/test, and confirms no child process remains afterward.
