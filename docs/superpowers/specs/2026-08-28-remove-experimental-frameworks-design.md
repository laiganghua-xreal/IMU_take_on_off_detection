# Remove Experimental Frameworks Design

## Goal

Reduce the project to two supported workflows:

1. the original EgoCHARM seven-class reproduction; and
2. the Xreal wear-state and wear-event classification workflow.

Later full-grid, dual-accelerometer, and impact-branch experiments are no longer
part of the maintained project.

## Retained source code

The original EgoCHARM workflow retains:

- `config.py`
- `dataset.py`
- `model.py`
- `engine.py`
- `checkpoint.py`
- `metrics.py`
- `train.py`
- `evaluate.py`

The Xreal wear workflow retains:

- `wear_dataset.py`
- `wear_model.py`
- `wear_evaluation.py`
- `train_wear.py`
- `evaluate_wear.py`

The ordinary preprocessing, comparison scripts, YAML configurations, caches,
checkpoints, and reports used by these two workflows remain in place.

## Removed source code

Remove the following later experiment implementations:

- `src/egocharm/full_grid/`
- `src/egocharm/dual_acc/`
- `src/egocharm/impact_model.py`
- `src/egocharm/impact_experiment.py`

Remove the dynamic `dual_acc_impact` construction branch from
`wear_model.build_random_wear_model`. The ordinary random, frozen-LLE, and
fine-tuned-LLE model paths remain unchanged.

## Removed experiment data and results

Delete these experiment directories in full, including their source entry
points, caches, checkpoints, logs, reports, and metadata:

- `experiments/imu_wear_full_grid_v1/`
- `experiments/xreal_dual_acc_v1/`
- `experiments/xreal_dual_acc_four_class_v1/`
- `experiments/xreal_dual_acc_impact_500hz_v1/`

These directories occupy approximately 1.56 GB. The project is not a Git
repository, so their generated contents cannot be recovered through Git.

## Removed tests and documentation

Delete tests whose only subject is `full_grid`, `dual_acc`, or the impact
experiment. Preserve every test for the original reproduction and ordinary
Xreal wear workflow.

Delete the four matching design documents and four matching implementation
plans dated 2026-08-20 and 2026-08-21. Retain this cleanup design document so
the removal boundary remains documented.

Remove stale references to deleted experiments from maintained documentation.
Delete Python bytecode caches and pytest caches after the source cleanup.

## Verification

After deletion:

1. search the retained source, scripts, configs, tests, and README for imports
   of `egocharm.full_grid`, `egocharm.dual_acc`, `impact_model`, and
   `impact_experiment`;
2. compile the retained Python packages and scripts;
3. run the remaining pytest suite;
4. verify that `egocharm.train`, `egocharm.evaluate`, `egocharm.train_wear`,
   and `egocharm.evaluate_wear` import successfully; and
5. verify that no retained Xreal configuration selects the removed
   `dual_acc_impact` architecture.

The cleanup is complete only if both supported workflows remain importable and
their retained tests pass.
