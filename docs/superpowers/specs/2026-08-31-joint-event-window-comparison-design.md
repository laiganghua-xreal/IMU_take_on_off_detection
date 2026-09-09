# Joint nine-class event-window comparison

## Goal

Locate every test window from the joint nine-class model that belongs to one
of these four groups and visualize its six-axis IMU signal:

- false-positive `PUT_ON`;
- false-positive `TAKE_OFF`;
- true-positive `PUT_ON`;
- true-positive `TAKE_OFF`.

The figures are diagnostic artifacts. They do not modify training data,
checkpoints, model configuration, or existing evaluation metrics.

## Window selection

Recreate `EvaluationJointDataset` with the resolved joint configuration and
the same combined Ego and Xreal test manifests used by `evaluate_joint.py`.
Load `joint_9class/best.pt`, run deterministic inference without augmentation,
and select windows using these definitions:

- true positive: true label and predicted label are the same event class;
- false positive: predicted label is an event class while the true label is
  different.

The existing confusion matrix predicts 2 false-positive `PUT_ON` windows,
5 false-positive `TAKE_OFF` windows, 10 true-positive `PUT_ON` windows, and
10 true-positive `TAKE_OFF` windows. Regenerated predictions must match these
counts before the artifacts are accepted.

## Outputs

Write all new artifacts below:

`artifacts/ego_xreal_joint_9class/runs/joint_9class/test/event-window-comparison/`

The output contains:

- one six-axis figure per selected window, grouped under
  `false_positive/PUT_ON`, `false_positive/TAKE_OFF`,
  `true_positive/PUT_ON`, and `true_positive/TAKE_OFF`;
- a `put-on-comparison.png` overview;
- a `take-off-comparison.png` overview;
- `window-index.csv` containing group, true label, predicted label,
  recording UID, start sample/time, confidence, and image path;
- `summary.json` containing counts and plotting metadata.

## Plot format

Each plot has two synchronized time-series panels:

- accelerometer XYZ in m/s²;
- gyroscope XYZ in rad/s.

All individual plots use consistent axis limits within each sensor modality,
calculated from all selected windows. This allows direct amplitude comparison.
Titles identify TP/FP status, true class, predicted class, recording UID,
window start time, and prediction confidence.

The two overview figures place false positives and true positives of the same
predicted event class in one grid, using the same sensor axis limits as the
individual figures.

## Verification

- Prediction counts must reproduce the event columns of the saved test
  confusion matrix.
- The CSV must contain 27 rows: 7 false positives and 20 true positives.
- Every CSV row must reference an existing non-empty PNG.
- The four output directories must contain 2, 5, 10, and 10 individual PNGs.
- Existing test metrics and confusion-matrix files must remain unchanged.
