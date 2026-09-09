"""三种 Xreal 标签任务共用的事件、方向和连续误报指标。"""

from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


EVENT_LABELS = ("PUT_ON", "TAKE_OFF")
STABLE_LABELS = ("WORN", "NOT_WORN")


def _event_class_indices(class_names):
    names = tuple(class_names)
    if "EVENT" in names:
        return (names.index("EVENT"),)
    if "PUT_ON" in names and "TAKE_OFF" in names:
        return (names.index("PUT_ON"), names.index("TAKE_OFF"))
    raise ValueError(f"类别中缺少事件输出: {names}")


def event_probabilities(logits, class_names):
    """把任务 logits 转为统一的 EVENT 概率。"""
    probabilities = torch.softmax(torch.as_tensor(logits), dim=1)
    event_indices = _event_class_indices(class_names)
    values = probabilities[:, event_indices].sum(dim=1)
    return values.detach().cpu().numpy()


def select_event_threshold(labels, probabilities):
    """只根据给定标签选择 Event F1 最大的确定性阈值。"""
    label_values = np.asarray(labels, dtype=np.int64)
    probability_values = np.asarray(probabilities, dtype=np.float64)
    if label_values.ndim != 1 or probability_values.ndim != 1:
        raise ValueError("labels 和 probabilities 必须是一维")
    if len(label_values) == 0 or len(label_values) != len(probability_values):
        raise ValueError("labels 和 probabilities 必须等长且非空")

    candidates = sorted(set(probability_values.tolist()) | {0.5})
    scored_candidates = []
    for threshold in candidates:
        predictions = probability_values >= threshold
        score = f1_score(
            label_values,
            predictions,
            zero_division=0,
        )
        scored_candidates.append((float(score), -float(threshold), float(threshold)))
    return max(scored_candidates)[2]


def _merge_intervals(intervals):
    merged = []
    for start, stop in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return [(start, stop) for start, stop in merged]


def _stable_metrics(
    records,
    event_predictions,
    sampling_rate_hz,
    window_seconds,
):
    window_samples = sampling_rate_hz * window_seconds
    stable_rows = [
        (record, bool(prediction))
        for record, prediction in zip(records, event_predictions)
        if record.label in STABLE_LABELS
    ]

    def summarize(rows):
        exposure_by_source = defaultdict(list)
        alarms_by_source = defaultdict(list)
        false_positive_windows = 0
        for record, predicted_event in rows:
            source_key = (
                str(record.source_path)
                if record.source_path is not None
                else f"uid:{record.example_uid}"
            )
            interval = (
                int(record.start_sample),
                int(record.start_sample) + window_samples,
            )
            exposure_by_source[source_key].append(interval)
            if predicted_event:
                alarms_by_source[source_key].append(interval)
                false_positive_windows += 1

        exposure_samples = sum(
            stop - start
            for intervals in exposure_by_source.values()
            for start, stop in _merge_intervals(intervals)
        )
        merged_false_alarms = sum(
            len(_merge_intervals(intervals))
            for intervals in alarms_by_source.values()
        )
        exposure_seconds = exposure_samples / sampling_rate_hz
        false_alarms_per_hour = 0.0
        if exposure_seconds > 0:
            false_alarms_per_hour = (
                merged_false_alarms * 3600 / exposure_seconds
            )
        number_of_windows = len(rows)
        false_positive_rate = 0.0
        if number_of_windows > 0:
            false_positive_rate = false_positive_windows / number_of_windows
        return {
            "number_of_windows": number_of_windows,
            "false_positive_windows": false_positive_windows,
            "window_false_positive_rate": float(false_positive_rate),
            "merged_false_alarms": merged_false_alarms,
            "exposure_seconds": float(exposure_seconds),
            "false_alarms_per_hour": float(false_alarms_per_hour),
        }

    overall = summarize(stable_rows)
    overall["by_original_label"] = {
        label: summarize(
            [row for row in stable_rows if row[0].label == label]
        )
        for label in STABLE_LABELS
    }
    return overall


def calculate_common_metrics(
    records,
    task_probabilities,
    class_names,
    threshold,
    sampling_rate_hz,
    window_seconds,
):
    """计算可跨四/三/二分类直接比较的指标。"""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold 必须位于 [0, 1]")
    probabilities = np.asarray(task_probabilities, dtype=np.float64)
    if probabilities.shape != (len(records), len(class_names)):
        raise ValueError("task_probabilities 形状与 records/class_names 不一致")

    event_indices = _event_class_indices(class_names)
    event_scores = probabilities[:, event_indices].sum(axis=1)
    event_truth = np.asarray(
        [record.label in EVENT_LABELS for record in records],
        dtype=np.int64,
    )
    event_predictions = event_scores >= threshold
    precision, recall, event_f1, _ = precision_recall_fscore_support(
        event_truth,
        event_predictions,
        average="binary",
        zero_division=0,
    )
    common_event = {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(event_f1),
        "miss_rate": float(1.0 - recall),
        "support": int(event_truth.sum()),
        "predicted_events": int(event_predictions.sum()),
    }

    direction = None
    names = tuple(class_names)
    if "PUT_ON" in names and "TAKE_OFF" in names:
        put_on_index = names.index("PUT_ON")
        take_off_index = names.index("TAKE_OFF")
        event_rows = [
            index
            for index, record in enumerate(records)
            if record.label in EVENT_LABELS
        ]
        direction_truth = np.asarray(
            [0 if records[index].label == "PUT_ON" else 1 for index in event_rows]
        )
        direction_predictions = np.asarray(
            [
                int(
                    probabilities[index, take_off_index]
                    > probabilities[index, put_on_index]
                )
                for index in event_rows
            ]
        )
        predicted_task_indices = probabilities.argmax(axis=1)
        correct_end_to_end = sum(
            predicted_task_indices[index]
            == (put_on_index if records[index].label == "PUT_ON" else take_off_index)
            for index in event_rows
        )
        support = len(event_rows)
        direction = {
            "accuracy": float(
                accuracy_score(direction_truth, direction_predictions)
                if support
                else 0.0
            ),
            "macro_f1": float(
                f1_score(
                    direction_truth,
                    direction_predictions,
                    labels=[0, 1],
                    average="macro",
                    zero_division=0,
                )
                if support
                else 0.0
            ),
            "support": support,
            "end_to_end_accuracy": float(
                correct_end_to_end / support if support else 0.0
            ),
        }

    return {
        "common_event": common_event,
        "direction": direction,
        "stable": _stable_metrics(
            records,
            event_predictions,
            sampling_rate_hz,
            window_seconds,
        ),
    }
