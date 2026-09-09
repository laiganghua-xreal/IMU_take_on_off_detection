"""在 Xreal val/test 上评估冻结 LLE 的佩戴任务模型。"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from egocharm.checkpoint import load_checkpoint
from egocharm.config import load_config, project_path
from egocharm.dataset import create_data_loader
from egocharm.engine import choose_device
from egocharm.metrics import calculate_metrics, save_metrics
from egocharm.model import count_trainable_parameters
from egocharm.wear_dataset import (
    BINARY_EVENT_MODE,
    WEAR_CLASS_NAMES,
    WearDataset,
    get_label_mode,
    get_task_class_names,
    load_wear_selection,
)
from egocharm.wear_model import (
    build_wear_model,
    validate_wear_model_config,
)
from egocharm.wear_evaluation import (
    calculate_common_metrics,
    select_event_threshold,
)


def build_argument_parser():
    parser = argparse.ArgumentParser(description="评估 Xreal 佩戴状态模型")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/xreal_wear_frozen_lle.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--high-level-hidden-size", type=int)
    parser.add_argument("--event-threshold", type=float)
    parser.add_argument("--select-event-threshold", action="store_true")
    return parser


def _event_scores(task_probabilities, class_names):
    probabilities = np.asarray(task_probabilities, dtype=np.float64)
    if tuple(class_names) == ("NO_EVENT", "EVENT"):
        return probabilities[:, class_names.index("EVENT")]
    return (
        probabilities[:, class_names.index("PUT_ON")]
        + probabilities[:, class_names.index("TAKE_OFF")]
    )


def _write_predictions(
    output_path,
    records,
    task_predictions,
    event_scores,
    event_threshold,
    class_names,
):
    output_path.mkdir(parents=True, exist_ok=True)
    with (output_path / "predictions.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            (
                "example_uid",
                "original_label",
                "source_path",
                "start_sample",
                "predicted_label",
                "event_probability",
                "predicted_event",
            )
        )
        for record, prediction, score in zip(
            records,
            task_predictions,
            event_scores,
        ):
            writer.writerow(
                (
                    record.example_uid,
                    record.label,
                    str(record.source_path) if record.source_path else "",
                    record.start_sample,
                    class_names[prediction],
                    float(score),
                    int(score >= event_threshold),
                )
            )


def calculate_original_label_breakdown(
    records,
    predictions,
    event_class_index,
    window_seconds,
):
    """按原始四类统计事件召回和稳定窗口误报。"""
    if len(records) != len(predictions):
        raise ValueError("records 和 predictions 的长度必须相同")
    if window_seconds <= 0:
        raise ValueError("window_seconds 必须大于零")

    grouped_predictions = {label: [] for label in WEAR_CLASS_NAMES}
    for record, prediction in zip(records, predictions):
        grouped_predictions[record.label].append(prediction)

    breakdown = {}
    for original_label, label_predictions in grouped_predictions.items():
        number_of_windows = len(label_predictions)
        detected_event_count = sum(
            prediction == event_class_index
            for prediction in label_predictions
        )
        if original_label in ("WORN", "NOT_WORN"):
            duration_hours = number_of_windows * window_seconds / 3600
            false_events_per_hour = 0.0
            if duration_hours > 0:
                false_events_per_hour = detected_event_count / duration_hours
            false_event_rate = 0.0
            if number_of_windows > 0:
                false_event_rate = detected_event_count / number_of_windows
            breakdown[original_label] = {
                "number_of_windows": number_of_windows,
                "false_event_count": detected_event_count,
                "false_event_rate": false_event_rate,
                "false_events_per_hour": false_events_per_hour,
            }
        else:
            event_recall = 0.0
            if number_of_windows > 0:
                event_recall = detected_event_count / number_of_windows
            breakdown[original_label] = {
                "number_of_windows": number_of_windows,
                "detected_event_count": detected_event_count,
                "event_recall": event_recall,
            }
    return breakdown


def main(arguments=None):
    args = build_argument_parser().parse_args(arguments)
    config = load_config(args.config)
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    if args.high_level_hidden_size is not None:
        model_config["high_level_hidden_size"] = args.high_level_hidden_size
    label_mode = get_label_mode(config)
    class_names = get_task_class_names(label_mode)
    validate_wear_model_config(model_config, label_mode)

    device = choose_device(str(training_config["device"]))
    source_checkpoint = None
    if model_config.get("initialization", "pretrained_frozen_lle") == (
        "pretrained_frozen_lle"
    ):
        source_checkpoint = project_path(
            model_config["pretrained_lle_checkpoint"]
        )
    model = build_wear_model(
        model_config,
        label_mode=label_mode,
        checkpoint_path=source_checkpoint,
    )
    model.to(device)
    load_checkpoint(
        args.checkpoint,
        model,
        expected_class_names=class_names,
    )

    records = load_wear_selection(
        project_path(data_config["selection"]),
        args.split,
    )
    dataset = WearDataset(
        records,
        sampling_rate_hz=int(data_config["sampling_rate_hz"]),
        window_seconds=int(data_config["window_seconds"]),
        normalization=str(data_config.get("normalization", "none")),
        label_mode=label_mode,
    )
    data_loader = create_data_loader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        number_of_workers=int(training_config["number_of_workers"]),
    )

    labels = []
    predictions = []
    probability_rows = []
    model.eval()
    with torch.no_grad():
        for features, batch_labels in data_loader:
            logits = model(features.to(device, non_blocking=True))
            labels.extend(batch_labels.tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            probability_rows.extend(
                torch.softmax(logits, dim=1).cpu().tolist()
            )

    task_probabilities = np.asarray(probability_rows, dtype=np.float64)
    scores = _event_scores(task_probabilities, class_names)
    event_truth = np.asarray(
        [record.label in ("PUT_ON", "TAKE_OFF") for record in records],
        dtype=np.int64,
    )
    if args.select_event_threshold:
        if args.split != "val":
            raise ValueError("只能在 val 上选择 event threshold")
        if args.event_threshold is not None:
            raise ValueError("不能同时指定和搜索 event threshold")
        event_threshold = select_event_threshold(event_truth, scores)
    elif args.event_threshold is None:
        event_threshold = 0.5
    else:
        event_threshold = args.event_threshold

    metrics = calculate_metrics(labels, predictions, class_names)
    metrics["number_of_samples"] = len(labels)
    metrics["number_of_trainable_parameters"] = count_trainable_parameters(model)
    metrics["split"] = args.split
    metrics["label_mode"] = label_mode
    metrics["event_threshold"] = float(event_threshold)
    metrics.update(
        calculate_common_metrics(
            records,
            task_probabilities,
            class_names,
            event_threshold,
            int(data_config["sampling_rate_hz"]),
            int(data_config["window_seconds"]),
        )
    )
    output_directory = args.output or args.checkpoint.parent / args.split
    save_metrics(metrics, output_directory)
    _write_predictions(
        output_directory,
        records,
        predictions,
        scores,
        event_threshold,
        class_names,
    )
    if args.select_event_threshold:
        (output_directory / "event-threshold.json").write_text(
            json.dumps(
                {"event_threshold": float(event_threshold)},
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"已评估 {len(labels)} 个五秒窗口，结果保存在 {output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
