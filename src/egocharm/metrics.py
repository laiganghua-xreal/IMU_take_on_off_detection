"""评估指标计算和结果输出。"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

matplotlib.use("Agg")
from matplotlib import pyplot as plot  # noqa: E402


def _validate_metric_inputs(labels, predictions, class_names):
    label_values = list(labels)
    prediction_values = list(predictions)
    configured_class_names = list(class_names)

    if not label_values or not prediction_values:
        raise ValueError("labels 和 predictions 都不能为空")
    if len(label_values) != len(prediction_values):
        raise ValueError("labels 和 predictions 的长度必须相同")
    if not configured_class_names or any(
        not isinstance(class_name, str) or not class_name.strip()
        for class_name in configured_class_names
    ):
        raise ValueError("class_names 必须是非空类别名称序列")
    if len(set(configured_class_names)) != len(configured_class_names):
        raise ValueError("class_names 不能包含重复类别名称")

    number_of_classes = len(configured_class_names)
    for values_name, values in (
        ("labels", label_values),
        ("predictions", prediction_values),
    ):
        for value in values:
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or not 0 <= int(value) < number_of_classes
            ):
                raise ValueError(
                    f"{values_name} 必须是 [0, {number_of_classes}) 范围内的整数"
                )

    return label_values, prediction_values, configured_class_names


def calculate_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    class_names: Sequence[str],
) -> dict[str, Any]:
    """按固定类别顺序计算 accuracy、macro-F1 和逐类别指标。"""
    labels, predictions, class_names = _validate_metric_inputs(
        labels,
        predictions,
        class_names,
    )
    class_indices = list(range(len(class_names)))
    precision, recall, class_f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=class_indices,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions, labels=class_indices)

    per_class: dict[str, dict[str, float | int]] = {}
    for class_index, class_name in enumerate(class_names):
        per_class[class_name] = {
            "precision": float(precision[class_index]),
            "recall": float(recall[class_index]),
            "f1": float(class_f1[class_index]),
            "support": int(support[class_index]),
        }

    return {
        "class_names": list(class_names),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=class_indices,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class": per_class,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
            encoding="utf-8",
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(
                payload,
                temporary_file,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _atomic_class_csv(path: Path, metrics: dict[str, Any]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            suffix=".tmp",
            delete=False,
            newline="",
            encoding="utf-8",
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            writer = csv.writer(temporary_file)
            writer.writerow(("class", "precision", "recall", "f1", "support"))
            for class_name in metrics["class_names"]:
                values = metrics["per_class"][class_name]
                writer.writerow(
                    (
                        class_name,
                        values["precision"],
                        values["recall"],
                        values["f1"],
                        values["support"],
                    )
                )
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def save_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
    normalize: bool,
) -> None:
    values = np.asarray(matrix, dtype=np.float64 if normalize else np.int64)
    if normalize:
        row_totals = values.sum(axis=1, keepdims=True)
        values = np.divide(
            values,
            row_totals,
            out=np.zeros_like(values),
            where=row_totals > 0,
        )

    figure, axis = plot.subplots(figsize=(8, 7))
    image = axis.imshow(values, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xlabel="Predicted label",
        ylabel="True label",
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plot.setp(axis.get_xticklabels(), rotation=45, ha="right")

    threshold = float(values.max()) / 2 if values.size else 0.0
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            if normalize:
                text = f"{value:.2f}"
            else:
                text = str(int(value))
            if value > threshold:
                text_color = "white"
            else:
                text_color = "black"
            axis.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                color=text_color,
            )

    figure.tight_layout()
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        figure.savefig(temporary_path, dpi=150, format="png")
        os.replace(temporary_path, output_path)
    finally:
        plot.close(figure)
        temporary_path.unlink(missing_ok=True)


def save_metrics(metrics: dict[str, Any], output_directory: Path) -> None:
    """保存 JSON、逐类别 CSV 和两张混淆矩阵图。"""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    _atomic_json(output_path / "metrics.json", metrics)
    _atomic_class_csv(output_path / "per-class.csv", metrics)

    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    class_names = tuple(metrics["class_names"])
    save_confusion_matrix(
        matrix,
        class_names,
        output_path / "confusion-matrix.png",
        normalize=False,
    )
    save_confusion_matrix(
        matrix,
        class_names,
        output_path / "confusion-matrix-normalized.png",
        normalize=True,
    )
