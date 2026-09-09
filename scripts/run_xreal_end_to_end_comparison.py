"""运行 Xreal HLE 容量搜索和三种标签任务的端到端对比。"""

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "artifacts/xreal_wear_detection/runs/xreal_end_to_end"
COMPARISON_ROOT = OUTPUT_ROOT / "comparison"
SELECTED_HLE_PATH = COMPARISON_ROOT / "selected-hle.json"
HLE_SIZES = (16, 32, 64)
SEEDS = (42, 43, 44)
MODES = ("four_class", "three_class", "binary_event")
CONFIG_BY_MODE = {
    "four_class": PROJECT_ROOT / "configs/xreal_end_to_end_four_class.yaml",
    "three_class": PROJECT_ROOT / "configs/xreal_end_to_end_three_class.yaml",
    "binary_event": PROJECT_ROOT / "configs/xreal_end_to_end_binary_event.yaml",
}
SUMMARY_METRICS = (
    "task_macro_f1",
    "event_precision",
    "event_recall",
    "event_f1",
    "direction_macro_f1",
    "stable_window_fpr",
    "false_alarms_per_hour",
)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def select_hle_size(candidate_metrics):
    """按联合分数选择 HLE；0.005 内优先更小模型。"""
    candidates = {}
    for hidden_size, metrics in sorted(candidate_metrics.items()):
        event_f1 = float(metrics["common_event"]["f1"])
        direction_f1 = float(metrics["direction"]["macro_f1"])
        candidates[str(hidden_size)] = {
            "event_f1": event_f1,
            "direction_macro_f1": direction_f1,
            "selection_score": (event_f1 + direction_f1) / 2,
        }

    best_score = max(item["selection_score"] for item in candidates.values())
    eligible_sizes = [
        int(hidden_size)
        for hidden_size, item in candidates.items()
        if best_score - item["selection_score"] <= 0.005 + 1e-12
    ]
    return {
        "high_level_hidden_size": min(eligible_sizes),
        "tie_tolerance": 0.005,
        "candidates": candidates,
    }


def build_formal_experiments(high_level_hidden_size):
    return [
        {
            "mode": mode,
            "seed": seed,
            "high_level_hidden_size": high_level_hidden_size,
        }
        for mode in MODES
        for seed in SEEDS
    ]


def aggregate_results(rows, metric_names=SUMMARY_METRICS):
    summary = {}
    for mode in sorted({row["mode"] for row in rows}):
        mode_rows = [row for row in rows if row["mode"] == mode]
        summary[mode] = {}
        for metric_name in metric_names:
            values = [
                float(row[metric_name])
                for row in mode_rows
                if row.get(metric_name) is not None
            ]
            if not values:
                summary[mode][metric_name] = None
                continue
            summary[mode][metric_name] = {
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values),
            }
    return summary


def _run(command):
    print("运行:", " ".join(str(value) for value in command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _train(config_path, run_name, seed, hidden_size):
    _run(
        [
            sys.executable,
            "-m",
            "egocharm.train_wear",
            "--config",
            str(config_path),
            "--run-name",
            run_name,
            "--seed",
            str(seed),
            "--high-level-hidden-size",
            str(hidden_size),
        ]
    )


def _evaluate(
    config_path,
    checkpoint_path,
    split,
    output_path,
    hidden_size,
    event_threshold=None,
    select_threshold=False,
):
    command = [
        sys.executable,
        "-m",
        "egocharm.evaluate_wear",
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--split",
        split,
        "--output",
        str(output_path),
        "--high-level-hidden-size",
        str(hidden_size),
    ]
    if select_threshold:
        command.append("--select-event-threshold")
    if event_threshold is not None:
        command.extend(("--event-threshold", str(event_threshold)))
    _run(command)


def run_hle_search():
    config_path = CONFIG_BY_MODE["three_class"]
    candidate_metrics = {}
    for hidden_size in HLE_SIZES:
        run_name = f"hle_search/three_class/hidden_{hidden_size}"
        run_directory = OUTPUT_ROOT / run_name
        _train(config_path, run_name, seed=42, hidden_size=hidden_size)
        _evaluate(
            config_path,
            run_directory / "best.pt",
            "val",
            run_directory / "val",
            hidden_size,
            select_threshold=True,
        )
        candidate_metrics[hidden_size] = json.loads(
            (run_directory / "val/metrics.json").read_text(encoding="utf-8")
        )

    selected = select_hle_size(candidate_metrics)
    _write_json(SELECTED_HLE_PATH, selected)
    print(
        "选定 HLE hidden size:",
        selected["high_level_hidden_size"],
        flush=True,
    )
    return selected


def _result_row(mode, seed, run_directory, test_metrics):
    direction = test_metrics.get("direction")
    checkpoint = torch.load(
        run_directory / "best.pt",
        map_location="cpu",
        weights_only=False,
    )
    return {
        "mode": mode,
        "seed": seed,
        "best_epoch": int(checkpoint["training_state"]["epoch"]),
        "number_of_trainable_parameters": int(
            test_metrics["number_of_trainable_parameters"]
        ),
        "task_macro_f1": float(test_metrics["macro_f1"]),
        "event_precision": float(test_metrics["common_event"]["precision"]),
        "event_recall": float(test_metrics["common_event"]["recall"]),
        "event_f1": float(test_metrics["common_event"]["f1"]),
        "direction_macro_f1": (
            float(direction["macro_f1"]) if direction is not None else None
        ),
        "stable_window_fpr": float(
            test_metrics["stable"]["window_false_positive_rate"]
        ),
        "false_alarms_per_hour": float(
            test_metrics["stable"]["false_alarms_per_hour"]
        ),
        "event_threshold": float(test_metrics["event_threshold"]),
    }


def _write_summary(rows, aggregate, hidden_size):
    COMPARISON_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(
        COMPARISON_ROOT / "summary.json",
        {
            "high_level_hidden_size": hidden_size,
            "seeds": list(SEEDS),
            "per_seed": rows,
            "aggregate": aggregate,
        },
    )
    field_names = list(rows[0])
    with (COMPARISON_ROOT / "summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def run_formal_comparison():
    if not SELECTED_HLE_PATH.is_file():
        raise FileNotFoundError(
            f"请先运行 hle-search: {SELECTED_HLE_PATH}"
        )
    selected = json.loads(SELECTED_HLE_PATH.read_text(encoding="utf-8"))
    hidden_size = int(selected["high_level_hidden_size"])
    rows = []
    for experiment in build_formal_experiments(hidden_size):
        mode = experiment["mode"]
        seed = experiment["seed"]
        config_path = CONFIG_BY_MODE[mode]
        run_name = f"{mode}/seed_{seed}"
        run_directory = OUTPUT_ROOT / run_name
        _train(config_path, run_name, seed, hidden_size)
        _evaluate(
            config_path,
            run_directory / "best.pt",
            "val",
            run_directory / "val",
            hidden_size,
            select_threshold=True,
        )
        threshold_payload = json.loads(
            (run_directory / "val/event-threshold.json").read_text(
                encoding="utf-8"
            )
        )
        threshold = float(threshold_payload["event_threshold"])
        _evaluate(
            config_path,
            run_directory / "best.pt",
            "test",
            run_directory / "test",
            hidden_size,
            event_threshold=threshold,
        )
        test_metrics = json.loads(
            (run_directory / "test/metrics.json").read_text(encoding="utf-8")
        )
        rows.append(_result_row(mode, seed, run_directory, test_metrics))

    aggregate = aggregate_results(rows)
    _write_summary(rows, aggregate, hidden_size)
    return rows, aggregate


def build_argument_parser():
    parser = argparse.ArgumentParser(description="运行 Xreal 端到端标签对比")
    parser.add_argument(
        "--stage",
        choices=("hle-search", "formal", "all"),
        default="all",
    )
    return parser


def main(arguments=None):
    args = build_argument_parser().parse_args(arguments)
    if args.stage in ("hle-search", "all"):
        run_hle_search()
    if args.stage in ("formal", "all"):
        run_formal_comparison()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
