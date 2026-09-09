"""顺序训练并评估无归一化与重力归一化两组EgoCHARM实验。"""

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml

from egocharm.config import load_config, project_path


VARIANT_NAMES = ("none", "gravity")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_variant_config(base_config, normalization, output_root, epochs):
    """复制基础配置，只设置实验归一化、轮数和统一输出根目录。"""
    if epochs <= 0:
        raise ValueError("epochs 必须大于零")
    if normalization not in VARIANT_NAMES:
        raise ValueError(f"不支持的 normalization: {normalization}")

    variant_config = copy.deepcopy(base_config)
    variant_config["data"]["normalization"] = normalization
    variant_config["training"]["number_of_epochs"] = epochs
    variant_config["training"]["output_root"] = str(Path(output_root).resolve())
    return variant_config


def validate_variant_state(variant_directory, resume):
    """阻止意外覆盖权重，并确保恢复训练确实存在last.pt。"""
    variant_path = Path(variant_directory)
    last_checkpoint = variant_path / "last.pt"
    best_checkpoint = variant_path / "best.pt"

    if resume:
        if not last_checkpoint.is_file():
            raise FileNotFoundError(f"恢复训练需要 checkpoint: {last_checkpoint}")
        return

    for checkpoint_path in (best_checkpoint, last_checkpoint):
        if checkpoint_path.exists():
            raise FileExistsError(f"拒绝覆盖已有 checkpoint: {checkpoint_path}")


def write_yaml_atomic(path, payload):
    """通过同目录临时文件原子写入YAML配置。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            yaml.safe_dump(
                payload,
                temporary_file,
                allow_unicode=True,
                sort_keys=False,
            )
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def train_command(config_path, variant_name, resume=False):
    """构造单个变体的训练命令。"""
    command = [
        sys.executable,
        "-m",
        "egocharm.train",
        "--config",
        str(Path(config_path).resolve()),
        "--run-name",
        variant_name,
    ]
    if resume:
        command.append("--resume")
    return command


def evaluate_command(config_path, checkpoint_path, split, output_path):
    """构造单个变体在val或test上的评估命令。"""
    return [
        sys.executable,
        "-m",
        "egocharm.evaluate",
        "--config",
        str(Path(config_path).resolve()),
        "--checkpoint",
        str(Path(checkpoint_path).resolve()),
        "--split",
        split,
        "--output",
        str(Path(output_path).resolve()),
    ]


def run_command(command, log_path):
    """运行子命令，同时把合并输出写到终端和独立日志。"""
    output_path = Path(log_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()

    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def _require_file(path):
    required_path = Path(path)
    if not required_path.is_file():
        raise FileNotFoundError(f"命令完成但缺少必要产物: {required_path}")


def execute_variant(
    variant_name,
    config_path,
    output_root,
    resume,
    command_runner=run_command,
):
    """顺序完成一个变体的训练、val评估和test评估。"""
    if variant_name not in VARIANT_NAMES:
        raise ValueError(f"不支持的变体: {variant_name}")

    comparison_root = Path(output_root).resolve()
    variant_directory = comparison_root / variant_name
    validate_variant_state(variant_directory, resume)

    command_runner(
        train_command(config_path, variant_name, resume),
        variant_directory / "train.log",
    )

    best_checkpoint = variant_directory / "best.pt"
    last_checkpoint = variant_directory / "last.pt"
    _require_file(best_checkpoint)
    _require_file(last_checkpoint)

    for split in ("val", "test"):
        split_directory = variant_directory / split
        command_runner(
            evaluate_command(
                config_path,
                best_checkpoint,
                split,
                split_directory,
            ),
            split_directory / "evaluate.log",
        )
        _require_file(split_directory / "metrics.json")


def _read_finite_number(payload, field_name, source_path):
    value = payload.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source_path} 的 {field_name} 必须是数值")
    float_value = float(value)
    if not math.isfinite(float_value):
        raise ValueError(f"{source_path} 的 {field_name} 必须是有限值")
    return float_value


def collect_comparison_rows(output_root):
    """读取两组best checkpoint与四份评估指标，构造固定顺序的汇总行。"""
    comparison_root = Path(output_root).resolve()
    rows = []
    for variant_name in VARIANT_NAMES:
        variant_directory = comparison_root / variant_name
        checkpoint_path = variant_directory / "best.pt"
        _require_file(checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        training_state = checkpoint.get("training_state")
        if not isinstance(training_state, dict):
            raise ValueError(f"checkpoint 缺少 training_state: {checkpoint_path}")

        best_epoch = training_state.get("epoch")
        if isinstance(best_epoch, bool) or not isinstance(best_epoch, int):
            raise ValueError(f"checkpoint epoch 必须是整数: {checkpoint_path}")
        best_val_macro_f1 = _read_finite_number(
            training_state,
            "best_macro_f1",
            checkpoint_path,
        )

        for split in ("val", "test"):
            metrics_path = variant_directory / split / "metrics.json"
            _require_file(metrics_path)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if not isinstance(metrics, dict):
                raise ValueError(f"评估指标必须是JSON对象: {metrics_path}")
            rows.append(
                {
                    "normalization": variant_name,
                    "split": split,
                    "accuracy": _read_finite_number(
                        metrics, "accuracy", metrics_path
                    ),
                    "macro_f1": _read_finite_number(
                        metrics, "macro_f1", metrics_path
                    ),
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val_macro_f1,
                    "checkpoint": str(checkpoint_path.resolve()),
                }
            )
    return rows


def _atomic_text(path, write_content):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            write_content(temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_comparison_summaries(output_root, base_config_path, rows):
    """原子写入内容一致的JSON和CSV对比汇总。"""
    comparison_root = Path(output_root).resolve()
    summary = {
        "base_config": str(Path(base_config_path).resolve()),
        "output_root": str(comparison_root),
        "rows": rows,
    }

    def write_json(output_file):
        json.dump(summary, output_file, ensure_ascii=False, indent=2, allow_nan=False)
        output_file.write("\n")

    field_names = [
        "normalization",
        "split",
        "accuracy",
        "macro_f1",
        "best_epoch",
        "best_val_macro_f1",
        "checkpoint",
    ]

    def write_csv(output_file):
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)

    _atomic_text(comparison_root / "comparison.json", write_json)
    _atomic_text(comparison_root / "comparison.csv", write_csv)


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="公平对比none与gravity两种EgoCHARM输入归一化"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/egocharm.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/comparisons/normalization"),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _planned_commands(config_path, variant_name, output_root, resume):
    variant_directory = Path(output_root).resolve() / variant_name
    best_checkpoint = variant_directory / "best.pt"
    return [
        train_command(config_path, variant_name, resume),
        evaluate_command(
            config_path,
            best_checkpoint,
            "val",
            variant_directory / "val",
        ),
        evaluate_command(
            config_path,
            best_checkpoint,
            "test",
            variant_directory / "test",
        ),
    ]


def main(arguments=None, command_runner=run_command):
    parsed_arguments = build_argument_parser().parse_args(arguments)
    base_config_path = Path(parsed_arguments.config).resolve()
    base_config = load_config(base_config_path)
    output_root = project_path(parsed_arguments.output).resolve()

    # 在写入任何配置前检查两个目标，拒绝留下半更新的对比目录。
    for variant_name in VARIANT_NAMES:
        validate_variant_state(
            output_root / variant_name,
            parsed_arguments.resume,
        )

    variant_config_paths = {}
    dry_run_variants = []
    for variant_name in VARIANT_NAMES:
        variant_config = build_variant_config(
            base_config,
            variant_name,
            output_root,
            parsed_arguments.epochs,
        )
        config_path = output_root / variant_name / "config.yaml"
        write_yaml_atomic(config_path, variant_config)
        variant_config_paths[variant_name] = config_path
        dry_run_variants.append(
            {
                "normalization": variant_name,
                "config": str(config_path.resolve()),
                "commands": _planned_commands(
                    config_path,
                    variant_name,
                    output_root,
                    parsed_arguments.resume,
                ),
            }
        )

    if parsed_arguments.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "output_root": str(output_root),
                    "variants": dry_run_variants,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    for variant_name in VARIANT_NAMES:
        execute_variant(
            variant_name,
            variant_config_paths[variant_name],
            output_root,
            parsed_arguments.resume,
            command_runner=command_runner,
        )

    rows = collect_comparison_rows(output_root)
    write_comparison_summaries(output_root, base_config_path, rows)
    print(f"对比完成，汇总保存在 {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
