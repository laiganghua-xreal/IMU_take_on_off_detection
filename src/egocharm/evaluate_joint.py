"""Evaluate a joint nine-class checkpoint on deterministic windows."""

import argparse
from pathlib import Path

import torch

from egocharm.checkpoint import load_checkpoint
from egocharm.config import load_config, project_path
from egocharm.engine import choose_device
from egocharm.joint_cache import load_joint_manifest
from egocharm.joint_dataset import (
    JOINT_CLASS_NAMES,
    EvaluationJointDataset,
    create_joint_data_loader,
)
from egocharm.metrics import calculate_metrics, save_metrics
from egocharm.model import build_model, count_trainable_parameters
from egocharm.train_joint import validate_joint_config


def build_argument_parser():
    parser = argparse.ArgumentParser(description="评估 Ego-Xreal 九分类模型")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ego_xreal_joint_9class.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--output", type=Path)
    return parser


def main(arguments=None):
    args = build_argument_parser().parse_args(arguments)
    config = load_config(args.config)
    validate_joint_config(config)
    data_config = config["data"]
    training_config = config["training"]
    recordings = load_joint_manifest(
        project_path(data_config["ego_manifest"])
    ) + load_joint_manifest(project_path(data_config["xreal_event_manifest"]))

    maximum_samples = None
    if args.split == "val":
        maximum_samples = int(
            data_config["maximum_validation_samples_per_class"]
        )
    dataset = EvaluationJointDataset(
        recordings,
        split=args.split,
        sampling_rate_hz=int(data_config["sampling_rate_hz"]),
        window_seconds=int(data_config["window_seconds"]),
        evaluation_step_seconds=float(data_config["evaluation_step_seconds"]),
        maximum_samples_per_class=maximum_samples,
    )
    data_loader = create_joint_data_loader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        number_of_workers=int(training_config["number_of_workers"]),
        seed=int(training_config["seed"]),
    )

    device = choose_device(str(training_config["device"]))
    model = build_model(config["model"]).to(device)
    load_checkpoint(
        args.checkpoint,
        model,
        expected_class_names=JOINT_CLASS_NAMES,
    )

    labels = []
    predictions = []
    model.eval()
    with torch.no_grad():
        for features, batch_labels in data_loader:
            logits = model(features.to(device, non_blocking=True))
            labels.extend(batch_labels.tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())

    metrics = calculate_metrics(labels, predictions, JOINT_CLASS_NAMES)
    metrics["split"] = args.split
    metrics["number_of_parameters"] = count_trainable_parameters(model)
    output_directory = args.output
    if output_directory is None:
        output_directory = args.checkpoint.parent / args.split
    save_metrics(metrics, output_directory)
    print(f"已评估 {len(labels)} 个窗口，结果保存在 {output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
