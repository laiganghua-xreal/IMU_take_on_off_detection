"""单独加载 checkpoint，在 official val 或 test 上评估 EgoCHARM。"""

import argparse
from pathlib import Path

import torch

from egocharm.checkpoint import load_checkpoint
from egocharm.config import load_config, project_path
from egocharm.dataset import (
    EgoCharmDataset,
    build_window_index,
    create_data_loader,
    load_selection,
    validate_model_class_config,
)
from egocharm.engine import choose_device
from egocharm.metrics import calculate_metrics, save_metrics
from egocharm.model import build_model, count_trainable_parameters


def build_argument_parser():
    parser = argparse.ArgumentParser(description="评估 EgoCHARM checkpoint")
    parser.add_argument("--config", type=Path, default=Path("configs/egocharm.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--output", type=Path)
    return parser


def main(arguments=None):
    parsed_arguments = build_argument_parser().parse_args(arguments)
    config = load_config(parsed_arguments.config)
    model_config = config["model"]
    validate_model_class_config(model_config)
    data_config = config["data"]
    training_config = config["training"]
    normalization = str(data_config.get("normalization", "none"))

    device = choose_device(str(training_config["device"]))
    model = build_model(model_config)
    model.to(device)
    load_checkpoint(
        parsed_arguments.checkpoint,
        model,
        expected_class_names=model_config["class_names"],
    )

    records = load_selection(project_path(data_config["selection"]))
    window_index = build_window_index(
        records,
        project_path(data_config["cache_root"]),
        parsed_arguments.split,
        int(data_config["sampling_rate_hz"]),
        int(data_config["window_seconds"]),
        int(data_config["stride_seconds"]),
    )
    dataset = EgoCharmDataset(
        window_index,
        int(data_config["sampling_rate_hz"]),
        int(data_config["window_seconds"]),
        normalization=normalization,
    )
    data_loader = create_data_loader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=False,
        number_of_workers=int(training_config["number_of_workers"]),
    )

    all_labels: list[int] = []
    all_predictions: list[int] = []
    model.eval()
    with torch.no_grad():
        for features, labels in data_loader:
            # features: (B, S, C, T)
            features = features.to(device, non_blocking=True)
            logits = model(features)  # (B, K)
            predictions = logits.argmax(dim=1).cpu()  # (B,)
            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.tolist())

    class_names = list(model_config["class_names"])
    metrics = calculate_metrics(all_labels, all_predictions, class_names)
    metrics["number_of_parameters"] = count_trainable_parameters(model)
    metrics["split"] = parsed_arguments.split

    if parsed_arguments.output is None:
        output_directory = parsed_arguments.checkpoint.parent / parsed_arguments.split
    else:
        output_directory = parsed_arguments.output
    save_metrics(metrics, output_directory)
    print(f"已评估 {len(all_labels)} 个窗口，结果保存在 {output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
