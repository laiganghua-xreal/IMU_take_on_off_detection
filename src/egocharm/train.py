"""使用预处理缓存训练 EgoCHARM。

运行方式：
    python -m egocharm.train --config configs/egocharm.yaml --run-name baseline
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from egocharm.checkpoint import TrainingState, load_checkpoint, save_checkpoint
from egocharm.config import load_config, project_path, set_random_seed
from egocharm.dataset import (
    EgoCharmDataset,
    build_window_index,
    create_data_loader,
    load_selection,
    validate_model_class_config,
)
from egocharm.engine import (
    EpochMetrics,
    calculate_class_weights,
    choose_device,
    train_one_epoch,
    validate_one_epoch,
)
from egocharm.model import build_model, count_trainable_parameters


def build_argument_parser():
    parser = argparse.ArgumentParser(description="训练 EgoCHARM")
    parser.add_argument("--config", type=Path, default=Path("configs/egocharm.yaml"))
    parser.add_argument("--run-name", default="baseline")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _create_gradient_scaler(device, enabled):
    """为当前 PyTorch 版本创建 CUDA AMP gradient scaler。"""
    if not enabled or device.type != "cuda":
        return None

    modern_gradient_scaler = getattr(torch.amp, "GradScaler", None)
    if modern_gradient_scaler is not None:
        return modern_gradient_scaler("cuda")
    return torch.cuda.amp.GradScaler()


def build_checkpoint_metadata(
    model_config,
    data_config,
    model,
    train_metrics,
    validation_metrics,
):
    """记录重现 checkpoint 输入定义所需的模型与数据元信息。"""
    return {
        "class_names": list(model_config["class_names"]),
        "number_of_parameters": count_trainable_parameters(model),
        "coordinate_frame": str(data_config.get("coordinate_frame", "sensor")),
        "sampling_rate_hz": int(data_config["sampling_rate_hz"]),
        "window_seconds": int(data_config["window_seconds"]),
        "normalization": str(data_config.get("normalization", "none")),
        "train_metrics": asdict(train_metrics),
        "validation_metrics": asdict(validation_metrics),
    }


def main(arguments=None):
    # 解析参数
    parsed_arguments = build_argument_parser().parse_args(arguments)

    # 加载配置
    config = load_config(parsed_arguments.config)
    model_config = config["model"]
    validate_model_class_config(model_config)
    data_config = config["data"]
    training_config = config["training"]
    normalization = str(data_config.get("normalization", "none"))
    output_root = project_path(training_config["output_root"])
    run_directory = output_root / parsed_arguments.run_name

    # 处理 dry-run
    if parsed_arguments.dry_run:
        result = {"run_directory": str(run_directory), "training_started": False}
        print(json.dumps(result, ensure_ascii=False))
        return 0

    # 设置随机种子和设备
    seed = int(training_config["seed"])
    set_random_seed(seed)
    device = choose_device(str(training_config["device"]))

    # 构建数据集和加载器
    records = load_selection(project_path(data_config["selection"]))
    cache_root = project_path(data_config["cache_root"])
    sampling_rate_hz = int(data_config["sampling_rate_hz"])
    window_seconds = int(data_config["window_seconds"])
    stride_seconds = int(data_config["stride_seconds"])
    train_index = build_window_index(
        records,
        cache_root,
        "train",
        sampling_rate_hz,
        window_seconds,
        stride_seconds,
    )
    validation_index = build_window_index(
        records,
        cache_root,
        "val",
        sampling_rate_hz,
        window_seconds,
        stride_seconds,
    )
    train_dataset = EgoCharmDataset(
        train_index,
        sampling_rate_hz,
        window_seconds,
        normalization=normalization,
    )
    validation_dataset = EgoCharmDataset(
        validation_index,
        sampling_rate_hz,
        window_seconds,
        normalization=normalization,
    )
    batch_size = int(training_config["batch_size"])
    num_workers = int(training_config["number_of_workers"])
    train_loader = create_data_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        number_of_workers=num_workers,
    )
    validation_loader = create_data_loader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        number_of_workers=num_workers,
    )

    # 构建模型、损失和优化组件
    model = build_model(model_config)
    model.to(device)
    num_classes = int(model_config["number_of_classes"])
    class_weights = calculate_class_weights(
        train_dataset.class_labels,
        num_classes,
    ).to(device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(training_config["scheduler_step_size"]),
        gamma=float(training_config["scheduler_gamma"]),
    )

    gradient_scaler = _create_gradient_scaler(
        device,
        bool(training_config["use_mixed_precision"]),
    )

    run_directory.mkdir(parents=True, exist_ok=True)
    resolved_config_path = run_directory / "resolved-config.json"
    resolved_config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # 恢复断点
    state = TrainingState()
    last_checkpoint_path = run_directory / "last.pt"
    if parsed_arguments.resume and last_checkpoint_path.is_file():
        state = load_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            scheduler,
            gradient_scaler,
            expected_class_names=model_config["class_names"],
        )

    # 训练各轮
    num_epochs = int(training_config["number_of_epochs"])
    max_grad_norm = training_config["maximum_gradient_norm"]
    for epoch_index in range(state.epoch, num_epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            device,
            gradient_scaler,
            max_grad_norm,
            num_classes,
        )
        validation_metrics = validate_one_epoch(
            model,
            validation_loader,
            loss_function,
            device,
            num_classes,
        )
        scheduler.step()

        is_best_checkpoint = validation_metrics.macro_f1 >= state.best_macro_f1
        best_macro_f1 = max(state.best_macro_f1, validation_metrics.macro_f1)
        state = TrainingState(
            epoch=epoch_index + 1,
            best_macro_f1=best_macro_f1,
        )
        metadata = build_checkpoint_metadata(
            model_config,
            data_config,
            model,
            train_metrics,
            validation_metrics,
        )

        # 保存最新和最佳断点
        save_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            scheduler,
            gradient_scaler,
            state,
            metadata,
        )
        if is_best_checkpoint:
            save_checkpoint(
                run_directory / "best.pt",
                model,
                optimizer,
                scheduler,
                gradient_scaler,
                state,
                metadata,
            )

        print(
            f"epoch={state.epoch:03d} "
            f"train_loss={train_metrics.loss:.4f} "
            f"val_loss={validation_metrics.loss:.4f} "
            f"val_macro_f1={validation_metrics.macro_f1:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
