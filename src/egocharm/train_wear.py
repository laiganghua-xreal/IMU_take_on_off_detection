"""冻结 CPF 预训练 LLE，训练 Xreal 佩戴任务 HLE。"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from egocharm.checkpoint import TrainingState, load_checkpoint, save_checkpoint
from egocharm.config import load_config, project_path, set_random_seed
from egocharm.dataset import create_data_loader
from egocharm.engine import (
    calculate_class_weights,
    choose_device,
    train_one_epoch,
    validate_one_epoch,
)
from egocharm.model import count_trainable_parameters
from egocharm.train import _create_gradient_scaler
from egocharm.wear_dataset import (
    WearDataset,
    get_label_mode,
    get_task_class_names,
    load_wear_selection,
)
from egocharm.wear_model import (
    build_wear_model,
    validate_wear_model_config,
)


def build_argument_parser():
    parser = argparse.ArgumentParser(description="训练 Xreal 佩戴状态分类器")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/xreal_wear_frozen_lle.yaml"),
    )
    parser.add_argument("--run-name", default="frozen_lle")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--high-level-hidden-size", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def create_optimizer(model, training_config):
    """只把 requires_grad=True 的新 HLE 参数交给 Adam。"""
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    return torch.optim.Adam(
        trainable_parameters,
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )


def _build_dataset(config, split, label_mode):
    data_config = config["data"]
    records = load_wear_selection(
        project_path(data_config["selection"]),
        split,
    )
    return WearDataset(
        records,
        sampling_rate_hz=int(data_config["sampling_rate_hz"]),
        window_seconds=int(data_config["window_seconds"]),
        normalization=str(data_config.get("normalization", "none")),
        label_mode=label_mode,
    )


def main(arguments=None):
    args = build_argument_parser().parse_args(arguments)
    config = load_config(args.config)
    model_config = config["model"]
    training_config = config["training"]
    if args.seed is not None:
        training_config["seed"] = args.seed
    if args.high_level_hidden_size is not None:
        model_config["high_level_hidden_size"] = args.high_level_hidden_size
    label_mode = get_label_mode(config)
    class_names = get_task_class_names(label_mode)
    validate_wear_model_config(model_config, label_mode)
    initialization = str(
        model_config.get("initialization", "pretrained_frozen_lle")
    )
    run_directory = project_path(training_config["output_root"]) / args.run_name

    if args.dry_run:
        print(
            json.dumps(
                {
                    "run_directory": str(run_directory),
                    "training_started": False,
                    "label_mode": label_mode,
                    "class_names": list(class_names),
                    "initialization": initialization,
                    "seed": training_config.get("seed"),
                    "high_level_hidden_size": model_config.get(
                        "high_level_hidden_size"
                    ),
                },
                ensure_ascii=False,
            )
        )
        return 0

    set_random_seed(int(training_config["seed"]))
    device = choose_device(str(training_config["device"]))
    source_checkpoint = None
    if initialization == "pretrained_frozen_lle":
        source_checkpoint = project_path(
            model_config["pretrained_lle_checkpoint"]
        )
    model = build_wear_model(
        model_config,
        label_mode=label_mode,
        checkpoint_path=source_checkpoint,
    )
    model.to(device)

    train_dataset = _build_dataset(config, "train", label_mode)
    validation_dataset = _build_dataset(config, "val", label_mode)
    batch_size = int(training_config["batch_size"])
    number_of_workers = int(training_config["number_of_workers"])
    train_loader = create_data_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        number_of_workers=number_of_workers,
    )
    validation_loader = create_data_loader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        number_of_workers=number_of_workers,
    )

    class_weights = calculate_class_weights(
        train_dataset.class_labels,
        len(class_names),
    ).to(device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = create_optimizer(model, training_config)
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
    (run_directory / "resolved-config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    state = TrainingState()
    last_checkpoint_path = run_directory / "last.pt"
    if args.resume:
        if not last_checkpoint_path.is_file():
            raise FileNotFoundError(f"恢复训练需要 {last_checkpoint_path}")
        state = load_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            scheduler,
            gradient_scaler,
            expected_class_names=class_names,
        )

    number_of_epochs = int(training_config["number_of_epochs"])
    maximum_gradient_norm = training_config["maximum_gradient_norm"]
    for epoch_index in range(state.epoch, number_of_epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            device,
            gradient_scaler,
            maximum_gradient_norm,
            len(class_names),
        )
        validation_metrics = validate_one_epoch(
            model,
            validation_loader,
            loss_function,
            device,
            len(class_names),
        )
        scheduler.step()

        is_best = validation_metrics.macro_f1 >= state.best_macro_f1
        state = TrainingState(
            epoch=epoch_index + 1,
            best_macro_f1=max(
                state.best_macro_f1,
                validation_metrics.macro_f1,
            ),
        )
        metadata = {
            "class_names": list(class_names),
            "label_mode": label_mode,
            "number_of_trainable_parameters": count_trainable_parameters(model),
            "initialization": initialization,
            "pretrained_lle_checkpoint": (
                str(source_checkpoint.resolve())
                if source_checkpoint is not None
                else None
            ),
            "seed": int(training_config["seed"]),
            "coordinate_frame": "cpf",
            "normalization": config["data"].get("normalization", "none"),
            "sampling_rate_hz": int(config["data"]["sampling_rate_hz"]),
            "window_seconds": int(config["data"]["window_seconds"]),
            "train_metrics": asdict(train_metrics),
            "validation_metrics": asdict(validation_metrics),
        }
        save_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            scheduler,
            gradient_scaler,
            state,
            metadata,
        )
        if is_best:
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
