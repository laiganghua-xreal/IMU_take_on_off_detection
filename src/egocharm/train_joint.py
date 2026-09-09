"""Train the nine-class Ego activity and Xreal wear-event model."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from egocharm.checkpoint import TrainingState, load_checkpoint, save_checkpoint
from egocharm.config import load_config, project_path, set_random_seed
from egocharm.engine import (
    calculate_class_weights,
    choose_device,
    train_one_epoch,
    validate_one_epoch,
)
from egocharm.joint_cache import load_joint_manifest
from egocharm.joint_dataset import (
    JOINT_CLASS_NAMES,
    BalancedJointDataset,
    EvaluationJointDataset,
    ImuAugmentation,
    create_joint_data_loader,
)
from egocharm.model import build_model, count_trainable_parameters
from egocharm.train import _create_gradient_scaler


def build_argument_parser():
    parser = argparse.ArgumentParser(description="训练 Ego-Xreal 九分类模型")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ego_xreal_joint_9class.yaml"),
    )
    parser.add_argument("--run-name", default="joint_9class")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def validate_joint_config(config):
    data_config = config["data"]
    model_config = config["model"]
    if tuple(model_config.get("class_names", ())) != JOINT_CLASS_NAMES:
        raise ValueError(f"联合类别顺序必须是 {JOINT_CLASS_NAMES}")
    if int(model_config.get("number_of_classes", 0)) != len(JOINT_CLASS_NAMES):
        raise ValueError("联合模型必须输出九个类别")
    if int(data_config["sampling_rate_hz"]) != 200:
        raise ValueError("联合实验采样率必须是 200 Hz")
    if int(data_config["window_seconds"]) != 5:
        raise ValueError("联合实验窗口必须是 5 秒")
    if float(data_config["candidate_step_seconds"]) != 0.025:
        raise ValueError("联合实验训练候选步进必须是 0.025 秒")
    if str(data_config.get("normalization", "none")) != "none":
        raise ValueError("联合实验不进行输入归一化")


def _load_recordings(data_config):
    ego_recordings = load_joint_manifest(project_path(data_config["ego_manifest"]))
    xreal_recordings = load_joint_manifest(
        project_path(data_config["xreal_event_manifest"])
    )
    return ego_recordings + xreal_recordings


def _build_augmentation(config):
    values = config.get("augmentation", {})
    return ImuAugmentation(
        add_bias_noise=bool(values.get("add_bias_noise", False)),
        accel_bias_range=float(values.get("accel_bias_range", 0.0)),
        gyro_bias_range=float(values.get("gyro_bias_range", 0.0)),
        add_gravity_noise=bool(values.get("add_gravity_noise", False)),
        gravity_noise_theta_range=float(
            values.get("gravity_noise_theta_range", 0.0)
        ),
        add_gaussian_noise=bool(values.get("add_gaussian_noise", False)),
        accel_noise_sigma=float(values.get("accel_noise_sigma", 0.0)),
        gyro_noise_sigma=float(values.get("gyro_noise_sigma", 0.0)),
    )


def _build_datasets(config):
    data_config = config["data"]
    training_config = config["training"]
    recordings = _load_recordings(data_config)
    sampling_rate_hz = int(data_config["sampling_rate_hz"])
    window_seconds = int(data_config["window_seconds"])
    train_dataset = BalancedJointDataset(
        recordings,
        sampling_rate_hz=sampling_rate_hz,
        window_seconds=window_seconds,
        candidate_step_seconds=float(data_config["candidate_step_seconds"]),
        samples_per_class=int(training_config["samples_per_class_per_epoch"]),
        augmentation=_build_augmentation(config),
    )
    validation_dataset = EvaluationJointDataset(
        recordings,
        split="val",
        sampling_rate_hz=sampling_rate_hz,
        window_seconds=window_seconds,
        evaluation_step_seconds=float(data_config["evaluation_step_seconds"]),
        maximum_samples_per_class=int(
            data_config["maximum_validation_samples_per_class"]
        ),
    )
    missing_validation_classes = sorted(
        set(range(len(JOINT_CLASS_NAMES)))
        - set(validation_dataset.class_labels)
    )
    if missing_validation_classes:
        names = [JOINT_CLASS_NAMES[index] for index in missing_validation_classes]
        raise ValueError(f"验证集缺少类别: {names}")
    return train_dataset, validation_dataset


def _write_history(path, epoch, train_metrics, validation_metrics):
    row = {
        "epoch": epoch,
        "train": asdict(train_metrics),
        "validation": asdict(validation_metrics),
    }
    with Path(path).open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _smoke_test(model, data_loader, optimizer, loss_function, device, scaler):
    model.train()
    features, labels = next(iter(data_loader))
    features = features.to(device)
    labels = labels.to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(
        device_type=device.type,
        enabled=scaler is not None,
    ):
        logits = model(features)
        loss = loss_function(logits, labels)
    if scaler is None:
        loss.backward()
        optimizer.step()
    else:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    return {
        "device": str(device),
        "features_shape": list(features.shape),
        "logits_shape": list(logits.shape),
        "loss": float(loss.detach().cpu()),
    }


def main(arguments=None):
    args = build_argument_parser().parse_args(arguments)
    config = load_config(args.config)
    validate_joint_config(config)
    data_config = config["data"]
    model_config = config["model"]
    training_config = config["training"]
    run_directory = project_path(training_config["output_root"]) / args.run_name

    if args.dry_run:
        print(
            json.dumps(
                {
                    "training_started": False,
                    "run_directory": str(run_directory),
                    "class_names": list(JOINT_CLASS_NAMES),
                    "input_shape": [
                        int(data_config["window_seconds"]),
                        int(model_config["input_channels"]),
                        int(data_config["sampling_rate_hz"]),
                    ],
                },
                ensure_ascii=False,
            )
        )
        return 0

    seed = int(training_config["seed"])
    set_random_seed(seed)
    device = choose_device(str(training_config["device"]))
    train_dataset, validation_dataset = _build_datasets(config)
    batch_size = int(training_config["batch_size"])
    number_of_workers = int(training_config["number_of_workers"])
    train_loader = create_joint_data_loader(
        train_dataset,
        batch_size,
        shuffle=True,
        number_of_workers=number_of_workers,
        seed=seed,
    )
    validation_loader = create_joint_data_loader(
        validation_dataset,
        batch_size,
        shuffle=False,
        number_of_workers=number_of_workers,
        seed=seed,
    )

    model = build_model(model_config).to(device)
    class_weights = calculate_class_weights(
        train_dataset.class_labels,
        len(JOINT_CLASS_NAMES),
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

    if args.smoke_test:
        report = _smoke_test(
            model,
            train_loader,
            optimizer,
            loss_function,
            device,
            gradient_scaler,
        )
        report["train_candidates"] = {
            class_name: train_dataset.number_of_candidates(class_name)
            for class_name in JOINT_CLASS_NAMES
        }
        report["validation_samples"] = len(validation_dataset)
        print(json.dumps(report, ensure_ascii=False))
        return 0

    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "resolved-config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    state = TrainingState()
    last_checkpoint_path = run_directory / "last.pt"
    if args.resume:
        if not last_checkpoint_path.is_file():
            raise FileNotFoundError(f"找不到断点: {last_checkpoint_path}")
        state = load_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            scheduler,
            gradient_scaler,
            expected_class_names=JOINT_CLASS_NAMES,
        )

    number_of_epochs = int(training_config["number_of_epochs"])
    for epoch_index in range(state.epoch, number_of_epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            device,
            gradient_scaler,
            training_config["maximum_gradient_norm"],
            len(JOINT_CLASS_NAMES),
            accumulation_steps=int(
                training_config.get("gradient_accumulation_steps", 1)
            ),
        )
        validation_metrics = validate_one_epoch(
            model,
            validation_loader,
            loss_function,
            device,
            len(JOINT_CLASS_NAMES),
        )
        scheduler.step()

        is_best = validation_metrics.macro_f1 >= state.best_macro_f1
        state = TrainingState(
            epoch=epoch_index + 1,
            best_macro_f1=max(state.best_macro_f1, validation_metrics.macro_f1),
        )
        metadata = {
            "class_names": list(JOINT_CLASS_NAMES),
            "number_of_parameters": count_trainable_parameters(model),
            "coordinate_frame": data_config["coordinate_frame"],
            "sampling_rate_hz": int(data_config["sampling_rate_hz"]),
            "window_seconds": int(data_config["window_seconds"]),
            "candidate_step_seconds": float(
                data_config["candidate_step_seconds"]
            ),
            "normalization": "none",
            "augmentation": config.get("augmentation", {}),
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
        _write_history(
            run_directory / "history.jsonl",
            state.epoch,
            train_metrics,
            validation_metrics,
        )
        print(
            f"epoch={state.epoch:03d} "
            f"train_loss={train_metrics.loss:.4f} "
            f"train_f1={train_metrics.macro_f1:.4f} "
            f"val_loss={validation_metrics.loss:.4f} "
            f"val_f1={validation_metrics.macro_f1:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
