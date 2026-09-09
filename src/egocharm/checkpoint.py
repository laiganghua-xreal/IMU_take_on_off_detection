"""训练 checkpoint 的原子保存和完整恢复。"""

from __future__ import annotations

import os
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class TrainingState:
    """恢复训练时需要的 epoch 和最佳验证指标。"""

    epoch: int = 0
    best_macro_f1: float = float("-inf")
    selection_score: float = float("-inf")
    stale_epochs: int = 0


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    gradient_scaler: torch.amp.GradScaler | None,
    state: TrainingState,
    metadata: dict[str, Any],
) -> None:
    """原子保存模型、优化器和所有可恢复训练状态。"""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")

    random_states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": None,
    }
    if torch.cuda.is_available():
        random_states["torch_cuda"] = torch.cuda.get_rng_state_all()

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "gradient_scaler": None,
        "training_state": asdict(state),
        "metadata": metadata,
        "random_states": random_states,
    }
    if gradient_scaler is not None:
        checkpoint["gradient_scaler"] = gradient_scaler.state_dict()

    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    gradient_scaler: torch.amp.GradScaler | None = None,
    expected_class_names: Sequence[str] | None = None,
) -> TrainingState:
    """恢复 checkpoint，并返回继续训练的进度。"""
    checkpoint = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=False,
    )

    if expected_class_names is not None:
        metadata = checkpoint.get("metadata")
        saved_class_names = (
            metadata.get("class_names") if isinstance(metadata, dict) else None
        )
        class_names_match = (
            isinstance(saved_class_names, Sequence)
            and not isinstance(saved_class_names, (str, bytes))
            and list(saved_class_names) == list(expected_class_names)
        )
        if not class_names_match:
            raise ValueError(
                "checkpoint metadata class_names 与当前配置不一致: "
                f"expected={list(expected_class_names)!r}, "
                f"checkpoint={saved_class_names!r}"
            )

    model.load_state_dict(checkpoint["model"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    saved_scaler = checkpoint.get("gradient_scaler")
    if gradient_scaler is not None and saved_scaler is not None:
        gradient_scaler.load_state_dict(saved_scaler)

    random_states = checkpoint.get("random_states", {})
    if "python" in random_states:
        random.setstate(random_states["python"])
    if "numpy" in random_states:
        np.random.set_state(random_states["numpy"])
    if "torch_cpu" in random_states:
        torch.set_rng_state(random_states["torch_cpu"])
    saved_cuda_states = random_states.get("torch_cuda")
    if torch.cuda.is_available() and saved_cuda_states is not None:
        torch.cuda.set_rng_state_all(saved_cuda_states)

    return TrainingState(**checkpoint["training_state"])
