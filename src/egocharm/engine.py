"""训练和验证 epoch 的纯执行逻辑。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EpochMetrics:
    """一个训练或验证 epoch 的汇总指标。"""

    loss: float
    accuracy: float
    macro_f1: float
    number_of_samples: int


def choose_device(device_name: str) -> torch.device:
    """根据配置明确选择 CPU 或 CUDA。"""
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def calculate_class_weights(
    labels: Sequence[int], number_of_classes: int
) -> torch.Tensor:
    """按 ``样本总数 / (类别数 × 当前类别数)`` 计算类别权重。"""
    if number_of_classes <= 0:
        raise ValueError("number_of_classes 必须大于零")
    if len(labels) == 0:
        raise ValueError("至少需要一个训练标签")

    label_tensor = torch.as_tensor(labels, dtype=torch.long)
    if label_tensor.min().item() < 0:
        raise ValueError("类别标签不能小于零")
    if label_tensor.max().item() >= number_of_classes:
        raise ValueError("类别标签超出配置的类别数量")

    class_counts = torch.bincount(
        label_tensor,
        minlength=number_of_classes,
    )
    for class_index, class_count in enumerate(class_counts):
        if class_count.item() == 0:
            raise ValueError(f"训练窗口中缺少类别 {class_index}")

    total_samples = label_tensor.numel()
    float_counts = class_counts.to(torch.float32)
    weights = total_samples / (number_of_classes * float_counts)
    return weights


def _summarize_epoch(
    total_loss: float,
    labels: list[int],
    predictions: list[int],
    number_of_classes: int,
) -> EpochMetrics:
    number_of_samples = len(labels)
    if number_of_samples == 0:
        raise ValueError("DataLoader 没有产生任何样本")

    number_of_correct_predictions = 0
    for label, prediction in zip(labels, predictions):
        if label == prediction:
            number_of_correct_predictions += 1

    accuracy = number_of_correct_predictions / number_of_samples
    class_indices = list(range(number_of_classes))
    macro_f1 = f1_score(
        labels,
        predictions,
        labels=class_indices,
        average="macro",
        zero_division=0,
    )
    return EpochMetrics(
        loss=total_loss / number_of_samples,
        accuracy=float(accuracy),
        macro_f1=float(macro_f1),
        number_of_samples=number_of_samples,
    )


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    gradient_scaler: torch.amp.GradScaler | None,
    maximum_gradient_norm: float | None,
    number_of_classes: int,
    accumulation_steps: int = 1,
) -> EpochMetrics:
    """完成一个训练 epoch，并返回样本加权指标。"""
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps 必须大于零")
    model.train()
    total_loss = 0.0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    optimizer.zero_grad(set_to_none=True)
    microbatches_in_group = 0
    number_of_batches = len(data_loader)

    for batch_index, (features, labels) in enumerate(data_loader):
        # features: (B, S, C, T), labels: (B,)
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        use_mixed_precision = gradient_scaler is not None
        with torch.amp.autocast(
            device_type=device.type,
            enabled=use_mixed_precision,
        ):
            # logits: (B, K), loss: scalar
            logits = model(features)
            loss = loss_function(logits, labels)
            backward_loss = loss / accumulation_steps

        if gradient_scaler is None:
            backward_loss.backward()
        else:
            # CUDA AMP 先放大 loss，降低 float16 梯度下溢风险。
            gradient_scaler.scale(backward_loss).backward()

        microbatches_in_group += 1
        group_is_complete = microbatches_in_group == accumulation_steps
        epoch_is_complete = batch_index + 1 == number_of_batches
        if group_is_complete or epoch_is_complete:
            # A final short group was divided by the configured accumulation
            # count.  Restore its mean gradient before stepping.
            if microbatches_in_group < accumulation_steps:
                correction = accumulation_steps / microbatches_in_group
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)

            if gradient_scaler is None:
                if maximum_gradient_norm is not None:
                    nn.utils.clip_grad_norm_(
                        model.parameters(),
                        maximum_gradient_norm,
                    )
                optimizer.step()
            else:
                if maximum_gradient_norm is not None:
                    gradient_scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        model.parameters(),
                        maximum_gradient_norm,
                    )
                gradient_scaler.step(optimizer)
                gradient_scaler.update()
            optimizer.zero_grad(set_to_none=True)
            microbatches_in_group = 0

        batch_size = labels.shape[0]
        total_loss += loss.detach().item() * batch_size
        all_labels.extend(labels.detach().cpu().tolist())
        batch_predictions = logits.detach().argmax(dim=1)
        all_predictions.extend(batch_predictions.cpu().tolist())

    return _summarize_epoch(
        total_loss,
        all_labels,
        all_predictions,
        number_of_classes,
    )


def validate_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
    number_of_classes: int,
) -> EpochMetrics:
    """关闭梯度和训练态层，完成一个验证 epoch。"""
    model.eval()
    total_loss = 0.0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    # 验证不需要反向传播；no_grad 可以降低显存和计算开销。
    with torch.no_grad():
        for features, labels in data_loader:
            # features: (B, S, C, T), labels: (B,)
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # logits: (B, K), loss: scalar
            logits = model(features)
            loss = loss_function(logits, labels)

            batch_size = labels.shape[0]
            total_loss += loss.item() * batch_size
            all_labels.extend(labels.cpu().tolist())
            batch_predictions = logits.argmax(dim=1)
            all_predictions.extend(batch_predictions.cpu().tolist())

    return _summarize_epoch(
        total_loss,
        all_labels,
        all_predictions,
        number_of_classes,
    )
