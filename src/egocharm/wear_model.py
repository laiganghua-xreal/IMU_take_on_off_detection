"""加载 CPF 预训练 LLE，并构造严格冻结的 Xreal 分类模型。"""

from pathlib import Path

import torch

from egocharm.dataset import CLASS_NAMES as EGO_CLASS_NAMES
from egocharm.model import EgoCHARM, HighLevelClassifier, LowLevelEncoder
from egocharm.wear_dataset import (
    BINARY_EVENT_MODE,
    FOUR_CLASS_MODE,
    THREE_CLASS_MODE,
    THREE_OTHER_DIRECTION_MODE,
    THREE_STATE_EVENT_MODE,
    get_task_class_names,
)


class FrozenLowLevelEgoCHARM(EgoCHARM):
    """训练 HLE 时始终关闭 LLE 的 Dropout 和 BatchNorm 更新。"""

    def train(self, mode=True):
        super().train(mode)
        self.low_level_encoder.eval()
        return self


def validate_wear_model_config(model_config, label_mode=FOUR_CLASS_MODE):
    """确保 checkpoint、Dataset 和分类头使用相同类别顺序。"""
    expected_class_names = get_task_class_names(label_mode)
    if tuple(model_config.get("class_names", ())) != expected_class_names:
        constant_names = {
            FOUR_CLASS_MODE: "WEAR_CLASS_NAMES",
            THREE_CLASS_MODE: "THREE_CLASS_NAMES",
            THREE_OTHER_DIRECTION_MODE: "THREE_CLASS_NAMES",
            THREE_STATE_EVENT_MODE: "THREE_STATE_EVENT_NAMES",
            BINARY_EVENT_MODE: "EVENT_CLASS_NAMES",
        }
        constant_name = constant_names[label_mode]
        raise ValueError(
            f"model.class_names 必须严格遵循 {constant_name}: "
            f"{expected_class_names}"
        )
    if model_config.get("number_of_classes") != len(expected_class_names):
        raise ValueError(
            "model.number_of_classes 必须等于当前标签模式的类别数量"
        )


def _build_low_level_encoder(model_config):
    return LowLevelEncoder(
        input_channels=int(model_config["input_channels"]),
        branch_channels=int(model_config["branch_channels"]),
        kernel_size=int(model_config["kernel_size"]),
        dilations=tuple(int(value) for value in model_config["dilations"]),
        number_of_blocks=int(model_config["number_of_blocks"]),
        hidden_size=int(model_config["low_level_hidden_size"]),
        dropout=float(model_config["dropout"]),
    )


def load_pretrained_low_level_encoder(encoder, checkpoint_path):
    """只从 Ego 活动 checkpoint 加载 `low_level_encoder.*`。"""
    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    metadata = checkpoint.get("metadata", {})
    if metadata.get("class_names") != list(EGO_CLASS_NAMES):
        raise ValueError("预训练 checkpoint 不是 Ego-Exo4D 七分类模型")
    if metadata.get("coordinate_frame") != "cpf":
        raise ValueError("预训练 checkpoint 必须使用 CPF 坐标方向")

    prefix = "low_level_encoder."
    encoder_state = {
        key.removeprefix(prefix): value
        for key, value in checkpoint["model"].items()
        if key.startswith(prefix)
    }
    encoder.load_state_dict(encoder_state, strict=True)


def build_frozen_wear_model(
    model_config,
    checkpoint_path,
    label_mode=FOUR_CLASS_MODE,
):
    """创建任务分类 HLE，加载并严格冻结 CPF 预训练 LLE。"""
    low_level_encoder = _build_low_level_encoder(model_config)
    load_pretrained_low_level_encoder(low_level_encoder, checkpoint_path)
    for parameter in low_level_encoder.parameters():
        parameter.requires_grad = False

    high_level_classifier = HighLevelClassifier(
        embedding_size=int(model_config["low_level_hidden_size"]),
        hidden_size=int(model_config["high_level_hidden_size"]),
        num_classes=len(get_task_class_names(label_mode)),
    )
    model = FrozenLowLevelEgoCHARM(
        low_level_encoder,
        high_level_classifier,
    )
    model.low_level_encoder.eval()
    return model


def build_random_wear_model(model_config, label_mode=FOUR_CLASS_MODE):
    """随机初始化完整 Xreal 模型，并让 LLE/HLE 全部参与训练。"""
    low_level_encoder = _build_low_level_encoder(model_config)
    high_level_classifier = HighLevelClassifier(
        embedding_size=int(model_config["low_level_hidden_size"]),
        hidden_size=int(model_config["high_level_hidden_size"]),
        num_classes=len(get_task_class_names(label_mode)),
    )
    return EgoCHARM(low_level_encoder, high_level_classifier)


def build_finetuned_wear_model(
    model_config,
    checkpoint_path,
    label_mode=FOUR_CLASS_MODE,
):
    """Load Ego LLE weights, replace the HLE, and train both levels."""
    low_level_encoder = _build_low_level_encoder(model_config)
    load_pretrained_low_level_encoder(low_level_encoder, checkpoint_path)
    high_level_classifier = HighLevelClassifier(
        embedding_size=int(model_config["low_level_hidden_size"]),
        hidden_size=int(model_config["high_level_hidden_size"]),
        num_classes=len(get_task_class_names(label_mode)),
    )
    return EgoCHARM(low_level_encoder, high_level_classifier)


def build_wear_model(
    model_config,
    label_mode=FOUR_CLASS_MODE,
    checkpoint_path=None,
):
    """按配置创建随机端到端或预训练冻结 LLE 模型。"""
    initialization = model_config.get(
        "initialization",
        "pretrained_frozen_lle",
    )
    if initialization == "random_end_to_end":
        return build_random_wear_model(model_config, label_mode)
    if initialization == "pretrained_frozen_lle":
        source_checkpoint = checkpoint_path
        if source_checkpoint is None:
            source_checkpoint = model_config.get("pretrained_lle_checkpoint")
        if source_checkpoint is None:
            raise ValueError(
                "pretrained_frozen_lle 需要 pretrained_lle_checkpoint"
            )
        return build_frozen_wear_model(
            model_config,
            source_checkpoint,
            label_mode,
        )
    if initialization == "pretrained_finetune_lle":
        source_checkpoint = checkpoint_path
        if source_checkpoint is None:
            source_checkpoint = model_config.get("pretrained_lle_checkpoint")
        if source_checkpoint is None:
            raise ValueError(
                "pretrained_finetune_lle 需要 pretrained_lle_checkpoint"
            )
        return build_finetuned_wear_model(
            model_config,
            source_checkpoint,
            label_mode,
        )
    raise ValueError(f"不支持的 model.initialization: {initialization}")
