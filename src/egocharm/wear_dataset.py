"""Xreal 佩戴状态与摘戴事件 Dataset。"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from egocharm.config import project_path
from egocharm.dataset import normalize_window


WEAR_CLASS_NAMES = ("WORN", "NOT_WORN", "PUT_ON", "TAKE_OFF")
THREE_CLASS_NAMES = ("OTHERS", "PUT_ON", "TAKE_OFF")
THREE_STATE_EVENT_NAMES = ("WORN", "NOT_WORN", "EVENT")
EVENT_CLASS_NAMES = ("NO_EVENT", "EVENT")
WEAR_SPLITS = ("train", "val", "test")
FOUR_CLASS_MODE = "four_class"
THREE_CLASS_MODE = "three_class"
THREE_OTHER_DIRECTION_MODE = "three_other_direction"
THREE_STATE_EVENT_MODE = "three_state_event"
BINARY_EVENT_MODE = "binary_event"

_TASK_CLASS_NAMES = {
    FOUR_CLASS_MODE: WEAR_CLASS_NAMES,
    THREE_CLASS_MODE: THREE_CLASS_NAMES,
    THREE_OTHER_DIRECTION_MODE: THREE_CLASS_NAMES,
    THREE_STATE_EVENT_MODE: THREE_STATE_EVENT_NAMES,
    BINARY_EVENT_MODE: EVENT_CLASS_NAMES,
}
_THREE_CLASS_LABELS = {
    "WORN": "OTHERS",
    "NOT_WORN": "OTHERS",
    "PUT_ON": "PUT_ON",
    "TAKE_OFF": "TAKE_OFF",
}
_BINARY_EVENT_LABELS = {
    "WORN": "NO_EVENT",
    "NOT_WORN": "NO_EVENT",
    "PUT_ON": "EVENT",
    "TAKE_OFF": "EVENT",
}
_THREE_STATE_EVENT_LABELS = {
    "WORN": "WORN",
    "NOT_WORN": "NOT_WORN",
    "PUT_ON": "EVENT",
    "TAKE_OFF": "EVENT",
}


def get_task_class_names(label_mode):
    """返回标签模式对应的有序类别名称。"""
    try:
        return _TASK_CLASS_NAMES[label_mode]
    except KeyError as error:
        raise ValueError(f"不支持的 label_mode: {label_mode}") from error


def get_label_mode(config):
    """读取任务标签模式；旧配置默认保持四分类。"""
    task_config = config.get("task", {})
    label_mode = str(task_config.get("label_mode", FOUR_CLASS_MODE))
    get_task_class_names(label_mode)
    return label_mode


def map_wear_label(label, label_mode):
    """把 selection 中的四分类标签映射为当前任务标签。"""
    if label not in WEAR_CLASS_NAMES:
        raise ValueError(f"不支持的佩戴标签: {label}")
    if label_mode == FOUR_CLASS_MODE:
        return label
    if label_mode in (THREE_CLASS_MODE, THREE_OTHER_DIRECTION_MODE):
        return _THREE_CLASS_LABELS[label]
    if label_mode == THREE_STATE_EVENT_MODE:
        return _THREE_STATE_EVENT_LABELS[label]
    if label_mode == BINARY_EVENT_MODE:
        return _BINARY_EVENT_LABELS[label]
    raise ValueError(f"不支持的 label_mode: {label_mode}")


@dataclass(frozen=True)
class WearRecord:
    """一个固定五秒 Xreal cache 的索引信息。"""

    example_uid: str
    cache_path: Path
    label: str
    split: str
    source_path: Path | None = None
    start_sample: int = 0


def load_wear_selection(path, split):
    """读取 selection 中指定 split 的样本。"""
    if split not in WEAR_SPLITS:
        raise ValueError(f"不支持的 split: {split}")
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if tuple(payload.get("class_names", ())) != WEAR_CLASS_NAMES:
        raise ValueError("selection class_names 与 WEAR_CLASS_NAMES 不一致")

    records = []
    for entry in payload.get("entries", ()):
        if entry["split"] != split:
            continue
        label = str(entry["label"])
        if label not in WEAR_CLASS_NAMES:
            raise ValueError(f"不支持的佩戴标签: {label}")
        cache_path = Path(entry["cache_path"])
        if not cache_path.is_absolute():
            cache_path = project_path(cache_path)
        source_path_value = entry.get("source_path")
        source_path = None
        if source_path_value is not None:
            source_path = Path(source_path_value)
            if not source_path.is_absolute():
                source_path = project_path(source_path)
        records.append(
            WearRecord(
                example_uid=str(entry["example_uid"]),
                cache_path=cache_path,
                label=label,
                split=str(entry["split"]),
                source_path=source_path,
                start_sample=int(entry.get("start_sample", 0)),
            )
        )
    return records


class WearDataset(Dataset):
    """读取固定窗口，并返回 `(秒, 通道, 每秒采样点)`。"""

    def __init__(
        self,
        records,
        sampling_rate_hz,
        window_seconds,
        normalization="gravity",
        label_mode=FOUR_CLASS_MODE,
    ):
        super().__init__()
        self.records = tuple(records)
        self.sampling_rate_hz = sampling_rate_hz
        self.window_seconds = window_seconds
        self.normalization = normalization
        self.label_mode = label_mode
        self.class_names = get_task_class_names(label_mode)
        self.class_to_index = {
            class_name: class_index
            for class_index, class_name in enumerate(self.class_names)
        }

    @property
    def class_labels(self):
        labels = []
        for record in self.records:
            target_label = map_wear_label(record.label, self.label_mode)
            labels.append(self.class_to_index[target_label])
        return labels

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item_index):
        record = self.records[item_index]
        with np.load(record.cache_path, allow_pickle=False) as cache:
            values = np.asarray(cache["values"], dtype=np.float32)

        number_of_samples = self.sampling_rate_hz * self.window_seconds
        expected_shape = (number_of_samples, 6)
        if values.shape != expected_shape:
            raise ValueError(
                f"cache 形状必须是 {expected_shape}: {record.cache_path}"
            )

        values = normalize_window(values, self.normalization)
        # (S*T, C) -> (S, T, C) -> (S, C, T)
        values = values.reshape(self.window_seconds, self.sampling_rate_hz, 6)
        values = np.ascontiguousarray(values.transpose(0, 2, 1))
        features = torch.from_numpy(values)
        target_label = map_wear_label(record.label, self.label_mode)
        label = torch.tensor(self.class_to_index[target_label], dtype=torch.long)
        return features, label
