"""从预处理 NPZ 缓存构造 PyTorch Dataset 和 DataLoader。

本文件不读取 VRS，也不依赖 Project Aria Tools。
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


CHANNEL_NAMES = (
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)
CLASS_NAMES = (
    "Basketball",
    "Soccer",
    "Dance",
    "Rock Climbing",
    "Cooking",
    "Bike Repair",
    "Music",
)
ALLOWED_SPLITS = ("train", "val", "test")
GRAVITY_METERS_PER_SECOND_SQUARED = 9.8


def _keep_raw_window(values):
    """保持窗口中的六轴IMU原值。"""
    return values


def _divide_acceleration_by_gravity(values):
    """将三个加速度通道转换为重力加速度倍数，保持陀螺仪原值。"""
    normalized_values = values.copy()
    normalized_values[:, :3] /= GRAVITY_METERS_PER_SECOND_SQUARED
    return normalized_values


WINDOW_NORMALIZERS = {
    "none": _keep_raw_window,
    "gravity": _divide_acceleration_by_gravity,
}


def normalize_window(values, method):
    """使用命名策略处理形状为 ``(S * T, 6)`` 的完整IMU窗口。"""
    try:
        normalizer = WINDOW_NORMALIZERS[method]
    except KeyError as error:
        raise ValueError(f"不支持的 normalization: {method}") from error
    return normalizer(values)


def validate_model_class_config(model_config) -> None:
    """验证模型类别数量、唯一性和 Dataset 的固定类别顺序。"""
    configured_class_names = model_config.get("class_names")
    if (
        not isinstance(configured_class_names, Sequence)
        or isinstance(configured_class_names, (str, bytes))
    ):
        raise ValueError("model.class_names 必须是非空类别名称序列")

    class_names = tuple(configured_class_names)
    if not class_names or any(
        not isinstance(class_name, str) or not class_name.strip()
        for class_name in class_names
    ):
        raise ValueError("model.class_names 不能包含空类别名称")
    if len(set(class_names)) != len(class_names):
        raise ValueError("model.class_names 不能包含重复类别名称")

    number_of_classes = model_config.get("number_of_classes")
    if number_of_classes != len(class_names):
        raise ValueError(
            "model.number_of_classes 必须等于 model.class_names 的长度"
        )
    if class_names != CLASS_NAMES:
        raise ValueError("model.class_names 必须严格遵循 dataset.CLASS_NAMES 顺序")


@dataclass(frozen=True)
class TakeRecord:
    """Dataset 建立窗口索引需要的 take 元数据。"""

    take_uid: str
    activity: str
    split: str
    participant_uid: int


@dataclass(frozen=True)
class WindowReference:
    """一个训练窗口在缓存文件中的位置。"""

    cache_path: Path
    start_sample: int
    class_index: int


def load_selection(path: Path) -> list[TakeRecord]:
    """从项目 selection 文件读取训练所需的四个字段。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"selection 缺少 entries 列表: {path}")

    records: list[TakeRecord] = []
    seen_take_uids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("selection 中的每一项都必须是字典")

        take_uid = str(entry["take_uid"])
        split = str(entry["official_split"])
        activity = str(entry["class"])

        if take_uid in seen_take_uids:
            raise ValueError(f"selection 中存在重复 take UID: {take_uid}")
        if split not in ALLOWED_SPLITS:
            raise ValueError(f"take {take_uid} 的 split 无效: {split}")
        if activity not in CLASS_NAMES:
            raise ValueError(f"take {take_uid} 的活动类别无效: {activity}")

        seen_take_uids.add(take_uid)
        records.append(
            TakeRecord(
                take_uid=take_uid,
                activity=activity,
                split=split,
                participant_uid=int(entry["participant_uid"]),
            )
        )
    return records


def _load_cache_index_information(
    cache_path: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    if not cache_path.is_file():
        raise FileNotFoundError(f"找不到预处理缓存: {cache_path}")

    with np.load(cache_path, allow_pickle=False) as cache:
        segment_bounds = np.asarray(cache["segment_bounds"], dtype=np.int64)
        metadata = json.loads(str(cache["metadata_json"].item()))

    if segment_bounds.ndim != 2 or segment_bounds.shape[1] != 2:
        raise ValueError(f"缓存的 segment_bounds 形状不正确: {cache_path}")
    if tuple(metadata.get("channel_names", ())) != CHANNEL_NAMES:
        raise ValueError(f"缓存的通道顺序不正确: {cache_path}")
    return segment_bounds, metadata


def build_window_index(
    records: Sequence[TakeRecord],
    cache_root: Path,
    split: str,
    sampling_rate_hz: int,
    window_seconds: int,
    stride_seconds: int,
) -> list[WindowReference]:
    """在每个连续片段内部建立固定长度窗口索引。"""
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"不支持的数据 split: {split}")
    if sampling_rate_hz <= 0 or window_seconds <= 0 or stride_seconds <= 0:
        raise ValueError("采样率、窗口长度和步长都必须大于零")

    class_to_index = {
        class_name: class_index
        for class_index, class_name in enumerate(CLASS_NAMES)
    }
    window_samples = sampling_rate_hz * window_seconds
    stride_samples = sampling_rate_hz * stride_seconds
    window_references: list[WindowReference] = []

    for record in records:
        if record.split != split:
            continue

        cache_path = Path(cache_root) / f"{record.take_uid}.npz"
        segment_bounds, metadata = _load_cache_index_information(cache_path)
        if metadata.get("take_uid") != record.take_uid:
            raise ValueError(f"缓存 take UID 不匹配: {cache_path}")
        if metadata.get("split") != record.split:
            raise ValueError(f"缓存 split 不匹配: {cache_path}")

        class_index = class_to_index[record.activity]
        for segment_start, segment_stop in segment_bounds:
            segment_length = int(segment_stop - segment_start)
            if segment_length < window_samples:
                continue

            last_start = segment_length - window_samples
            local_starts = range(0, last_start + 1, stride_samples)
            for local_start in local_starts:
                absolute_start = int(segment_start) + local_start
                window_references.append(
                    WindowReference(
                        cache_path=cache_path,
                        start_sample=absolute_start,
                        class_index=class_index,
                    )
                )
    return window_references


class EgoCharmDataset(Dataset):
    """按窗口索引惰性读取六轴IMU，并应用可选的窗口归一化策略。"""

    def __init__(
        self,
        window_references,
        sampling_rate_hz,
        window_seconds,
        normalization="none",
    ):
        super().__init__()

        self.window_references = tuple(window_references)
        self.sampling_rate_hz = sampling_rate_hz
        self.window_seconds = window_seconds
        self.normalization = normalization

        if sampling_rate_hz <= 0 or window_seconds <= 0:
            raise ValueError("采样率和窗口长度必须大于零")
        if normalization not in WINDOW_NORMALIZERS:
            raise ValueError(f"不支持的 normalization: {normalization}")

    @property
    def class_labels(self):
        """返回全部窗口标签，用于计算训练类别权重。"""
        labels: list[int] = []
        for reference in self.window_references:
            labels.append(reference.class_index)
        return labels

    def __len__(self):
        return len(self.window_references)

    def __getitem__(self, item_index):
        reference = self.window_references[item_index]
        number_of_samples = self.sampling_rate_hz * self.window_seconds
        stop_sample = reference.start_sample + number_of_samples

        with np.load(reference.cache_path, allow_pickle=False) as cache:
            cached_values = np.asarray(cache["values"], dtype=np.float32)
            # NPZ values: (N, C)，当前切片: (S * T, C)。
            values = cached_values[reference.start_sample:stop_sample]

        expected_shape = (number_of_samples, len(CHANNEL_NAMES))
        if values.shape != expected_shape:
            raise IndexError(f"缓存窗口不完整: {reference.cache_path}")
        if not np.isfinite(values).all():
            raise ValueError(f"缓存窗口包含 NaN 或 Inf: {reference.cache_path}")

        # values: (S * T, 6)。gravity策略只缩放前三个加速度通道。
        normalized_values = normalize_window(values, self.normalization)

        # (S * T, C) -> (S, T, C) -> (S, C, T)，默认 (1500, 6) -> (30, 50, 6) -> (30, 6, 50)。
        values_by_second = normalized_values.reshape(
            self.window_seconds,
            self.sampling_rate_hz,
            len(CHANNEL_NAMES),
        )
        hierarchical_values = values_by_second.transpose(0, 2, 1)
        contiguous_values = np.ascontiguousarray(
            hierarchical_values,
            dtype=np.float32,
        )

        features = torch.from_numpy(contiguous_values)
        label = torch.tensor(reference.class_index, dtype=torch.long)
        return features, label


def create_data_loader(
    dataset: EgoCharmDataset,
    batch_size: int,
    shuffle: bool,
    number_of_workers: int,
) -> DataLoader:
    """构造 DataLoader：单样本 features (S, C, T)、target scalar；batch 为 (B, S, C, T) 和 (B,)。"""
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于零")
    if number_of_workers < 0:
        raise ValueError("number_of_workers 不能小于零")

    use_pinned_memory = torch.cuda.is_available()
    keep_workers_alive = number_of_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=number_of_workers,
        pin_memory=use_pinned_memory,
        persistent_workers=keep_workers_alive,
    )
