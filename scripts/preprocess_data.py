"""把 Ego-Exo4D 左侧 IMU 从 VRS 转换成训练使用的 NPZ 缓存。

这个文件是项目中唯一读取 VRS 的地方。训练代码只读取本脚本生成的缓存。
"""

import argparse
import json
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from egocharm.config import load_config, project_path


CHANNEL_NAMES = (
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)
ALLOWED_SPLITS = ("train", "val", "test")


def rotate_imu_vectors(values: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """用同一个三维旋转矩阵转换加速度和角速度方向。"""
    samples = np.asarray(values)
    rotation_matrix = np.asarray(rotation, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 6:
        raise ValueError("IMU values 必须是 (N, 6)")
    if rotation_matrix.shape != (3, 3):
        raise ValueError("rotation 必须是 (3, 3)")

    transformed = np.empty_like(samples)
    transformed[:, :3] = samples[:, :3] @ rotation_matrix.T
    transformed[:, 3:] = samples[:, 3:] @ rotation_matrix.T
    return transformed


@dataclass(frozen=True)
class SelectedTake:
    """选择清单中的一个 Ego-Exo4D take。"""

    take_uid: str
    take_name: str
    participant_uid: int
    activity: str
    split: str
    duration_seconds: float
    size_bytes: int
    relative_path: str


@dataclass(frozen=True)
class RawImu:
    """从 VRS 读取、但还没有清洗和重采样的左侧 IMU。"""

    timestamps_seconds: np.ndarray
    values: np.ndarray
    coordinate_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreprocessSettings:
    """预处理需要的路径和信号参数。"""

    dataset_root: Path
    cache_root: Path
    sampling_rate_hz: int
    maximum_gap_seconds: float
    minimum_segment_seconds: float
    coordinate_frame: str = "sensor"


def load_selected_takes(path: Path) -> list[SelectedTake]:
    """读取 available selection，并转换成有明确字段的数据对象。"""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise ValueError(f"selection 缺少 entries 列表: {path}")

    selected_takes: list[SelectedTake] = []
    seen_take_uids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("selection 中的每一项都必须是字典")

        take_uid = str(entry["take_uid"])
        split = str(entry["official_split"])
        relative_path = str(entry["relative_path"])
        path_parts = PurePosixPath(relative_path)

        if split not in ALLOWED_SPLITS:
            raise ValueError(f"take {take_uid} 的 split 无效: {split}")
        if path_parts.is_absolute() or ".." in path_parts.parts:
            raise ValueError(f"take {take_uid} 的相对路径不安全")
        if not relative_path.endswith("_noimagestreams.vrs"):
            raise ValueError(f"take {take_uid} 不是无图像 VRS")
        if take_uid in seen_take_uids:
            raise ValueError(f"selection 中存在重复 take UID: {take_uid}")

        seen_take_uids.add(take_uid)
        selected_takes.append(
            SelectedTake(
                take_uid=take_uid,
                take_name=str(entry["take_name"]),
                participant_uid=int(entry["participant_uid"]),
                activity=str(entry["class"]),
                split=split,
                duration_seconds=float(entry["duration_sec"]),
                size_bytes=int(entry["size_bytes"]),
                relative_path=relative_path,
            )
        )
    return selected_takes


def read_left_imu(path: Path, coordinate_frame: str = "sensor") -> RawImu:
    """使用 Project Aria Tools 只读取 VRS 中的 ``imu-left``。"""
    if not path.name.endswith("_noimagestreams.vrs"):
        raise ValueError(f"只接受无图像 VRS 文件: {path}")
    if coordinate_frame not in ("sensor", "cpf"):
        raise ValueError(f"不支持的 coordinate_frame: {coordinate_frame}")

    # 延迟导入让 Dataset、训练和单元测试不依赖 VRS 读取环境。
    try:
        from projectaria_tools.core import data_provider
    except ImportError as error:
        raise RuntimeError(
            "读取 VRS 需要 projectaria-tools>=2.1,<2.2"
        ) from error

    provider = data_provider.create_vrs_data_provider(str(path))
    if provider is None:
        raise RuntimeError(f"Project Aria 无法打开 VRS: {path}")

    stream_id = provider.get_stream_id_from_label("imu-left")
    if stream_id is None:
        raise ValueError(f"VRS 中不存在 imu-left: {path}")

    timestamps_seconds: list[float] = []
    imu_values: list[list[float]] = []
    number_of_samples = provider.get_num_data(stream_id)
    for sample_index in range(number_of_samples):
        sample = provider.get_imu_data_by_index(stream_id, sample_index)
        if not sample.accel_valid or not sample.gyro_valid:
            continue

        # VRS 时间戳单位为纳秒；NumPy 处理时统一转换为秒。
        timestamp_seconds = int(sample.capture_timestamp_ns) * 1e-9
        timestamps_seconds.append(timestamp_seconds)

        # 通道顺序固定为加速度 XYZ，随后是角速度 XYZ。
        row = [
            float(sample.accel_msec2[0]),
            float(sample.accel_msec2[1]),
            float(sample.accel_msec2[2]),
            float(sample.gyro_radsec[0]),
            float(sample.gyro_radsec[1]),
            float(sample.gyro_radsec[2]),
        ]
        imu_values.append(row)

    if not timestamps_seconds:
        raise ValueError(f"imu-left 中没有有效采样: {path}")

    values = np.asarray(imu_values, dtype=np.float32)
    coordinate_metadata: dict[str, Any] = {}
    if coordinate_frame == "cpf":
        calibration = provider.get_device_calibration()
        transform_cpf_imu = calibration.get_transform_cpf_sensor("imu-left")
        if transform_cpf_imu is None:
            raise ValueError(f"VRS 缺少 T_Cpf_ImuLeft 标定: {path}")

        transform_matrix = np.asarray(
            transform_cpf_imu.to_matrix(),
            dtype=np.float64,
        )
        rotation_matrix = transform_matrix[:3, :3]
        values = rotate_imu_vectors(values, rotation_matrix)
        coordinate_metadata = {
            "device_serial": provider.get_file_tags().get("device_serial", ""),
            "source_coordinate_frame": "imu-left",
            "target_coordinate_frame": "cpf",
            "transform_cpf_imu": transform_matrix.tolist(),
            "rotation_cpf_imu": rotation_matrix.tolist(),
            "translation_applied": False,
        }

    # timestamps: (N,), values: (N, 6).
    return RawImu(
        timestamps_seconds=np.asarray(timestamps_seconds, dtype=np.float64),
        values=values,
        coordinate_metadata=coordinate_metadata,
    )


def clean_samples(
    timestamps_seconds: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """检查、排序 IMU，并在重复时间戳处保留最后一条记录。"""
    # timestamps: (N,), values: (N, 6).
    timestamps = np.asarray(timestamps_seconds, dtype=np.float64)
    samples = np.asarray(values)

    if timestamps.ndim != 1:
        raise ValueError("时间戳必须是一维数组")
    if samples.ndim != 2:
        raise ValueError("IMU 必须是 (采样点, 通道) 二维数组")
    if len(timestamps) != len(samples) or len(timestamps) == 0:
        raise ValueError("时间戳和 IMU 的采样数量必须相同且非空")
    if not np.isfinite(timestamps).all() or not np.isfinite(samples).all():
        raise ValueError("时间戳和 IMU 不能包含 NaN 或 Inf")

    sorting_order = np.argsort(timestamps, kind="stable")
    sorted_timestamps = timestamps[sorting_order]
    sorted_samples = samples[sorting_order]

    # 稳定排序后，同一时间戳的最后一条记录位于重复组末尾。
    keep_sample = np.r_[
        sorted_timestamps[1:] != sorted_timestamps[:-1],
        True,
    ]
    return sorted_timestamps[keep_sample], sorted_samples[keep_sample]


def find_continuous_segments(
    timestamps_seconds: np.ndarray, maximum_gap_seconds: float
) -> list[slice]:
    """返回连续信号片段，避免在较大的数据断点之间插值。"""
    timestamps = np.asarray(timestamps_seconds, dtype=np.float64)
    if timestamps.ndim != 1 or len(timestamps) == 0:
        raise ValueError("时间戳必须是非空一维数组")
    if maximum_gap_seconds <= 0:
        raise ValueError("maximum_gap_seconds 必须大于零")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("清洗后的时间戳必须严格递增")

    gap_locations = np.flatnonzero(
        np.diff(timestamps) > maximum_gap_seconds
    ) + 1
    segment_starts = np.r_[0, gap_locations]
    segment_stops = np.r_[gap_locations, len(timestamps)]

    segments: list[slice] = []
    for start, stop in zip(segment_starts, segment_stops):
        segments.append(slice(int(start), int(stop)))
    return segments


def resample_segment(
    timestamps_seconds: np.ndarray,
    values: np.ndarray,
    sampling_rate_hz: int,
) -> tuple[np.ndarray, np.ndarray]:
    """将一个连续片段线性重采样到固定频率。"""
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz 必须大于零")

    timestamps, samples = clean_samples(timestamps_seconds, values)
    if len(timestamps) == 1:
        return timestamps.copy(), samples.astype(np.float32, copy=True)

    duration_seconds = timestamps[-1] - timestamps[0]
    number_of_intervals = int(
        np.floor(duration_seconds * sampling_rate_hz + 1e-9)
    )
    output_timestamps = timestamps[0] + (
        np.arange(number_of_intervals + 1, dtype=np.float64) / sampling_rate_hz
    )

    number_of_channels = samples.shape[1]
    output_values = np.empty(
        (len(output_timestamps), number_of_channels), dtype=np.float32
    )
    for channel_index in range(number_of_channels):
        output_values[:, channel_index] = np.interp(
            output_timestamps,
            timestamps,
            samples[:, channel_index],
        )
    return output_timestamps, output_values


def _cache_metadata(
    take: SelectedTake,
    settings: PreprocessSettings,
    source_information: os.stat_result,
):
    metadata = {
        "take_uid": take.take_uid,
        "take_name": take.take_name,
        "participant_uid": take.participant_uid,
        "activity": take.activity,
        "split": take.split,
        "relative_path": take.relative_path,
        "source_size_bytes": source_information.st_size,
        "source_modification_time_ns": source_information.st_mtime_ns,
        "channel_names": list(CHANNEL_NAMES),
        "sampling_rate_hz": settings.sampling_rate_hz,
        "maximum_gap_seconds": settings.maximum_gap_seconds,
        "minimum_segment_seconds": settings.minimum_segment_seconds,
    }
    if settings.coordinate_frame == "cpf":
        metadata["source_coordinate_frame"] = "imu-left"
        metadata["target_coordinate_frame"] = "cpf"
        metadata["translation_applied"] = False
    return metadata


def _existing_cache_matches(path: Path, expected_metadata: dict[str, Any]):
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as cache:
            actual_metadata = json.loads(str(cache["metadata_json"].item()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return all(
        actual_metadata.get(key) == value
        for key, value in expected_metadata.items()
    )


def preprocess_take(
    take: SelectedTake,
    settings: PreprocessSettings,
    reader: Callable[[Path], RawImu] | None = None,
) -> Path:
    """预处理一个 take，并原子发布 NPZ 缓存。"""
    source_path = settings.dataset_root / take.relative_path
    if not source_path.is_file():
        raise FileNotFoundError(f"找不到完整 VRS: {source_path}")

    source_information = source_path.stat()
    if source_information.st_size != take.size_bytes:
        raise ValueError(
            f"VRS 大小不匹配，预期 {take.size_bytes}，"
            f"实际 {source_information.st_size}: {source_path}"
        )

    metadata = _cache_metadata(take, settings, source_information)
    settings.cache_root.mkdir(parents=True, exist_ok=True)
    output_path = settings.cache_root / f"{take.take_uid}.npz"
    if _existing_cache_matches(output_path, metadata):
        return output_path

    if reader is None:
        raw_imu = read_left_imu(source_path, settings.coordinate_frame)
    else:
        raw_imu = reader(source_path)
    metadata.update(raw_imu.coordinate_metadata)
    timestamps, values = clean_samples(
        raw_imu.timestamps_seconds,
        raw_imu.values,
    )

    output_timestamps: list[np.ndarray] = []
    output_values: list[np.ndarray] = []
    segment_bounds: list[tuple[int, int]] = []
    output_offset = 0

    segments = find_continuous_segments(
        timestamps,
        settings.maximum_gap_seconds,
    )
    for segment in segments:
        segment_timestamps = timestamps[segment]
        # segment_values: (M, 6).
        segment_values = values[segment]
        segment_duration = segment_timestamps[-1] - segment_timestamps[0]
        if segment_duration < settings.minimum_segment_seconds:
            continue

        # resampled_values: (M', 6).
        resampled_timestamps, resampled_values = resample_segment(
            segment_timestamps,
            segment_values,
            settings.sampling_rate_hz,
        )
        output_timestamps.append(resampled_timestamps)
        output_values.append(resampled_values)

        segment_stop = output_offset + len(resampled_timestamps)
        segment_bounds.append((output_offset, segment_stop))
        output_offset = segment_stop

    if not output_values:
        raise ValueError(f"take {take.take_uid} 没有达到最短时长的连续片段")

    combined_timestamps = np.concatenate(output_timestamps)
    combined_values = np.concatenate(output_values)
    # segment_bounds: (num_segments, 2).
    bounds_array = np.asarray(segment_bounds, dtype=np.int64)
    metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=settings.cache_root,
            prefix=f".{take.take_uid}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.savez_compressed(
                temporary_file,
                timestamps_seconds=combined_timestamps,
                values=combined_values,
                segment_bounds=bounds_array,
                metadata_json=np.asarray(metadata_json),
            )
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        # 发布前重新打开一次，确认四个必要字段都能被读取。
        with np.load(temporary_path, allow_pickle=False) as temporary_cache:
            required_fields = {
                "timestamps_seconds",
                "values",
                "segment_bounds",
                "metadata_json",
            }
            if not required_fields.issubset(temporary_cache.files):
                raise ValueError("临时缓存缺少必要字段")

        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return output_path


def settings_from_config(config: dict[str, Any]) -> PreprocessSettings:
    data_config = config["data"]
    return PreprocessSettings(
        dataset_root=project_path(data_config["dataset_root"]),
        cache_root=project_path(data_config["cache_root"]),
        sampling_rate_hz=int(data_config["sampling_rate_hz"]),
        maximum_gap_seconds=float(data_config["maximum_gap_seconds"]),
        minimum_segment_seconds=float(data_config["minimum_segment_seconds"]),
        coordinate_frame=str(data_config.get("coordinate_frame", "sensor")),
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="把 Ego-Exo4D 左侧 IMU 预处理为 EgoCHARM 缓存"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/egocharm.yaml"))
    parser.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="all",
    )
    parser.add_argument("--take-uid", action="append", default=[])
    parser.add_argument("--max-takes", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    parsed_arguments = parser.parse_args(arguments)
    config = load_config(parsed_arguments.config)
    settings = settings_from_config(config)

    selection_path = project_path(config["data"]["selection"])
    selected_takes = load_selected_takes(selection_path)

    if parsed_arguments.split != "all":
        matching_split: list[SelectedTake] = []
        for take in selected_takes:
            if take.split == parsed_arguments.split:
                matching_split.append(take)
        selected_takes = matching_split

    requested_uids = set(parsed_arguments.take_uid)
    if requested_uids:
        matching_uids: list[SelectedTake] = []
        for take in selected_takes:
            if take.take_uid in requested_uids:
                matching_uids.append(take)
        selected_takes = matching_uids

    if parsed_arguments.max_takes is not None:
        if parsed_arguments.max_takes < 0:
            parser.error("--max-takes 不能小于零")
        selected_takes = selected_takes[: parsed_arguments.max_takes]

    if parsed_arguments.dry_run:
        result = {
            "selected": len(selected_takes),
            "take_uids": [take.take_uid for take in selected_takes],
            "vrs_opened": False,
        }
        print(json.dumps(result, ensure_ascii=False))
        return 0

    failed_takes = 0
    for take in selected_takes:
        try:
            output_path = preprocess_take(take, settings)
            print(f"完成 {take.take_uid}: {output_path}")
        except Exception as error:
            failed_takes += 1
            print(f"失败 {take.take_uid}: {type(error).__name__}: {error}")
            if parsed_arguments.fail_fast:
                break

    if failed_takes > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
