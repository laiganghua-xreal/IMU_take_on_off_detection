"""把 Xreal 六轴 IMU CSV 转换为五秒 CPF 训练样本。"""

import argparse
import csv
import json
import os
import random
import tempfile
from array import array
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


CLASS_NAMES = ("WORN", "NOT_WORN", "PUT_ON", "TAKE_OFF")
STABLE_LABELS = ("WORN", "NOT_WORN")
EVENT_LABELS = ("PUT_ON", "TAKE_OFF")
XREAL_TO_CPF_ROTATION = np.diag([-1.0, 1.0, -1.0])
DEFAULT_SOURCE_ROOTS = (
    Path("/home/xreal/record_data_tools/data/20260818/free_record_only_imu"),
    Path("/home/xreal/record_data_tools/data/20260819/free_record_only_imu"),
)


@dataclass(frozen=True)
class RawXrealImu:
    """时间对齐后的 Xreal IMU，通道顺序为 acc XYZ、gyro XYZ。"""

    timestamps_ns: np.ndarray
    values: np.ndarray


@dataclass(frozen=True)
class ConvertedExample:
    """一个已经写入磁盘的五秒样本。"""

    example_uid: str
    cache_path: Path
    label: str
    split: str
    source_path: Path
    start_sample: int

    def selection_entry(self):
        return {
            "example_uid": self.example_uid,
            "cache_path": str(self.cache_path.resolve()),
            "label": self.label,
            "split": self.split,
            "source_path": str(self.source_path.resolve()),
            "start_sample": self.start_sample,
        }


def rotate_xreal_to_cpf(values):
    """把右、上、后方向的 acc/gyro 转成 CPF 左、上、前。"""
    samples = np.asarray(values)
    if samples.ndim != 2 or samples.shape[1] != 6:
        raise ValueError("Xreal IMU values 必须是 (N, 6)")

    transformed = np.empty_like(samples)
    transformed[:, :3] = samples[:, :3] @ XREAL_TO_CPF_ROTATION.T
    transformed[:, 3:] = samples[:, 3:] @ XREAL_TO_CPF_ROTATION.T
    return transformed


def read_xreal_imu(path):
    """读取 type 1/2，并按共同时间戳组合成六轴 IMU。"""
    gyro_timestamps = array("q")
    accel_timestamps = array("q")
    gyro_values = array("f")
    accel_values = array("f")

    with Path(path).open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.reader(input_file)
        next(reader)
        for row in reader:
            sensor_type = row[1]
            if sensor_type == "1":
                gyro_timestamps.append(int(row[0]))
                gyro_values.extend(float(value) for value in row[2:5])
            elif sensor_type == "2":
                accel_timestamps.append(int(row[0]))
                accel_values.extend(float(value) for value in row[2:5])

    gyro_times = np.frombuffer(gyro_timestamps, dtype=np.int64).copy()
    accel_times = np.frombuffer(accel_timestamps, dtype=np.int64).copy()
    if not np.array_equal(gyro_times, accel_times):
        raise ValueError(f"gyro 和 acc 时间戳不一致: {path}")
    if len(accel_times) < 2:
        raise ValueError(f"IMU 有效采样不足: {path}")

    acceleration = np.frombuffer(accel_values, dtype=np.float32).reshape(-1, 3)
    gyroscope = np.frombuffer(gyro_values, dtype=np.float32).reshape(-1, 3)
    values = np.concatenate((acceleration, gyroscope), axis=1)
    return RawXrealImu(timestamps_ns=accel_times, values=values)


def resample_xreal_imu(raw, sampling_rate_hz):
    """用带抗混叠滤波的 polyphase 方法降采样到目标频率。"""
    duration_seconds = (raw.timestamps_ns[-1] - raw.timestamps_ns[0]) * 1e-9
    source_rate_hz = int(round((len(raw.timestamps_ns) - 1) / duration_seconds))
    output_values = resample_poly(
        raw.values,
        up=sampling_rate_hz,
        down=source_rate_hz,
        axis=0,
    ).astype(np.float32, copy=False)
    output_timestamps = raw.timestamps_ns[0] + np.rint(
        np.arange(len(output_values), dtype=np.float64)
        * (1e9 / sampling_rate_hz)
    ).astype(np.int64)
    return RawXrealImu(output_timestamps, output_values)


def allocate_split_counts(number_of_items, split_ratios=(7, 1, 2)):
    """按最大余数法把整数样本确定性地分配到 train/val/test。"""
    if number_of_items < 0:
        raise ValueError("number_of_items 不能小于零")
    if len(split_ratios) != 3 or any(ratio < 0 for ratio in split_ratios):
        raise ValueError("split_ratios 必须包含三个非负数")
    ratio_sum = sum(split_ratios)
    if ratio_sum <= 0:
        raise ValueError("split_ratios 之和必须大于零")

    exact_counts = [
        number_of_items * ratio / ratio_sum
        for ratio in split_ratios
    ]
    counts = [int(value) for value in exact_counts]
    remainder = number_of_items - sum(counts)
    allocation_order = sorted(
        range(len(counts)),
        key=lambda index: (
            -(exact_counts[index] - counts[index]),
            index,
        ),
    )
    for index in allocation_order[:remainder]:
        counts[index] += 1
    return tuple(counts)


def assign_stable_window_splits(number_of_windows, split_ratios=(7, 1, 2)):
    """按时间顺序划分长序列，并在 split 边界各跳过一个窗口。"""
    if number_of_windows < 5:
        raise ValueError("稳定录制至少需要五个窗口")
    usable_windows = number_of_windows - 2
    train_count, validation_count, test_count = allocate_split_counts(
        usable_windows,
        split_ratios,
    )
    return (
        ["train"] * train_count
        + [None]
        + ["val"] * validation_count
        + [None]
        + ["test"] * test_count
    )


def build_stable_window_assignments(
    number_of_samples,
    sampling_rate_hz,
    window_seconds,
    stride_seconds,
    split_ratios=(7, 1, 2),
):
    """生成稳定状态窗口的起点和 split，不让窗口跨越 split 边界。"""
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz 必须大于零")
    if window_seconds <= 0:
        raise ValueError("window_seconds 必须大于零")
    if stride_seconds <= 0:
        raise ValueError("stride_seconds 必须大于零")

    window_samples = sampling_rate_hz * window_seconds
    stride_samples = sampling_rate_hz * stride_seconds
    number_of_base_windows = number_of_samples // window_samples
    split_by_base_window = assign_stable_window_splits(
        number_of_base_windows,
        split_ratios,
    )

    assignments = []
    region_start = 0
    while region_start < len(split_by_base_window):
        split = split_by_base_window[region_start]
        region_stop = region_start + 1
        while (
            region_stop < len(split_by_base_window)
            and split_by_base_window[region_stop] == split
        ):
            region_stop += 1

        if split is not None:
            first_sample = region_start * window_samples
            final_start = region_stop * window_samples - window_samples
            for start_sample in range(
                first_sample,
                final_start + 1,
                stride_samples,
            ):
                assignments.append((start_sample, split))

        region_start = region_stop

    return assignments


def assign_event_splits(paths, seed, split_ratios=(7, 1, 2)):
    """以文件为单位确定事件录制的 train/val/test。"""
    shuffled_paths = list(sorted(Path(path) for path in paths))
    random.Random(seed).shuffle(shuffled_paths)
    train_count, validation_count, _ = allocate_split_counts(
        len(shuffled_paths),
        split_ratios,
    )
    assignments = {}
    for index, path in enumerate(shuffled_paths):
        if index < train_count:
            split = "train"
        elif index < train_count + validation_count:
            split = "val"
        else:
            split = "test"
        assignments[path] = split
    return assignments


def infer_label(path):
    """从采集目录名称读取文件级标签。"""
    directory_name = Path(path).parent.name
    if "_WALK_ON_" in directory_name:
        return "WORN"
    if "_WALK_OFF_" in directory_name:
        return "NOT_WORN"
    if "_WORKING_WEAR_" in directory_name:
        return "WORN"
    if "_NOWEAR_" in directory_name:
        return "NOT_WORN"
    if "_ON_" in directory_name:
        return "PUT_ON"
    if "_OFF_" in directory_name:
        return "TAKE_OFF"
    raise ValueError(f"无法从目录名称识别标签: {directory_name}")


def _write_example(
    values,
    timestamps_ns,
    source_path,
    label,
    split,
    cache_root,
    start_sample,
    sampling_rate_hz,
    window_seconds,
):
    source_name = Path(source_path).parent.name
    example_uid = f"{source_name}__{start_sample:09d}"
    output_path = Path(cache_root) / f"{example_uid}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stop_sample = start_sample + sampling_rate_hz * window_seconds
    window_values = values[start_sample:stop_sample]
    window_timestamps = timestamps_ns[start_sample:stop_sample]
    metadata = {
        "example_uid": example_uid,
        "label": label,
        "split": split,
        "source_path": str(Path(source_path).resolve()),
        "sampling_rate_hz": sampling_rate_hz,
        "window_seconds": window_seconds,
        "source_coordinate_frame": "xreal-right-up-back",
        "target_coordinate_frame": "cpf-left-up-forward",
        "rotation_cpf_xreal": XREAL_TO_CPF_ROTATION.tolist(),
        "translation_applied": False,
        "channel_names": [
            "accel_x",
            "accel_y",
            "accel_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
        ],
    }

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent,
            prefix=f".{example_uid}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.savez_compressed(
                temporary_file,
                timestamps_ns=window_timestamps,
                values=window_values,
                metadata_json=np.asarray(
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True)
                ),
            )
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return ConvertedExample(
        example_uid=example_uid,
        cache_path=output_path,
        label=label,
        split=split,
        source_path=Path(source_path),
        start_sample=start_sample,
    )


def _load_resampled_cpf(path, sampling_rate_hz):
    raw = read_xreal_imu(path)
    cpf_values = rotate_xreal_to_cpf(raw.values)
    return resample_xreal_imu(
        RawXrealImu(raw.timestamps_ns, cpf_values),
        sampling_rate_hz,
    )


def preprocess_recording(
    source_path,
    label,
    split,
    cache_root,
    sampling_rate_hz=50,
    window_seconds=5,
):
    """转换一个录制；稳定录制产生多个窗口，事件录制产生居中窗口。"""
    resampled = _load_resampled_cpf(source_path, sampling_rate_hz)
    window_samples = sampling_rate_hz * window_seconds
    if len(resampled.values) < window_samples:
        raise ValueError(f"录制不足 {window_seconds} 秒: {source_path}")

    if label in EVENT_LABELS:
        starts = [(len(resampled.values) - window_samples) // 2]
    elif label in STABLE_LABELS:
        starts = list(range(0, len(resampled.values) - window_samples + 1, window_samples))
    else:
        raise ValueError(f"不支持的标签: {label}")

    return [
        _write_example(
            resampled.values,
            resampled.timestamps_ns,
            source_path,
            label,
            split,
            cache_root,
            start_sample,
            sampling_rate_hz,
            window_seconds,
        )
        for start_sample in starts
    ]


def _save_plot(raw, source_path, plot_root):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plot

    time_seconds = (raw.timestamps_ns - raw.timestamps_ns[0]) * 1e-9
    figure, axes = plot.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(time_seconds, raw.values[:, :3])
    axes[0].set_ylabel("acceleration")
    axes[0].legend(("left", "up", "forward"))
    axes[1].plot(time_seconds, raw.values[:, 3:])
    axes[1].set(xlabel="time (s)", ylabel="angular velocity")
    axes[1].legend(("left", "up", "forward"))
    figure.tight_layout()
    output_path = Path(plot_root) / f"{Path(source_path).parent.name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=130)
    plot.close(figure)


def _write_selection(path, examples, source_roots):
    if isinstance(source_roots, (str, Path)):
        roots = [Path(source_roots)]
    else:
        roots = [Path(root) for root in source_roots]
    payload = {
        "class_names": list(CLASS_NAMES),
        "source_root": str(roots[0].resolve()),
        "source_roots": [str(root.resolve()) for root in roots],
        "coordinate_frame": "cpf-left-up-forward",
        "entries": [example.selection_entry() for example in examples],
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_existing_examples(selection_path, labels):
    """从已有 selection 恢复指定标签，不读取或改写对应 NPZ。"""
    path = Path(selection_path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到可复用的 selection: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = []
    for entry in payload.get("entries", []):
        if entry.get("label") not in labels:
            continue
        examples.append(
            ConvertedExample(
                example_uid=entry["example_uid"],
                cache_path=Path(entry["cache_path"]),
                label=entry["label"],
                split=entry["split"],
                source_path=Path(entry["source_path"]),
                start_sample=entry["start_sample"],
            )
        )
    return examples


def _replace_stable_cache(staged_examples, cache_root, old_stable_examples):
    """安装新的稳定样本，并删除旧 selection 中已失效的稳定样本。"""
    output_root = Path(cache_root)
    output_root.mkdir(parents=True, exist_ok=True)
    installed_examples = []

    for example in staged_examples:
        output_path = output_root / example.cache_path.name
        os.replace(example.cache_path, output_path)
        installed_examples.append(
            ConvertedExample(
                example_uid=example.example_uid,
                cache_path=output_path,
                label=example.label,
                split=example.split,
                source_path=example.source_path,
                start_sample=example.start_sample,
            )
        )

    new_paths = {example.cache_path.resolve() for example in installed_examples}
    resolved_output_root = output_root.resolve()
    removed_count = 0
    for example in old_stable_examples:
        old_path = example.cache_path.resolve()
        if old_path in new_paths or old_path.parent != resolved_output_root:
            continue
        if old_path.is_file():
            old_path.unlink()
            removed_count += 1

    return installed_examples, removed_count


def build_argument_parser():
    parser = argparse.ArgumentParser(description="转换 Xreal IMU 为 CPF 五秒窗口")
    parser.add_argument(
        "--source-root",
        dest="source_roots",
        action="append",
        type=Path,
        default=None,
        help="可重复指定；默认合并 20260818 和 20260819",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("artifacts/xreal_wear_detection/cache/xreal_cpf"),
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("Xreal_datasets/metadata/xreal_wear_selection.json"),
    )
    parser.add_argument("--sampling-rate-hz", type=int, default=50)
    parser.add_argument("--window-seconds", type=int, default=5)
    parser.add_argument(
        "--stable-stride-seconds",
        type=int,
        default=None,
        help="稳定状态窗口的起点间隔；默认等于 window-seconds",
    )
    parser.add_argument(
        "--reuse-existing-events",
        action="store_true",
        help="保留已有 selection 中的 PUT_ON/TAKE_OFF 样本，不重写其缓存",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--plot-root",
        type=Path,
        default=Path("artifacts/xreal_wear_detection/plots"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(arguments=None):
    args = build_argument_parser().parse_args(arguments)
    source_roots = args.source_roots or list(DEFAULT_SOURCE_ROOTS)
    source_paths = sorted(
        source_path
        for source_root in source_roots
        for source_path in source_root.glob("*/imu_0.csv")
    )
    grouped_paths = {label: [] for label in CLASS_NAMES}
    for source_path in source_paths:
        grouped_paths[infer_label(source_path)].append(source_path)

    if args.dry_run:
        print(
            json.dumps(
                {label: len(paths) for label, paths in grouped_paths.items()},
                ensure_ascii=False,
            )
        )
        return 0

    if args.stable_stride_seconds is None:
        stride_seconds = args.window_seconds
    else:
        stride_seconds = args.stable_stride_seconds

    old_stable_examples = []
    if args.selection.is_file():
        old_stable_examples = _load_existing_examples(args.selection, STABLE_LABELS)
    if args.reuse_existing_events:
        event_examples = _load_existing_examples(args.selection, EVENT_LABELS)
    else:
        event_examples = []

    args.cache_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=args.cache_root.parent,
        prefix=".xreal-stable-",
    ) as staging_directory:
        staged_examples = []
        for label in STABLE_LABELS:
            for source_path in grouped_paths[label]:
                resampled = _load_resampled_cpf(source_path, args.sampling_rate_hz)
                if args.plot:
                    _save_plot(resampled, source_path, args.plot_root)
                assignments = build_stable_window_assignments(
                    number_of_samples=len(resampled.values),
                    sampling_rate_hz=args.sampling_rate_hz,
                    window_seconds=args.window_seconds,
                    stride_seconds=stride_seconds,
                )
                for start_sample, split in assignments:
                    staged_examples.append(
                        _write_example(
                            resampled.values,
                            resampled.timestamps_ns,
                            source_path,
                            label,
                            split,
                            staging_directory,
                            start_sample,
                            args.sampling_rate_hz,
                            args.window_seconds,
                        )
                    )

        stable_examples, removed_count = _replace_stable_cache(
            staged_examples,
            args.cache_root,
            old_stable_examples,
        )

    examples = list(stable_examples)

    if args.reuse_existing_events:
        examples.extend(event_examples)
    else:
        for label in EVENT_LABELS:
            split_by_path = {}
            for source_root in source_roots:
                stratum_paths = [
                    source_path
                    for source_path in grouped_paths[label]
                    if source_path.parents[1] == source_root
                ]
                split_by_path.update(
                    assign_event_splits(stratum_paths, args.seed)
                )
            for source_path in grouped_paths[label]:
                converted = preprocess_recording(
                    source_path,
                    label,
                    split_by_path[source_path],
                    args.cache_root,
                    args.sampling_rate_hz,
                    args.window_seconds,
                )
                examples.extend(converted)
                if args.plot:
                    _save_plot(
                        _load_resampled_cpf(source_path, args.sampling_rate_hz),
                        source_path,
                        args.plot_root,
                    )

    _write_selection(args.selection, examples, source_roots)
    counts = {split: 0 for split in ("train", "val", "test")}
    for example in examples:
        counts[example.split] += 1
    print(
        json.dumps(
            {
                "examples": len(examples),
                "splits": counts,
                "removed_stale_stable_examples": removed_count,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
