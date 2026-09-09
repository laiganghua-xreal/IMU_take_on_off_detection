"""Create isolated 200 Hz caches for Ego activities and Xreal wear events."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from egocharm.config import load_config, project_path
from egocharm.joint_cache import (
    JointRecording,
    load_joint_manifest,
    write_joint_manifest,
    write_recording_cache,
)
from egocharm.joint_preprocessing import resample_uniform_imu
from scripts.preprocess_data import (
    clean_samples,
    find_continuous_segments,
    load_selected_takes,
    read_left_imu,
)
from Xreal_datasets.preprocess_xreal_imu import (
    EVENT_LABELS,
    assign_event_splits,
    infer_label,
    read_xreal_imu,
    rotate_xreal_to_cpf,
)


CPF_FRAME = "cpf-left-up-forward"


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description="生成 Ego 七分类与 Xreal 摘戴事件的联合训练缓存"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ego_xreal_joint_9class.yaml"),
    )
    parser.add_argument("--max-ego-takes", type=int)
    parser.add_argument("--skip-ego", action="store_true")
    parser.add_argument("--skip-xreal", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _load_reusable_recordings(manifest_path):
    path = Path(manifest_path)
    if not path.is_file():
        return {}
    return {
        recording.recording_uid: recording
        for recording in load_joint_manifest(path)
        if recording.values_path.is_file()
    }


def _can_reuse(recording, label, split, sampling_rate_hz):
    return (
        recording.label == label
        and recording.split == split
        and recording.sampling_rate_hz == sampling_rate_hz
        and recording.coordinate_frame == CPF_FRAME
        and recording.cache_version == 4
        and recording.values_path.is_file()
    )


def _preprocess_ego_take(take, dataset_root, cache_root, data_config):
    source_path = dataset_root / take.relative_path
    if not source_path.is_file():
        raise FileNotFoundError(f"找不到 VRS: {source_path}")

    raw = read_left_imu(source_path, coordinate_frame="cpf")
    timestamps, values = clean_samples(raw.timestamps_seconds, raw.values)
    sampling_rate_hz = int(data_config["sampling_rate_hz"])
    minimum_segment_seconds = float(data_config["minimum_segment_seconds"])
    maximum_gap_seconds = float(data_config["maximum_gap_seconds"])

    resampled_values = []
    segment_bounds = []
    output_start = 0
    for segment in find_continuous_segments(timestamps, maximum_gap_seconds):
        segment_timestamps = timestamps[segment]
        if segment_timestamps[-1] - segment_timestamps[0] < minimum_segment_seconds:
            continue
        _, output_values = resample_uniform_imu(
            segment_timestamps,
            values[segment],
            sampling_rate_hz,
        )
        output_stop = output_start + len(output_values)
        resampled_values.append(output_values)
        segment_bounds.append((output_start, output_stop))
        output_start = output_stop

    if not resampled_values:
        raise ValueError(f"{take.take_uid} 没有满足长度要求的连续 IMU")

    return write_recording_cache(
        cache_root=cache_root,
        recording_uid=take.take_uid,
        source="ego",
        label=take.activity,
        split=take.split,
        values=np.concatenate(resampled_values),
        segment_bounds=segment_bounds,
        sampling_rate_hz=sampling_rate_hz,
        coordinate_frame=CPF_FRAME,
    )


def _preprocess_xreal_event(
    source_path,
    label,
    split,
    cache_root,
    sampling_rate_hz,
    window_seconds,
):
    raw = read_xreal_imu(source_path)
    cpf_values = rotate_xreal_to_cpf(raw.values)
    _, sampled_values = resample_uniform_imu(
        raw.timestamps_ns.astype(np.float64) * 1e-9,
        cpf_values,
        sampling_rate_hz,
    )
    window_samples = sampling_rate_hz * window_seconds
    if len(sampled_values) < window_samples:
        raise ValueError(f"录制不足 {window_seconds} 秒: {source_path}")
    start_sample = (len(sampled_values) - window_samples) // 2
    values = sampled_values[start_sample : start_sample + window_samples]
    recording_uid = source_path.parent.name
    return write_recording_cache(
        cache_root=cache_root,
        recording_uid=recording_uid,
        source="xreal",
        label=label,
        split=split,
        values=values,
        segment_bounds=((0, window_samples),),
        sampling_rate_hz=sampling_rate_hz,
        coordinate_frame=CPF_FRAME,
    )


def _find_xreal_events(source_roots):
    grouped_paths = {label: [] for label in EVENT_LABELS}
    for source_root in source_roots:
        for source_path in sorted(Path(source_root).glob("*/imu_0.csv")):
            label = infer_label(source_path)
            if label in EVENT_LABELS:
                grouped_paths[label].append(source_path)
    return grouped_paths


def _event_splits_by_source_root(grouped_paths, source_roots, seed):
    split_by_path = {}
    for label in EVENT_LABELS:
        for source_root in source_roots:
            paths = [
                path
                for path in grouped_paths[label]
                if path.parents[1].resolve() == Path(source_root).resolve()
            ]
            split_by_path.update(assign_event_splits(paths, seed))
    return split_by_path


def _process_ego(data_config, max_ego_takes):
    selection_path = project_path(data_config["ego_selection"])
    selected_takes = load_selected_takes(selection_path)
    if max_ego_takes is not None:
        selected_takes = selected_takes[:max_ego_takes]

    dataset_root = project_path(data_config["ego_dataset_root"])
    manifest_path = project_path(data_config["ego_manifest"])
    cache_root = manifest_path.parent
    reusable = _load_reusable_recordings(manifest_path)
    recordings = []
    failures = 0
    for take in selected_takes:
        existing = reusable.get(take.take_uid)
        if existing is not None and _can_reuse(
            existing,
            take.activity,
            take.split,
            int(data_config["sampling_rate_hz"]),
        ):
            recording = existing
        else:
            try:
                recording = _preprocess_ego_take(
                    take,
                    dataset_root,
                    cache_root,
                    data_config,
                )
            except Exception as error:
                failures += 1
                print(f"Ego 失败 {take.take_uid}: {type(error).__name__}: {error}")
                continue
        recordings.append(recording)
        write_joint_manifest(manifest_path, recordings)
        print(f"Ego {len(recordings)}/{len(selected_takes)}: {take.take_uid}")
    return recordings, failures


def _process_xreal(data_config, seed):
    source_roots = [project_path(path) for path in data_config["xreal_source_roots"]]
    grouped_paths = _find_xreal_events(source_roots)
    split_by_path = _event_splits_by_source_root(grouped_paths, source_roots, seed)
    manifest_path = project_path(data_config["xreal_event_manifest"])
    cache_root = manifest_path.parent
    reusable = _load_reusable_recordings(manifest_path)
    sampling_rate_hz = int(data_config["sampling_rate_hz"])
    window_seconds = int(data_config["window_seconds"])

    recordings = []
    failures = 0
    for label in EVENT_LABELS:
        for source_path in grouped_paths[label]:
            recording_uid = source_path.parent.name
            split = split_by_path[source_path]
            existing = reusable.get(recording_uid)
            if existing is not None and _can_reuse(
                existing,
                label,
                split,
                sampling_rate_hz,
            ):
                recording = existing
            else:
                try:
                    recording = _preprocess_xreal_event(
                        source_path,
                        label,
                        split,
                        cache_root,
                        sampling_rate_hz,
                        window_seconds,
                    )
                except Exception as error:
                    failures += 1
                    print(
                        f"Xreal 失败 {recording_uid}: "
                        f"{type(error).__name__}: {error}"
                    )
                    continue
            recordings.append(recording)
            write_joint_manifest(manifest_path, recordings)
            print(f"Xreal {len(recordings)}: {recording_uid}")
    return recordings, failures


def main(arguments=None):
    args = build_argument_parser().parse_args(arguments)
    config = load_config(args.config)
    data_config = config["data"]
    source_roots = [project_path(path) for path in data_config["xreal_source_roots"]]
    grouped_paths = _find_xreal_events(source_roots)

    if args.dry_run:
        selection_path = project_path(data_config["ego_selection"])
        number_of_ego_takes = 0
        if selection_path.is_file():
            number_of_ego_takes = len(load_selected_takes(selection_path))
        print(
            json.dumps(
                {
                    "ego_takes": number_of_ego_takes,
                    "xreal_event_recordings": {
                        label: len(grouped_paths[label]) for label in EVENT_LABELS
                    },
                },
                ensure_ascii=False,
            )
        )
        return 0

    total_failures = 0
    if not args.skip_ego:
        _, failures = _process_ego(data_config, args.max_ego_takes)
        total_failures += failures
    if not args.skip_xreal:
        _, failures = _process_xreal(config["data"], int(config["training"]["seed"]))
        total_failures += failures
    return 2 if total_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
