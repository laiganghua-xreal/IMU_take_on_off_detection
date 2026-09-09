"""Memory-mapped cache records shared by Ego and Xreal IMU data."""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from egocharm.dataset import CHANNEL_NAMES


@dataclass(frozen=True)
class JointRecording:
    """One continuous IMU recording and its training metadata."""

    recording_uid: str
    source: str
    label: str
    split: str
    values_path: Path
    segment_bounds: tuple[tuple[int, int], ...]
    sampling_rate_hz: int
    coordinate_frame: str
    cache_version: int = 4


def write_recording_cache(
    cache_root,
    recording_uid,
    source,
    label,
    split,
    values,
    segment_bounds,
    sampling_rate_hz,
    coordinate_frame,
    cache_version=4,
):
    """Write one float32 ``(N, 6)`` array that NumPy can memory-map."""
    samples = np.asarray(values)
    if samples.ndim != 2 or samples.shape[1] != len(CHANNEL_NAMES):
        raise ValueError("IMU values 必须是 (N, 6)")

    bounds = tuple((int(start), int(stop)) for start, stop in segment_bounds)
    output_root = Path(cache_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{recording_uid}.npy"

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_root,
            prefix=f".{recording_uid}.",
            suffix=".npy",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            np.save(temporary_file, samples.astype(np.float32, copy=False))
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return JointRecording(
        recording_uid=str(recording_uid),
        source=str(source),
        label=str(label),
        split=str(split),
        values_path=output_path.resolve(),
        segment_bounds=bounds,
        sampling_rate_hz=int(sampling_rate_hz),
        coordinate_frame=str(coordinate_frame),
        cache_version=int(cache_version),
    )


def write_joint_manifest(path, recordings):
    """Atomically write a portable JSON manifest for cached recordings."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for recording in recordings:
        relative_values_path = os.path.relpath(
            recording.values_path,
            output_path.parent.resolve(),
        )
        entries.append(
            {
                "recording_uid": recording.recording_uid,
                "source": recording.source,
                "label": recording.label,
                "split": recording.split,
                "values_path": relative_values_path,
                "segment_bounds": [list(bound) for bound in recording.segment_bounds],
                "sampling_rate_hz": recording.sampling_rate_hz,
                "coordinate_frame": recording.coordinate_frame,
                "cache_version": recording.cache_version,
                "channel_names": list(CHANNEL_NAMES),
            }
        )

    payload = {"entries": entries}
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_joint_manifest(path):
    """Load a joint manifest and resolve value paths from its directory."""
    manifest_path = Path(path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    recordings = []
    for entry in payload["entries"]:
        if tuple(entry["channel_names"]) != CHANNEL_NAMES:
            raise ValueError("联合缓存的通道顺序不正确")
        values_path = Path(entry["values_path"])
        if not values_path.is_absolute():
            values_path = manifest_path.parent / values_path
        recordings.append(
            JointRecording(
                recording_uid=str(entry["recording_uid"]),
                source=str(entry["source"]),
                label=str(entry["label"]),
                split=str(entry["split"]),
                values_path=values_path.resolve(),
                segment_bounds=tuple(
                    (int(start), int(stop))
                    for start, stop in entry["segment_bounds"]
                ),
                sampling_rate_hz=int(entry["sampling_rate_hz"]),
                coordinate_frame=str(entry["coordinate_frame"]),
                cache_version=int(entry.get("cache_version", 1)),
            )
        )
    return recordings
