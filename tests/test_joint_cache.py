import numpy as np

from egocharm.joint_cache import (
    JointRecording,
    load_joint_manifest,
    write_joint_manifest,
    write_recording_cache,
)


def test_recording_cache_is_float32_and_memory_mappable(tmp_path):
    values = np.arange(72, dtype=np.float64).reshape(12, 6)

    recording = write_recording_cache(
        cache_root=tmp_path / "cache",
        recording_uid="take-001",
        source="ego",
        label="Cooking",
        split="train",
        values=values,
        segment_bounds=((0, 12),),
        sampling_rate_hz=200,
        coordinate_frame="cpf-left-up-forward",
    )

    memory_mapped_values = np.load(recording.values_path, mmap_mode="r")
    assert recording.cache_version == 4
    assert isinstance(memory_mapped_values, np.memmap)
    assert memory_mapped_values.dtype == np.float32
    assert memory_mapped_values.shape == (12, 6)
    np.testing.assert_array_equal(memory_mapped_values, values.astype(np.float32))


def test_manifest_round_trip_resolves_relative_value_paths(tmp_path):
    cache_root = tmp_path / "cache"
    recording = write_recording_cache(
        cache_root=cache_root,
        recording_uid="event-001",
        source="xreal",
        label="PUT_ON",
        split="val",
        values=np.zeros((1000, 6), dtype=np.float32),
        segment_bounds=((0, 1000),),
        sampling_rate_hz=200,
        coordinate_frame="cpf-left-up-forward",
    )
    manifest_path = cache_root / "manifest.json"

    write_joint_manifest(manifest_path, [recording])
    loaded = load_joint_manifest(manifest_path)

    assert loaded == [
        JointRecording(
            recording_uid="event-001",
            source="xreal",
            label="PUT_ON",
            split="val",
            values_path=(cache_root / "event-001.npy").resolve(),
            segment_bounds=((0, 1000),),
            sampling_rate_hz=200,
            coordinate_frame="cpf-left-up-forward",
            cache_version=4,
        )
    ]


def test_recording_cache_rejects_wrong_channel_count(tmp_path):
    values = np.zeros((1000, 3), dtype=np.float32)

    try:
        write_recording_cache(
            cache_root=tmp_path,
            recording_uid="bad",
            source="ego",
            label="Dance",
            split="train",
            values=values,
            segment_bounds=((0, 1000),),
            sampling_rate_hz=200,
            coordinate_frame="cpf-left-up-forward",
        )
    except ValueError as error:
        assert "(N, 6)" in str(error)
    else:
        raise AssertionError("three-channel IMU input must be rejected")
