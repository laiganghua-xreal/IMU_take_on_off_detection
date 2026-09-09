import numpy as np

from egocharm.joint_cache import write_recording_cache
from egocharm.joint_dataset import (
    JOINT_CLASS_NAMES,
    BalancedJointDataset,
    EvaluationJointDataset,
    ImuAugmentation,
)


def _recording(tmp_path, label, values, split="train", uid=None):
    recording_uid = uid or label.lower().replace(" ", "-")
    return write_recording_cache(
        cache_root=tmp_path,
        recording_uid=recording_uid,
        source="xreal" if label in ("PUT_ON", "TAKE_OFF") else "ego",
        label=label,
        split=split,
        values=values,
        segment_bounds=((0, len(values)),),
        sampling_rate_hz=200,
        coordinate_frame="cpf-left-up-forward",
    )


def test_balanced_dataset_has_equal_labels_and_expected_shape(tmp_path):
    values = np.arange(6000, dtype=np.float32).reshape(1000, 6)
    recordings = [
        _recording(tmp_path, label, values, uid=f"recording-{index}")
        for index, label in enumerate(JOINT_CLASS_NAMES)
    ]
    dataset = BalancedJointDataset(
        recordings,
        sampling_rate_hz=200,
        window_seconds=5,
        candidate_step_seconds=0.025,
        samples_per_class=2,
    )

    features, label = dataset[0]

    assert features.shape == (5, 6, 200)
    assert label.ndim == 0
    assert np.bincount(dataset.class_labels, minlength=9).tolist() == [2] * 9
    np.testing.assert_array_equal(
        features.numpy().transpose(0, 2, 1).reshape(1000, 6),
        values,
    )


def test_candidate_step_uses_five_samples_at_200_hz(tmp_path):
    values = np.zeros((1010, 6), dtype=np.float32)
    recordings = [
        _recording(tmp_path, label, values, uid=f"recording-{index}")
        for index, label in enumerate(JOINT_CLASS_NAMES)
    ]

    dataset = BalancedJointDataset(
        recordings,
        sampling_rate_hz=200,
        window_seconds=5,
        candidate_step_seconds=0.025,
        samples_per_class=1,
    )

    assert dataset.candidate_step_samples == 5
    assert dataset.number_of_candidates("Basketball") == 3


def test_bias_augmentation_is_constant_with_separate_sensor_ranges():
    augmentation = ImuAugmentation(
        add_bias_noise=True,
        accel_bias_range=0.2,
        gyro_bias_range=0.02,
    )
    values = np.zeros((100, 6), dtype=np.float32)

    augmented = augmentation(values, np.random.default_rng(7))

    np.testing.assert_allclose(augmented, np.broadcast_to(augmented[0], values.shape))
    assert np.max(np.abs(augmented[:, :3])) <= 0.2
    assert np.max(np.abs(augmented[:, 3:])) <= 0.02


def test_tilt_augmentation_rotates_accel_and_gyro_by_same_matrix():
    augmentation = ImuAugmentation(
        add_gravity_noise=True,
        gravity_noise_theta_range=5.0,
    )
    values = np.tile(
        np.asarray([1.0, 2.0, 3.0, 2.0, 4.0, 6.0], dtype=np.float32),
        (20, 1),
    )

    augmented = augmentation(values, np.random.default_rng(11))

    np.testing.assert_allclose(augmented[:, 3:], augmented[:, :3] * 2, atol=1e-6)
    np.testing.assert_allclose(
        np.linalg.norm(augmented[:, :3], axis=1),
        np.linalg.norm(values[:, :3], axis=1),
        atol=1e-6,
    )


def test_evaluation_dataset_uses_fixed_non_overlapping_windows(tmp_path):
    values = np.zeros((2200, 6), dtype=np.float32)
    recording = _recording(
        tmp_path,
        "PUT_ON",
        values,
        split="val",
        uid="event-validation",
    )

    dataset = EvaluationJointDataset(
        [recording],
        split="val",
        sampling_rate_hz=200,
        window_seconds=5,
        evaluation_step_seconds=5,
    )

    assert len(dataset) == 2
    assert dataset.window_start_samples == (0, 1000)
