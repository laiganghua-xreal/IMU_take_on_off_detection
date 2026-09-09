"""Virtual-window datasets for joint Ego activity and Xreal event training."""

from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from egocharm.dataset import CLASS_NAMES as EGO_CLASS_NAMES


JOINT_CLASS_NAMES = EGO_CLASS_NAMES + ("PUT_ON", "TAKE_OFF")


@dataclass(frozen=True)
class _WindowSource:
    recording: object
    segment_start: int
    number_of_windows: int


@dataclass(frozen=True)
class _EvaluationReference:
    recording: object
    start_sample: int
    class_index: int


class _MemoryMapStore:
    """Keep a small worker-local LRU cache of open NPY memory maps."""

    def __init__(self, maximum_open_files=16):
        self.maximum_open_files = maximum_open_files
        self._arrays = OrderedDict()

    def load(self, path):
        resolved_path = str(path)
        if resolved_path in self._arrays:
            values = self._arrays.pop(resolved_path)
            self._arrays[resolved_path] = values
            return values

        values = np.load(resolved_path, mmap_mode="r")
        self._arrays[resolved_path] = values
        if len(self._arrays) > self.maximum_open_files:
            self._arrays.popitem(last=False)
        return values


@dataclass(frozen=True)
class ImuAugmentation:
    """Train-only bias, small tilt, and Gaussian IMU augmentation."""

    add_bias_noise: bool = False
    accel_bias_range: float = 0.0
    gyro_bias_range: float = 0.0
    add_gravity_noise: bool = False
    gravity_noise_theta_range: float = 0.0
    add_gaussian_noise: bool = False
    accel_noise_sigma: float = 0.0
    gyro_noise_sigma: float = 0.0

    def __call__(self, values, random_generator):
        augmented = np.array(values, dtype=np.float32, copy=True)

        if self.add_bias_noise:
            bias = np.empty(6, dtype=np.float32)
            bias[:3] = random_generator.uniform(
                -self.accel_bias_range,
                self.accel_bias_range,
                size=3,
            )
            bias[3:] = random_generator.uniform(
                -self.gyro_bias_range,
                self.gyro_bias_range,
                size=3,
            )
            augmented += bias

        if self.add_gravity_noise:
            axis_angle = random_generator.uniform(0.0, 2.0 * np.pi)
            axis = np.asarray(
                [np.cos(axis_angle), np.sin(axis_angle), 0.0],
                dtype=np.float64,
            )
            theta = np.deg2rad(
                random_generator.uniform(0.0, self.gravity_noise_theta_range)
            )
            axis_cross = np.asarray(
                [
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ]
            )
            rotation = (
                np.eye(3)
                + np.sin(theta) * axis_cross
                + (1.0 - np.cos(theta)) * (axis_cross @ axis_cross)
            )
            augmented[:, :3] = augmented[:, :3] @ rotation.T
            augmented[:, 3:] = augmented[:, 3:] @ rotation.T

        if self.add_gaussian_noise:
            if self.accel_noise_sigma > 0:
                augmented[:, :3] += random_generator.normal(
                    0.0,
                    self.accel_noise_sigma,
                    size=augmented[:, :3].shape,
                )
            if self.gyro_noise_sigma > 0:
                augmented[:, 3:] += random_generator.normal(
                    0.0,
                    self.gyro_noise_sigma,
                    size=augmented[:, 3:].shape,
                )

        return augmented


def _to_hierarchical_tensor(values, sampling_rate_hz, window_seconds):
    # (S*T, C) -> (S, T, C) -> (S, C, T)
    values = values.reshape(window_seconds, sampling_rate_hz, 6)
    values = np.ascontiguousarray(values.transpose(0, 2, 1))
    return torch.from_numpy(values)


class BalancedJointDataset(Dataset):
    """Sample equal counts from nine classes without expanding window indices."""

    def __init__(
        self,
        recordings,
        sampling_rate_hz,
        window_seconds,
        candidate_step_seconds,
        samples_per_class,
        augmentation=None,
    ):
        super().__init__()
        self.sampling_rate_hz = int(sampling_rate_hz)
        self.window_seconds = int(window_seconds)
        self.window_samples = self.sampling_rate_hz * self.window_seconds
        self.candidate_step_samples = int(
            round(self.sampling_rate_hz * float(candidate_step_seconds))
        )
        self.samples_per_class = int(samples_per_class)
        self.augmentation = augmentation
        self._memory_maps = _MemoryMapStore()

        if self.candidate_step_samples <= 0:
            raise ValueError("候选窗口步进必须至少包含一个采样点")
        if self.samples_per_class <= 0:
            raise ValueError("samples_per_class 必须大于零")

        self._sources_by_class = {name: [] for name in JOINT_CLASS_NAMES}
        for recording in recordings:
            if recording.split != "train":
                continue
            if recording.sampling_rate_hz != self.sampling_rate_hz:
                raise ValueError(f"采样率不匹配: {recording.recording_uid}")
            if recording.label not in self._sources_by_class:
                continue
            for segment_start, segment_stop in recording.segment_bounds:
                segment_length = segment_stop - segment_start
                if segment_length < self.window_samples:
                    continue
                number_of_windows = (
                    (segment_length - self.window_samples)
                    // self.candidate_step_samples
                    + 1
                )
                self._sources_by_class[recording.label].append(
                    _WindowSource(recording, segment_start, number_of_windows)
                )

        self._cumulative_counts = {}
        for class_name, sources in self._sources_by_class.items():
            if not sources:
                raise ValueError(f"训练数据缺少类别: {class_name}")
            self._cumulative_counts[class_name] = np.cumsum(
                [source.number_of_windows for source in sources],
                dtype=np.int64,
            )

    @property
    def class_labels(self):
        return [index % len(JOINT_CLASS_NAMES) for index in range(len(self))]

    def number_of_candidates(self, class_name):
        return int(self._cumulative_counts[class_name][-1])

    def __len__(self):
        return self.samples_per_class * len(JOINT_CLASS_NAMES)

    def __getitem__(self, item_index):
        class_index = int(item_index) % len(JOINT_CLASS_NAMES)
        class_name = JOINT_CLASS_NAMES[class_index]
        cumulative_counts = self._cumulative_counts[class_name]

        random_seed = torch.randint(0, 2**32 - 1, ()).item()
        random_generator = np.random.default_rng(random_seed)
        virtual_index = int(
            random_generator.integers(0, int(cumulative_counts[-1]))
        )
        source_index = bisect_right(cumulative_counts, virtual_index)
        previous_count = 0 if source_index == 0 else cumulative_counts[source_index - 1]
        source = self._sources_by_class[class_name][source_index]
        local_window_index = virtual_index - int(previous_count)
        start_sample = (
            source.segment_start
            + local_window_index * self.candidate_step_samples
        )
        stop_sample = start_sample + self.window_samples

        cached_values = self._memory_maps.load(source.recording.values_path)
        values = np.array(
            cached_values[start_sample:stop_sample],
            dtype=np.float32,
            copy=True,
        )
        if self.augmentation is not None:
            values = self.augmentation(values, random_generator)

        features = _to_hierarchical_tensor(
            values,
            self.sampling_rate_hz,
            self.window_seconds,
        )
        label = torch.tensor(class_index, dtype=torch.long)
        return features, label


class EvaluationJointDataset(Dataset):
    """Use deterministic, low-overlap windows for validation or testing."""

    def __init__(
        self,
        recordings,
        split,
        sampling_rate_hz,
        window_seconds,
        evaluation_step_seconds,
        maximum_samples_per_class=None,
    ):
        super().__init__()
        self.sampling_rate_hz = int(sampling_rate_hz)
        self.window_seconds = int(window_seconds)
        self.window_samples = self.sampling_rate_hz * self.window_seconds
        step_samples = int(
            round(self.sampling_rate_hz * float(evaluation_step_seconds))
        )
        self._memory_maps = _MemoryMapStore()

        references_by_class = {index: [] for index in range(len(JOINT_CLASS_NAMES))}
        class_to_index = {
            class_name: class_index
            for class_index, class_name in enumerate(JOINT_CLASS_NAMES)
        }
        for recording in recordings:
            if recording.split != split or recording.label not in class_to_index:
                continue
            class_index = class_to_index[recording.label]
            for segment_start, segment_stop in recording.segment_bounds:
                final_start = segment_stop - self.window_samples
                for start_sample in range(
                    segment_start,
                    final_start + 1,
                    step_samples,
                ):
                    references_by_class[class_index].append(
                        _EvaluationReference(recording, start_sample, class_index)
                    )

        self.references = []
        for class_index in range(len(JOINT_CLASS_NAMES)):
            class_references = references_by_class[class_index]
            if (
                maximum_samples_per_class is not None
                and len(class_references) > maximum_samples_per_class
            ):
                selected_indices = np.linspace(
                    0,
                    len(class_references) - 1,
                    num=maximum_samples_per_class,
                    dtype=np.int64,
                )
                class_references = [
                    class_references[index] for index in selected_indices
                ]
            self.references.extend(class_references)

    @property
    def class_labels(self):
        return [reference.class_index for reference in self.references]

    @property
    def window_start_samples(self):
        return tuple(reference.start_sample for reference in self.references)

    def __len__(self):
        return len(self.references)

    def __getitem__(self, item_index):
        reference = self.references[item_index]
        stop_sample = reference.start_sample + self.window_samples
        cached_values = self._memory_maps.load(reference.recording.values_path)
        values = np.array(
            cached_values[reference.start_sample:stop_sample],
            dtype=np.float32,
            copy=True,
        )
        features = _to_hierarchical_tensor(
            values,
            self.sampling_rate_hz,
            self.window_seconds,
        )
        label = torch.tensor(reference.class_index, dtype=torch.long)
        return features, label


def create_joint_data_loader(
    dataset,
    batch_size,
    shuffle,
    number_of_workers,
    seed,
):
    """Build a reproducible DataLoader for a joint dataset."""
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(number_of_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=int(number_of_workers) > 0,
        generator=generator,
    )
