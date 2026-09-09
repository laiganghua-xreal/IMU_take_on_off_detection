"""Signal processing shared by joint Ego and Xreal training caches."""

from math import gcd

import numpy as np
from scipy.signal import resample_poly


def resample_uniform_imu(timestamps_seconds, values, sampling_rate_hz):
    """Resample near-uniform IMU with a polyphase anti-aliasing filter."""
    timestamps = np.asarray(timestamps_seconds, dtype=np.float64)
    samples = np.asarray(values)
    if timestamps.ndim != 1 or len(timestamps) < 2:
        raise ValueError("时间戳必须是至少包含两个采样的一维数组")
    if samples.ndim != 2 or samples.shape != (len(timestamps), 6):
        raise ValueError("IMU values 必须是 (N, 6)")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz 必须大于零")

    duration_seconds = timestamps[-1] - timestamps[0]
    source_rate_hz = int(round((len(timestamps) - 1) / duration_seconds))
    common_divisor = gcd(source_rate_hz, int(sampling_rate_hz))
    up = int(sampling_rate_hz) // common_divisor
    down = source_rate_hz // common_divisor
    output_values = resample_poly(
        samples,
        up=up,
        down=down,
        axis=0,
    ).astype(np.float32, copy=False)
    output_timestamps = timestamps[0] + (
        np.arange(len(output_values), dtype=np.float64) / sampling_rate_hz
    )
    return output_timestamps, output_values
