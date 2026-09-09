import numpy as np

from egocharm.joint_preprocessing import resample_uniform_imu


def _frequency_amplitude(values, sampling_rate_hz, frequency_hz):
    spectrum = np.fft.rfft(values)
    frequencies = np.fft.rfftfreq(len(values), d=1.0 / sampling_rate_hz)
    frequency_index = np.argmin(np.abs(frequencies - frequency_hz))
    return 2.0 * np.abs(spectrum[frequency_index]) / len(values)


def test_resampling_preserves_motion_and_filters_above_new_nyquist():
    source_rate_hz = 1000
    target_rate_hz = 200
    timestamps = np.arange(source_rate_hz, dtype=np.float64) / source_rate_hz
    low_frequency = np.sin(2.0 * np.pi * 20.0 * timestamps)
    high_frequency = np.sin(2.0 * np.pi * 280.0 * timestamps)
    values = np.tile((low_frequency + high_frequency)[:, None], (1, 6))

    output_timestamps, output_values = resample_uniform_imu(
        timestamps,
        values,
        target_rate_hz,
    )

    assert output_timestamps.shape == (200,)
    assert output_values.shape == (200, 6)
    assert output_values.dtype == np.float32
    assert _frequency_amplitude(output_values[:, 0], 200, 20) > 0.9
    assert _frequency_amplitude(output_values[:, 0], 200, 80) < 0.05
