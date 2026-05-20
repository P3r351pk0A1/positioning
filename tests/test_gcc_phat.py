import numpy as np
import pytest

from positioning.config import load_config
from positioning.gcc_phat import estimate_tdoa_gcc_phat, gcc_phat_delay
from positioning.models import AudioFrame


def _pulse_channel(length: int, start: int, pulse: np.ndarray) -> tuple[float, ...]:
    data = np.zeros(length, dtype=np.float64)
    data[start : start + pulse.size] = pulse
    return tuple(float(value) for value in data)


def test_gcc_phat_delay_recovers_integer_sample_shift():
    sample_rate = 48_000
    rng = np.random.default_rng(42)
    pulse = rng.normal(0.0, 1.0, size=64)
    reference = _pulse_channel(512, 120, pulse)
    delayed = _pulse_channel(512, 137, pulse)

    delay, diagnostics = gcc_phat_delay(delayed, reference, sample_rate, max_tau=0.001, interpolation=1)

    assert delay == pytest.approx(17 / sample_rate, abs=0.5 / sample_rate)
    assert diagnostics.quality > 0.5
    assert diagnostics.peak_to_sidelobe > 4.0


def test_estimate_tdoa_gcc_phat_returns_all_microphone_pairs():
    config = load_config()
    sample_rate = config.audio.sample_rate
    rng = np.random.default_rng(7)
    pulse = rng.normal(0.0, 1.0, size=80)
    sample_offsets = [50, 64, 74, 59]
    channels = tuple(_pulse_channel(1024, 180 + offset, pulse) for offset in sample_offsets)
    frame = AudioFrame(channels, sample_rate)

    result = estimate_tdoa_gcc_phat(
        frame,
        config.microphone_array,
        config.environment.sound_speed,
        interpolation=1,
    )

    delays = {item.pair: item.delay_seconds for item in result.estimate.pair_delays}
    assert len(delays) == 6
    for i, j in delays:
        expected = (sample_offsets[i] - sample_offsets[j]) / sample_rate
        assert delays[(i, j)] == pytest.approx(expected, abs=0.5 / sample_rate)
    assert all(item.quality > 0.5 for item in result.estimate.pair_delays)
