"""Synthetic data generation for validation and experiments."""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np

from .geometry import Vec3, distance
from .localization import DEFAULT_SPEED_OF_SOUND
from .models import AppConfig, AudioFrame


def make_arrival_time(
    microphones: Sequence[Vec3],
    target: Vec3,
    emission_time: float,
    *,
    speed_of_sound: float = DEFAULT_SPEED_OF_SOUND,
    noise_std: float = 0.0,
    rng: random.Random | None = None,
) -> list[float]:
    if rng is None:
        rng = random.Random()

    arrival: list[float] = []
    for microphone in microphones:
        noise = rng.gauss(0.0, noise_std) if noise_std > 0.0 else 0.0
        arrival.append(emission_time + distance(target, microphone) / speed_of_sound + noise)
    return arrival


def generate_synthetic_signal(
    config: AppConfig,
    target: Vec3,
    *,
    duration_s: float = 1.0,
    source_kind: str = "sine",
    source_frequency_hz: float = 700.0,
    source_amplitude: float = 1.0,
    emission_time_s: float = 0.02,
    noise_std: float = 0.0,
    attenuation: bool = True,
    outlier_channel: int | None = None,
    outlier_time_s: float = 0.2,
    outlier_amplitude: float = 0.0,
    reflection_delay_s: float = 0.0,
    reflection_gain: float = 0.0,
    seed: int = 42,
) -> tuple[AudioFrame, dict[str, object]]:
    """Generate a reproducible four-channel synthetic acoustic signal."""
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if source_frequency_hz <= 0.0:
        raise ValueError("source_frequency_hz must be positive")
    supported_source_kinds = ("sine", "pulse", "chirp", "multitone_burst", "band_noise", "am_fm_tone")
    if source_kind not in supported_source_kinds:
        raise ValueError(f"source_kind must be one of {supported_source_kinds}")
    if source_amplitude <= 0.0:
        raise ValueError("source_amplitude must be positive")
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")
    if reflection_delay_s < 0.0:
        raise ValueError("reflection_delay_s must be non-negative")
    if outlier_channel is not None and not 0 <= outlier_channel < 4:
        raise ValueError("outlier_channel must be in range [0, 3]")

    sample_rate = config.audio.sample_rate
    sample_count = int(round(duration_s * sample_rate))
    time = np.arange(sample_count, dtype=np.float64) / float(sample_rate)
    speed = config.environment.sound_speed
    arrivals = make_arrival_time(config.microphone_array.microphones, target, emission_time_s, speed_of_sound=speed)
    rng = np.random.default_rng(seed)

    channels: list[np.ndarray] = []
    channel_metadata = []
    for microphone, arrival in zip(config.microphone_array.microphones, arrivals):
        propagation_distance = distance(target, microphone)
        gain = source_amplitude / max(propagation_distance, 1.0) if attenuation else source_amplitude
        shifted_time = time - arrival
        direct = _source_waveform(shifted_time, source_kind, source_frequency_hz, gain)
        signal = direct
        if reflection_gain > 0.0 and reflection_delay_s > 0.0:
            reflected_time = time - arrival - reflection_delay_s
            reflected = _source_waveform(reflected_time, source_kind, source_frequency_hz, gain * reflection_gain)
            signal = signal + reflected
        if noise_std > 0.0:
            signal = signal + rng.normal(0.0, noise_std, size=sample_count)
        channels.append(signal.astype(np.float64))
        channel_metadata.append({"arrival_time_s": arrival, "distance_m": propagation_distance, "gain": gain})

    if outlier_channel is not None and outlier_amplitude != 0.0:
        outlier_index = int(round(outlier_time_s * sample_rate))
        if 0 <= outlier_index < sample_count:
            channels[outlier_channel][outlier_index] += outlier_amplitude

    frame = AudioFrame(tuple(tuple(float(x) for x in channel) for channel in channels), sample_rate)  # type: ignore[arg-type]
    metadata: dict[str, object] = {
        "kind": "synthetic_signal",
        "sample_rate": sample_rate,
        "duration_s": duration_s,
        "sample_count": sample_count,
        "target": {"x": target.x, "y": target.y, "z": target.z},
        "source": {
            "kind": source_kind,
            "frequency_hz": source_frequency_hz,
            "amplitude": source_amplitude,
            "emission_time_s": emission_time_s,
        },
        "environment": {
            "temperature_c": config.environment.temperature_c,
            "humidity_percent": config.environment.humidity_percent,
            "sound_speed_m_s": speed,
        },
        "noise_std": noise_std,
        "attenuation": attenuation,
        "reflection": {"delay_s": reflection_delay_s, "gain": reflection_gain},
        "outlier": {
            "channel": outlier_channel,
            "time_s": outlier_time_s,
            "amplitude": outlier_amplitude,
        },
        "channels": channel_metadata,
    }
    return frame, metadata


def generate_synthetic_trajectory_signal(
    config: AppConfig,
    start: Vec3,
    end: Vec3,
    *,
    duration_s: float = 1.0,
    point_count: int = 12,
    source_kind: str = "pulse",
    source_frequency_hz: float = 700.0,
    source_amplitude: float = 1.0,
    first_emission_time_s: float = 0.08,
    last_emission_time_s: float | None = None,
    noise_std: float = 0.0,
    attenuation: bool = True,
    seed: int = 42,
) -> tuple[AudioFrame, dict[str, object]]:
    """Generate a moving-source synthetic signal with trajectory metadata."""
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive")
    if point_count < 2:
        raise ValueError("point_count must be at least two")
    if source_frequency_hz <= 0.0:
        raise ValueError("source_frequency_hz must be positive")
    supported_source_kinds = ("sine", "pulse", "chirp", "multitone_burst", "band_noise", "am_fm_tone")
    if source_kind not in supported_source_kinds:
        raise ValueError(f"source_kind must be one of {supported_source_kinds}")
    if source_amplitude <= 0.0:
        raise ValueError("source_amplitude must be positive")
    if first_emission_time_s < 0.0:
        raise ValueError("first_emission_time_s must be non-negative")
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")

    sample_rate = config.audio.sample_rate
    sample_count = int(round(duration_s * sample_rate))
    time = np.arange(sample_count, dtype=np.float64) / float(sample_rate)
    speed = config.environment.sound_speed
    if last_emission_time_s is None:
        last_emission_time_s = max(first_emission_time_s, duration_s - 0.12)
    if last_emission_time_s <= first_emission_time_s:
        raise ValueError("last_emission_time_s must be greater than first_emission_time_s")
    if last_emission_time_s >= duration_s:
        raise ValueError("last_emission_time_s must be less than duration_s")

    channels = [np.zeros(sample_count, dtype=np.float64) for _ in range(4)]
    trajectory: list[dict[str, float]] = []
    events: list[dict[str, object]] = []
    emission_times = np.linspace(first_emission_time_s, last_emission_time_s, point_count)

    for index, emission_time in enumerate(emission_times):
        fraction = index / float(point_count - 1)
        target = _lerp_vec3(start, end, fraction)
        trajectory.append(
            {
                "timestamp": float(emission_time),
                "x": target.x,
                "y": target.y,
                "z": target.z,
            }
        )
        event_channels = []
        for microphone_index, microphone in enumerate(config.microphone_array.microphones):
            propagation_distance = distance(target, microphone)
            arrival = float(emission_time + propagation_distance / speed)
            gain = source_amplitude / max(propagation_distance, 1.0) if attenuation else source_amplitude
            channels[microphone_index] += _source_waveform(time - arrival, source_kind, source_frequency_hz, gain)
            event_channels.append({"arrival_time_s": arrival, "distance_m": propagation_distance, "gain": gain})
        events.append({"index": index, "emission_time_s": float(emission_time), "target": trajectory[-1], "channels": event_channels})

    rng = np.random.default_rng(seed)
    if noise_std > 0.0:
        for index in range(4):
            channels[index] = channels[index] + rng.normal(0.0, noise_std, size=sample_count)

    frame = AudioFrame(tuple(tuple(float(x) for x in channel) for channel in channels), sample_rate)  # type: ignore[arg-type]
    metadata: dict[str, object] = {
        "kind": "synthetic_trajectory_signal",
        "sample_rate": sample_rate,
        "duration_s": duration_s,
        "sample_count": sample_count,
        "target": {"x": start.x, "y": start.y, "z": start.z},
        "trajectory": trajectory,
        "source": {
            "kind": source_kind,
            "frequency_hz": source_frequency_hz,
            "amplitude": source_amplitude,
            "first_emission_time_s": first_emission_time_s,
            "last_emission_time_s": last_emission_time_s,
        },
        "environment": {
            "temperature_c": config.environment.temperature_c,
            "humidity_percent": config.environment.humidity_percent,
            "sound_speed_m_s": speed,
        },
        "noise_std": noise_std,
        "attenuation": attenuation,
        "events": events,
    }
    return frame, metadata


def _source_waveform(shifted_time: np.ndarray, source_kind: str, frequency_hz: float, gain: float) -> np.ndarray:
    if source_kind == "sine":
        duration_s = 0.03
        t = shifted_time
        active = (t >= 0.0) & (t <= duration_s)
        local_t = np.clip(t, 0.0, duration_s)
        window = 0.5 - 0.5 * np.cos(2.0 * np.pi * local_t / duration_s)
        return np.where(active, gain * np.sin(2.0 * np.pi * frequency_hz * local_t) * window, 0.0)

    if source_kind == "chirp":
        duration_s = 0.02
        t = shifted_time
        active = (t >= 0.0) & (t <= duration_s)
        local_t = np.clip(t, 0.0, duration_s)
        f0 = max(80.0, 0.5 * frequency_hz)
        f1 = max(f0 + 50.0, 2.5 * frequency_hz)
        k = (f1 - f0) / duration_s
        phase = 2.0 * np.pi * (f0 * local_t + 0.5 * k * local_t * local_t)
        window = 0.5 - 0.5 * np.cos(2.0 * np.pi * local_t / duration_s)
        signal = gain * np.sin(phase) * window
        return np.where(active, signal, 0.0)

    if source_kind == "multitone_burst":
        duration_s = 0.025
        t = shifted_time
        active = (t >= 0.0) & (t <= duration_s)
        local_t = np.clip(t, 0.0, duration_s)
        freqs = (0.85 * frequency_hz, 1.0 * frequency_hz, 1.35 * frequency_hz)
        signal = np.zeros_like(local_t)
        for ratio, freq in enumerate(freqs):
            phase_shift = ratio * np.pi / 3.0
            signal += np.sin(2.0 * np.pi * max(30.0, freq) * local_t + phase_shift)
        signal /= float(len(freqs))
        am = 0.6 + 0.4 * np.sin(2.0 * np.pi * 22.0 * local_t)
        window = 0.5 - 0.5 * np.cos(2.0 * np.pi * local_t / duration_s)
        return np.where(active, gain * signal * am * window, 0.0)

    if source_kind == "band_noise":
        duration_s = 0.025
        t = shifted_time
        active = (t >= 0.0) & (t <= duration_s)
        local_t = np.clip(t, 0.0, duration_s)
        freqs = (
            0.55 * frequency_hz,
            0.8 * frequency_hz,
            1.1 * frequency_hz,
            1.45 * frequency_hz,
            1.9 * frequency_hz,
        )
        signal = np.zeros_like(local_t)
        for idx, freq in enumerate(freqs):
            phase = (idx + 1) * 1.61803398875
            signal += np.sin(2.0 * np.pi * max(20.0, freq) * local_t + phase)
        signal /= float(len(freqs))
        window = 0.5 - 0.5 * np.cos(2.0 * np.pi * local_t / duration_s)
        return np.where(active, gain * signal * window, 0.0)

    if source_kind == "am_fm_tone":
        duration_s = 0.035
        t = shifted_time
        active = (t >= 0.0) & (t <= duration_s)
        local_t = np.clip(t, 0.0, duration_s)
        fm_dev1 = 0.25 * frequency_hz
        fm_dev2 = 0.12 * frequency_hz
        fm_rate1 = 16.0
        fm_rate2 = 31.0
        phase_base = 2.0 * np.pi * (
            frequency_hz * local_t
            - (fm_dev1 / (2.0 * np.pi * fm_rate1)) * np.cos(2.0 * np.pi * fm_rate1 * local_t)
            - (fm_dev2 / (2.0 * np.pi * fm_rate2)) * np.cos(2.0 * np.pi * fm_rate2 * local_t)
        )
        # Harmonic stack keeps tone-like character but broadens spectrum for robust TDOA.
        tone = (
            0.62 * np.sin(phase_base)
            + 0.25 * np.sin(2.0 * phase_base + 0.7)
            + 0.13 * np.sin(3.0 * phase_base + 1.4)
        )
        am = 0.50 + 0.50 * np.sin(2.0 * np.pi * 13.0 * local_t + np.pi / 7.0)
        window = 0.5 - 0.5 * np.cos(2.0 * np.pi * local_t / duration_s)
        return np.where(active, gain * tone * am * window, 0.0)

    pulse_center_s = 0.006
    pulse_sigma_s = max(0.00025, min(0.0012, 1.0 / (2.0 * np.pi * frequency_hz)))
    normalized = (shifted_time - pulse_center_s) / pulse_sigma_s
    pulse = gain * (1.0 - normalized * normalized) * np.exp(-0.5 * normalized * normalized)
    return np.where((shifted_time >= 0.0) & (shifted_time <= pulse_center_s + 5.0 * pulse_sigma_s), pulse, 0.0)


def _lerp_vec3(start: Vec3, end: Vec3, fraction: float) -> Vec3:
    return Vec3(
        start.x + (end.x - start.x) * fraction,
        start.y + (end.y - start.y) * fraction,
        start.z + (end.z - start.z) * fraction,
    )
