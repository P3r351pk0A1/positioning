"""Windowing and preprocessing for four-channel acoustic frames."""

from __future__ import annotations

import numpy as np

from .models import AudioConfig, AudioFrame, PreprocessDiagnostics


def split_frames(frame: AudioFrame, window_size: int, overlap: float) -> list[AudioFrame]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in the range [0, 1)")

    hop = max(1, int(round(window_size * (1.0 - overlap))))
    data = _to_array(frame)
    windows: list[AudioFrame] = []
    for start in range(0, frame.sample_count - window_size + 1, hop):
        chunk = data[:, start : start + window_size]
        timestamp = frame.timestamp + start / float(frame.sample_rate)
        windows.append(_from_array(chunk, frame.sample_rate, timestamp))
    return windows


def preprocess_frame(
    frame: AudioFrame,
    audio_config: AudioConfig,
    *,
    min_mean_rms: float = 1e-3,
    min_mean_abs_correlation: float = 0.05,
) -> tuple[AudioFrame, PreprocessDiagnostics]:
    data = _to_array(frame)
    data = remove_dc(data)
    data = apply_window(data, "hann")
    data = apply_frequency_mask(data, frame.sample_rate, audio_config.frequency_band)
    diagnostics = compute_diagnostics(data, min_mean_rms, min_mean_abs_correlation)
    data = normalize_channels(data)
    return _from_array(data, frame.sample_rate, frame.timestamp), diagnostics


def remove_dc(data: np.ndarray) -> np.ndarray:
    return data - np.mean(data, axis=1, keepdims=True)


def normalize_channels(data: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(data), axis=1, keepdims=True)
    peak = np.where(peak > 0.0, peak, 1.0)
    return data / peak


def apply_window(data: np.ndarray, kind: str = "hann") -> np.ndarray:
    if kind != "hann":
        raise ValueError("only hann window is currently supported")
    if data.shape[1] == 0:
        return data
    return data * np.hanning(data.shape[1])


def apply_frequency_mask(data: np.ndarray, sample_rate: int, frequency_band: tuple[float, float]) -> np.ndarray:
    low, high = frequency_band
    if low < 0.0 or high <= low:
        raise ValueError("frequency_band must be [low, high] with 0 <= low < high")
    if high > sample_rate / 2.0:
        raise ValueError("frequency_band high value must not exceed Nyquist frequency")

    spectrum = np.fft.rfft(data, axis=1)
    frequencies = np.fft.rfftfreq(data.shape[1], d=1.0 / sample_rate)
    mask = (frequencies >= low) & (frequencies <= high)
    spectrum = spectrum * mask[np.newaxis, :]
    return np.fft.irfft(spectrum, n=data.shape[1], axis=1).real


def compute_diagnostics(
    data: np.ndarray,
    min_mean_rms: float = 1e-4,
    min_mean_abs_correlation: float = 0.01,
) -> PreprocessDiagnostics:
    rms = np.sqrt(np.mean(data * data, axis=1))
    mean_rms = float(np.mean(rms))
    correlations = []
    for i in range(4):
        for j in range(i + 1, 4):
            if rms[i] == 0.0 or rms[j] == 0.0:
                correlations.append(0.0)
            else:
                corr = float(np.corrcoef(data[i], data[j])[0, 1])
                if np.isfinite(corr):
                    correlations.append(abs(corr))
                else:
                    correlations.append(0.0)
    mean_abs_correlation = float(np.mean(correlations)) if correlations else 0.0
    informative = mean_rms >= min_mean_rms and mean_abs_correlation >= min_mean_abs_correlation
    reason = ""
    if mean_rms < min_mean_rms:
        reason = "low_energy"
    elif mean_abs_correlation < min_mean_abs_correlation:
        reason = "low_interchannel_correlation"
    return PreprocessDiagnostics(
        rms_by_channel=tuple(float(value) for value in rms),  # type: ignore[arg-type]
        mean_rms=mean_rms,
        mean_abs_correlation=mean_abs_correlation,
        informative=informative,
        reason=reason,
    )


def _to_array(frame: AudioFrame) -> np.ndarray:
    return np.asarray(frame.channels, dtype=np.float64)


def _from_array(data: np.ndarray, sample_rate: int, timestamp: float) -> AudioFrame:
    return AudioFrame(tuple(tuple(float(x) for x in channel) for channel in data), sample_rate, timestamp)  # type: ignore[arg-type]
