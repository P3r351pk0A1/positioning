"""GCC-PHAT delay estimation for four-channel audio frames."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

import numpy as np

from .geometry import distance
from .models import AudioFrame, MicrophoneArray, PairDelay, TdoaEstimate
from .tdoa import TDOA_PAIRS


@dataclass(frozen=True)
class GccPhatPairDiagnostics:
    pair: tuple[int, int]
    delay_seconds: float
    lag_samples: float
    peak_value: float
    peak_to_sidelobe: float
    quality: float
    policy: str = "manual"
    subbands_used: int = 1
    continuity_applied: bool = False


@dataclass(frozen=True)
class GccPhatEstimate:
    estimate: TdoaEstimate
    diagnostics: tuple[GccPhatPairDiagnostics, ...]
    policy_counts: dict[str, int] | None = None


def gcc_phat_delay(
    signal: Sequence[float],
    reference: Sequence[float],
    sample_rate: int,
    *,
    max_tau: float | None = None,
    interpolation: int = 8,
    phat_weight: float = 1.0,
) -> tuple[float, GccPhatPairDiagnostics]:
    """Estimate delay between two channels using GCC-PHAT.

    Positive delay means that ``signal`` arrives later than ``reference``.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if interpolation < 1:
        raise ValueError("interpolation must be positive")
    if not 0.0 <= phat_weight <= 1.0:
        raise ValueError("phat_weight must be in the range [0, 1]")

    sig = _prepare_channel(signal)
    ref = _prepare_channel(reference)
    if sig.size != ref.size:
        raise ValueError("signal and reference must have the same length")
    if sig.size < 2:
        raise ValueError("at least two samples are required for GCC-PHAT")
    if not np.any(sig) or not np.any(ref):
        diagnostics = GccPhatPairDiagnostics((0, 1), 0.0, 0.0, 0.0, 0.0, 0.0)
        return 0.0, diagnostics

    fft_size = _next_power_of_two(sig.size + ref.size)
    sig_fft = np.fft.rfft(sig, n=fft_size)
    ref_fft = np.fft.rfft(ref, n=fft_size)
    cross_power = sig_fft * np.conj(ref_fft)
    cross_power_abs = np.abs(cross_power)
    if phat_weight > 0.0:
        cross_power = cross_power / np.maximum(cross_power_abs, 1e-15) ** phat_weight

    interpolation_factor = max(1, int(interpolation))
    correlation_size = fft_size * interpolation_factor
    correlation = np.fft.irfft(cross_power, n=correlation_size)
    max_shift = correlation_size // 2
    if max_tau is not None:
        if max_tau <= 0.0:
            raise ValueError("max_tau must be positive")
        max_shift = min(max_shift, int(round(interpolation_factor * sample_rate * max_tau)))
    shifted = np.concatenate((correlation[-max_shift:], correlation[: max_shift + 1]))
    abs_shifted = np.abs(shifted)
    peak_index = int(np.argmax(abs_shifted))
    shift = peak_index - max_shift
    fractional = _parabolic_peak_offset(abs_shifted, peak_index) if interpolation_factor > 1 else 0.0
    lag_samples = (shift + fractional) / float(interpolation_factor)
    delay = lag_samples / float(sample_rate)

    peak_value = float(abs_shifted[peak_index])
    peak_to_sidelobe = _peak_to_sidelobe(abs_shifted, peak_index, guard=max(1, interpolation))
    quality = _quality_from_peak_ratio(peak_to_sidelobe)
    diagnostics = GccPhatPairDiagnostics(
        pair=(0, 1),
        delay_seconds=delay,
        lag_samples=lag_samples,
        peak_value=peak_value,
        peak_to_sidelobe=peak_to_sidelobe,
        quality=quality,
    )
    return delay, diagnostics


def estimate_tdoa_gcc_phat(
    frame: AudioFrame,
    microphone_array: MicrophoneArray,
    speed_of_sound: float,
    *,
    interpolation: int = 8,
    min_quality: float = 0.0,
    phat_weight: float = 1.0,
    subbands: int = 1,
    continuity_reference: dict[tuple[int, int], float] | None = None,
    policy_mode: str = "manual",
) -> GccPhatEstimate:
    """Estimate all six TDOA pairs from a four-channel audio frame."""
    if speed_of_sound <= 0.0:
        raise ValueError("speed_of_sound must be positive")
    if frame.sample_count < 2:
        raise ValueError("audio frame must contain at least two samples")

    channels = [np.asarray(channel, dtype=np.float64) for channel in frame.channels]
    pair_delays: list[PairDelay] = []
    diagnostics: list[GccPhatPairDiagnostics] = []
    policy_counts: dict[str, int] = {"narrowband": 0, "wideband": 0, "manual": 0}
    for pair in TDOA_PAIRS:
        i, j = pair
        max_tau = distance(microphone_array.microphones[i], microphone_array.microphones[j]) / speed_of_sound
        policy_used = "manual"
        if policy_mode == "auto":
            policy_used = _classify_pair_policy(channels[i], channels[j], frame.sample_rate)
            policy_counts[policy_used] = policy_counts.get(policy_used, 0) + 1
        else:
            policy_counts["manual"] = policy_counts.get("manual", 0) + 1
        delay, raw_diag = _estimate_pair_delay(
            channels[i],
            channels[j],
            frame.sample_rate,
            max_tau=max_tau,
            interpolation=interpolation,
            phat_weight=phat_weight,
            subbands=subbands,
            pair=pair,
            continuity_reference=continuity_reference,
            policy_used=policy_used,
        )
        quality = raw_diag.quality
        if quality < min_quality:
            weight = 0.0
        else:
            weight = quality
        pair_delays.append(PairDelay(pair, delay, quality=quality, weight=weight))
        diagnostics.append(
            GccPhatPairDiagnostics(
                pair=pair,
                delay_seconds=delay,
                lag_samples=raw_diag.lag_samples,
                peak_value=raw_diag.peak_value,
                peak_to_sidelobe=raw_diag.peak_to_sidelobe,
                quality=quality,
                policy=raw_diag.policy,
                subbands_used=raw_diag.subbands_used,
                continuity_applied=raw_diag.continuity_applied,
            )
        )
    return GccPhatEstimate(TdoaEstimate(tuple(pair_delays), used_pairs=TDOA_PAIRS), tuple(diagnostics), policy_counts)


def _estimate_pair_delay(
    signal: np.ndarray,
    reference: np.ndarray,
    sample_rate: int,
    *,
    max_tau: float,
    interpolation: int,
    phat_weight: float,
    subbands: int,
    pair: tuple[int, int],
    continuity_reference: dict[tuple[int, int], float] | None,
    policy_used: str,
) -> tuple[float, GccPhatPairDiagnostics]:
    previous = continuity_reference.get(pair) if continuity_reference else None
    effective_phat_weight = 0.4 if policy_used == "narrowband" else phat_weight
    effective_subbands = max(2, subbands) if policy_used == "wideband" else subbands
    if effective_subbands <= 1 or policy_used == "narrowband":
        delay, diag = _gcc_phat_delay_candidates(
            signal,
            reference,
            sample_rate,
            max_tau=max_tau,
            interpolation=interpolation,
            phat_weight=effective_phat_weight,
            previous_delay=previous,
        )
        diag = GccPhatPairDiagnostics(
            pair=diag.pair,
            delay_seconds=diag.delay_seconds,
            lag_samples=diag.lag_samples,
            peak_value=diag.peak_value,
            peak_to_sidelobe=diag.peak_to_sidelobe,
            quality=diag.quality,
            policy=policy_used,
            subbands_used=1,
            continuity_applied=previous is not None,
        )
    else:
        low_hz, high_hz = _active_frequency_span(signal, reference, sample_rate)
        # If the source is effectively narrowband, splitting into subbands is unstable.
        if high_hz - low_hz < 220.0:
            delay, diag = gcc_phat_delay(
                signal,
                reference,
                sample_rate,
                max_tau=max_tau,
                interpolation=interpolation,
                phat_weight=effective_phat_weight,
            )
            return delay, GccPhatPairDiagnostics(
                pair=diag.pair,
                delay_seconds=delay,
                lag_samples=diag.lag_samples,
                peak_value=diag.peak_value,
                peak_to_sidelobe=diag.peak_to_sidelobe,
                quality=diag.quality,
                policy="narrowband",
                subbands_used=1,
                continuity_applied=previous is not None,
            )
        band_edges = np.linspace(low_hz, high_hz, effective_subbands + 1)
        estimates: list[tuple[float, GccPhatPairDiagnostics]] = []
        band_energies: list[float] = []
        for band_index in range(effective_subbands):
            band_low = float(band_edges[band_index])
            band_high = float(band_edges[band_index + 1])
            filtered_signal = _bandpass_by_fft(signal, sample_rate, band_low, band_high)
            filtered_reference = _bandpass_by_fft(reference, sample_rate, band_low, band_high)
            band_energy = float(np.mean(filtered_signal * filtered_signal) + np.mean(filtered_reference * filtered_reference))
            band_energies.append(band_energy)
            band_delay, band_diag = gcc_phat_delay(
                filtered_signal,
                filtered_reference,
                sample_rate,
                max_tau=max_tau,
                interpolation=interpolation,
                phat_weight=effective_phat_weight,
            )
            estimates.append((band_delay, band_diag))
        if estimates:
            energy_floor = max(band_energies) * 0.08
            estimates = [
                item
                for item, energy in zip(estimates, band_energies)
                if energy >= energy_floor and np.isfinite(item[0])
            ]
        if not estimates:
            delay, diag = gcc_phat_delay(
                signal,
                reference,
                sample_rate,
                max_tau=max_tau,
                interpolation=interpolation,
                phat_weight=phat_weight,
            )
            return delay, diag
        delay, diag = _combine_subband_delays(estimates, pair)
        full_delay, full_diag = _gcc_phat_delay_candidates(
            signal,
            reference,
            sample_rate,
            max_tau=max_tau,
            interpolation=interpolation,
            phat_weight=0.7,
            previous_delay=previous,
        )
        disagreement_limit = 1.0 / sample_rate
        fallback_to_fullband = False
        if abs(delay - full_delay) > disagreement_limit:
            delay, diag = full_delay, full_diag
            fallback_to_fullband = True
        diag = GccPhatPairDiagnostics(
            pair=diag.pair,
            delay_seconds=diag.delay_seconds,
            lag_samples=diag.lag_samples,
            peak_value=diag.peak_value,
            peak_to_sidelobe=diag.peak_to_sidelobe,
            quality=diag.quality,
            policy=("narrowband" if fallback_to_fullband else "wideband"),
            subbands_used=max(1, len(estimates)),
            continuity_applied=previous is not None,
        )

    return delay, diag


def _classify_pair_policy(signal: np.ndarray, reference: np.ndarray, sample_rate: int) -> str:
    n = signal.size
    if n < 16:
        return "narrowband"
    sig_fft = np.fft.rfft(signal)
    ref_fft = np.fft.rfft(reference)
    power = np.abs(sig_fft) ** 2 + np.abs(ref_fft) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    valid = (freqs >= 40.0) & (freqs <= 0.48 * sample_rate)
    if not np.any(valid):
        return "narrowband"
    p = power[valid]
    if float(np.sum(p)) <= 1e-18:
        return "narrowband"
    pn = p / float(np.sum(p))
    spectral_entropy = float(-np.sum(pn * np.log(pn + 1e-18)) / np.log(max(2, pn.size)))
    threshold = max(1e-18, float(np.max(p)) * 0.03)
    active = np.where(p >= threshold)[0]
    if active.size == 0:
        return "narrowband"
    f = freqs[valid]
    bandwidth = float(f[active[-1]] - f[active[0]])
    if bandwidth < 180.0 or spectral_entropy < 0.35:
        return "narrowband"
    return "wideband"


def _gcc_phat_delay_candidates(
    signal: Sequence[float],
    reference: Sequence[float],
    sample_rate: int,
    *,
    max_tau: float | None,
    interpolation: int,
    phat_weight: float,
    previous_delay: float | None,
) -> tuple[float, GccPhatPairDiagnostics]:
    sig = _prepare_channel(signal)
    ref = _prepare_channel(reference)
    fft_size = _next_power_of_two(sig.size + ref.size)
    sig_fft = np.fft.rfft(sig, n=fft_size)
    ref_fft = np.fft.rfft(ref, n=fft_size)
    cross_power = sig_fft * np.conj(ref_fft)
    cross_power_abs = np.abs(cross_power)
    if phat_weight > 0.0:
        cross_power = cross_power / np.maximum(cross_power_abs, 1e-15) ** phat_weight
    interpolation_factor = max(1, int(interpolation))
    correlation_size = fft_size * interpolation_factor
    correlation = np.fft.irfft(cross_power, n=correlation_size)
    max_shift = correlation_size // 2
    if max_tau is not None:
        max_shift = min(max_shift, int(round(interpolation_factor * sample_rate * max_tau)))
    shifted = np.concatenate((correlation[-max_shift:], correlation[: max_shift + 1]))
    abs_shifted = np.abs(shifted)
    candidate_indices = _candidate_peak_indices(abs_shifted, limit=11)
    if not candidate_indices:
        candidate_indices = [int(np.argmax(abs_shifted))]
    best_index = candidate_indices[0]
    best_score = -1e18
    max_step = max(8.0 / sample_rate, 0.35 * max_tau) if previous_delay is not None else None
    for idx in candidate_indices:
        shift = idx - max_shift
        frac = _parabolic_peak_offset(abs_shifted, idx) if interpolation_factor > 1 else 0.0
        delay = (shift + frac) / float(interpolation_factor * sample_rate)
        p2s = _peak_to_sidelobe(abs_shifted, idx, guard=max(1, interpolation))
        score = abs_shifted[idx] + 0.6 * p2s
        if previous_delay is not None:
            delta = abs(delay - previous_delay)
            score -= 40.0 * delta
            if max_step is not None and delta > max_step:
                score -= 0.5
        if score > best_score:
            best_score = float(score)
            best_index = idx
    shift = best_index - max_shift
    fractional = _parabolic_peak_offset(abs_shifted, best_index) if interpolation_factor > 1 else 0.0
    lag_samples = (shift + fractional) / float(interpolation_factor)
    delay = lag_samples / float(sample_rate)
    peak_value = float(abs_shifted[best_index])
    peak_to_sidelobe = _peak_to_sidelobe(abs_shifted, best_index, guard=max(1, interpolation))
    quality = _quality_from_peak_ratio(peak_to_sidelobe)
    diag = GccPhatPairDiagnostics(
        pair=(0, 1),
        delay_seconds=delay,
        lag_samples=lag_samples,
        peak_value=peak_value,
        peak_to_sidelobe=peak_to_sidelobe,
        quality=quality,
    )
    return delay, diag


def _candidate_peak_indices(values: np.ndarray, limit: int = 11) -> list[int]:
    if values.size < 3:
        return [int(np.argmax(values))] if values.size else []
    peaks: list[int] = []
    for i in range(1, values.size - 1):
        if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
            peaks.append(i)
    if not peaks:
        return [int(np.argmax(values))]
    peaks.sort(key=lambda i: float(values[i]), reverse=True)
    return peaks[:limit]


def _combine_subband_delays(
    estimates: Sequence[tuple[float, GccPhatPairDiagnostics]],
    pair: tuple[int, int],
) -> tuple[float, GccPhatPairDiagnostics]:
    if not estimates:
        empty_diag = GccPhatPairDiagnostics(pair, 0.0, 0.0, 0.0, 0.0, 0.0)
        return 0.0, empty_diag
    delays = np.asarray([item[0] for item in estimates], dtype=np.float64)
    weights = np.asarray([max(1e-9, item[1].quality) for item in estimates], dtype=np.float64)
    order = np.argsort(delays)
    sorted_delays = delays[order]
    sorted_weights = weights[order]
    cdf = np.cumsum(sorted_weights) / float(np.sum(sorted_weights))
    median_index = int(np.searchsorted(cdf, 0.5, side="left"))
    combined_delay = float(sorted_delays[min(max(0, median_index), sorted_delays.size - 1)])
    spread = float(np.std(delays))
    base_quality = float(np.mean([item[1].quality for item in estimates]))
    spread_penalty = max(0.0, min(0.4, spread * 800.0))
    quality = max(0.0, min(1.0, base_quality - spread_penalty))
    best_diag = max(estimates, key=lambda item: item[1].quality)[1]
    diag = GccPhatPairDiagnostics(
        pair=pair,
        delay_seconds=combined_delay,
        lag_samples=best_diag.lag_samples,
        peak_value=best_diag.peak_value,
        peak_to_sidelobe=best_diag.peak_to_sidelobe,
        quality=max(0.0, min(1.0, quality)),
    )
    return combined_delay, diag


def _bandpass_by_fft(signal: np.ndarray, sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    spectrum = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / float(sample_rate))
    mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    spectrum = spectrum * mask
    return np.fft.irfft(spectrum, n=signal.size).real


def _active_frequency_span(signal: np.ndarray, reference: np.ndarray, sample_rate: int) -> tuple[float, float]:
    n = signal.size
    if n < 8:
        return 80.0, max(320.0, 0.45 * sample_rate)
    sig_fft = np.fft.rfft(signal)
    ref_fft = np.fft.rfft(reference)
    power = np.abs(sig_fft) ** 2 + np.abs(ref_fft) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    valid = (freqs >= 40.0) & (freqs <= 0.48 * sample_rate)
    if not np.any(valid):
        return 80.0, max(320.0, 0.45 * sample_rate)
    p = power[valid]
    f = freqs[valid]
    threshold = max(1e-18, float(np.max(p)) * 0.03)
    active = p >= threshold
    if not np.any(active):
        return 80.0, max(320.0, 0.45 * sample_rate)
    low = float(np.min(f[active]))
    high = float(np.max(f[active]))
    if high - low < 120.0:
        center = 0.5 * (low + high)
        half = 120.0
        low = max(40.0, center - half)
        high = min(0.48 * sample_rate, center + half)
    return low, high

def _prepare_channel(channel: Sequence[float]) -> np.ndarray:
    data = np.asarray(channel, dtype=np.float64)
    if data.ndim != 1:
        raise ValueError("GCC-PHAT expects one-dimensional channels")
    data = data - np.mean(data)
    energy = sqrt(float(np.mean(data * data)))
    if energy > 0.0:
        data = data / energy
    return data


def _peak_to_sidelobe(values: np.ndarray, peak_index: int, guard: int) -> float:
    mask = np.ones(values.shape, dtype=bool)
    start = max(0, peak_index - guard)
    end = min(values.size, peak_index + guard + 1)
    mask[start:end] = False
    sidelobes = values[mask]
    if sidelobes.size == 0:
        return float("inf")
    sidelobe_rms = sqrt(float(np.mean(sidelobes * sidelobes)))
    return float(values[peak_index] / max(sidelobe_rms, 1e-15))


def _quality_from_peak_ratio(peak_to_sidelobe: float) -> float:
    if not np.isfinite(peak_to_sidelobe):
        return 1.0
    return max(0.0, min(1.0, peak_to_sidelobe / (peak_to_sidelobe + 4.0)))


def _parabolic_peak_offset(values: np.ndarray, peak_index: int) -> float:
    if peak_index <= 0 or peak_index >= values.size - 1:
        return 0.0
    left = float(values[peak_index - 1])
    center = float(values[peak_index])
    right = float(values[peak_index + 1])
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-15:
        return 0.0
    return max(-0.5, min(0.5, 0.5 * (left - right) / denominator))


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()
