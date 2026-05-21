"""Generate 4-channel trajectory WAV files from a real FPV rotor reference audio."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

from positioning.config import load_config
from positioning.geometry import Vec3
from positioning.models import AudioFrame
from positioning.offline_io import write_metadata, write_wav_audio


def _read_wav_any_channels(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.getnframes()
        data = wav_file.readframes(frames)
    if sample_width == 2:
        dtype = np.int16
        scale = 32768.0
    elif sample_width == 4:
        dtype = np.int32
        scale = 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")
    raw = np.frombuffer(data, dtype=dtype).astype(np.float64) / scale
    if channels <= 0:
        raise ValueError("WAV must have at least one channel")
    shaped = raw.reshape(-1, channels)
    mono = shaped.mean(axis=1)
    return mono, sample_rate


def _resample_linear(signal: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return signal.astype(np.float64, copy=True)
    if signal.size < 2:
        raise ValueError("Reference signal is too short for resampling")
    duration_s = signal.size / float(src_rate)
    dst_count = max(2, int(round(duration_s * dst_rate)))
    src_x = np.arange(signal.size, dtype=np.float64)
    dst_x = np.linspace(0.0, float(signal.size - 1), num=dst_count, dtype=np.float64)
    return np.interp(dst_x, src_x, signal).astype(np.float64)


def _prepare_reference(reference: np.ndarray, sample_count: int) -> np.ndarray:
    if reference.size >= sample_count:
        prepared = reference[:sample_count]
    else:
        reps = int(np.ceil(sample_count / float(reference.size)))
        prepared = np.tile(reference, reps)[:sample_count]
    prepared = prepared - float(np.mean(prepared))
    peak = float(np.max(np.abs(prepared)))
    if peak > 1e-12:
        prepared = 0.85 * (prepared / peak)
    return prepared


def _generate_scenario(
    source: np.ndarray,
    fs: int,
    microphones: tuple[Vec3, ...],
    speed_of_sound: float,
    start: Vec3,
    end: Vec3,
    duration_s: float,
) -> tuple[AudioFrame, dict[str, object]]:
    sample_count = int(round(duration_s * fs))
    t = np.arange(sample_count, dtype=np.float64) / float(fs)
    base = _prepare_reference(source, sample_count)
    x0 = np.arange(sample_count, dtype=np.float64)

    direction = Vec3(end.x - start.x, end.y - start.y, end.z - start.z)
    denom = max(duration_s, 1e-12)
    vx = direction.x / denom
    vy = direction.y / denom
    vz = direction.z / denom

    frac = np.clip(t / duration_s, 0.0, 1.0)
    pos_x = start.x + (end.x - start.x) * frac
    pos_y = start.y + (end.y - start.y) * frac
    pos_z = start.z + (end.z - start.z) * frac

    channels: list[np.ndarray] = []
    for mic in microphones:
        dist = np.sqrt((pos_x - mic.x) ** 2 + (pos_y - mic.y) ** 2 + (pos_z - mic.z) ** 2)
        delay_samples = dist / speed_of_sound * fs
        query = x0 - delay_samples
        delayed = np.interp(query, x0, base, left=0.0, right=0.0)
        gain = 1.0 / np.maximum(dist, 1.0)
        channels.append((delayed * gain).astype(np.float64))

    all_data = np.vstack(channels)
    peak = float(np.max(np.abs(all_data)))
    if peak > 0.98:
        all_data *= 0.98 / peak

    frame = AudioFrame(tuple(tuple(float(v) for v in ch) for ch in all_data), fs)

    point_count = 30
    trajectory = []
    for i in range(point_count):
        f = i / float(point_count - 1)
        ts = f * duration_s
        trajectory.append(
            {
                "timestamp": ts,
                "x": start.x + (end.x - start.x) * f,
                "y": start.y + (end.y - start.y) * f,
                "z": start.z + (end.z - start.z) * f,
            }
        )

    metadata: dict[str, object] = {
        "kind": "synthetic_trajectory_signal",
        "sample_rate": fs,
        "duration_s": duration_s,
        "sample_count": sample_count,
        "target": {"x": start.x, "y": start.y, "z": start.z},
        "trajectory": trajectory,
        "source": {
            "kind": "fpv_reference",
            "file": "FPV дрон.wav",
            "note": "continuous variable-delay projection of mono rotor track",
        },
        "environment": {"sound_speed_m_s": speed_of_sound},
        "attenuation": True,
        "trajectory_velocity_m_s": {"x": vx, "y": vy, "z": vz},
    }
    return frame, metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate near/far/500m+ FPV trajectory WAV signals.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--reference", default="../FPV дрон.wav")
    parser.add_argument("--output-dir", default="data/sources")
    parser.add_argument("--duration", type=float, default=6.0)
    args = parser.parse_args()

    config = load_config(args.config)
    reference_path = Path(args.reference)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mono, src_rate = _read_wav_any_channels(reference_path)
    resampled = _resample_linear(mono, src_rate, config.audio.sample_rate)

    scenarios = [
        (
            "01_FPV_траектория_ближнее_поле_45град.wav",
            Vec3(2.0, -1.2, 0.6),
            Vec3(4.8, 1.6, 0.6),
        ),
        (
            "02_FPV_траектория_дальнее_поле_45град.wav",
            Vec3(20.0, -10.0, 4.0),
            Vec3(60.0, 30.0, 4.0),
        ),
        (
            "03_FPV_траектория_сверхдальняя_500м_plus.wav",
            Vec3(520.0, -120.0, 80.0),
            Vec3(680.0, 40.0, 80.0),
        ),
    ]

    for filename, start, end in scenarios:
        frame, metadata = _generate_scenario(
            resampled,
            config.audio.sample_rate,
            config.microphone_array.microphones,
            config.environment.sound_speed,
            start,
            end,
            args.duration,
        )
        wav_path = output_dir / filename
        meta_path = Path(str(wav_path) + ".meta.json")
        write_wav_audio(frame, wav_path)
        write_metadata(metadata, meta_path)
        print(f"Saved: {wav_path}")
        print(f"Saved: {meta_path}")
        print(f"Start={start}, End={end}")

    summary_path = output_dir / "01_02_03_fpv_generation_summary.json"
    summary = {
        "config": args.config,
        "reference": str(reference_path),
        "output_dir": str(output_dir),
        "sample_rate_hz": config.audio.sample_rate,
        "duration_s": args.duration,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
