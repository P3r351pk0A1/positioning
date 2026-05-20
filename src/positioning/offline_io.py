"""Offline audio input/output helpers for CSV, WAV and metadata files."""

from __future__ import annotations

import csv
import json
import struct
import wave
from pathlib import Path
from typing import Any

import numpy as np

from .models import AudioFrame


def write_csv_audio(frame: AudioFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(frame.channels, dtype=float).T
    with output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ch0", "ch1", "ch2", "ch3"])
        writer.writerows(data.tolist())


def read_csv_audio(path: str | Path, sample_rate: int, timestamp: float = 0.0) -> AudioFrame:
    input_path = Path(path)
    columns = [[], [], [], []]
    with input_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        first_row = next(reader, None)
        if first_row is None:
            raise ValueError(f"CSV audio file '{input_path}' is empty")
        rows = reader if _looks_like_header(first_row) else [first_row, *reader]
        for row_index, row in enumerate(rows, 1):
            if len(row) != 4:
                raise ValueError(f"CSV row {row_index} must contain exactly four channels")
            for channel_index, value in enumerate(row):
                columns[channel_index].append(float(value))
    return AudioFrame(tuple(tuple(channel) for channel in columns), sample_rate, timestamp)  # type: ignore[arg-type]


def write_wav_audio(frame: AudioFrame, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(frame.channels, dtype=float).T
    clipped = np.clip(data, -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(4)
        wav.setsampwidth(2)
        wav.setframerate(frame.sample_rate)
        wav.writeframes(pcm.tobytes())


def read_wav_audio(path: str | Path, timestamp: float = 0.0) -> AudioFrame:
    input_path = Path(path)
    with wave.open(str(input_path), "rb") as wav:
        channel_count = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        frame_count = wav.getnframes()
        raw = wav.readframes(frame_count)

    if channel_count != 4:
        raise ValueError(f"WAV audio file '{input_path}' must contain exactly four channels")

    samples = _decode_pcm(raw, sample_width)
    if samples.size % 4 != 0:
        raise ValueError(f"WAV audio file '{input_path}' has incomplete 4-channel frames")
    data = samples.reshape((-1, 4)).T
    return AudioFrame(tuple(tuple(float(x) for x in channel) for channel in data), sample_rate, timestamp)  # type: ignore[arg-type]


def write_metadata(metadata: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def read_metadata(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def metadata_path_for_signal(path: str | Path) -> Path:
    signal_path = Path(path)
    return signal_path.with_suffix(signal_path.suffix + ".meta.json")


def _looks_like_header(row: list[str]) -> bool:
    try:
        [float(value) for value in row]
    except ValueError:
        return True
    return False


def _decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        return (data - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if sample_width == 3:
        values = []
        for offset in range(0, len(raw), 3):
            chunk = raw[offset : offset + 3]
            sign = b"\xff" if chunk[2] & 0x80 else b"\x00"
            values.append(struct.unpack("<i", chunk + sign)[0])
        return np.asarray(values, dtype=np.float64) / 8388608.0
    if sample_width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    raise ValueError(f"unsupported WAV sample width: {sample_width} bytes")
