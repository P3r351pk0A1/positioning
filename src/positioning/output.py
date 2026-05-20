"""Structured JSON output and CSV window logging."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

from .geometry import Vec3
from .models import AppConfig, LocalizationResult, PairDelay
from .tracking import TrackedResult


def localization_result_to_dict(result: LocalizationResult) -> dict[str, Any]:
    return {
        "mode": result.mode,
        "status": result.status,
        "position": _vec_to_dict(result.position),
        "direction": _vec_to_dict(result.direction),
        "azimuth_deg": result.azimuth_deg,
        "elevation_deg": result.elevation_deg,
        "residual_error2": result.residual_error2,
        "used_pairs": result.used_pairs,
        "rejected_pairs": result.rejected_pairs,
        "reliable": result.reliable,
        "confidence": result.confidence,
        "message": result.message,
    }


def tracked_result_to_dict(item: TrackedResult) -> dict[str, Any]:
    data = localization_result_to_dict(item.raw)
    data.update(
        {
            "timestamp": item.timestamp,
            "accepted": item.accepted,
            "smoothed_azimuth_deg": item.smoothed_azimuth_deg,
            "smoothed_elevation_deg": item.smoothed_elevation_deg,
            "azimuth_rate_deg_s": item.azimuth_rate_deg_s,
            "elevation_rate_deg_s": item.elevation_rate_deg_s,
            "aggregated_confidence": item.aggregated_confidence,
            "tracking_message": item.message,
        }
    )
    return data


def write_json_result(result: LocalizationResult | TrackedResult, path: str | Path, config: AppConfig | None = None) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any]
    if isinstance(result, TrackedResult):
        payload = tracked_result_to_dict(result)
    else:
        payload = localization_result_to_dict(result)
    if config is not None:
        payload["environment"] = {
            "temperature_c": config.environment.temperature_c,
            "humidity_percent": config.environment.humidity_percent,
            "sound_speed_m_s": config.environment.sound_speed,
        }
        payload["audio"] = {
            "sample_rate": config.audio.sample_rate,
            "window_size": config.audio.window_size,
            "overlap": config.audio.overlap,
            "frequency_band": list(config.audio.frequency_band),
        }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class CsvWindowLogger:
    def __init__(self, path: str | Path, config: AppConfig) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.config = config
        self._file = self.path.open("w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[
                "window_index",
                "timestamp",
                "status",
                "mode",
                "accepted",
                "azimuth_deg",
                "elevation_deg",
                "smoothed_azimuth_deg",
                "smoothed_elevation_deg",
                "azimuth_rate_deg_s",
                "elevation_rate_deg_s",
                "confidence",
                "aggregated_confidence",
                "residual_error2",
                "used_pairs",
                "rejected_pairs",
                "reliable",
                "message",
                "tracking_message",
                "temperature_c",
                "humidity_percent",
                "sound_speed_m_s",
                "window_size",
                "overlap",
                "tdoa_pairs_json",
            ],
        )
        self._writer.writeheader()

    def write(self, window_index: int, item: TrackedResult, pair_delays: Sequence[PairDelay] = ()) -> None:
        result = item.raw
        self._writer.writerow(
            {
                "window_index": window_index,
                "timestamp": item.timestamp,
                "status": result.status,
                "mode": result.mode,
                "accepted": item.accepted,
                "azimuth_deg": result.azimuth_deg,
                "elevation_deg": result.elevation_deg,
                "smoothed_azimuth_deg": item.smoothed_azimuth_deg,
                "smoothed_elevation_deg": item.smoothed_elevation_deg,
                "azimuth_rate_deg_s": item.azimuth_rate_deg_s,
                "elevation_rate_deg_s": item.elevation_rate_deg_s,
                "confidence": result.confidence,
                "aggregated_confidence": item.aggregated_confidence,
                "residual_error2": result.residual_error2,
                "used_pairs": result.used_pairs,
                "rejected_pairs": result.rejected_pairs,
                "reliable": result.reliable,
                "message": result.message,
                "tracking_message": item.message,
                "temperature_c": self.config.environment.temperature_c,
                "humidity_percent": self.config.environment.humidity_percent,
                "sound_speed_m_s": self.config.environment.sound_speed,
                "window_size": self.config.audio.window_size,
                "overlap": self.config.audio.overlap,
                "tdoa_pairs_json": json.dumps([_pair_to_dict(pair) for pair in pair_delays], ensure_ascii=False),
            }
        )

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "CsvWindowLogger":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _vec_to_dict(value: Vec3 | None) -> dict[str, float] | None:
    if value is None:
        return None
    return {"x": value.x, "y": value.y, "z": value.z}


def _pair_to_dict(pair: PairDelay) -> dict[str, Any]:
    return {
        "pair": list(pair.pair),
        "delay_seconds": pair.delay_seconds,
        "quality": pair.quality,
        "weight": pair.weight,
    }
