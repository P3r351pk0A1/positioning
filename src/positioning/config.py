"""JSON configuration loading and validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .geometry import Vec3
from .models import AppConfig, AudioConfig, Environment, LocalizationConfig, microphone_array_from_vectors

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.json"


class ConfigError(ValueError):
    """Raised when a configuration file is missing required data or is invalid."""


def load_config(path: str | Path | None = None) -> AppConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"failed to read config '{config_path}': {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"failed to parse config '{config_path}': {exc}") from exc

    try:
        return parse_config(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"invalid config '{config_path}': {exc}") from exc


def parse_config(data: dict[str, Any]) -> AppConfig:
    if not isinstance(data, dict):
        raise TypeError("top-level config value must be an object")

    array_data = _required_object(data, "array")
    environment_data = _required_object(data, "environment")
    audio_data = _required_object(data, "audio")
    localization_data = _required_object(data, "localization")

    microphones_raw = _required_array(array_data, "microphones")
    microphones = [_parse_vec3(item, f"array.microphones[{index}]") for index, item in enumerate(microphones_raw)]

    frequency_band_raw = _required_array(audio_data, "frequency_band")
    if len(frequency_band_raw) != 2:
        raise ValueError("audio.frequency_band must contain exactly two numbers")

    return AppConfig(
        microphone_array=microphone_array_from_vectors(microphones),
        environment=Environment(
            temperature_c=_number(environment_data, "temperature_c"),
            humidity_percent=_number(environment_data, "humidity_percent"),
        ),
        audio=AudioConfig(
            sample_rate=_positive_int(audio_data, "sample_rate"),
            window_size=_positive_int(audio_data, "window_size"),
            overlap=_number(audio_data, "overlap"),
            frequency_band=(_number_at(frequency_band_raw, 0, "audio.frequency_band"), _number_at(frequency_band_raw, 1, "audio.frequency_band")),
        ),
        localization=LocalizationConfig(mode=_string(localization_data, "mode")),  # type: ignore[arg-type]
    )


def _required_object(data: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in data:
        raise KeyError(f"missing required section '{key}'")
    value = data[key]
    if not isinstance(value, dict):
        raise TypeError(f"section '{key}' must be an object")
    return value


def _required_array(data: dict[str, Any], key: str) -> list[Any]:
    if key not in data:
        raise KeyError(f"missing required field '{key}'")
    value = data[key]
    if not isinstance(value, list):
        raise TypeError(f"field '{key}' must be an array")
    return value


def _parse_vec3(data: Any, path: str) -> Vec3:
    if not isinstance(data, dict):
        raise TypeError(f"{path} must be an object")
    return Vec3(
        x=_number(data, "x", path),
        y=_number(data, "y", path),
        z=_number(data, "z", path),
    )


def _number(data: dict[str, Any], key: str, prefix: str | None = None) -> float:
    if key not in data:
        name = f"{prefix}.{key}" if prefix else key
        raise KeyError(f"missing required field '{name}'")
    value = data[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        name = f"{prefix}.{key}" if prefix else key
        raise TypeError(f"field '{name}' must be a number")
    return float(value)


def _number_at(data: list[Any], index: int, path: str) -> float:
    value = data[index]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{path}[{index}] must be a number")
    return float(value)


def _positive_int(data: dict[str, Any], key: str) -> int:
    value = _number(data, key)
    if int(value) != value or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return int(value)


def _string(data: dict[str, Any], key: str) -> str:
    if key not in data:
        raise KeyError(f"missing required field '{key}'")
    value = data[key]
    if not isinstance(value, str):
        raise TypeError(f"field '{key}' must be a string")
    return value
