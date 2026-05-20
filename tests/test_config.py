import json

import pytest

from positioning.config import ConfigError, load_config, parse_config


def test_default_config_loads():
    config = load_config()

    assert len(config.microphone_array.microphones) == 4
    assert config.audio.sample_rate == 48000
    assert config.audio.window_size == 2048
    assert config.audio.frequency_band == (100.0, 5000.0)
    assert config.localization.mode == "position"
    assert config.environment.sound_speed == pytest.approx(343.42)
    assert config.microphone_array.is_coplanar


def test_invalid_microphone_count_is_rejected():
    data = {
        "array": {"microphones": [{"x": 0, "y": 0, "z": 0}]},
        "environment": {"temperature_c": 20, "humidity_percent": 0},
        "audio": {"sample_rate": 48000, "window_size": 1024, "overlap": 0.5, "frequency_band": [100, 5000]},
        "localization": {"mode": "position"},
    }

    with pytest.raises(ConfigError, match="exactly four"):
        try:
            parse_config(data)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc


def test_invalid_overlap_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "array": {
                    "microphones": [
                        {"x": 0, "y": 0, "z": 0},
                        {"x": 0.2, "y": 0, "z": 0},
                        {"x": 0, "y": 0.2, "z": 0},
                        {"x": 0, "y": 0, "z": 0.2},
                    ]
                },
                "environment": {"temperature_c": 20, "humidity_percent": 0},
                "audio": {"sample_rate": 48000, "window_size": 1024, "overlap": 1.0, "frequency_band": [100, 5000]},
                "localization": {"mode": "position"},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="overlap"):
        load_config(path)


def test_invalid_json_has_context(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(ConfigError, match="failed to parse config"):
        load_config(path)
