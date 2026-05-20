"""Program and test method (PMI) automated protocol."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .config import ConfigError, load_config, parse_config
from .gcc_phat import estimate_tdoa_gcc_phat
from .geometry import Vec3, dot, norm, position_error
from .localization import estimate_localization_from_tdoa
from .models import AppConfig, AudioFrame, PairDelay, TdoaEstimate
from .offline_io import (
    read_csv_audio,
    read_metadata,
    read_wav_audio,
    write_csv_audio,
    write_metadata,
    write_wav_audio,
)
from .output import CsvWindowLogger, write_json_result
from .preprocess import preprocess_frame, split_frames
from .simulation import generate_synthetic_signal, make_arrival_time
from .tracking import ResultTracker


@dataclass(frozen=True)
class PmiCaseResult:
    number: int
    name: str
    passed: bool
    details: str
    metrics: dict[str, float | int | str | bool] = field(default_factory=dict)


def run_pmi(config_path: str | Path | None = None, output_dir: str | Path = "data/pmi", repeat_count: int = 100) -> list[PmiCaseResult]:
    config = load_config(config_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    context = {"config": config, "output_dir": output, "repeat_count": repeat_count}
    cases: list[tuple[str, Callable[[dict[str, object]], tuple[bool, str, dict[str, float | int | str | bool]]]]] = [
        ("Запуск программы", _case_program_start),
        ("Прием четырех каналов и метаданных", _case_four_channels_and_metadata),
        ("Обработка файлового режима", _case_file_mode),
        ("Учет температуры и влажности", _case_environment),
        ("Фильтрация сигнала с шумом", _case_noisy_filtering),
        ("Оценка TDOA из аудио через GCC-PHAT", _case_gcc_phat_audio_tdoa),
        ("Корректность TDOA на синтетике", _case_synthetic_tdoa),
        ("Подавление отраженного пути", _case_reflection_rejection),
        ("Вычисление направления и координат", _case_direction_and_position),
        ("JSON/CSV-выход и журнал", _case_output_and_logs),
        ("Время обработки окна", _case_window_processing_time),
        ("Отсутствие одного канала", _case_missing_channel),
        ("Некорректные параметры среды", _case_invalid_environment),
        ("Недостаточная информативность сигнала", _case_uninformative_signal),
        ("Длительный повторный прогон", _case_long_repeat),
    ]

    results = []
    for index, (name, fn) in enumerate(cases, 1):
        try:
            passed, details, metrics = fn(context)
        except Exception as exc:  # noqa: BLE001 - PMI protocol should capture unexpected failures.
            passed = False
            details = f"unexpected error: {exc}"
            metrics = {}
        results.append(PmiCaseResult(index, name, passed, details, metrics))

    _write_protocols(results, config, output)
    return results


def _case_program_start(context: dict[str, object]) -> tuple[bool, str, dict[str, float | int | str | bool]]:
    output_dir = context["output_dir"]
    config = context["config"]
    assert isinstance(output_dir, Path)
    assert isinstance(config, AppConfig)
    target = Vec3(0.6, 0.4, 0.3)
    arrival = make_arrival_time(config.microphone_array.microphones, target, 0.1, speed_of_sound=config.environment.sound_speed)
    result = estimate_localization_from_tdoa(config.microphone_array, _arrival_to_tdoa(arrival), config.environment.sound_speed, config.localization.mode)
    json_path = output_dir / "program_start_result.json"
    write_json_result(result, json_path, config)
    return result.status == "ok" and json_path.exists(), "program entry scenario produced a valid JSON result", {"status": result.status}


def _case_four_channels_and_metadata(context: dict[str, object]) -> tuple[bool, str, dict[str, float | int | str | bool]]:
    config = _config(context)
    output_dir = _output_dir(context)
    frame, metadata = generate_synthetic_signal(config, Vec3(0.6, 0.4, 0.3), duration_s=0.05, noise_std=0.001)
    signal_path = output_dir / "four_channels.csv"
    metadata_path = output_dir / "four_channels.csv.meta.json"
    write_csv_audio(frame, signal_path)
    write_metadata(metadata, metadata_path)
    loaded = read_csv_audio(signal_path, config.audio.sample_rate)
    loaded_metadata = read_metadata(metadata_path)
    passed = len(loaded.channels) == 4 and loaded.sample_count == frame.sample_count and loaded_metadata["kind"] == "synthetic_signal"
    return passed, "4-channel CSV and sidecar metadata were written and read", {"samples": loaded.sample_count}


def _case_file_mode(context: dict[str, object]) -> tuple[bool, str, dict[str, float | int | str | bool]]:
    config = _config(context)
    output_dir = _output_dir(context)
    frame, _ = generate_synthetic_signal(config, Vec3(0.6, 0.4, 0.3), duration_s=0.05)
    csv_path = output_dir / "file_mode.csv"
    wav_path = output_dir / "file_mode.wav"
    write_csv_audio(frame, csv_path)
    write_wav_audio(frame, wav_path)
    csv_frame = read_csv_audio(csv_path, config.audio.sample_rate)
    wav_frame = read_wav_audio(wav_path)
    passed = csv_frame.sample_count == frame.sample_count and wav_frame.sample_count == frame.sample_count and wav_frame.sample_rate == config.audio.sample_rate
    return passed, "CSV and WAV file modes were accepted", {"csv_samples": csv_frame.sample_count, "wav_samples": wav_frame.sample_count}


def _case_environment(context: dict[str, object]) -> tuple[bool, str, dict[str, float | int | str | bool]]:
    base = parse_config(_config_dict(20.0, 0.0))
    warm = parse_config(_config_dict(30.0, 50.0))
    passed = warm.environment.sound_speed > base.environment.sound_speed
    return passed, "sound speed changes with temperature and humidity", {"base_speed": base.environment.sound_speed, "warm_speed": warm.environment.sound_speed}


def _case_noisy_filtering(context: dict[str, object]) -> tuple[bool, str, dict[str, float | int | str | bool]]:
    config = _config(context)
    frame, _ = generate_synthetic_signal(config, Vec3(0.6, 0.4, 0.3), duration_s=0.12, noise_std=0.02)
    window = split_frames(frame, config.audio.window_size, config.audio.overlap)[1]
    processed, diagnostics = preprocess_frame(window, config.audio)
    passed = processed.sample_count == config.audio.window_size and diagnostics.informative and diagnostics.mean_rms > 0.0
    return passed, "noisy signal window was filtered and marked informative", {"mean_rms": diagnostics.mean_rms, "mean_abs_correlation": diagnostics.mean_abs_correlation}


def _case_gcc_phat_audio_tdoa(context: dict[str, object]) -> tuple[bool, str, dict[str, float | int | str | bool]]:
    config = _config(context)
    sample_offsets = [50, 64, 74, 59]
    rng = np.random.default_rng(17)
    pulse = rng.normal(0.0, 1.0, size=80)
    channels = tuple(_pulse_channel(1024, 180 + offset, pulse) for offset in sample_offsets)
    frame = AudioFrame(channels, config.audio.sample_rate)
    estimate = estimate_tdoa_gcc_phat(frame, config.microphone_array, config.environment.sound_speed, interpolation=1)
    errors = []
    for item in estimate.estimate.pair_delays:
        i, j = item.pair
        expected = (sample_offsets[i] - sample_offsets[j]) / config.audio.sample_rate
        errors.append(abs(item.delay_seconds - expected))
    max_error = max(errors)
    min_quality = min(item.quality for item in estimate.estimate.pair_delays)
    passed = len(estimate.estimate.pair_delays) == 6 and max_error <= 0.5 / config.audio.sample_rate and min_quality > 0.5
    return passed, "GCC-PHAT recovered TDOA delays from a synthetic four-channel pulse", {"max_delay_error_s": max_error, "min_quality": min_quality}


def _case_synthetic_tdoa(context: dict[str, object]) -> tuple[bool, str, dict[str, float | int | str | bool]]:
    config = _config(context)
    direction = _unit(Vec3(0.7, 0.5, 0.2))
    result = estimate_localization_from_tdoa(config.microphone_array, TdoaEstimate(tuple(_direction_pair_delays(config, direction))), config.environment.sound_speed, "direction")
    direction_error = 1.0 - dot(result.direction or Vec3(), direction)
    passed = result.status == "ok" and direction_error < 1e-6
    return passed, "synthetic TDOA direction estimate matches the reference direction", {"direction_error": direction_error, "used_pairs": result.used_pairs}


def _case_reflection_rejection(context: dict[str, object]) -> tuple[bool, str, dict[str, float | int | str | bool]]:
    config = _config(context)
    direction = _unit(Vec3(0.7, 0.5, 0.2))
    pairs = []
    for item in _direction_pair_delays(config, direction):
        delay = item.delay_seconds + (3e-4 if item.pair == (0, 3) else 0.0)
        pairs.append(PairDelay(item.pair, delay, quality=1.0, weight=1.0))
    result = estimate_localization_from_tdoa(config.microphone_array, TdoaEstimate(tuple(pairs)), config.environment.sound_speed, "direction")
    direction_error = 1.0 - dot(result.direction or Vec3(), direction)
    passed = result.status == "ok" and result.rejected_pairs >= 1 and direction_error < 1e-4
    return passed, "single reflected/false TDOA pair was rejected", {"rejected_pairs": result.rejected_pairs, "direction_error": direction_error}


def _case_direction_and_position(context: dict[str, object]) -> tuple[bool, str, dict[str, float | int | str | bool]]:
    config = _config(context)
    direction = _unit(Vec3(0.7, 0.4, 0.3))
    direction_result = estimate_localization_from_tdoa(config.microphone_array, TdoaEstimate(tuple(_direction_pair_delays(config, direction))), config.environment.sound_speed, "direction")
    target = Vec3(0.6, 0.4, 0.3)
    arrival = make_arrival_time(config.microphone_array.microphones, target, 0.1, speed_of_sound=config.environment.sound_speed)
    position_result = estimate_localization_from_tdoa(config.microphone_array, _arrival_to_tdoa(arrival), config.environment.sound_speed, "position")
    pos_error = position_error(position_result.position or Vec3(), target)
    dir_error = 1.0 - dot(direction_result.direction or Vec3(), direction)
    passed = direction_result.status == "ok" and position_result.status == "ok" and pos_error < 1e-5 and dir_error < 1e-6
    return passed, "direction and allowed near-field 3D position modes computed successfully", {"position_error_m": pos_error, "direction_error": dir_error}


def _case_output_and_logs(context: dict[str, object]) -> tuple[bool, str, dict[str, float | int | str | bool]]:
    config = _config(context)
    output_dir = _output_dir(context)
    tracker = ResultTracker()
    direction = _unit(Vec3(0.7, 0.4, 0.3))
    result = estimate_localization_from_tdoa(config.microphone_array, TdoaEstimate(tuple(_direction_pair_delays(config, direction))), config.environment.sound_speed, "direction")
    tracked = tracker.update(result, 0.0)
    json_path = output_dir / "single_result.json"
    csv_path = output_dir / "window_log.csv"
    write_json_result(tracked, json_path, config)
    with CsvWindowLogger(csv_path, config) as logger:
        logger.write(0, tracked, _direction_pair_delays(config, direction))
    passed = json_path.exists() and csv_path.exists() and "tdoa_pairs_json" in csv_path.read_text(encoding="utf-8")
    return passed, "JSON result and CSV window log were written", {"json_bytes": json_path.stat().st_size, "csv_bytes": csv_path.stat().st_size}


def _case_window_processing_time(context: dict[str, object]) -> tuple[bool, str, dict[str, float | int | str | bool]]:
    config = _config(context)
    frame, _ = generate_synthetic_signal(config, Vec3(0.6, 0.4, 0.3), duration_s=0.12, noise_std=0.01)
    window = split_frames(frame, config.audio.window_size, config.audio.overlap)[1]
    started = time.perf_counter()
    preprocess_frame(window, config.audio)
    elapsed = time.perf_counter() - started
    window_duration = config.audio.window_size / config.audio.sample_rate
    return elapsed < window_duration, "single-window preprocessing time is below the window duration", {"elapsed_s": elapsed, "window_duration_s": window_duration}


def _case_missing_channel(context: dict[str, object]) -> tuple[bool, str, dict[str, float | int | str | bool]]:
    config = _config(context)
    output_dir = _output_dir(context)
    path = output_dir / "missing_channel.csv"
    path.write_text("ch0,ch1,ch2\n1,2,3\n", encoding="utf-8")
    try:
        read_csv_audio(path, config.audio.sample_rate)
    except ValueError as exc:
        return "exactly four channels" in str(exc), "missing channel was rejected with a clear error", {"error": str(exc)}
    return False, "missing channel was not rejected", {}


def _case_invalid_environment(context: dict[str, object]) -> tuple[bool, str, dict[str, float | int | str | bool]]:
    try:
        config = parse_config(_config_dict(200.0, 0.0))
        _ = config.environment.sound_speed
    except ConfigError as exc:
        return True, "invalid environment was rejected", {"error": str(exc)}
    except ValueError as exc:
        return True, "invalid environment was rejected", {"error": str(exc)}
    return False, "invalid environment was accepted", {}


def _case_uninformative_signal(context: dict[str, object]) -> tuple[bool, str, dict[str, float | int | str | bool]]:
    config = _config(context)
    zero = tuple(0.0 for _ in range(config.audio.window_size))
    frame = AudioFrame((zero, zero, zero, zero), config.audio.sample_rate)
    _processed, diagnostics = preprocess_frame(frame, config.audio)
    passed = not diagnostics.informative and diagnostics.reason == "low_energy"
    return passed, "silent window was marked insufficiently informative", {"mean_rms": diagnostics.mean_rms, "reason": diagnostics.reason}


def _case_long_repeat(context: dict[str, object]) -> tuple[bool, str, dict[str, float | int | str | bool]]:
    config = _config(context)
    repeat_count = int(context["repeat_count"])
    tracker = ResultTracker()
    accepted = 0
    started = time.perf_counter()
    for index in range(repeat_count):
        direction = _unit(Vec3(0.7, 0.4 + 0.001 * index, 0.3))
        result = estimate_localization_from_tdoa(config.microphone_array, TdoaEstimate(tuple(_direction_pair_delays(config, direction))), config.environment.sound_speed, "direction")
        tracked = tracker.update(result, index * config.audio.window_size / config.audio.sample_rate)
        if tracked.accepted:
            accepted += 1
    elapsed = time.perf_counter() - started
    passed = accepted == repeat_count
    return passed, "repeated tracking run completed without rejected valid windows", {"repeat_count": repeat_count, "accepted": accepted, "elapsed_s": elapsed}


def _write_protocols(results: list[PmiCaseResult], config: AppConfig, output_dir: Path) -> None:
    json_path = output_dir / "pmi_results.json"
    csv_path = output_dir / "pmi_results.csv"
    md_path = output_dir / "ПРОТОКОЛ_ПМИ.md"
    payload = {
        "summary": {
            "passed": sum(1 for item in results if item.passed),
            "total": len(results),
            "all_passed": all(item.passed for item in results),
        },
        "environment": {
            "temperature_c": config.environment.temperature_c,
            "humidity_percent": config.environment.humidity_percent,
            "sound_speed_m_s": config.environment.sound_speed,
        },
        "audio": {
            "sample_rate": config.audio.sample_rate,
            "window_size": config.audio.window_size,
            "overlap": config.audio.overlap,
            "frequency_band": list(config.audio.frequency_band),
        },
        "results": [_case_to_dict(item) for item in results],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["number", "name", "passed", "details", "metrics_json"])
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "number": item.number,
                    "name": item.name,
                    "passed": item.passed,
                    "details": item.details,
                    "metrics_json": json.dumps(item.metrics, ensure_ascii=False),
                }
            )
    lines = [
        "# Протокол испытаний по ПМИ",
        "",
        f"Итог: {payload['summary']['passed']} / {payload['summary']['total']} проверок пройдено.",
        "",
        "| № | Проверка | Статус | Детали |",
        "| --- | --- | --- | --- |",
    ]
    for item in results:
        status = "пройдено" if item.passed else "не пройдено"
        lines.append(f"| {item.number} | {item.name} | {status} | {item.details} |")
    lines.extend(
        [
            "",
            "## Файлы протокола",
            "",
            f"- JSON: `{json_path.name}`",
            f"- CSV: `{csv_path.name}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _case_to_dict(item: PmiCaseResult) -> dict[str, object]:
    return {
        "number": item.number,
        "name": item.name,
        "passed": item.passed,
        "details": item.details,
        "metrics": item.metrics,
    }


def _config(context: dict[str, object]) -> AppConfig:
    return context["config"]  # type: ignore[return-value]


def _output_dir(context: dict[str, object]) -> Path:
    return context["output_dir"]  # type: ignore[return-value]


def _unit(value: Vec3) -> Vec3:
    value_norm = norm(value)
    return value * (1.0 / value_norm)


def _direction_pair_delays(config: AppConfig, direction: Vec3) -> list[PairDelay]:
    pairs = []
    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        baseline = config.microphone_array.microphones[j] - config.microphone_array.microphones[i]
        delay = dot(baseline, direction) / config.environment.sound_speed
        pairs.append(PairDelay((i, j), delay, quality=1.0, weight=1.0))
    return pairs


def _arrival_to_tdoa(arrival: list[float]) -> TdoaEstimate:
    return TdoaEstimate(
        tuple(PairDelay((i, j), arrival[i] - arrival[j]) for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)))
    )


def _config_dict(temperature_c: float, humidity_percent: float) -> dict[str, object]:
    return {
        "array": {
            "microphones": [
                {"x": 0.0, "y": 0.0, "z": 0.0},
                {"x": 0.2, "y": 0.0, "z": 0.0},
                {"x": 0.2, "y": 0.2, "z": 0.0},
                {"x": 0.0, "y": 0.2, "z": 0.0},
            ]
        },
        "audio": {"sample_rate": 48000, "window_size": 2048, "overlap": 0.5, "frequency_band": [100.0, 5000.0]},
        "environment": {"temperature_c": temperature_c, "humidity_percent": humidity_percent},
        "localization": {"mode": "position"},
    }


def _pulse_channel(length: int, start: int, pulse: np.ndarray) -> tuple[float, ...]:
    data = np.zeros(length, dtype=np.float64)
    data[start : start + pulse.size] = pulse
    return tuple(float(value) for value in data)
