"""Streamlit dashboard for interactive positioning demos."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
import inspect
from datetime import datetime
from pathlib import Path
import time
from typing import Sequence
import json
import plotly.graph_objects as go
import numpy as np
import streamlit as st

from positioning.config import ConfigError, load_config
from positioning.gcc_phat import estimate_tdoa_gcc_phat
from positioning.geometry import Vec3, distance, norm
from positioning.localization import estimate_localization_from_tdoa
from positioning.models import AppConfig, AudioFrame, LocalizationResult, PairDelay
from positioning.offline_io import metadata_path_for_signal, read_csv_audio, read_metadata, read_wav_audio
from positioning.preprocess import preprocess_frame, split_frames
from positioning.visualization import TruthTrajectoryPoint, metadata_truth_samples


DEFAULT_CONFIG = Path("configs/side_looking_demo.json")
DEFAULT_SIGNAL = Path("data/synthetic/side_looking_good_trajectory.csv")


@dataclass(frozen=True)
class WindowEstimate:
    window_index: int
    timestamp: float
    result: LocalizationResult
    display_point: Vec3 | None
    pair_delays: tuple[PairDelay, ...]
    informative: bool
    truth_sample: TruthTrajectoryPoint | None = None
    truth_index: int | None = None
    reason: str = ""
    tdoa_policy: str = "manual"
    preprocess_s: float = 0.0
    gcc_s: float = 0.0
    localize_s: float = 0.0
    total_s: float = 0.0
    runtime_error: bool = False


def main() -> None:
    st.set_page_config(page_title="Позиционирование по TDOA", layout="wide")
    st.title("Панель позиционирования по TDOA")

    options = _sidebar()
    try:
        config = load_config(options["config_path"])
        frame = _read_signal(Path(options["signal_path"]), config)
        metadata = _read_optional_metadata(Path(options["metadata_path"]))
        if int(options["window_size_samples"]) > 0:
            config = replace(
                config,
                audio=replace(config.audio, window_size=int(options["window_size_samples"])),
            )
    except (ConfigError, OSError, ValueError) as exc:
        st.error(f"Ошибка подготовки данных: {exc}")
        return

    truth_samples = metadata_truth_samples(metadata)
    windows = _split_frames_for_ui(frame, config)
    selected_windows = _select_windows_for_ui(
        config,
        windows,
        truth_samples,
        metadata,
        str(options["window_selection"]),
    )
    estimates = _estimate_windows(config, selected_windows, options, truth_samples, metadata)
    accepted_raw = [
        item
        for item in estimates
        if item.display_point is not None
        and (item.result.status == "ok" or (options["allow_low_confidence"] and item.result.status == "low_confidence"))
        and item.result.confidence >= options["min_confidence"]
    ]
    accepted = _filter_display_estimates(
        accepted_raw,
        str(options["mode"]),
        str(options["window_selection"]),
        float(options["display_max_speed"]),
        bool(options.get("stabilize_first_direction", True)),
        float(options.get("max_direction_jump_deg", 20.0)),
        float(options.get("max_direction_rate_deg_s", 180.0)),
    )

    _summary(config, frame, windows, selected_windows, estimates, accepted, truth_samples, str(options["mode"]))

    left, right = st.columns([2.5, 1.0], gap="large")
    with left:
        fig = _build_figure(config, truth_samples, accepted, options)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        _result_panel(accepted)

    _table(estimates, truth_samples, str(options["mode"]), config.microphone_array.center)
    csv_log_path, json_log_path = _persist_streamlit_logs(
        config=config,
        options=options,
        estimates=estimates,
        accepted=accepted,
    )
    st.caption(f"Логи Streamlit сохранены: CSV `{csv_log_path}`, JSON `{json_log_path}`")

def _weather_caption_for_config(config_path: Path) -> str:
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        environment = data.get("environment", {})
        temperature_c = float(environment.get("temperature_c", 20.0))
        humidity_percent = float(environment.get("humidity_percent", 50.0))
        loaded = load_config(str(config_path))
        sound_speed = loaded.environment.sound_speed
        return (
            "Метаданные погоды: "
            f"температура {temperature_c:.1f} °C, "
            f"влажность {humidity_percent:.1f} %, "
            f"скорость звука {sound_speed:.2f} м/с"
        )
    except (OSError, ValueError, TypeError, ConfigError):
        return "Метаданные погоды: недоступно"


def _persist_streamlit_logs(
    *,
    config: AppConfig,
    options: dict[str, object],
    estimates: Sequence[WindowEstimate],
    accepted: Sequence[WindowEstimate],
) -> tuple[str, str]:
    logs_dir = Path("data/streamlit_logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    csv_path = logs_dir / f"streamlit_windows_{run_id}.csv"
    json_path = logs_dir / f"streamlit_result_{run_id}.json"

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "window_index",
                "timestamp_s",
                "status",
                "confidence",
                "azimuth_deg",
                "elevation_deg",
                "x_m",
                "y_m",
                "z_m",
                "used_pairs",
                "rejected_pairs",
                "tdoa_policy",
                "informative",
                "message",
                "reason",
                "preprocess_s",
                "gcc_s",
                "localize_s",
                "total_s",
                "runtime_error",
            ],
        )
        writer.writeheader()
        for item in estimates:
            writer.writerow(
                {
                    "window_index": item.window_index,
                    "timestamp_s": item.timestamp,
                    "status": item.result.status,
                    "confidence": item.result.confidence,
                    "azimuth_deg": item.result.azimuth_deg,
                    "elevation_deg": item.result.elevation_deg,
                    "x_m": item.display_point.x if item.display_point else None,
                    "y_m": item.display_point.y if item.display_point else None,
                    "z_m": item.display_point.z if item.display_point else None,
                    "used_pairs": item.result.used_pairs,
                    "rejected_pairs": item.result.rejected_pairs,
                    "tdoa_policy": item.tdoa_policy,
                    "informative": item.informative,
                    "message": item.result.message,
                    "reason": item.reason,
                    "preprocess_s": item.preprocess_s,
                    "gcc_s": item.gcc_s,
                    "localize_s": item.localize_s,
                    "total_s": item.total_s,
                    "runtime_error": item.runtime_error,
                }
            )

    final_item = accepted[-1] if accepted else (estimates[-1] if estimates else None)
    payload: dict[str, object] = {
        "run_id": run_id,
        "saved_at_local": datetime.now().isoformat(timespec="seconds"),
        "streamlit_options": {
            "config_path": str(options.get("config_path", "")),
            "signal_path": str(options.get("signal_path", "")),
            "metadata_path": str(options.get("metadata_path", "")),
            "mode": str(options.get("mode", "")),
            "window_selection": str(options.get("window_selection", "")),
            "window_size_samples": int(options.get("window_size_samples", 0)),
            "interpolation": int(options.get("interpolation", 0)),
            "subbands": int(options.get("subbands", 0)),
            "tdoa_policy": str(options.get("tdoa_policy", "")),
            "min_quality": float(options.get("min_quality", 0.0)),
            "min_confidence": float(options.get("min_confidence", 0.0)),
        },
        "summary": {
            "windows_total": len(estimates),
            "windows_displayed": len(accepted),
            "windows_ok": sum(1 for item in estimates if item.result.status == "ok"),
            "windows_informative": sum(1 for item in estimates if item.informative),
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
    }
    if final_item is not None:
        payload["final_result"] = {
            "timestamp_s": final_item.timestamp,
            "status": final_item.result.status,
            "mode": final_item.result.mode,
            "confidence": final_item.result.confidence,
            "azimuth_deg": final_item.result.azimuth_deg,
            "elevation_deg": final_item.result.elevation_deg,
            "position": (
                {"x": final_item.result.position.x, "y": final_item.result.position.y, "z": final_item.result.position.z}
                if final_item.result.position is not None
                else None
            ),
            "direction": (
                {"x": final_item.result.direction.x, "y": final_item.result.direction.y, "z": final_item.result.direction.z}
                if final_item.result.direction is not None
                else None
            ),
            "residual_error2": final_item.result.residual_error2,
            "used_pairs": final_item.result.used_pairs,
            "rejected_pairs": final_item.result.rejected_pairs,
            "reliable": final_item.result.reliable,
            "message": final_item.result.message,
        }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(csv_path), str(json_path)

def _sidebar() -> dict[str, object]:
    st.sidebar.header("Входные данные")
    config_path = _path_selectbox("Конфигурация JSON", Path("configs"), ("*.json",), DEFAULT_CONFIG)
    st.sidebar.caption(_weather_caption_for_config(config_path))
    signal_path = _path_selectbox("Сигнал CSV/WAV", Path("data/synthetic"), ("*.csv", "*.wav"), DEFAULT_SIGNAL)
    metadata_default = metadata_path_for_signal(signal_path)
    metadata_path = _path_selectbox("Метаданные JSON", Path("data/synthetic"), ("*.meta.json",), metadata_default)

    st.sidebar.header("Режим")
    mode = st.sidebar.radio(
        "Режим локализации",
        ["direction", "position"],
        index=0,
        horizontal=True,
        format_func=_mode_label,
    )
    window_selection = st.sidebar.radio(
        "Окна",
        ["trajectory", "first"],
        index=0,
        horizontal=True,
        format_func=_window_selection_label,
    )
    window_size_samples = st.sidebar.slider("Длина окна, отсчетов", min_value=1024, max_value=32768, value=12288, step=512)

    st.sidebar.header("3D-отображение")
    direction_range = st.sidebar.number_input("Длина луча направления, м", min_value=0.1, max_value=500.0, value=10.0, step=0.5)
    show_truth = st.sidebar.checkbox("Истинная траектория", value=True)
    show_position_points = st.sidebar.checkbox("Оценки положения", value=True)
    show_direction_rays = st.sidebar.checkbox("Лучи направления", value=True)

    with st.sidebar.expander("TDOA", expanded=False):
        interpolation = st.slider("Интерполяция GCC-PHAT", min_value=1, max_value=64, value=16, step=1)
        subbands = st.slider("Субполосы GCC", min_value=1, max_value=8, value=1, step=1)
        tdoa_policy = st.selectbox(
            "Политика TDOA",
            ["auto", "manual"],
            index=0,
            format_func=_tdoa_policy_mode_label,
        )
        min_quality = st.slider("Мин. качество пары", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        min_confidence = st.slider("Мин. уверенность результата", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        allow_low_confidence = st.checkbox("Показывать низкую уверенность", value=False)
        enable_delay_continuity = st.checkbox("Непрерывность задержки (развертка)", value=True)
        display_max_speed = st.slider("Макс. скорость отображения, м/с", min_value=0.5, max_value=10.0, value=3.0, step=0.5)
        stabilize_first_direction = st.checkbox("Стабилизация режима 'первые' (направление)", value=True)
        max_direction_jump_deg = st.slider("Макс. скачок направления, град", min_value=5, max_value=90, value=20, step=1)
        max_direction_rate_deg_s = st.slider("Макс. угловая скорость, град/с", min_value=10, max_value=720, value=180, step=10)

    return {
        "config_path": str(config_path),
        "signal_path": str(signal_path),
        "metadata_path": str(metadata_path),
        "mode": mode,
        "window_selection": window_selection,
        "window_size_samples": int(window_size_samples),
        "interpolation": int(interpolation),
        "subbands": int(subbands),
        "tdoa_policy": str(tdoa_policy),
        "min_quality": float(min_quality),
        "min_confidence": float(min_confidence),
        "allow_low_confidence": bool(allow_low_confidence),
        "enable_delay_continuity": bool(enable_delay_continuity),
        "display_max_speed": float(display_max_speed),
        "stabilize_first_direction": bool(stabilize_first_direction),
        "max_direction_jump_deg": float(max_direction_jump_deg),
        "max_direction_rate_deg_s": float(max_direction_rate_deg_s),
        "direction_range": float(direction_range),
        "show_truth": bool(show_truth),
        "show_position_points": bool(show_position_points),
        "show_direction_rays": bool(show_direction_rays),
    }


def _path_selectbox(label: str, root: Path, patterns: tuple[str, ...], default: Path) -> Path:
    files = _list_files(root, patterns)
    choices = [str(path) for path in files]
    default_text = str(default)
    if default_text not in choices:
        choices.insert(0, default_text)
    choices.append("Свой путь...")
    index = choices.index(default_text) if default_text in choices else 0
    selected = st.sidebar.selectbox(label, choices, index=index)
    if selected == "Свой путь...":
        return Path(st.sidebar.text_input(f"{label}: путь", value=default_text))
    return Path(selected)


def _list_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    if not root.exists():
        return []
    result: list[Path] = []
    for pattern in patterns:
        result.extend(path for path in root.rglob(pattern) if path.is_file())
    return sorted(result, key=lambda path: str(path).lower())


def _read_signal(path: Path, config: AppConfig) -> AudioFrame:
    if path.suffix.lower() == ".wav":
        return read_wav_audio(path)
    return read_csv_audio(path, config.audio.sample_rate)


def _read_optional_metadata(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return read_metadata(path)


def _split_frames_for_ui(frame: AudioFrame, config: AppConfig) -> list[AudioFrame]:
    windows = split_frames(frame, config.audio.window_size, config.audio.overlap)
    if windows:
        return windows
    return [frame] if frame.sample_count >= 2 else []


def _select_windows_for_ui(
    config: AppConfig,
    windows: Sequence[AudioFrame],
    truth_samples: Sequence[TruthTrajectoryPoint],
    metadata: dict[str, object] | None,
    selection_mode: str,
) -> list[AudioFrame]:
    if not windows:
        return []
    start_index, end_index = _metadata_window_bounds_indices(config, windows, truth_samples, metadata)
    if end_index <= start_index:
        return []
    if selection_mode == "first":
        return list(windows[start_index:end_index])
    if truth_samples:
        timed_truth = [item for item in truth_samples if item.timestamp is not None]
    else:
        timed_truth = []
    if not timed_truth:
        return list(windows[start_index:end_index])

    expected_offset_s = _source_timing_offset_from_metadata(metadata)
    selected: list[AudioFrame] = []
    next_window_index = start_index
    for truth in sorted(timed_truth, key=lambda item: float(item.timestamp or 0.0)):
        expected_time = (
            float(truth.timestamp or 0.0)
            + distance(truth.point, config.microphone_array.center) / config.environment.sound_speed
            + expected_offset_s
        )
        if next_window_index >= end_index:
            break
        best_index = min(
            range(next_window_index, end_index),
            key=lambda index: _window_time_distance(windows[index], expected_time),
        )
        selected.append(windows[best_index])
        next_window_index = best_index + 1
    return selected if selected else list(windows[start_index:end_index])


def _metadata_window_bounds_indices(
    config: AppConfig,
    windows: Sequence[AudioFrame],
    truth_samples: Sequence[TruthTrajectoryPoint],
    metadata: dict[str, object] | None,
) -> tuple[int, int]:
    timed_truth = [item for item in truth_samples if item.timestamp is not None]
    if not timed_truth:
        return 0, len(windows)
    expected_offset_s = _source_timing_offset_from_metadata(metadata)
    sorted_truth = sorted(timed_truth, key=lambda item: float(item.timestamp or 0.0))
    first_truth = sorted_truth[0]
    last_truth = sorted_truth[-1]
    expected_start_time = (
        float(first_truth.timestamp or 0.0)
        + distance(first_truth.point, config.microphone_array.center) / config.environment.sound_speed
        + expected_offset_s
    )
    expected_end_time = (
        float(last_truth.timestamp or 0.0)
        + distance(last_truth.point, config.microphone_array.center) / config.environment.sound_speed
        + expected_offset_s
    )
    start_index = min(range(len(windows)), key=lambda index: _window_time_distance(windows[index], expected_start_time))
    end_index_inclusive = min(range(start_index, len(windows)), key=lambda index: _window_time_distance(windows[index], expected_end_time))
    return start_index, min(len(windows), end_index_inclusive + 1)


def _window_time_distance(window: AudioFrame, timestamp: float) -> float:
    center = window.timestamp + 0.5 * window.duration
    return abs(timestamp - center)


def _estimate_windows(
    config: AppConfig,
    windows: Sequence[AudioFrame],
    options: dict[str, object],
    truth_samples: Sequence[TruthTrajectoryPoint],
    metadata: dict[str, object] | None,
) -> list[WindowEstimate]:
    estimates: list[WindowEstimate] = []
    continuity_reference: dict[tuple[int, int], float] = {}
    for index, window in enumerate(windows):
        started_total = time.perf_counter()
        started_pre = time.perf_counter()
        processed, diagnostics = preprocess_frame(window, config.audio)
        preprocess_s = time.perf_counter() - started_pre
        if not diagnostics.informative:
            result = LocalizationResult(
                mode=options["mode"],  # type: ignore[arg-type]
                status="insufficient_signal",
                message=f"окно неинформативно: {diagnostics.reason}",
            )
            truth_index, truth = _truth_for_estimate(index, processed.timestamp, truth_samples, str(options["window_selection"]))
            total_s = time.perf_counter() - started_total
            estimates.append(
                WindowEstimate(
                    index,
                    processed.timestamp,
                    result,
                    None,
                    tuple(),
                    False,
                    truth,
                    truth_index,
                    diagnostics.reason,
                    "manual",
                    preprocess_s,
                    0.0,
                    0.0,
                    total_s,
                    False,
                )
            )
            continue

        try:
            truth_index, truth = _truth_for_estimate(index, processed.timestamp, truth_samples, str(options["window_selection"]))
            started_gcc = time.perf_counter()
            gcc = _estimate_tdoa_compat(
                window,
                config,
                options,
                continuity_reference=continuity_reference if bool(options["enable_delay_continuity"]) else None,
            )
            gcc_s = time.perf_counter() - started_gcc
            if bool(options["enable_delay_continuity"]):
                continuity_reference = {
                    item.pair: item.delay_seconds
                    for item in gcc.estimate.pair_delays
                    if item.weight > 0.0
                }
            started_loc = time.perf_counter()
            result = estimate_localization_from_tdoa(
                config.microphone_array,
                gcc.estimate,
                config.environment.sound_speed,
                options["mode"],  # type: ignore[arg-type]
                min_quality=float(options["min_quality"]),
            )
            localize_s = time.perf_counter() - started_loc
            display_point = _display_point(config, result, truth, float(options["direction_range"]))
            policy_label = _policy_label_from_gcc(gcc)
            total_s = time.perf_counter() - started_total
            estimates.append(
                WindowEstimate(
                    index,
                    processed.timestamp,
                    result,
                    display_point,
                    gcc.estimate.pair_delays,
                    True,
                    truth,
                    truth_index,
                    "",
                    policy_label,
                    preprocess_s,
                    gcc_s,
                    localize_s,
                    total_s,
                    False,
                )
            )
        except ValueError as exc:
            result = LocalizationResult(
                mode=options["mode"],
                status="invalid_input",
                message=f"Некорректные данные: {exc}",
            )  # type: ignore[arg-type]
            truth_index, truth = _truth_for_estimate(index, processed.timestamp, truth_samples, str(options["window_selection"]))
            total_s = time.perf_counter() - started_total
            estimates.append(
                WindowEstimate(
                    index,
                    processed.timestamp,
                    result,
                    None,
                    tuple(),
                    True,
                    truth,
                    truth_index,
                    str(exc),
                    "manual",
                    preprocess_s,
                    0.0,
                    0.0,
                    total_s,
                    True,
                )
            )
    return estimates


def _phat_weight_for_mode(mode: str) -> float:
    _ = mode
    return 1.0


def _estimate_tdoa_compat(
    window: AudioFrame,
    config: AppConfig,
    options: dict[str, object],
    *,
    continuity_reference: dict[tuple[int, int], float] | None,
):
    kwargs: dict[str, object] = {
        "interpolation": int(options["interpolation"]),
        "phat_weight": _phat_weight_for_mode(str(options["mode"])),
        "policy_mode": str(options.get("tdoa_policy", "auto")),
    }
    signature = inspect.signature(estimate_tdoa_gcc_phat)
    if "subbands" in signature.parameters:
        kwargs["subbands"] = int(options["subbands"])
    if "continuity_reference" in signature.parameters:
        kwargs["continuity_reference"] = continuity_reference
    return estimate_tdoa_gcc_phat(
        window,
        config.microphone_array,
        config.environment.sound_speed,
        **kwargs,
    )


def _source_timing_offset_from_metadata(metadata: dict[str, object] | None) -> float:
    if not isinstance(metadata, dict):
        return 0.0
    source = metadata.get("source")
    if not isinstance(source, dict):
        return 0.0
    kind = str(source.get("kind", "")).strip().lower()
    if kind == "pulse":
        return 0.006
    if kind == "sine":
        return 0.015
    if kind == "chirp":
        return 0.01
    if kind in {"multitone_burst", "band_noise"}:
        return 0.0125
    if kind == "am_fm_tone":
        return 0.0175
    return 0.0


def _display_point(
    config: AppConfig,
    result: LocalizationResult,
    truth: TruthTrajectoryPoint | None,
    direction_range: float,
) -> Vec3 | None:
    if result.mode == "position" and result.position is not None:
        return result.position
    if result.direction is not None:
        display_range = direction_range
        if truth is not None:
            truth_range = norm(truth.point - config.microphone_array.center)
            if truth_range > 1e-9:
                display_range = truth_range
        return config.microphone_array.center + result.direction * display_range
    return None


def _truth_for_estimate(
    index: int,
    timestamp: float,
    truth_samples: Sequence[TruthTrajectoryPoint],
    selection_mode: str,
) -> tuple[int | None, TruthTrajectoryPoint | None]:
    if selection_mode == "trajectory" and index < len(truth_samples):
        return index, truth_samples[index]
    nearest = _nearest_truth(timestamp, truth_samples)
    if nearest is None:
        return None, None
    try:
        return truth_samples.index(nearest), nearest
    except ValueError:
        return None, nearest


def _filter_display_estimates(
    estimates: Sequence[WindowEstimate],
    mode: str,
    window_selection: str,
    max_speed_m_s: float,
    stabilize_first_direction: bool,
    max_direction_jump_deg: float,
    max_direction_rate_deg_s: float,
) -> list[WindowEstimate]:
    if len(estimates) < 2:
        return list(estimates)
    ordered = sorted(estimates, key=lambda estimate: estimate.timestamp)

    if mode == "position":
        filtered: list[WindowEstimate] = []
        for item in ordered:
            if item.display_point is None:
                continue
            if not filtered:
                filtered.append(item)
                continue
            previous = filtered[-1]
            dt = item.timestamp - previous.timestamp
            if dt <= 1e-9:
                continue
            if previous.display_point is None:
                filtered.append(item)
                continue
            speed = norm(item.display_point - previous.display_point) / dt
            if speed > max_speed_m_s:
                continue
            filtered.append(item)
        return filtered

    if mode == "direction" and window_selection == "first" and stabilize_first_direction:
        filtered: list[WindowEstimate] = []
        pending: list[WindowEstimate] = []
        for item in ordered:
            if item.display_point is None or item.result.direction is None:
                continue
            if not filtered:
                filtered.append(item)
                continue
            previous = filtered[-1]
            if previous.result.direction is None:
                filtered.append(item)
                continue
            dt = item.timestamp - previous.timestamp
            if dt <= 1e-9:
                continue
            jump_deg = _direction_jump_deg(previous, item)
            rate_deg_s = jump_deg / dt
            # In "first" mode we do not align windows to truth timestamps, so
            # we suppress implausible direction spikes that form normal-axis artifacts.
            direct_ok = jump_deg <= max_direction_jump_deg or item.result.confidence >= 0.9
            rate_ok = rate_deg_s <= max_direction_rate_deg_s or item.result.confidence >= 0.9
            if direct_ok and rate_ok:
                if pending:
                    filtered.extend(pending)
                    pending = []
                filtered.append(item)
                continue

            # A real trajectory can move to a new branch after a bad or weakly
            # observable window. Accept it after consecutive raw estimates agree.
            if pending:
                pending_jump = _direction_jump_deg(pending[-1], item)
                if pending_jump <= max_direction_jump_deg:
                    pending.append(item)
                    if len(pending) >= 2:
                        filtered.extend(pending)
                        pending = []
                    continue
                pending = [item]
                continue
            pending = [item]
        if pending and len(filtered) < 2:
            filtered.extend(pending)
        return filtered

    return list(ordered)


def _direction_jump_deg(first: WindowEstimate, second: WindowEstimate) -> float:
    if first.result.direction is None or second.result.direction is None:
        return 180.0
    dot_value = (
        first.result.direction.x * second.result.direction.x
        + first.result.direction.y * second.result.direction.y
        + first.result.direction.z * second.result.direction.z
    )
    dot_value = max(-1.0, min(1.0, dot_value))
    return float(np.degrees(np.arccos(dot_value)))


def _summary(
    config: AppConfig,
    frame: AudioFrame,
    windows: Sequence[AudioFrame],
    selected_windows: Sequence[AudioFrame],
    estimates: Sequence[WindowEstimate],
    accepted: Sequence[WindowEstimate],
    truth_samples: Sequence[TruthTrajectoryPoint],
    mode: str,
) -> None:
    ok_count = sum(1 for item in estimates if item.result.status == "ok")
    informative_count = sum(1 for item in estimates if item.informative)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Каналы", len(frame.channels))
    c2.metric("Окна", len(windows))
    c3.metric("Выбрано", len(selected_windows))
    c4.metric("Успешно", ok_count)
    c5.metric("Показано", len(accepted))
    if estimates:
        narrowband_count = sum(1 for item in estimates if item.tdoa_policy == "narrowband")
        wideband_count = sum(1 for item in estimates if item.tdoa_policy == "wideband")
        if narrowband_count or wideband_count:
            st.caption(f"Баланс политики TDOA: узкополосный={narrowband_count}, широкополосный={wideband_count}")
    rms_value, rms_count, rms_unit = _rms_effectiveness_metric(accepted, truth_samples, mode, config.microphone_array.center)
    if rms_value is not None:
        st.caption(f"СКО ошибки ({rms_count} точек): {rms_value:.3f} {rms_unit}")

    st.caption(
        f"Центр решетки: {_fmt_vec(config.microphone_array.center)} | "
        f"скорость звука: {config.environment.sound_speed:.3f} м/с | "
        f"опорные точки: {len(truth_samples)} | информативных окон: {informative_count}"
    )
    informative_timings = [item.total_s for item in estimates if item.informative]
    if informative_timings:
        avg_window_s = float(np.mean(informative_timings))
        max_window_s = float(np.max(informative_timings))
        window_duration_s = config.audio.window_size / config.audio.sample_rate
        st.caption(
            f"Время обработки окна: avg={avg_window_s:.6f} c, max={max_window_s:.6f} c, "
            f"длительность окна={window_duration_s:.6f} c"
        )
    timed_truth_count = sum(1 for item in truth_samples if item.timestamp is not None)
    if timed_truth_count and len(selected_windows) < timed_truth_count:
        st.warning(
            f"Траектория усечена лимитом окон: выбрано {len(selected_windows)} из {timed_truth_count} опорных точек. "
            "Увеличьте лимит окон, чтобы показать всю траекторию."
        )
def _result_panel(estimates: Sequence[WindowEstimate]) -> None:
    st.subheader("Текущий результат")
    if not estimates:
        st.info("Нет принятых оценок.")
        return
    last = estimates[-1]
    result = last.result
    st.metric("Статус", _status_label(result.status))
    st.metric("Уверенность", f"{result.confidence:.3f}")
    st.metric("Время", f"{last.timestamp:.4f} с")
    if result.azimuth_deg is not None:
        st.metric("Азимут", f"{result.azimuth_deg:.2f} град")
    if result.elevation_deg is not None:
        st.metric("Угол места", f"{result.elevation_deg:.2f} град")
    if result.position is not None:
        st.write("Положение:", _fmt_vec(result.position))
    if result.direction is not None:
        st.write("Направление:", _fmt_vec(result.direction))
    if result.message:
        st.warning(f"Сообщение: {result.message}")


def _table(
    estimates: Sequence[WindowEstimate],
    truth_samples: Sequence[TruthTrajectoryPoint],
    mode: str,
    center: Vec3,
) -> None:
    rows = []
    for item in estimates:
        truth = item.truth_sample or _nearest_truth(item.timestamp, truth_samples)
        error_m = distance(item.display_point, truth.point) if truth and item.display_point else None
        error_deg = None
        if mode == "direction" and truth and item.result.direction is not None:
            truth_vec = truth.point - center
            truth_norm = norm(truth_vec)
            if truth_norm > 1e-12:
                truth_dir = truth_vec / truth_norm
                dot_value = max(
                    -1.0,
                    min(
                        1.0,
                        item.result.direction.x * truth_dir.x
                        + item.result.direction.y * truth_dir.y
                        + item.result.direction.z * truth_dir.z,
                    ),
                )
                error_deg = float(np.degrees(np.arccos(dot_value)))
        rows.append(
            {
                "окно": item.window_index,
                "время, с": item.timestamp,
                "статус": _status_label(item.result.status),
                "уверенность": item.result.confidence,
                "азимут, град": item.result.azimuth_deg,
                "угол места, град": item.result.elevation_deg,
                "x, м": item.display_point.x if item.display_point else None,
                "y, м": item.display_point.y if item.display_point else None,
                "z, м": item.display_point.z if item.display_point else None,
                "ошибка по истине, м": error_m,
                "угловая ошибка, град": error_deg,
                "использованные пары": item.result.used_pairs,
                "отброшенные пары": item.result.rejected_pairs,
                "сообщение": item.result.message or item.reason,
                "политика TDOA": _tdoa_policy_label(item.tdoa_policy),
            }
        )
    st.subheader("Оценки по окнам")
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _build_figure(
    config: AppConfig,
    truth_samples: Sequence[TruthTrajectoryPoint],
    estimates: Sequence[WindowEstimate],
    options: dict[str, object],
) -> go.Figure:
    center = config.microphone_array.center
    microphones = list(config.microphone_array.microphones)
    mode = str(options["mode"])
    direction_range = float(options["direction_range"])
    fig = go.Figure()

    _add_points(fig, microphones, "Микрофоны", "#2563eb", size=5)
    if len(microphones) >= 4:
        array_loop = [*microphones, microphones[0]]
        fig.add_trace(
            go.Scatter3d(
                x=[p.x for p in array_loop],
                y=[p.y for p in array_loop],
                z=[p.z for p in array_loop],
                mode="lines",
                name="Плоскость решетки",
                line={"color": "#2563eb", "width": 4},
            )
        )

    if options["show_truth"] and truth_samples:
        truth_points = [item.point for item in truth_samples]
        truth_trace_index = len(fig.data)
        fig.add_trace(
            go.Scatter3d(
                x=[p.x for p in truth_points],
                y=[p.y for p in truth_points],
                z=[p.z for p in truth_points],
                mode="lines+markers",
                name="Истинная траектория",
                line={"color": "#16a34a", "width": 5},
                marker={"size": 4, "color": "#16a34a"},
            )
        )
    else:
        truth_trace_index = None

    estimate_points = [item.display_point for item in estimates if item.display_point is not None]
    if options["show_position_points"] and estimate_points:
        estimate_trace_index = len(fig.data)
        fig.add_trace(
            go.Scatter3d(
                x=[p.x for p in estimate_points],
                y=[p.y for p in estimate_points],
                z=[p.z for p in estimate_points],
                mode="lines+markers",
                name="Показанные оценки",
                line={"color": "#dc2626", "width": 4},
                marker={
                    "size": [5 + 5 * max(0.0, min(1.0, item.result.confidence)) for item in estimates if item.display_point],
                    "color": "#dc2626",
                },
                text=[
                    _hover_text(item)
                    for item in estimates
                    if item.display_point is not None
                ],
                hoverinfo="text",
            )
        )
    else:
        estimate_trace_index = None

    if options["show_direction_rays"]:
        for item in estimates:
            if item.result.direction is None:
                continue
            endpoint = center + item.result.direction * float(options["direction_range"])
            fig.add_trace(
                go.Scatter3d(
                    x=[center.x, endpoint.x],
                    y=[center.y, endpoint.y],
                    z=[center.z, endpoint.z],
                    mode="lines",
                    name="Луч направления",
                    showlegend=False,
                    line={"color": "rgba(220,38,38,0.35)", "width": 3},
                    hoverinfo="text",
                    text=[_hover_text(item), _hover_text(item)],
                )
            )

    all_points = [center, *microphones, *(item.point for item in truth_samples), *estimate_points]
    if mode == "direction":
        all_points = [center, *microphones, *estimate_points]
    axis_range = _axis_range(all_points)
    fig.update_layout(
        height=760,
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        scene={
            "xaxis": {"title": "X, м", "range": axis_range[0]},
            "yaxis": {"title": "Y, м", "range": axis_range[1]},
            "zaxis": {"title": "Z, м", "range": axis_range[2]},
            "aspectmode": "data",
        },
        legend={"orientation": "h", "y": 1.02, "x": 0.0},
    )
    return fig


def _add_points(fig: go.Figure, points: Sequence[Vec3], name: str, color: str, size: int) -> None:
    fig.add_trace(
        go.Scatter3d(
            x=[p.x for p in points],
            y=[p.y for p in points],
            z=[p.z for p in points],
            mode="markers+text",
            name=name,
            marker={"size": size, "color": color},
            text=[f"M{index}" for index, _ in enumerate(points)],
            textposition="top center",
        )
    )


def _axis_range(points: Sequence[Vec3]) -> tuple[list[float], list[float], list[float]]:
    if not points:
        points = [Vec3()]
    xs = [p.x for p in points]
    ys = [p.y for p in points]
    zs = [p.z for p in points]
    span = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs), 1.0)
    pad = span * 0.15
    return [min(xs) - pad, max(xs) + pad], [min(ys) - pad, max(ys) + pad], [min(zs) - pad, max(zs) + pad]


def _nearest_truth(timestamp: float, truth_samples: Sequence[TruthTrajectoryPoint]) -> TruthTrajectoryPoint | None:
    timed = [item for item in truth_samples if item.timestamp is not None]
    if timed:
        return min(timed, key=lambda item: abs(float(item.timestamp or 0.0) - timestamp))
    return truth_samples[0] if truth_samples else None


def _hover_text(item: WindowEstimate) -> str:
    result = item.result
    return (
        f"t={item.timestamp:.4f} с<br>"
        f"статус={_status_label(result.status)}<br>"
        f"уверенность={result.confidence:.3f}<br>"
        f"азимут={_fmt_optional(result.azimuth_deg)} град<br>"
        f"угол места={_fmt_optional(result.elevation_deg)} град"
    )


def _fmt_optional(value: float | None) -> str:
    return "н/д" if value is None else f"{value:.2f}"


def _fmt_vec(value: Vec3) -> str:
    return f"({value.x:.3f}, {value.y:.3f}, {value.z:.3f})"


def _policy_label_from_gcc(gcc) -> str:
    counts = getattr(gcc, "policy_counts", None)
    if not counts:
        return "manual"
    narrow = int(counts.get("narrowband", 0))
    wide = int(counts.get("wideband", 0))
    if narrow > wide:
        return "narrowband"
    if wide > narrow:
        return "wideband"
    return "manual"


def _status_label(status: str) -> str:
    labels = {
        "ok": "ок",
        "low_confidence": "низкая уверенность",
        "insufficient_signal": "недостаточный сигнал",
        "invalid_input": "некорректные данные",
    }
    return labels.get(status, status)


def _tdoa_policy_label(value: str) -> str:
    labels = {
        "manual": "ручной",
        "auto": "авто",
        "narrowband": "узкополосный",
        "wideband": "широкополосный",
    }
    return labels.get(value, value)


def _mode_label(value: str) -> str:
    return {
        "direction": "направление (дальнее поле)",
        "position": "позиция (ближнее поле)",
    }.get(value, value)


def _window_selection_label(value: str) -> str:
    return {"trajectory": "траектория", "first": "авто (первые)"}.get(value, value)


def _tdoa_policy_mode_label(value: str) -> str:
    return {"auto": "авто", "manual": "вручную"}.get(value, value)


def _rms_effectiveness_metric(
    estimates: Sequence[WindowEstimate],
    truth_samples: Sequence[TruthTrajectoryPoint],
    mode: str,
    center: Vec3,
) -> tuple[float | None, int, str]:
    sq_errors: list[float] = []
    for item in estimates:
        if item.result.status != "ok":
            continue
        truth = item.truth_sample or _nearest_truth(item.timestamp, truth_samples)
        if truth is None:
            continue
        if mode == "direction":
            if item.result.direction is None:
                continue
            truth_vec = truth.point - center
            truth_norm = norm(truth_vec)
            if truth_norm <= 1e-12:
                continue
            truth_dir = truth_vec / truth_norm
            dot_value = max(
                -1.0,
                min(
                    1.0,
                    item.result.direction.x * truth_dir.x
                    + item.result.direction.y * truth_dir.y
                    + item.result.direction.z * truth_dir.z,
                ),
            )
            angle_deg = float(np.degrees(np.arccos(dot_value)))
            sq_errors.append(angle_deg * angle_deg)
        else:
            if item.result.position is None:
                continue
            error_m = distance(item.result.position, truth.point)
            sq_errors.append(error_m * error_m)
    if not sq_errors:
        return None, 0, "град" if mode == "direction" else "м"
    rms = float(np.sqrt(np.mean(np.asarray(sq_errors, dtype=np.float64))))
    return rms, len(sq_errors), ("град" if mode == "direction" else "м")


if __name__ == "__main__":
    main()
