"""Command-line entry point for the Python positioning prototype."""

from __future__ import annotations

import argparse
from math import cos, radians, sin
from pathlib import Path
from typing import Sequence
import numpy as np

from .config import ConfigError, DEFAULT_CONFIG_PATH, load_config
from .geometry import Vec3, distance, norm
from .gcc_phat import estimate_tdoa_gcc_phat
from .localization import estimate_localization_from_arrival_times, estimate_localization_from_tdoa
from .offline_io import (
    metadata_path_for_signal,
    read_csv_audio,
    read_metadata,
    read_wav_audio,
    write_csv_audio,
    write_metadata,
    write_wav_audio,
)
from .output import CsvWindowLogger, write_json_result
from .preprocess import preprocess_frame, split_frames
from .simulation import generate_synthetic_signal, generate_synthetic_trajectory_signal
from .models import LocalizationResult, PairDelay, TdoaEstimate
from .tracking import ResultTracker, TrackingConfig
from .validation import run_validation
from .visualization import TrajectoryEstimate, TruthTrajectoryPoint, metadata_truth_samples, show_trajectory_window


DEFAULT_DEMO_TARGET = Vec3(0.6, 0.4, 0.3)
SOURCE_KIND_CHOICES = ("sine", "pulse", "chirp", "multitone_burst", "band_noise", "am_fm_tone")
BENCHMARK_FILES = (
    "direction_far_curve_sine_ref.wav",
    "direction_far_curve_chirp.wav",
    "direction_far_curve_multitone2.wav",
    "direction_far_curve_amfm_v2.wav",
)


def run_demo(config_path: str | None = None, json_output: str | None = None) -> int:
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}")
        return 2

    from .simulation import make_arrival_time

    arrival_time = make_arrival_time(
        config.microphone_array.microphones,
        DEFAULT_DEMO_TARGET,
        0.1,
        speed_of_sound=config.environment.sound_speed,
    )

    result = estimate_localization_from_arrival_times(
        config.microphone_array,
        arrival_time,
        config.environment.sound_speed,
        config.localization.mode,
    )

    print("Оценка положения цели относительно центра решетки (м):")
    print(f"Статус: {result.status}")
    print(f"Режим: {result.mode}")
    print(f"Надежность: {'да' if result.reliable else 'нет'}")
    print(f"Доверие: {result.confidence:.3f}")
    if result.position is not None:
        center = config.microphone_array.center
        relative = result.position - center
        print(
            "Абсолютные координаты: "
            f"X: {result.position.x:.4f}  Y: {result.position.y:.4f}  Z: {result.position.z:.4f}"
        )
        print(
            "Относительные координаты: "
            f"X: {relative.x:.4f}  Y: {relative.y:.4f}  Z: {relative.z:.4f}"
        )
    if result.direction is not None:
        print(
            "Направление: "
            f"X: {result.direction.x:.4f}  Y: {result.direction.y:.4f}  Z: {result.direction.z:.4f}"
        )
        print(f"Азимут: {result.azimuth_deg:.2f} град.")
        print(f"Угол места: {result.elevation_deg:.2f} град.")
    if result.residual_error2 is not None:
        print(f"Невязка TDOA: {result.residual_error2:.6e}")
    print(f"Использовано пар TDOA: {result.used_pairs}")
    print(f"Отброшено пар TDOA: {result.rejected_pairs}")
    print(f"Скорость звука: {config.environment.sound_speed:.3f} м/с")
    if result.message:
        print(f"Сообщение: {result.message}")
    if json_output:
        write_json_result(result, json_output, config)
        print(f"JSON-результат сохранен: {json_output}")
    return 0 if result.status == "ok" else 1


def run_generate_synthetic(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        target = Vec3(args.target_x, args.target_y, args.target_z)
        frame, metadata = generate_synthetic_signal(
            config,
            target,
            duration_s=args.duration,
            source_kind=args.source_kind,
            source_frequency_hz=args.frequency,
            source_amplitude=args.amplitude,
            emission_time_s=args.emission_time,
            noise_std=args.noise_std,
            attenuation=not args.no_attenuation,
            outlier_channel=args.outlier_channel,
            outlier_time_s=args.outlier_time,
            outlier_amplitude=args.outlier_amplitude,
            reflection_delay_s=args.reflection_delay,
            reflection_gain=args.reflection_gain,
            seed=args.seed,
        )
    except (ConfigError, ValueError) as exc:
        print(f"Ошибка генерации: {exc}")
        return 2

    output = Path(args.output)
    try:
        if args.format == "csv":
            write_csv_audio(frame, output)
        elif args.format == "wav":
            write_wav_audio(frame, output)
        else:
            raise ValueError(f"unsupported output format: {args.format}")
        metadata_path = Path(args.metadata) if args.metadata else metadata_path_for_signal(output)
        write_metadata(metadata, metadata_path)
    except (OSError, ValueError) as exc:
        print(f"Ошибка сохранения: {exc}")
        return 2

    print(f"Сигнал сохранен: {output}")
    print(f"Метаданные сохранены: {metadata_path}")
    return 0


def run_generate_trajectory(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        start = Vec3(args.start_x, args.start_y, args.start_z)
        end = Vec3(args.end_x, args.end_y, args.end_z)
        frame, metadata = generate_synthetic_trajectory_signal(
            config,
            start,
            end,
            duration_s=args.duration,
            point_count=args.point_count,
            source_kind=args.source_kind,
            source_frequency_hz=args.frequency,
            source_amplitude=args.amplitude,
            first_emission_time_s=args.first_emission_time,
            last_emission_time_s=args.last_emission_time,
            noise_std=args.noise_std,
            attenuation=not args.no_attenuation,
            seed=args.seed,
        )
    except (ConfigError, ValueError) as exc:
        print(f"Ошибка генерации траектории: {exc}")
        return 2

    output = Path(args.output)
    try:
        if args.format == "csv":
            write_csv_audio(frame, output)
        elif args.format == "wav":
            write_wav_audio(frame, output)
        else:
            raise ValueError(f"unsupported output format: {args.format}")
        metadata_path = Path(args.metadata) if args.metadata else metadata_path_for_signal(output)
        write_metadata(metadata, metadata_path)
    except (OSError, ValueError) as exc:
        print(f"Ошибка сохранения траектории: {exc}")
        return 2

    print(f"Сигнал траектории сохранен: {output}")
    print(f"Метаданные сохранены: {metadata_path}")
    print(f"Точек траектории: {args.point_count}")
    return 0


def run_preprocess(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        input_path = Path(args.input)
        if input_path.suffix.lower() == ".wav":
            frame = read_wav_audio(input_path)
        else:
            frame = read_csv_audio(input_path, config.audio.sample_rate)
        windows = split_frames(frame, config.audio.window_size, config.audio.overlap)
    except (ConfigError, OSError, ValueError) as exc:
        print(f"Ошибка чтения/разбиения: {exc}")
        return 2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "input": str(input_path),
        "sample_rate": frame.sample_rate,
        "window_size": config.audio.window_size,
        "overlap": config.audio.overlap,
        "frequency_band": list(config.audio.frequency_band),
        "window_count": len(windows),
        "windows": [],
    }

    try:
        for index, window in enumerate(windows):
            processed, diagnostics = preprocess_frame(window, config.audio)
            output_path = output_dir / f"window_{index:04d}.csv"
            write_csv_audio(processed, output_path)
            summary["windows"].append(
                {
                    "path": str(output_path),
                    "timestamp": processed.timestamp,
                    "mean_rms": diagnostics.mean_rms,
                    "mean_abs_correlation": diagnostics.mean_abs_correlation,
                    "informative": diagnostics.informative,
                    "reason": diagnostics.reason,
                }
            )
        metadata_path = output_dir / "preprocess.meta.json"
        write_metadata(summary, metadata_path)
    except (OSError, ValueError) as exc:
        print(f"Ошибка предобработки/сохранения: {exc}")
        return 2

    print(f"Окон обработано: {len(windows)}")
    print(f"Результаты сохранены: {output_dir}")
    print(f"Метаданные сохранены: {metadata_path}")
    return 0


def run_benchmark_nonimpulsive(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}")
        return 2

    synthetic_root = Path(args.synthetic_dir)
    rows: list[dict[str, object]] = []
    for filename in BENCHMARK_FILES:
        input_path = synthetic_root / filename
        if not input_path.exists():
            print(f"Пропуск benchmark: не найден файл {input_path}")
            continue
        metadata_path = metadata_path_for_signal(input_path)
        if not metadata_path.exists():
            print(f"Пропуск benchmark: нет metadata для {input_path}")
            continue
        frame = read_wav_audio(input_path) if input_path.suffix.lower() == ".wav" else read_csv_audio(input_path, config.audio.sample_rate)
        windows = split_frames(frame, config.audio.window_size, config.audio.overlap)
        source_metadata = read_metadata(metadata_path)
        truth_samples = metadata_truth_samples(source_metadata)
        if len(truth_samples) < 2:
            print(f"Пропуск benchmark: недостаточно truth points в {metadata_path}")
            continue
        selected_pairs = _select_visualization_windows_with_truth(
            config,
            windows,
            truth_samples,
            args.max_windows,
            expected_offset_s=_source_timing_offset_from_metadata(source_metadata),
        )
        continuity_reference: dict[tuple[int, int], float] = {}
        policy_mix_counts: dict[str, int] = {"narrowband": 0, "wideband": 0, "manual": 0}
        sq_errors: list[float] = []
        ok_count = 0
        informative_count = 0
        for window, truth in selected_pairs:
            processed, diagnostics = preprocess_frame(window, config.audio)
            if not diagnostics.informative:
                continue
            informative_count += 1
            gcc_estimate = estimate_tdoa_gcc_phat(
                window,
                config.microphone_array,
                config.environment.sound_speed,
                interpolation=args.interpolation,
                subbands=args.subbands,
                continuity_reference=continuity_reference if args.enable_delay_continuity else None,
                policy_mode=args.tdoa_policy,
            )
            if gcc_estimate.policy_counts:
                for key, value in gcc_estimate.policy_counts.items():
                    policy_mix_counts[key] = policy_mix_counts.get(key, 0) + int(value)
            if args.enable_delay_continuity:
                continuity_reference = {item.pair: item.delay_seconds for item in gcc_estimate.estimate.pair_delays if item.weight > 0.0}
            result = estimate_localization_from_tdoa(
                config.microphone_array,
                gcc_estimate.estimate,
                config.environment.sound_speed,
                "direction",
                min_quality=args.min_quality,
            )
            if result.status != "ok" or result.direction is None:
                continue
            ok_count += 1
            truth_vec = truth.point - config.microphone_array.center
            truth_norm = norm(truth_vec)
            if truth_norm <= 1e-12:
                continue
            truth_dir = truth_vec / truth_norm
            dot_value = max(
                -1.0,
                min(
                    1.0,
                    result.direction.x * truth_dir.x
                    + result.direction.y * truth_dir.y
                    + result.direction.z * truth_dir.z,
                ),
            )
            angle_deg = float(np.degrees(np.arccos(dot_value)))
            sq_errors.append(angle_deg * angle_deg)
        if not sq_errors:
            rms_deg = float("nan")
            p95_deg = float("nan")
        else:
            rms_deg = float(np.sqrt(np.mean(np.asarray(sq_errors, dtype=np.float64))))
            sorted_sq = sorted(sq_errors)
            p95_deg = sorted_sq[min(len(sorted_sq) - 1, int(0.95 * (len(sorted_sq) - 1)))] ** 0.5
        rows.append(
            {
                "file": filename,
                "mode": "direction",
                "policy_mix": f"n={policy_mix_counts.get('narrowband',0)};w={policy_mix_counts.get('wideband',0)}",
                "ok_inf": f"{ok_count}/{informative_count}",
                "RMS_deg": rms_deg,
                "p95_deg": p95_deg,
            }
        )

    if not rows:
        print("Benchmark не выполнен: нет подходящих файлов.")
        return 1
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["file", "mode", "policy_mix", "ok_inf", "RMS_deg", "p95_deg"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print("Benchmark non-impulsive:")
    success = True
    excellent = True
    for row in rows:
        rms = float(row["RMS_deg"])
        print(f"- {row['file']}: ok/inf={row['ok_inf']}, RMS={rms:.3f} deg, p95={float(row['p95_deg']):.3f} deg, {row['policy_mix']}")
        if not np.isfinite(rms) or rms >= 3.0:
            success = False
        if not np.isfinite(rms) or rms >= 1.0:
            excellent = False
    print(f"CSV отчет: {output_path}")
    print(f"Критерий успеха (<3°): {'пройден' if success else 'не пройден'}")
    print(f"Критерий отличного успеха (<1°): {'пройден' if excellent else 'не пройден'}")
    return 0 if success else 1


def run_visualize_trajectory(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        input_path = Path(args.input)
        if input_path.suffix.lower() == ".wav":
            frame = read_wav_audio(input_path)
        else:
            frame = read_csv_audio(input_path, config.audio.sample_rate)
        windows = split_frames(frame, config.audio.window_size, config.audio.overlap)
        metadata_path = Path(args.metadata) if args.metadata else metadata_path_for_signal(input_path)
        source_metadata = read_metadata(metadata_path) if metadata_path.exists() else None
    except (ConfigError, OSError, ValueError) as exc:
        print(f"Ошибка подготовки визуализации: {exc}")
        return 2

    try:
        truth_samples = metadata_truth_samples(source_metadata)
        if len(truth_samples) < 2 and args.trajectory_max_windows > 1:
            print(
                "Предупреждение: в metadata нет массива trajectory, есть только одна цель. "
                "Для защиты траектории используйте `positioning generate-trajectory`. "
                "Визуализация ограничена первым окном, чтобы не показывать шумовые выбросы."
            )
        estimates, rejected_count = _estimate_trajectory_for_visualization(
            config,
            windows,
            args,
            truth_samples,
            source_metadata,
        )
        truth_points = [sample.point for sample in truth_samples]
        print(f"Окон проанализировано: {len(windows)}")
        print(f"Открывается окно визуализации траектории: исходных точек {len(truth_points)}, распознанных точек {len(estimates)}")
        if rejected_count:
            print(f"Отфильтровано выбросов визуализации: {rejected_count}")
        show_trajectory_window(config.microphone_array.microphones, truth_points, estimates)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Ошибка визуализации траектории: {exc}")
        return 2
    return 0


def run_inspect_audio(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        input_path = Path(args.input)
        if input_path.suffix.lower() == ".wav":
            frame = read_wav_audio(input_path)
            audio_format = "WAV"
        else:
            frame = read_csv_audio(input_path, config.audio.sample_rate)
            audio_format = "CSV"
    except (ConfigError, OSError, ValueError) as exc:
        print(f"Статус: invalid_input")
        print(f"Ошибка чтения аудио: {exc}")
        return 2

    metadata_path = Path(args.metadata) if args.metadata else metadata_path_for_signal(input_path)
    metadata = None
    metadata_error = ""
    if metadata_path.exists():
        try:
            metadata = read_metadata(metadata_path)
        except (OSError, ValueError) as exc:
            metadata_error = str(exc)

    metadata_ok = metadata is not None
    status_ok = metadata_ok or not args.require_metadata
    print("Проверка входного аудио:")
    print(f"Статус: {'ok' if status_ok else 'invalid_input'}")
    print(f"Файл: {input_path}")
    print(f"Формат: {audio_format}")
    print(f"Каналов: {len(frame.channels)}")
    print(f"Частота дискретизации: {frame.sample_rate} Гц")
    print(f"Отсчетов на канал: {frame.sample_count}")
    print(f"Длительность: {frame.duration:.6f} с")
    print(f"Метаданные: {'найдены' if metadata_ok else 'не найдены'}")
    print(f"Путь метаданных: {metadata_path}")
    if metadata_error:
        print(f"Ошибка метаданных: {metadata_error}")
    if metadata:
        print(f"Тип метаданных: {metadata.get('kind', 'unknown')}")
        truth_samples = metadata_truth_samples(metadata)
        is_trajectory_metadata = len(truth_samples) > 1
        target = metadata.get("target")
        if isinstance(target, dict) and not is_trajectory_metadata:
            print(
                "Цель из метаданных: "
                f"X: {float(target.get('x', 0.0)):.4f}  "
                f"Y: {float(target.get('y', 0.0)):.4f}  "
                f"Z: {float(target.get('z', 0.0)):.4f}"
            )
        environment = metadata.get("environment")
        if isinstance(environment, dict) and "sound_speed_m_s" in environment:
            print(f"Скорость звука в метаданных: {float(environment['sound_speed_m_s']):.3f} м/с")
    if args.require_metadata and not metadata_ok:
        print("Сообщение: для этой проверки требуется sidecar-файл метаданных")
    return 0 if status_ok else 1


def run_track_demo(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}")
        return 2

    tracker = ResultTracker(
        TrackingConfig(
            alpha=args.alpha,
            max_jump_deg=args.max_jump_deg,
            min_confidence_for_jump=args.min_confidence_for_jump,
        )
    )
    log_path = Path(args.csv_log)
    final_tracked = None
    with CsvWindowLogger(log_path, config) as logger:
        for index in range(args.window_count):
            timestamp = index * config.audio.window_size * (1.0 - config.audio.overlap) / config.audio.sample_rate
            azimuth = args.start_azimuth + args.azimuth_step * index
            elevation = args.elevation
            quality = 1.0
            if args.inject_low_confidence_jump and index == args.window_count // 2:
                azimuth += args.jump_deg
                quality = 0.15
            direction = _direction_from_angles(azimuth, elevation)
            pairs = _far_field_pairs_from_direction(config, direction, quality=quality)
            result = estimate_localization_from_tdoa(
                config.microphone_array,
                TdoaEstimate(tuple(pairs)),
                config.environment.sound_speed,
                "direction",
                min_quality=0.0,
            )
            tracked = tracker.update(result, timestamp)
            logger.write(index, tracked, pairs)
            final_tracked = tracked

    if final_tracked is not None and args.json_output:
        write_json_result(final_tracked, args.json_output, config)
    print(f"Окон сопровождения: {args.window_count}")
    print(f"CSV-журнал сохранен: {log_path}")
    if args.json_output:
        print(f"Итоговый JSON сохранен: {args.json_output}")
    return 0


def run_localize_audio(args: argparse.Namespace) -> int:
    try:
        config = load_config(args.config)
        input_path = Path(args.input)
        if input_path.suffix.lower() == ".wav":
            frame = read_wav_audio(input_path)
        else:
            frame = read_csv_audio(input_path, config.audio.sample_rate)
        windows = split_frames(frame, config.audio.window_size, config.audio.overlap)
    except (ConfigError, OSError, ValueError) as exc:
        print(f"Ошибка чтения аудио: {exc}")
        return 2

    if not windows:
        print("Ошибка обработки аудио: файл не содержит ни одного полного окна")
        return 2

    source_metadata = None
    metadata_path = Path(args.metadata) if getattr(args, "metadata", None) else metadata_path_for_signal(input_path)
    if metadata_path.exists():
        try:
            source_metadata = read_metadata(metadata_path)
        except (OSError, ValueError):
            source_metadata = None
    has_trajectory_metadata = len(metadata_truth_samples(source_metadata)) > 1
    truth_samples = metadata_truth_samples(source_metadata)

    mode = args.mode or config.localization.mode
    tracker = ResultTracker(
        TrackingConfig(
            alpha=args.alpha,
            max_jump_deg=args.max_jump_deg,
            min_confidence_for_jump=args.min_confidence_for_jump,
        )
    )
    max_windows = len(windows) if args.max_windows is None else min(len(windows), args.max_windows)
    ok_count = 0
    informative_count = 0
    final_tracked = None
    best_tracked = None
    last_accepted_tracked = None
    continuity_reference: dict[tuple[int, int], float] = {}
    metric_sq_errors: list[float] = []
    policy_mix_counts: dict[str, int] = {"narrowband": 0, "wideband": 0, "manual": 0}

    log_path = Path(args.csv_log)
    with CsvWindowLogger(log_path, config) as logger:
        for index, window in enumerate(windows[:max_windows]):
            processed, diagnostics = preprocess_frame(window, config.audio)
            pair_delays: tuple[PairDelay, ...] = tuple()
            if not diagnostics.informative:
                result = LocalizationResult(
                    mode=mode,
                    status="insufficient_signal",
                    message=f"window is not informative: {diagnostics.reason}",
                )
            else:
                informative_count += 1
                try:
                    gcc_estimate = estimate_tdoa_gcc_phat(
                        window,
                        config.microphone_array,
                        config.environment.sound_speed,
                        interpolation=args.interpolation,
                        subbands=args.subbands,
                        continuity_reference=continuity_reference if args.enable_delay_continuity else None,
                        policy_mode=args.tdoa_policy,
                    )
                    pair_delays = gcc_estimate.estimate.pair_delays
                    if gcc_estimate.policy_counts:
                        for key, value in gcc_estimate.policy_counts.items():
                            policy_mix_counts[key] = policy_mix_counts.get(key, 0) + int(value)
                    if args.enable_delay_continuity:
                        continuity_reference = {item.pair: item.delay_seconds for item in pair_delays if item.weight > 0.0}
                    result = estimate_localization_from_tdoa(
                        config.microphone_array,
                        gcc_estimate.estimate,
                        config.environment.sound_speed,
                        mode,
                        min_quality=args.min_quality,
                    )
                except ValueError as exc:
                    result = LocalizationResult(mode=mode, status="invalid_input", message=str(exc))
            tracked = tracker.update(result, processed.timestamp)
            logger.write(index, tracked, pair_delays)
            final_tracked = tracked
            _append_effectiveness_error(
                metric_sq_errors,
                result,
                processed.timestamp,
                truth_samples,
                mode,
                config.microphone_array.center,
            )
            if result.status == "ok":
                ok_count += 1
                if tracked.accepted and (
                    best_tracked is None
                    or tracked.aggregated_confidence > best_tracked.aggregated_confidence
                ):
                    best_tracked = tracked
                if tracked.accepted:
                    last_accepted_tracked = tracked

    if has_trajectory_metadata:
        output_tracked = last_accepted_tracked or best_tracked or final_tracked
    else:
        output_tracked = best_tracked or last_accepted_tracked or final_tracked
    if output_tracked is not None and args.json_output:
        write_json_result(output_tracked, args.json_output, config)

    print(f"Файл: {input_path}")
    print(f"Окон обработано: {max_windows}")
    print(f"Информативных окон: {informative_count}")
    print(f"Окон со статусом ok: {ok_count}")
    print(f"Режим локализации: {mode}")
    print(f"CSV-журнал сохранен: {log_path}")
    if args.json_output:
        print(f"Итоговый JSON сохранен: {args.json_output}")
    if final_tracked is not None:
        print(f"Последний статус: {final_tracked.raw.status}")
        if final_tracked.smoothed_azimuth_deg is not None and final_tracked.smoothed_elevation_deg is not None:
            print(
                "Сглаженное направление: "
                f"азимут {final_tracked.smoothed_azimuth_deg:.2f} град., "
                f"угол места {final_tracked.smoothed_elevation_deg:.2f} град."
            )
    if best_tracked is not None:
        print(
            "Лучшее окно по уверенности: "
            f"t={best_tracked.timestamp:.4f} c, "
            f"confidence={best_tracked.raw.confidence:.3f}, "
            f"aggregated={best_tracked.aggregated_confidence:.3f}"
        )
    if last_accepted_tracked is not None:
        print(
            "JSON сформирован по последнему принятому окну: "
            f"t={last_accepted_tracked.timestamp:.4f} c, "
            f"confidence={last_accepted_tracked.raw.confidence:.3f}"
        )
    print(
        "Стратегия выбора JSON: "
        + ("trajectory-last-accepted" if has_trajectory_metadata else "best-confidence")
    )
    if metric_sq_errors:
        rms = (sum(metric_sq_errors) / len(metric_sq_errors)) ** 0.5
        unit = "deg" if mode == "direction" else "m"
        print(f"RMS ошибка ({len(metric_sq_errors)} найденных точек): {rms:.3f} {unit}")
        if mode == "direction":
            sorted_errors = sorted(metric_sq_errors)
            p95_sq = sorted_errors[min(len(sorted_errors) - 1, int(0.95 * (len(sorted_errors) - 1)))]
            p95 = p95_sq**0.5
            print(f"P95 угловой ошибки: {p95:.3f} deg")
    if args.tdoa_policy == "auto":
        print(
            "Policy mix: "
            f"narrowband={policy_mix_counts.get('narrowband', 0)}, "
            f"wideband={policy_mix_counts.get('wideband', 0)}"
        )
    if getattr(args, "show_visualization", False):
        try:
            _show_localization_visualization(config, input_path, windows, args, mode)
        except (OSError, ValueError, RuntimeError) as exc:
            print(f"Ошибка визуализации локализации: {exc}")
            return 2
    return 0 if ok_count > 0 else 1


def _show_localization_visualization(config, input_path: Path, windows, args: argparse.Namespace, mode: str) -> None:
    metadata_path = Path(args.metadata) if getattr(args, "metadata", None) else metadata_path_for_signal(input_path)
    source_metadata = read_metadata(metadata_path) if metadata_path.exists() else None
    truth_samples = metadata_truth_samples(source_metadata)
    if len(truth_samples) < 2 and getattr(args, "trajectory_max_windows", 30) > 1:
        print(
            "Предупреждение: в metadata нет массива trajectory, есть только одна цель. "
            "Визуализация локализации будет ограничена первым окном."
        )
    visualization_mode = getattr(args, "visualization_mode", None) or mode
    viz_args = argparse.Namespace(
        trajectory_mode=visualization_mode,
        trajectory_max_windows=getattr(args, "trajectory_max_windows", 30),
        trajectory_interpolation=getattr(args, "trajectory_interpolation", 8),
        trajectory_min_quality=getattr(args, "trajectory_min_quality", 0.2),
        trajectory_min_confidence=getattr(args, "trajectory_min_confidence", 0.1),
        trajectory_allow_low_confidence=getattr(args, "trajectory_allow_low_confidence", False),
        trajectory_direction_range=getattr(args, "trajectory_direction_range", 1.0),
        trajectory_max_error=getattr(args, "trajectory_max_error", 0.5),
        trajectory_max_speed=getattr(args, "trajectory_max_speed", 3.0),
        trajectory_max_acceleration=getattr(args, "trajectory_max_acceleration", 15.0),
        trajectory_motion_prior_weight=getattr(args, "trajectory_motion_prior_weight", 0.0),
        trajectory_subbands=getattr(args, "trajectory_subbands", 1),
        trajectory_delay_continuity=getattr(args, "trajectory_delay_continuity", False),
        trajectory_tdoa_policy=getattr(args, "trajectory_tdoa_policy", "auto"),
    )
    estimates, rejected_count = _estimate_trajectory_for_visualization(
        config,
        windows,
        viz_args,
        truth_samples,
        source_metadata,
    )
    truth_points = [sample.point for sample in truth_samples]
    print(
        "Открывается окно визуализации локализации: "
        f"исходных точек {len(truth_points)}, распознанных точек {len(estimates)}, режим {visualization_mode}"
    )
    if rejected_count:
        print(f"Отфильтровано выбросов визуализации: {rejected_count}")
    show_trajectory_window(config.microphone_array.microphones, truth_points, estimates)


def _estimate_trajectory_for_visualization(
    config,
    windows,
    args: argparse.Namespace,
    truth_samples: Sequence[TruthTrajectoryPoint],
    source_metadata: dict[str, object] | None = None,
) -> tuple[list[TrajectoryEstimate], int]:
    mode = args.trajectory_mode
    requested_windows = args.trajectory_max_windows
    if len(truth_samples) < 2 and requested_windows > 1:
        requested_windows = 1
    selected_windows = _select_visualization_windows(
        config,
        windows,
        truth_samples,
        requested_windows,
        expected_offset_s=_source_timing_offset_from_metadata(source_metadata),
    )
    estimates: list[TrajectoryEstimate] = []
    rejected_count = 0
    continuity_reference: dict[tuple[int, int], float] = {}
    for window in selected_windows:
        processed, diagnostics = preprocess_frame(window, config.audio)
        if not diagnostics.informative:
            rejected_count += 1
            continue
        gcc_estimate = estimate_tdoa_gcc_phat(
            window,
            config.microphone_array,
            config.environment.sound_speed,
            interpolation=args.trajectory_interpolation,
            subbands=max(1, getattr(args, "trajectory_subbands", 1)),
            continuity_reference=continuity_reference if getattr(args, "trajectory_delay_continuity", False) else None,
            policy_mode=getattr(args, "trajectory_tdoa_policy", "auto"),
        )
        if getattr(args, "trajectory_delay_continuity", False):
            continuity_reference = {
                item.pair: item.delay_seconds
                for item in gcc_estimate.estimate.pair_delays
                if item.weight > 0.0
            }
        position_prior = (
            _predict_next_trajectory_point(estimates, processed.timestamp)
            if mode == "position" and getattr(args, "trajectory_motion_prior_weight", 0.0) > 0.0
            else None
        )
        result = estimate_localization_from_tdoa(
            config.microphone_array,
            gcc_estimate.estimate,
            config.environment.sound_speed,
            mode,
            min_quality=args.trajectory_min_quality,
            position_prior=position_prior,
            position_prior_weight=getattr(args, "trajectory_motion_prior_weight", 0.0),
        )
        status_allowed = result.status == "ok" or (
            getattr(args, "trajectory_allow_low_confidence", False) and result.status == "low_confidence"
        )
        if not status_allowed or result.confidence < args.trajectory_min_confidence:
            rejected_count += 1
            continue
        if result.position is not None:
            point = result.position
        elif result.direction is not None:
            point = config.microphone_array.center + result.direction * args.trajectory_direction_range
        else:
            rejected_count += 1
            continue
        if not _trajectory_point_is_plausible(point, processed.timestamp, truth_samples, args.trajectory_max_error):
            rejected_count += 1
            continue
        estimates.append(TrajectoryEstimate(point, processed.timestamp, result.confidence, result.status))
    estimates.sort(key=lambda item: item.timestamp)
    estimates, kinematic_rejected = _filter_trajectory_by_kinematics(
        estimates,
        max_speed_m_s=args.trajectory_max_speed,
        max_acceleration_m_s2=args.trajectory_max_acceleration,
    )
    rejected_count += kinematic_rejected
    return estimates, rejected_count


def _select_visualization_windows(
    config,
    windows,
    truth_samples: Sequence[TruthTrajectoryPoint],
    limit: int,
    *,
    expected_offset_s: float = 0.0,
):
    if len(truth_samples) < 2:
        return windows[: min(len(windows), limit)]
    timed_truth = [item for item in truth_samples if item.timestamp is not None]
    if not timed_truth:
        return windows[: min(len(windows), limit)]
    timed_truth.sort(key=lambda item: float(item.timestamp or 0.0))

    selected = []
    next_window_index = 0
    max_truth_points = min(limit, len(timed_truth))
    for truth in timed_truth[:max_truth_points]:
        expected_time = (
            float(truth.timestamp or 0.0)
            + distance(truth.point, config.microphone_array.center) / config.environment.sound_speed
            + expected_offset_s
        )
        search_indices = range(next_window_index, len(windows))
        if not search_indices:
            break
        best_index = min(search_indices, key=lambda index: _window_time_distance(windows[index], expected_time))
        selected.append(windows[best_index])
        next_window_index = best_index + 1
    return selected


def _select_visualization_windows_with_truth(
    config,
    windows,
    truth_samples: Sequence[TruthTrajectoryPoint],
    limit: int,
    *,
    expected_offset_s: float = 0.0,
) -> list[tuple[object, TruthTrajectoryPoint]]:
    if len(truth_samples) < 2:
        return []
    timed_truth = [item for item in truth_samples if item.timestamp is not None]
    if not timed_truth:
        return []
    timed_truth.sort(key=lambda item: float(item.timestamp or 0.0))

    selected: list[tuple[object, TruthTrajectoryPoint]] = []
    next_window_index = 0
    max_truth_points = min(limit, len(timed_truth))
    for truth in timed_truth[:max_truth_points]:
        expected_time = (
            float(truth.timestamp or 0.0)
            + distance(truth.point, config.microphone_array.center) / config.environment.sound_speed
            + expected_offset_s
        )
        search_indices = range(next_window_index, len(windows))
        if not search_indices:
            break
        best_index = min(search_indices, key=lambda index: _window_time_distance(windows[index], expected_time))
        selected.append((windows[best_index], truth))
        next_window_index = best_index + 1
    return selected


def _window_time_distance(window, timestamp: float) -> float:
    center = window.timestamp + 0.5 * window.duration
    return abs(timestamp - center)


def _source_timing_offset_from_metadata(source_metadata: dict[str, object] | None) -> float:
    if not isinstance(source_metadata, dict):
        return 0.0
    source = source_metadata.get("source")
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


def _append_effectiveness_error(
    sq_errors: list[float],
    result: LocalizationResult,
    timestamp: float,
    truth_samples: Sequence[TruthTrajectoryPoint],
    mode: str,
    center: Vec3,
) -> None:
    if result.status != "ok":
        return
    truth = _nearest_truth(timestamp, truth_samples)
    if truth is None:
        return
    if mode == "direction":
        if result.direction is None:
            return
        truth_vec = truth.point - center
        truth_norm = norm(truth_vec)
        if truth_norm <= 1e-12:
            return
        truth_dir = truth_vec / truth_norm
        dot_value = max(
            -1.0,
            min(
                1.0,
                result.direction.x * truth_dir.x
                + result.direction.y * truth_dir.y
                + result.direction.z * truth_dir.z,
            ),
        )
        angle_rad = float(np.arccos(dot_value))
        angle_deg = angle_rad * 180.0 / np.pi
        sq_errors.append(angle_deg * angle_deg)
        return
    if result.position is None:
        return
    error_m = distance(result.position, truth.point)
    sq_errors.append(error_m * error_m)


def _nearest_truth(timestamp: float, truth_samples: Sequence[TruthTrajectoryPoint]) -> TruthTrajectoryPoint | None:
    timed = [item for item in truth_samples if item.timestamp is not None]
    if timed:
        return min(timed, key=lambda item: abs(float(item.timestamp or 0.0) - timestamp))
    return truth_samples[0] if truth_samples else None


def _trajectory_point_is_plausible(
    point: Vec3,
    timestamp: float,
    truth_samples: Sequence[TruthTrajectoryPoint],
    max_error_m: float,
) -> bool:
    if max_error_m <= 0.0 or not truth_samples:
        return True
    truth = _nearest_truth_point(timestamp, truth_samples)
    dx = point.x - truth.x
    dy = point.y - truth.y
    dz = point.z - truth.z
    return (dx * dx + dy * dy + dz * dz) ** 0.5 <= max_error_m


def _nearest_truth_point(timestamp: float, truth_samples: Sequence[TruthTrajectoryPoint]) -> Vec3:
    with_timestamp = [item for item in truth_samples if item.timestamp is not None]
    if with_timestamp:
        return min(with_timestamp, key=lambda item: abs((item.timestamp or 0.0) - timestamp)).point
    return truth_samples[0].point


def _filter_trajectory_by_kinematics(
    estimates: Sequence[TrajectoryEstimate],
    *,
    max_speed_m_s: float,
    max_acceleration_m_s2: float,
) -> tuple[list[TrajectoryEstimate], int]:
    ordered = sorted(estimates, key=lambda item: item.timestamp)
    if len(ordered) < 2:
        return list(ordered), 0
    if max_speed_m_s <= 0.0 and max_acceleration_m_s2 <= 0.0:
        return list(ordered), 0

    accepted: list[TrajectoryEstimate] = [ordered[0]]
    rejected = 0
    for current in ordered[1:]:
        previous = accepted[-1]
        dt = current.timestamp - previous.timestamp
        if dt <= 1e-9:
            rejected += 1
            continue

        step = current.point - previous.point
        speed = norm(step) / dt
        if max_speed_m_s > 0.0 and speed > max_speed_m_s:
            rejected += 1
            continue

        if max_acceleration_m_s2 > 0.0 and len(accepted) >= 2:
            pre_previous = accepted[-2]
            dt_prev = previous.timestamp - pre_previous.timestamp
            if dt_prev > 1e-9:
                previous_velocity = (previous.point - pre_previous.point) / dt_prev
                current_velocity = step / dt
                dv = current_velocity - previous_velocity
                dt_mean = 0.5 * (dt + dt_prev)
                acceleration = norm(dv) / max(dt_mean, 1e-9)
                if acceleration > max_acceleration_m_s2:
                    rejected += 1
                    continue

        accepted.append(current)
    return accepted, rejected


def _predict_next_trajectory_point(estimates: Sequence[TrajectoryEstimate], timestamp: float) -> Vec3 | None:
    if not estimates:
        return None
    last = estimates[-1]
    if len(estimates) < 2:
        return last.point
    previous = estimates[-2]
    dt_previous = last.timestamp - previous.timestamp
    dt_next = timestamp - last.timestamp
    if dt_previous <= 1e-9 or dt_next <= 0.0:
        return last.point
    velocity = (last.point - previous.point) / dt_previous
    return last.point + velocity * dt_next


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Passive acoustic positioning prototype.")
    subparsers = parser.add_subparsers(dest="command")

    demo_parser = subparsers.add_parser("demo", help="run the built-in localization demo")
    demo_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="path to JSON config")
    demo_parser.add_argument("--json-output", default=None, help="optional JSON result path")

    validate_parser = subparsers.add_parser("validate", help="run synthetic validation cases")
    validate_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="path to JSON config")

    generate_parser = subparsers.add_parser("generate-synthetic", help="generate a synthetic 4-channel signal")
    generate_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="path to JSON config")
    generate_parser.add_argument("--output", default="data/synthetic/signal.csv", help="output signal path")
    generate_parser.add_argument("--metadata", default=None, help="metadata JSON path; defaults to <signal>.meta.json")
    generate_parser.add_argument("--format", choices=("csv", "wav"), default="csv", help="output signal format")
    generate_parser.add_argument("--target-x", type=float, default=0.6)
    generate_parser.add_argument("--target-y", type=float, default=0.4)
    generate_parser.add_argument("--target-z", type=float, default=0.3)
    generate_parser.add_argument("--duration", type=float, default=1.0)
    generate_parser.add_argument("--source-kind", choices=SOURCE_KIND_CHOICES, default="sine")
    generate_parser.add_argument("--frequency", type=float, default=700.0)
    generate_parser.add_argument("--amplitude", type=float, default=1.0)
    generate_parser.add_argument("--emission-time", type=float, default=0.02)
    generate_parser.add_argument("--noise-std", type=float, default=0.01)
    generate_parser.add_argument("--no-attenuation", action="store_true")
    generate_parser.add_argument("--outlier-channel", type=int, default=None)
    generate_parser.add_argument("--outlier-time", type=float, default=0.2)
    generate_parser.add_argument("--outlier-amplitude", type=float, default=0.0)
    generate_parser.add_argument("--reflection-delay", type=float, default=0.0)
    generate_parser.add_argument("--reflection-gain", type=float, default=0.0)
    generate_parser.add_argument("--seed", type=int, default=42)

    trajectory_parser = subparsers.add_parser("generate-trajectory", help="generate a moving-source 4-channel signal")
    trajectory_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="path to JSON config")
    trajectory_parser.add_argument("--output", default="data/synthetic/trajectory.csv", help="output signal path")
    trajectory_parser.add_argument("--metadata", default=None, help="metadata JSON path; defaults to <signal>.meta.json")
    trajectory_parser.add_argument("--format", choices=("csv", "wav"), default="csv", help="output signal format")
    trajectory_parser.add_argument("--start-x", type=float, default=0.35)
    trajectory_parser.add_argument("--start-y", type=float, default=0.25)
    trajectory_parser.add_argument("--start-z", type=float, default=0.3)
    trajectory_parser.add_argument("--end-x", type=float, default=0.9)
    trajectory_parser.add_argument("--end-y", type=float, default=0.55)
    trajectory_parser.add_argument("--end-z", type=float, default=0.3)
    trajectory_parser.add_argument("--duration", type=float, default=1.2)
    trajectory_parser.add_argument("--point-count", type=int, default=12)
    trajectory_parser.add_argument("--source-kind", choices=SOURCE_KIND_CHOICES, default="pulse")
    trajectory_parser.add_argument("--frequency", type=float, default=700.0)
    trajectory_parser.add_argument("--amplitude", type=float, default=1.0)
    trajectory_parser.add_argument("--first-emission-time", type=float, default=0.08)
    trajectory_parser.add_argument("--last-emission-time", type=float, default=None)
    trajectory_parser.add_argument("--noise-std", type=float, default=0.001)
    trajectory_parser.add_argument("--no-attenuation", action="store_true")
    trajectory_parser.add_argument("--seed", type=int, default=42)

    preprocess_parser = subparsers.add_parser("preprocess", help="split and preprocess an offline signal")
    preprocess_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="path to JSON config")
    preprocess_parser.add_argument("--input", required=True, help="input CSV or WAV signal")
    preprocess_parser.add_argument("--output-dir", default="data/processed", help="directory for processed windows")

    visualize_parser = subparsers.add_parser("visualize-trajectory", help="open a popup with truth and recognized trajectory")
    visualize_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="path to JSON config")
    visualize_parser.add_argument("--input", required=True, help="input CSV or WAV signal")
    visualize_parser.add_argument("--metadata", default=None, help="metadata JSON path; defaults to <signal>.meta.json")
    visualize_parser.add_argument("--trajectory-mode", choices=("direction", "position"), default="position", help="localization mode for popup trajectory")
    visualize_parser.add_argument("--trajectory-max-windows", type=int, default=30, help="maximum windows used for popup trajectory")
    visualize_parser.add_argument("--trajectory-interpolation", type=int, default=8, help="GCC-PHAT interpolation for popup trajectory")
    visualize_parser.add_argument("--trajectory-min-quality", type=float, default=0.2, help="minimum GCC-PHAT pair quality for popup trajectory")
    visualize_parser.add_argument("--trajectory-min-confidence", type=float, default=0.1, help="minimum localization confidence for popup trajectory")
    visualize_parser.add_argument("--trajectory-allow-low-confidence", action="store_true", help="allow low_confidence points in popup trajectory when confidence threshold is met")
    visualize_parser.add_argument("--trajectory-direction-range", type=float, default=1.0, help="display range for direction-only estimates")
    visualize_parser.add_argument("--trajectory-max-error", type=float, default=0.5, help="drop popup trajectory estimates farther than this from metadata truth; <=0 disables")
    visualize_parser.add_argument("--trajectory-max-speed", type=float, default=3.0, help="maximum allowed trajectory speed between neighboring recognized points in m/s; <=0 disables")
    visualize_parser.add_argument("--trajectory-max-acceleration", type=float, default=15.0, help="maximum allowed trajectory acceleration between neighboring recognized points in m/s^2; <=0 disables")
    visualize_parser.add_argument("--trajectory-motion-prior-weight", type=float, default=0.0, help="weight of constant-velocity motion prior for position visualization; 0 disables")
    visualize_parser.add_argument("--trajectory-subbands", type=int, default=1, help="number of GCC subbands for popup visualization")
    visualize_parser.add_argument("--trajectory-delay-continuity", action="store_true", help="enable temporal delay continuity (phase unwrap) in popup visualization")
    visualize_parser.add_argument("--trajectory-tdoa-policy", choices=("auto", "manual"), default="auto", help="TDOA policy mode for popup visualization")

    inspect_parser = subparsers.add_parser("inspect-audio", help="check 4-channel CSV/WAV audio and sidecar metadata")
    inspect_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="path to JSON config")
    inspect_parser.add_argument("--input", required=True, help="input CSV or WAV signal")
    inspect_parser.add_argument("--metadata", default=None, help="metadata JSON path; defaults to <signal>.meta.json")
    inspect_parser.add_argument("--require-metadata", action="store_true", help="fail if sidecar metadata is missing")

    track_parser = subparsers.add_parser("track-demo", help="simulate per-window tracking and write CSV/JSON logs")
    track_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="path to JSON config")
    track_parser.add_argument("--csv-log", default="data/logs/tracking.csv", help="CSV window log path")
    track_parser.add_argument("--json-output", default="data/logs/final_result.json", help="final tracked JSON path")
    track_parser.add_argument("--window-count", type=int, default=12)
    track_parser.add_argument("--start-azimuth", type=float, default=20.0)
    track_parser.add_argument("--azimuth-step", type=float, default=2.0)
    track_parser.add_argument("--elevation", type=float, default=35.0)
    track_parser.add_argument("--alpha", type=float, default=0.35)
    track_parser.add_argument("--max-jump-deg", type=float, default=35.0)
    track_parser.add_argument("--min-confidence-for-jump", type=float, default=0.4)
    track_parser.add_argument("--inject-low-confidence-jump", action="store_true")
    track_parser.add_argument("--jump-deg", type=float, default=90.0)

    audio_parser = subparsers.add_parser("localize-audio", help="estimate TDOA from CSV/WAV audio using GCC-PHAT")
    _add_localization_arguments(audio_parser)

    localization_parser = subparsers.add_parser("localization", help="run audio localization and optional visualization")
    _add_localization_arguments(localization_parser)

    benchmark_parser = subparsers.add_parser("benchmark-nonimpulsive", help="benchmark non-impulsive direction datasets")
    benchmark_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="path to JSON config")
    benchmark_parser.add_argument("--synthetic-dir", default="data/synthetic", help="directory with synthetic benchmark files")
    benchmark_parser.add_argument("--output", default="data/logs/nonimpulsive_benchmark.csv", help="CSV output path")
    benchmark_parser.add_argument("--max-windows", type=int, default=36)
    benchmark_parser.add_argument("--interpolation", type=int, default=16)
    benchmark_parser.add_argument("--subbands", type=int, default=2)
    benchmark_parser.add_argument("--min-quality", type=float, default=0.2)
    benchmark_parser.add_argument("--tdoa-policy", choices=("auto", "manual"), default="auto")
    benchmark_parser.add_argument("--enable-delay-continuity", action="store_true")

    args = parser.parse_args(argv)
    if args.command in (None, "demo"):
        return run_demo(getattr(args, "config", str(DEFAULT_CONFIG_PATH)), getattr(args, "json_output", None))
    if args.command == "validate":
        try:
            config = load_config(args.config)
        except ConfigError as exc:
            print(f"Ошибка конфигурации: {exc}")
            return 2
        run_validation(config)
        return 0
    if args.command == "generate-synthetic":
        return run_generate_synthetic(args)
    if args.command == "generate-trajectory":
        return run_generate_trajectory(args)
    if args.command == "preprocess":
        return run_preprocess(args)
    if args.command == "visualize-trajectory":
        return run_visualize_trajectory(args)
    if args.command == "inspect-audio":
        return run_inspect_audio(args)
    if args.command == "track-demo":
        return run_track_demo(args)
    if args.command in ("localize-audio", "localization"):
        return run_localize_audio(args)
    if args.command == "benchmark-nonimpulsive":
        return run_benchmark_nonimpulsive(args)

    parser.error(f"unknown command: {args.command}")
    return 2


def _add_localization_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="path to JSON config")
    parser.add_argument("--input", required=True, help="input CSV or WAV signal")
    parser.add_argument("--metadata", default=None, help="metadata JSON path; defaults to <signal>.meta.json")
    parser.add_argument("--mode", choices=("direction", "position"), default=None, help="override localization mode")
    parser.add_argument("--csv-log", default="data/logs/audio_localization.csv", help="CSV window log path")
    parser.add_argument("--json-output", default="data/logs/audio_final_result.json", help="final tracked JSON path")
    parser.add_argument("--max-windows", type=int, default=None, help="optional limit for processed windows")
    parser.add_argument("--interpolation", type=int, default=8, help="GCC-PHAT interpolation factor")
    parser.add_argument("--subbands", type=int, default=1, help="number of GCC subbands for consensus delay")
    parser.add_argument("--tdoa-policy", choices=("auto", "manual"), default="auto", help="TDOA policy mode")
    parser.add_argument("--min-quality", type=float, default=0.2, help="minimum GCC-PHAT pair quality")
    parser.add_argument("--enable-delay-continuity", action="store_true", help="enable temporal delay continuity (phase unwrap) using source frequency from metadata")
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--max-jump-deg", type=float, default=35.0)
    parser.add_argument("--min-confidence-for-jump", type=float, default=0.4)
    parser.add_argument(
        "--show-visualization",
        "--show-trajectory",
        dest="show_visualization",
        action="store_true",
        help="open a popup with truth and recognized localization result",
    )
    parser.add_argument("--visualization-mode", choices=("direction", "position"), default=None, help="override visualization mode; defaults to localization mode")
    parser.add_argument("--trajectory-max-windows", type=int, default=30, help="maximum windows used for popup visualization")
    parser.add_argument("--trajectory-interpolation", type=int, default=8, help="GCC-PHAT interpolation for popup visualization")
    parser.add_argument("--trajectory-min-quality", type=float, default=0.2, help="minimum GCC-PHAT pair quality for popup visualization")
    parser.add_argument("--trajectory-min-confidence", type=float, default=0.1, help="minimum localization confidence for popup visualization")
    parser.add_argument("--trajectory-allow-low-confidence", action="store_true", help="allow low_confidence points in popup visualization when confidence threshold is met")
    parser.add_argument("--trajectory-direction-range", type=float, default=1.0, help="display range for direction-only estimates")
    parser.add_argument("--trajectory-max-error", type=float, default=0.5, help="drop popup estimates farther than this from metadata truth; <=0 disables")
    parser.add_argument("--trajectory-max-speed", type=float, default=3.0, help="maximum allowed trajectory speed between neighboring popup points in m/s; <=0 disables")
    parser.add_argument("--trajectory-max-acceleration", type=float, default=15.0, help="maximum allowed trajectory acceleration between neighboring popup points in m/s^2; <=0 disables")
    parser.add_argument("--trajectory-motion-prior-weight", type=float, default=0.0, help="weight of constant-velocity motion prior for position popup visualization; 0 disables")
    parser.add_argument("--trajectory-subbands", type=int, default=1, help="number of GCC subbands for popup visualization")
    parser.add_argument("--trajectory-delay-continuity", action="store_true", help="enable temporal delay continuity in popup visualization")
    parser.add_argument("--trajectory-tdoa-policy", choices=("auto", "manual"), default="auto", help="TDOA policy mode for popup visualization")


def _direction_from_angles(azimuth_deg: float, elevation_deg: float) -> Vec3:
    azimuth = radians(azimuth_deg)
    elevation = radians(elevation_deg)
    return Vec3(cos(elevation) * cos(azimuth), cos(elevation) * sin(azimuth), sin(elevation))


def _far_field_pairs_from_direction(config, direction: Vec3, quality: float = 1.0) -> list[PairDelay]:
    pairs = []
    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        baseline = config.microphone_array.microphones[j] - config.microphone_array.microphones[i]
        delay = (baseline.x * direction.x + baseline.y * direction.y + baseline.z * direction.z) / config.environment.sound_speed
        pairs.append(PairDelay((i, j), delay, quality=quality, weight=1.0))
    return pairs


if __name__ == "__main__":
    raise SystemExit(main())
