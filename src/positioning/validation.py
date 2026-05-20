"""Synthetic validator for the Python TDOA solver."""

from __future__ import annotations

import argparse
import math
import random
from typing import Sequence

from .config import ConfigError, DEFAULT_CONFIG_PATH, load_config
from .geometry import Vec3, dot, norm, position_error
from .localization import estimate_localization_from_tdoa, estimate_position_tdoa
from .models import AppConfig, PairDelay, TdoaEstimate
from .simulation import make_arrival_time


def run_case(
    name: str,
    microphones: Sequence[Vec3],
    truth: Vec3,
    noise_std: float,
    *,
    speed_of_sound: float,
    inject_outlier: bool = False,
) -> float:
    rng = random.Random(42)
    arrival = make_arrival_time(
        microphones,
        truth,
        0.1,
        speed_of_sound=speed_of_sound,
        noise_std=noise_std,
        rng=rng,
    )

    if inject_outlier:
        arrival[3] += 2.5e-4

    try:
        estimate = estimate_position_tdoa(microphones, arrival, speed_of_sound)
        error = position_error(estimate, truth)
        print(name)
        print(f"  truth: ({truth.x:.6f}, {truth.y:.6f}, {truth.z:.6f})")
        print(f"  est  : ({estimate.x:.6f}, {estimate.y:.6f}, {estimate.z:.6f})")
        print(f"  err  : {error:.6f} m")
        return error
    except ValueError as exc:
        print(name)
        print(f"  ERROR: {exc}")
        return math.inf


def run_validation(config: AppConfig | None = None) -> None:
    if config is None:
        config = load_config()
    microphones = config.microphone_array.microphones
    sound_speed = config.environment.sound_speed
    if config.microphone_array.is_coplanar:
        run_planar_direction_validation(config)
        return

    near_cases = (
        Vec3(0.10, 0.12, 0.15),
        Vec3(0.60, 0.40, 0.30),
        Vec3(1.20, 0.90, 0.70),
        Vec3(-0.30, 0.25, 0.50),
        Vec3(0.05, 0.05, 0.05),
    )
    far_noiseless_cases = (
        Vec3(100.0, 0.0, 0.0),
        Vec3(220.0, 120.0, 40.0),
        Vec3(0.0, 400.0, 300.0),
        Vec3(288.675, 288.675, 288.675),
        Vec3(-320.0, 220.0, 300.0),
        Vec3(500.0, 0.0, 0.0),
    )

    print("=== Noise-free tests ===")
    for index, truth in enumerate(near_cases, 1):
        run_case(f"Case {index}", microphones, truth, 0.0, speed_of_sound=sound_speed)

    print("\n=== Far noise-free tests (distance <= 500 m) ===")
    max_far_error = 0.0
    sum_far_error = 0.0
    far_count = 0
    for index, truth in enumerate(far_noiseless_cases, 1):
        error = run_case(f"Far {index}", microphones, truth, 0.0, speed_of_sound=sound_speed)
        if math.isfinite(error):
            max_far_error = max(max_far_error, error)
            sum_far_error += error
            far_count += 1
    if far_count:
        print(f"  max err (far, no-noise): {max_far_error:.6f} m")
        print(f"  mean err (far, no-noise): {(sum_far_error / far_count):.6f} m")

    print("\n=== Small-noise tests (std=3e-6 s) ===")
    for index, truth in enumerate(near_cases, 1):
        run_case(f"Case {index}", microphones, truth, 3e-6, speed_of_sound=sound_speed)

    print("\n=== Outlier tests (single channel +250 us) ===")
    for index, truth in enumerate(near_cases, 1):
        run_case(f"Case {index}", microphones, truth, 3e-6, speed_of_sound=sound_speed, inject_outlier=True)

    print("\n=== Robust TDOA-pair outlier test (single pair +300 us) ===")
    run_robust_pair_outlier_case(config)


def run_planar_direction_validation(config: AppConfig) -> None:
    print("=== Planar direction tests ===")
    direction_cases = (
        Vec3(0.7, 0.4, 0.3),
        Vec3(0.4, 0.8, 0.4),
        Vec3(-0.3, 0.7, 0.5),
        Vec3(0.8, -0.2, 0.4),
    )
    max_direction_error = 0.0
    sum_direction_error = 0.0
    for index, direction in enumerate(direction_cases, 1):
        truth = direction * (1.0 / norm(direction))
        result = estimate_localization_from_tdoa(
            config.microphone_array,
            TdoaEstimate(tuple(_direction_pair_delays(config, truth))),
            config.environment.sound_speed,
            "direction",
        )
        if result.direction is None:
            direction_error = math.inf
        else:
            direction_error = max(0.0, 1.0 - dot(result.direction, truth))
        max_direction_error = max(max_direction_error, direction_error)
        sum_direction_error += direction_error
        print(f"Direction {index}")
        print(f"  status       : {result.status}")
        print(f"  reliable     : {result.reliable}")
        print(f"  used pairs   : {result.used_pairs}")
        print(f"  rejected     : {result.rejected_pairs}")
        print(f"  direction err: {direction_error:.6f}")
    print(f"  max direction err: {max_direction_error:.6f}")
    print(f"  mean direction err: {(sum_direction_error / len(direction_cases)):.6f}")
    print("\n=== Robust TDOA-pair outlier test (single pair +300 us) ===")
    run_robust_pair_outlier_case(config)


def run_robust_pair_outlier_case(config: AppConfig) -> None:
    direction = Vec3(0.7, 0.5, 0.2)
    direction = direction * (1.0 / norm(direction))
    delays = []
    for item in _direction_pair_delays(config, direction):
        delay = item.delay_seconds
        i, j = item.pair
        if (i, j) == (0, 3):
            delay += 3e-4
        delays.append(PairDelay((i, j), delay, quality=1.0, weight=1.0))
    result = estimate_localization_from_tdoa(
        config.microphone_array,
        TdoaEstimate(tuple(delays)),
        config.environment.sound_speed,
        "direction",
    )
    print(f"  status       : {result.status}")
    print(f"  reliable     : {result.reliable}")
    print(f"  used pairs   : {result.used_pairs}")
    print(f"  rejected     : {result.rejected_pairs}")
    if result.direction is not None:
        direction_error = 1.0 - dot(result.direction, direction)
        print(f"  direction err: {direction_error:.6f}")


def _direction_pair_delays(config: AppConfig, direction: Vec3) -> list[PairDelay]:
    delays = []
    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        baseline = config.microphone_array.microphones[j] - config.microphone_array.microphones[i]
        delay = dot(baseline, direction) / config.environment.sound_speed
        delays.append(PairDelay((i, j), delay, quality=1.0, weight=1.0))
    return delays


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run synthetic TDOA validation cases.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="path to JSON config")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"Ошибка конфигурации: {exc}")
        return 2
    run_validation(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
