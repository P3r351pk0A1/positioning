import random

import pytest

from positioning.config import load_config
from positioning.geometry import Vec3, distance, dot, norm, position_error
from positioning.localization import estimate_localization_from_arrival_times, estimate_localization_from_tdoa, estimate_position_tdoa
from positioning.models import PairDelay, TdoaEstimate
from positioning.simulation import make_arrival_time
from positioning.tdoa import tdoa_from_arrival_times, validate_tdoa


def test_tdoa_solver_matches_synthetic_target():
    config = load_config()
    truth = Vec3(0.6, 0.4, 0.3)
    arrival = make_arrival_time(
        config.microphone_array.microphones,
        truth,
        0.1,
        speed_of_sound=config.environment.sound_speed,
        rng=random.Random(42),
    )

    estimate = estimate_position_tdoa(
        config.microphone_array.microphones,
        arrival,
        config.environment.sound_speed,
    )

    assert position_error(estimate, truth) < 1e-6


def test_structured_localization_result_is_ok():
    config = load_config()
    truth = Vec3(0.6, 0.4, 0.3)
    arrival = make_arrival_time(
        config.microphone_array.microphones,
        truth,
        0.1,
        speed_of_sound=config.environment.sound_speed,
        rng=random.Random(42),
    )

    result = estimate_localization_from_arrival_times(
        config.microphone_array,
        arrival,
        config.environment.sound_speed,
        config.localization.mode,
    )

    assert result.status == "ok"
    assert result.mode == "position"
    assert result.position is not None
    assert result.direction is not None
    assert result.residual_error2 == pytest.approx(0.0, abs=1e-10)
    assert result.used_pairs == 6
    assert result.reliable
    assert result.confidence > 0.9


def test_direction_mode_estimates_unit_vector_for_far_field_tdoa():
    config = load_config()
    direction = Vec3(0.7, 0.5, 0.2)
    direction = direction * (1.0 / norm(direction))
    delays = []
    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        baseline = config.microphone_array.microphones[j] - config.microphone_array.microphones[i]
        delay = dot(baseline, direction) / config.environment.sound_speed
        delays.append(PairDelay((i, j), delay, quality=1.0, weight=1.0))

    result = estimate_localization_from_tdoa(
        config.microphone_array,
        TdoaEstimate(tuple(delays)),
        config.environment.sound_speed,
        "direction",
    )

    assert result.status == "ok"
    assert result.mode == "direction"
    assert result.position is None
    assert result.direction is not None
    assert dot(result.direction, direction) > 0.999
    assert result.reliable


def test_robust_direction_rejects_single_false_pair():
    config = load_config()
    direction = Vec3(0.7, 0.5, 0.2)
    direction = direction * (1.0 / norm(direction))
    delays = []
    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        baseline = config.microphone_array.microphones[j] - config.microphone_array.microphones[i]
        delay = dot(baseline, direction) / config.environment.sound_speed
        if (i, j) == (0, 3):
            delay += 3e-4
        delays.append(PairDelay((i, j), delay, quality=1.0, weight=1.0))

    naive = estimate_localization_from_tdoa(
        config.microphone_array,
        TdoaEstimate(tuple(delays)),
        config.environment.sound_speed,
        "direction",
    )

    assert naive.status == "ok"
    assert naive.direction is not None
    assert dot(naive.direction, direction) > 0.995
    assert naive.rejected_pairs >= 1
    assert naive.used_pairs < 6


def test_tdoa_validation_rejects_low_quality_and_physically_impossible_pair():
    config = load_config()
    estimate = TdoaEstimate(
        (
            PairDelay((0, 1), 0.0, quality=0.1),
            PairDelay((0, 2), 10.0, quality=1.0),
            PairDelay((0, 3), 0.0, quality=1.0),
        )
    )

    validation = validate_tdoa(config.microphone_array, estimate, config.environment.sound_speed)

    assert len(validation.accepted) == 1
    assert len(validation.rejected) == 2


def test_transitivity_check_rejects_inconsistent_pair():
    config = load_config()
    arrival = make_arrival_time(
        config.microphone_array.microphones,
        Vec3(0.6, 0.4, 0.3),
        0.1,
        speed_of_sound=config.environment.sound_speed,
        rng=random.Random(42),
    )
    estimate = tdoa_from_arrival_times(arrival)
    corrupted = []
    for item in estimate.pair_delays:
        if item.pair == (0, 3):
            corrupted.append(PairDelay(item.pair, item.delay_seconds + 3e-4, item.quality, item.weight))
        else:
            corrupted.append(item)

    validation = validate_tdoa(config.microphone_array, TdoaEstimate(tuple(corrupted)), config.environment.sound_speed)

    assert validation.transitivity_errors
    assert (0, 3) in {item.pair for item in validation.rejected}


def test_coplanar_far_field_position_is_marked_low_confidence():
    config = load_config("configs/side_looking_demo.json")
    truth = Vec3(30.0, 5.0, 3.0)
    arrival = [
        distance(truth, microphone) / config.environment.sound_speed
        for microphone in config.microphone_array.microphones
    ]

    result = estimate_localization_from_arrival_times(
        config.microphone_array,
        arrival,
        config.environment.sound_speed,
        "position",
    )

    assert result.status == "low_confidence"
    assert "not observable" in result.message
