"""TDOA localization solver translated from the original C++ prototype."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import atan2, degrees, fabs, isfinite, sqrt
from typing import Sequence

import numpy as np

from .geometry import Vec3, centroid, cross, dot, norm, solve3x3, try_solve3x3
from .models import LocalizationMode, LocalizationResult, MicrophoneArray, PairDelay, TdoaEstimate
from .tdoa import tdoa_from_arrival_times, validate_tdoa

DEFAULT_SPEED_OF_SOUND = 343.0
MAX_ITERATIONS = 120
TOLERANCE = 1e-8
INITIAL_LAMBDA = 1e-3
MIN_LAMBDA = 1e-12
MAX_LAMBDA = 1e6
LM_SEEDS_TO_TRY = 10
MICROPHONE_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
POSITION_PRIOR_WEIGHT = 1e-6
COPLANAR_MAX_RANGE_RATIO = 6.0
COPLANAR_MIN_NORMAL_RATIO = 0.5
COPLANAR_FAR_FIELD_RMS_M = 1e-3
COPLANAR_FAR_FIELD_MIN_RANGE_RATIO = 2.4
NEAR_FIELD_DIRECTION_MIN_NORMAL_RATIO = 0.5
NEAR_FIELD_DIRECTION_MIN_RANGE_RATIO = 1.0
NEAR_FIELD_DIRECTION_MAX_RANGE_RATIO = COPLANAR_MAX_RANGE_RATIO
NEAR_FIELD_DIRECTION_IMPROVEMENT_RATIO = 0.5
COPLANAR_DIRECTION_MIN_IN_PLANE_RATIO = 0.02
MIN_RELIABLE_CONFIDENCE = 0.55


@dataclass(frozen=True)
class LocalizationDiagnostics:
    residual_error2: float
    tried_seeds: int
    used_pairs: int = 6
    rejected_pairs: int = 0
    reliable: bool = False
    confidence: float = 0.0


def append_axis_seeds(seeds: list[Vec3], base: Vec3, scales: Sequence[float]) -> None:
    for scale in scales:
        seeds.extend(
            [
                base + Vec3(scale, 0.0, 0.0),
                base - Vec3(scale, 0.0, 0.0),
                base + Vec3(0.0, scale, 0.0),
                base - Vec3(0.0, scale, 0.0),
                base + Vec3(0.0, 0.0, scale),
                base - Vec3(0.0, 0.0, scale),
            ]
        )


def make_linear_init_candidates(
    microphones: Sequence[Vec3],
    delta_range: Sequence[float],
) -> list[Vec3]:
    candidates: list[Vec3] = []
    m0 = microphones[0]

    matrix: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(3)]
    rhs_b = [0.0, 0.0, 0.0]
    rhs_d = [float(delta_range[0]), float(delta_range[1]), float(delta_range[2])]

    for i in range(1, 4):
        dm = microphones[i] - m0
        matrix[i - 1][0] = dm.x
        matrix[i - 1][1] = dm.y
        matrix[i - 1][2] = dm.z

        mi2 = dot(microphones[i], microphones[i])
        m02 = dot(m0, m0)
        rhs_b[i - 1] = 0.5 * (mi2 - m02 - delta_range[i - 1] * delta_range[i - 1])

    u = try_solve3x3(matrix, rhs_b)
    w = try_solve3x3(matrix, rhs_d)
    if u is None or w is None:
        return candidates

    y = u - m0
    alpha = dot(w, w) - 1.0
    beta = -2.0 * dot(y, w)
    gamma = dot(y, y)
    eps = 1e-12

    roots: list[float] = []
    if fabs(alpha) < eps:
        if fabs(beta) > eps:
            roots.append(-gamma / beta)
    else:
        discriminant = beta * beta - 4.0 * alpha * gamma
        if discriminant >= -eps:
            discriminant = max(0.0, discriminant)
            sqrt_discriminant = sqrt(discriminant)
            roots.append((-beta + sqrt_discriminant) / (2.0 * alpha))
            roots.append((-beta - sqrt_discriminant) / (2.0 * alpha))

    for r0 in roots:
        if not isfinite(r0) or r0 <= 0.0:
            continue
        point = u - w * r0
        if all(isfinite(value) for value in point.as_tuple()):
            candidates.append(point)

    if not candidates and fabs(alpha) > eps:
        r0 = -beta / (2.0 * alpha)
        if isfinite(r0) and r0 > 0.0:
            point = u - w * r0
            if all(isfinite(value) for value in point.as_tuple()):
                candidates.append(point)

    return candidates


def build_seed_pool(
    microphones: Sequence[Vec3],
    delta_range_ref: Sequence[float],
) -> list[Vec3]:
    center = centroid(microphones)
    seeds = [center]

    linear_candidates = make_linear_init_candidates(microphones, delta_range_ref)
    seeds.extend(linear_candidates)

    append_axis_seeds(seeds, center, (0.5, 2.0, 10.0, 50.0, 200.0, 500.0))
    for point in linear_candidates:
        append_axis_seeds(seeds, point, (0.5, 2.0, 10.0, 50.0))

    return seeds


def compute_residual_error2(
    point: Vec3,
    microphones: Sequence[Vec3],
    pair_delta_range: Sequence[float],
) -> float:
    deltas: list[Vec3] = []
    ranges: list[float] = []
    for microphone in microphones:
        delta = point - microphone
        radius = norm(delta)
        if radius < 1e-12:
            return float("inf")
        deltas.append(delta)
        ranges.append(radius)

    error2 = 0.0
    for index, (i, j) in enumerate(MICROPHONE_PAIRS):
        predicted_diff = ranges[i] - ranges[j]
        residual = predicted_diff - pair_delta_range[index]
        error2 += residual * residual
    return error2


def compute_position_pair_residuals(
    point: Vec3,
    microphones: Sequence[Vec3],
    pair_delays: Sequence[PairDelay],
    speed_of_sound: float,
) -> list[float]:
    ranges = [norm(point - microphone) for microphone in microphones]
    residuals = []
    for item in pair_delays:
        i, j = item.pair
        residuals.append((ranges[i] - ranges[j]) - item.delay_seconds * speed_of_sound)
    return residuals


def compute_direction_pair_residuals(
    direction: Vec3,
    microphones: Sequence[Vec3],
    pair_delays: Sequence[PairDelay],
    speed_of_sound: float,
) -> list[float]:
    residuals = []
    for item in pair_delays:
        i, j = item.pair
        baseline = microphones[j] - microphones[i]
        residuals.append(dot(baseline, direction) - item.delay_seconds * speed_of_sound)
    return residuals


def weighted_error2(residuals: Sequence[float], pair_delays: Sequence[PairDelay]) -> float:
    return sum(_pair_weight(item) * residual * residual for item, residual in zip(pair_delays, residuals))


def huber_score(
    residuals: Sequence[float],
    pair_delays: Sequence[PairDelay],
    threshold_m: float,
) -> float:
    score = 0.0
    for item, residual in zip(pair_delays, residuals):
        abs_residual = abs(residual)
        if abs_residual <= threshold_m:
            loss = 0.5 * abs_residual * abs_residual
        else:
            loss = threshold_m * (abs_residual - 0.5 * threshold_m)
        score += _pair_weight(item) * loss
    return score


def solve_from_initial_guess(
    initial: Vec3,
    microphones: Sequence[Vec3],
    pair_delta_range: Sequence[float],
) -> Vec3:
    point = initial
    lm_lambda = INITIAL_LAMBDA
    previous_error2 = float("inf")

    for _ in range(MAX_ITERATIONS):
        jtj = [[0.0, 0.0, 0.0] for _ in range(3)]
        jtr = [0.0, 0.0, 0.0]

        deltas: list[Vec3] = []
        ranges: list[float] = []
        for microphone in microphones:
            delta = point - microphone
            radius = norm(delta)
            if radius < 1e-9:
                raise ValueError("estimated point coincides with one of the microphones")
            deltas.append(delta)
            ranges.append(radius)

        error2 = 0.0
        for index, (i, j) in enumerate(MICROPHONE_PAIRS):
            predicted_diff = ranges[i] - ranges[j]
            residual = predicted_diff - pair_delta_range[index]
            error2 += residual * residual

            grad = Vec3(
                deltas[i].x / ranges[i] - deltas[j].x / ranges[j],
                deltas[i].y / ranges[i] - deltas[j].y / ranges[j],
                deltas[i].z / ranges[i] - deltas[j].z / ranges[j],
            )
            g = grad.as_tuple()
            for row in range(3):
                jtr[row] += g[row] * residual
                for col in range(3):
                    jtj[row][col] += g[row] * g[col]

        accepted = False
        accepted_step = Vec3()
        for _attempt in range(10):
            matrix = [
                [jtj[0][0], jtj[0][1], jtj[0][2]],
                [jtj[1][0], jtj[1][1], jtj[1][2]],
                [jtj[2][0], jtj[2][1], jtj[2][2]],
            ]

            matrix[0][0] += lm_lambda * (1.0 + jtj[0][0])
            matrix[1][1] += lm_lambda * (1.0 + jtj[1][1])
            matrix[2][2] += lm_lambda * (1.0 + jtj[2][2])

            rhs = [-jtr[0], -jtr[1], -jtr[2]]
            step = solve3x3(matrix, rhs)
            candidate = point + step
            new_error2 = compute_residual_error2(candidate, microphones, pair_delta_range)

            if new_error2 < error2:
                point = candidate
                accepted_step = step
                accepted = True
                lm_lambda = max(MIN_LAMBDA, lm_lambda * 0.3)
                error2 = new_error2
                break

            lm_lambda = min(MAX_LAMBDA, lm_lambda * 10.0)

        if not accepted:
            break

        if norm(accepted_step) < TOLERANCE * (1.0 + norm(point)):
            break

        if isfinite(previous_error2) and (previous_error2 - error2) <= 1e-12 * (1.0 + previous_error2):
            break
        previous_error2 = error2

    return point


def build_seed_pool_from_tdoa(
    microphones: Sequence[Vec3],
    pair_delays: Sequence[PairDelay],
    speed_of_sound: float,
) -> list[Vec3]:
    center = centroid(microphones)
    seeds = [center]
    pair_map = {item.pair: item.delay_seconds * speed_of_sound for item in pair_delays}
    ref_pairs = ((0, 1), (0, 2), (0, 3))
    if all(pair in pair_map for pair in ref_pairs):
        linear_candidates = make_linear_init_candidates(microphones, [pair_map[pair] for pair in ref_pairs])
        seeds.extend(linear_candidates)
    else:
        linear_candidates = []
    append_axis_seeds(seeds, center, (0.5, 2.0, 10.0, 50.0, 200.0, 500.0))
    for point in linear_candidates:
        append_axis_seeds(seeds, point, (0.5, 2.0, 10.0, 50.0))
    return seeds


def solve_position_from_initial_guess_tdoa(
    initial: Vec3,
    microphones: Sequence[Vec3],
    pair_delays: Sequence[PairDelay],
    speed_of_sound: float,
) -> Vec3:
    point = initial
    lm_lambda = INITIAL_LAMBDA
    previous_error2 = float("inf")

    for _ in range(MAX_ITERATIONS):
        jtj = [[0.0, 0.0, 0.0] for _ in range(3)]
        jtr = [0.0, 0.0, 0.0]
        deltas = []
        ranges = []
        for microphone in microphones:
            delta = point - microphone
            radius = norm(delta)
            if radius < 1e-9:
                raise ValueError("estimated point coincides with one of the microphones")
            deltas.append(delta)
            ranges.append(radius)

        error2 = 0.0
        for item in pair_delays:
            i, j = item.pair
            residual = (ranges[i] - ranges[j]) - item.delay_seconds * speed_of_sound
            weight = _pair_weight(item)
            error2 += weight * residual * residual
            grad = Vec3(
                deltas[i].x / ranges[i] - deltas[j].x / ranges[j],
                deltas[i].y / ranges[i] - deltas[j].y / ranges[j],
                deltas[i].z / ranges[i] - deltas[j].z / ranges[j],
            )
            g = grad.as_tuple()
            for row in range(3):
                jtr[row] += weight * g[row] * residual
                for col in range(3):
                    jtj[row][col] += weight * g[row] * g[col]

        accepted = False
        accepted_step = Vec3()
        for _attempt in range(10):
            matrix = [
                [jtj[0][0], jtj[0][1], jtj[0][2]],
                [jtj[1][0], jtj[1][1], jtj[1][2]],
                [jtj[2][0], jtj[2][1], jtj[2][2]],
            ]
            matrix[0][0] += lm_lambda * (1.0 + jtj[0][0])
            matrix[1][1] += lm_lambda * (1.0 + jtj[1][1])
            matrix[2][2] += lm_lambda * (1.0 + jtj[2][2])
            rhs = [-jtr[0], -jtr[1], -jtr[2]]
            step = solve3x3(matrix, rhs)
            candidate = point + step
            residuals = compute_position_pair_residuals(candidate, microphones, pair_delays, speed_of_sound)
            new_error2 = weighted_error2(residuals, pair_delays)
            if new_error2 < error2:
                point = candidate
                accepted_step = step
                accepted = True
                lm_lambda = max(MIN_LAMBDA, lm_lambda * 0.3)
                error2 = new_error2
                break
            lm_lambda = min(MAX_LAMBDA, lm_lambda * 10.0)

        if not accepted:
            break
        if norm(accepted_step) < TOLERANCE * (1.0 + norm(point)):
            break
        if isfinite(previous_error2) and (previous_error2 - error2) <= 1e-12 * (1.0 + previous_error2):
            break
        previous_error2 = error2
    return point


def solve_position_tdoa(
    microphones: Sequence[Vec3],
    pair_delays: Sequence[PairDelay],
    speed_of_sound: float,
    *,
    position_prior: Vec3 | None = None,
    position_prior_weight: float = 0.0,
) -> Vec3:
    if len(pair_delays) < 4:
        raise ValueError("position mode requires at least four TDOA pairs")
    seeds = build_seed_pool_from_tdoa(microphones, pair_delays, speed_of_sound)
    if position_prior is not None and position_prior_weight > 0.0:
        seeds.append(position_prior)
        append_axis_seeds(seeds, position_prior, (0.1, 0.5, 1.0, 2.0))
    center = centroid(microphones)
    aperture = _array_aperture(microphones)
    ranked = []
    for seed in seeds:
        residuals = compute_position_pair_residuals(seed, microphones, pair_delays, speed_of_sound)
        error2 = weighted_error2(residuals, pair_delays)
        score = _position_score(error2, seed, center, aperture, position_prior, position_prior_weight)
        if isfinite(score):
            ranked.append((score, seed))
    if not ranked:
        raise ValueError("failed to build valid position seed points")
    ranked.sort(key=lambda item: item[0])

    best = ranked[0][1]
    best_score = float("inf")
    found = False
    for _, seed in ranked[: min(LM_SEEDS_TO_TRY, len(ranked))]:
        try:
            candidate = solve_position_from_initial_guess_tdoa(seed, microphones, pair_delays, speed_of_sound)
        except ValueError:
            continue
        residuals = compute_position_pair_residuals(candidate, microphones, pair_delays, speed_of_sound)
        error2 = weighted_error2(residuals, pair_delays)
        score = _position_score(error2, candidate, center, aperture, position_prior, position_prior_weight)
        if score < best_score:
            best = candidate
            best_score = score
            found = True
    if not found:
        raise ValueError("failed to estimate 3D position from TDOA pairs")
    return best


def solve_direction_tdoa(
    microphones: Sequence[Vec3],
    pair_delays: Sequence[PairDelay],
    speed_of_sound: float,
) -> Vec3:
    if len(pair_delays) < 3:
        raise ValueError("direction mode requires at least three TDOA pairs")
    rows = []
    targets = []
    weights = []
    for item in pair_delays:
        i, j = item.pair
        baseline = microphones[j] - microphones[i]
        rows.append(baseline.as_tuple())
        targets.append(item.delay_seconds * speed_of_sound)
        weights.append(sqrt(_pair_weight(item)))

    a = np.asarray(rows, dtype=np.float64)
    b = np.asarray(targets, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    aw = a * w[:, np.newaxis]
    bw = b * w
    solution, *_ = np.linalg.lstsq(aw, bw, rcond=None)
    direction = Vec3(float(solution[0]), float(solution[1]), float(solution[2]))
    direction_norm = norm(direction)
    rank = int(np.linalg.matrix_rank(aw, tol=1e-10))
    if rank < 3 and direction_norm < 1.0:
        normal = _array_plane_normal(microphones)
        normal_component = sqrt(max(0.0, 1.0 - direction_norm * direction_norm))
        direction = direction + normal * normal_component
        direction_norm = norm(direction)
    if direction_norm <= 1e-12:
        raise ValueError("failed to estimate non-zero direction vector")
    return direction * (1.0 / direction_norm)


def estimate_position_tdoa(
    microphones: Sequence[Vec3],
    arrival_time: Sequence[float],
    speed_of_sound: float = DEFAULT_SPEED_OF_SOUND,
    *,
    return_diagnostics: bool = False,
) -> Vec3 | tuple[Vec3, LocalizationDiagnostics]:
    """Estimate source position from four arrival timestamps.

    The function keeps the same mathematical behavior as the original C++
    prototype, but exposes optional diagnostics for logs and validation.
    """
    if len(microphones) != 4:
        raise ValueError("exactly four microphones are required")
    if len(arrival_time) != 4:
        raise ValueError("exactly four arrival timestamps are required")
    if speed_of_sound <= 0.0:
        raise ValueError("speed_of_sound must be positive")

    delta_range_ref = [
        (arrival_time[i] - arrival_time[0]) * speed_of_sound
        for i in range(1, 4)
    ]

    pair_delta_range = [
        (arrival_time[i] - arrival_time[j]) * speed_of_sound
        for i, j in MICROPHONE_PAIRS
    ]

    seed_pool = build_seed_pool(microphones, delta_range_ref)
    ranked_seeds = [
        (compute_residual_error2(seed, microphones, pair_delta_range), seed)
        for seed in seed_pool
    ]
    ranked_seeds = [(err, seed) for err, seed in ranked_seeds if isfinite(err)]

    if not ranked_seeds:
        raise ValueError("failed to build valid localization seed points")

    ranked_seeds.sort(key=lambda item: item[0])

    best = ranked_seeds[0][1]
    best_error2 = float("inf")
    found = False

    seeds_to_try = min(LM_SEEDS_TO_TRY, len(ranked_seeds))
    for _, seed in ranked_seeds[:seeds_to_try]:
        try:
            candidate = solve_from_initial_guess(seed, microphones, pair_delta_range)
            error2 = compute_residual_error2(candidate, microphones, pair_delta_range)
        except ValueError:
            continue

        if error2 < best_error2:
            best_error2 = error2
            best = candidate
            found = True
            if best_error2 < 1e-12:
                break

    if not found or not isfinite(best_error2):
        raise ValueError("failed to robustly estimate position from TDOA data")

    diagnostics = LocalizationDiagnostics(
        residual_error2=best_error2,
        tried_seeds=seeds_to_try,
        used_pairs=len(MICROPHONE_PAIRS),
    )
    if return_diagnostics:
        return best, diagnostics
    return best


def estimate_localization_from_arrival_times(
    microphone_array: MicrophoneArray,
    arrival_time: Sequence[float],
    speed_of_sound: float,
    mode: LocalizationMode = "position",
) -> LocalizationResult:
    """Estimate a structured localization result from arrival timestamps."""
    try:
        estimate = tdoa_from_arrival_times(arrival_time)
    except ValueError as exc:
        return LocalizationResult(mode=mode, status="invalid_input", message=str(exc))
    return estimate_localization_from_tdoa(microphone_array, estimate, speed_of_sound, mode)


def estimate_localization_from_tdoa(
    microphone_array: MicrophoneArray,
    estimate: TdoaEstimate,
    speed_of_sound: float,
    mode: LocalizationMode = "position",
    *,
    min_quality: float = 0.2,
    inlier_threshold_m: float = 0.03,
    position_prior: Vec3 | None = None,
    position_prior_weight: float = 0.0,
) -> LocalizationResult:
    """Estimate direction or position from TDOA pair delays with robust filtering."""
    try:
        validation = validate_tdoa(
            microphone_array,
            estimate,
            speed_of_sound,
            min_quality=min_quality,
            transitivity_tolerance_s=inlier_threshold_m / speed_of_sound,
        )
    except ValueError as exc:
        return LocalizationResult(mode=mode, status="invalid_input", message=str(exc))

    min_pairs = 3 if mode == "direction" else 4
    if len(validation.accepted) < min_pairs:
        return LocalizationResult(
            mode=mode,
            status="low_confidence",
            used_pairs=len(validation.accepted),
            rejected_pairs=len(validation.rejected),
            message=f"not enough accepted TDOA pairs for {mode} mode",
        )

    try:
        solution, used_pairs, residuals = _robust_solve(
            microphone_array.microphones,
            validation.accepted,
            speed_of_sound,
            mode,
            min_pairs,
            inlier_threshold_m,
            position_prior=position_prior,
            position_prior_weight=position_prior_weight,
        )
    except ValueError as exc:
        return LocalizationResult(
            mode=mode,
            status="low_confidence",
            used_pairs=len(validation.accepted),
            rejected_pairs=len(validation.rejected),
            message=str(exc),
        )

    if mode == "direction":
        direction = solution
        position = None
        near_field = _maybe_near_field_direction(
            microphone_array,
            validation.accepted,
            speed_of_sound,
            inlier_threshold_m,
            used_pairs,
            residuals,
        )
        if near_field is not None:
            direction, used_pairs, residuals = near_field
    else:
        position = solution
        direction = _direction_from_position(microphone_array, position)

    residual_error2 = weighted_error2(residuals, used_pairs)
    rms = sqrt(residual_error2 / max(1, len(used_pairs)))
    confidence = _confidence(len(used_pairs), len(estimate.pair_delays), rms, inlier_threshold_m)
    reliable = len(used_pairs) >= min_pairs and rms <= inlier_threshold_m
    range_message = ""
    if mode == "position" and position is not None and microphone_array.is_coplanar:
        aperture = _array_aperture(microphone_array.microphones)
        offset = position - microphone_array.center
        range_ratio = norm(offset) / aperture
        normal = microphone_array.plane_normal
        normal_ratio = abs(dot(offset, normal)) / aperture if normal is not None else 0.0
        max_ratio = COPLANAR_MAX_RANGE_RATIO
        if normal_ratio < COPLANAR_MIN_NORMAL_RATIO:
            confidence *= max(0.0, min(1.0, normal_ratio / COPLANAR_MIN_NORMAL_RATIO))
            reliable = False
            range_message = f"coplanar position range is not observable: normal ratio {normal_ratio:.2f}"
        if range_ratio > max_ratio:
            confidence *= max(0.0, min(1.0, max_ratio / max(range_ratio, 1e-12)))
            reliable = False
            if range_message:
                range_message += f"; range ratio too high: {range_ratio:.2f}"
            else:
                range_message = f"coplanar range ratio too high: {range_ratio:.2f}"
        far_field_rms = _direction_rms_for_pairs(
            microphone_array.microphones,
            validation.accepted,
            speed_of_sound,
        )
        if (
            range_ratio >= COPLANAR_FAR_FIELD_MIN_RANGE_RATIO
            and far_field_rms is not None
            and far_field_rms <= COPLANAR_FAR_FIELD_RMS_M
        ):
            confidence *= max(0.0, min(1.0, far_field_rms / COPLANAR_FAR_FIELD_RMS_M))
            reliable = False
            observability_message = f"coplanar position range is not observable: far-field RMS {far_field_rms:.6f} m"
            if range_message:
                range_message += f"; {observability_message}"
            else:
                range_message = observability_message
    if mode == "direction" and microphone_array.is_coplanar:
        normal = microphone_array.plane_normal
        if normal is not None:
            normal_component = abs(dot(direction, normal))
            in_plane_ratio = sqrt(max(0.0, 1.0 - normal_component * normal_component))
            if in_plane_ratio < COPLANAR_DIRECTION_MIN_IN_PLANE_RATIO:
                confidence *= max(0.0, min(1.0, in_plane_ratio / COPLANAR_DIRECTION_MIN_IN_PLANE_RATIO))
                reliable = False
                range_message = (
                    f"coplanar direction weakly observable: in-plane ratio {in_plane_ratio:.4f}"
                    if not range_message
                    else f"{range_message}; coplanar direction weakly observable: in-plane ratio {in_plane_ratio:.4f}"
                )
    if confidence < MIN_RELIABLE_CONFIDENCE:
        reliable = False
        low_confidence_message = f"confidence below reliable threshold: {confidence:.3f}"
        range_message = low_confidence_message if not range_message else f"{range_message}; {low_confidence_message}"
    status = "ok" if reliable else "low_confidence"
    azimuth_deg, elevation_deg = _angles_from_direction(direction)
    rejected_count = len(estimate.pair_delays) - len(used_pairs)

    messages = []
    if mode == "direction" and near_field is not None:
        messages.append("near-field bearing")
    if validation.transitivity_errors:
        messages.append(f"transitivity errors: {len(validation.transitivity_errors)}")
    if rejected_count:
        messages.append(f"rejected pairs: {rejected_count}")
    if range_message:
        messages.append(range_message)

    return LocalizationResult(
        mode=mode,
        status=status,
        position=position,
        direction=direction,
        azimuth_deg=azimuth_deg,
        elevation_deg=elevation_deg,
        residual_error2=residual_error2,
        used_pairs=len(used_pairs),
        rejected_pairs=rejected_count,
        reliable=reliable,
        confidence=confidence,
        message="; ".join(messages),
    )


def _direction_rms_for_pairs(
    microphones: Sequence[Vec3],
    pair_delays: Sequence[PairDelay],
    speed_of_sound: float,
) -> float | None:
    try:
        direction = solve_direction_tdoa(microphones, pair_delays, speed_of_sound)
    except ValueError:
        return None
    residuals = compute_direction_pair_residuals(direction, microphones, pair_delays, speed_of_sound)
    return sqrt(weighted_error2(residuals, pair_delays) / max(1, len(pair_delays)))


def _maybe_near_field_direction(
    microphone_array: MicrophoneArray,
    accepted_pairs: Sequence[PairDelay],
    speed_of_sound: float,
    inlier_threshold_m: float,
    far_field_pairs: Sequence[PairDelay],
    far_field_residuals: Sequence[float],
) -> tuple[Vec3, tuple[PairDelay, ...], list[float]] | None:
    if not microphone_array.is_coplanar or len(accepted_pairs) < 4:
        return None

    try:
        position, position_pairs, position_residuals = _robust_solve(
            microphone_array.microphones,
            accepted_pairs,
            speed_of_sound,
            "position",
            4,
            inlier_threshold_m,
        )
    except ValueError:
        return None

    center = microphone_array.center
    aperture = _array_aperture(microphone_array.microphones)
    normal = microphone_array.plane_normal
    if normal is None:
        return None

    offset = position - center
    range_ratio = norm(offset) / aperture
    normal_ratio = abs(dot(offset, normal)) / aperture
    if range_ratio < NEAR_FIELD_DIRECTION_MIN_RANGE_RATIO:
        return None
    if normal_ratio < NEAR_FIELD_DIRECTION_MIN_NORMAL_RATIO:
        return None
    if range_ratio > NEAR_FIELD_DIRECTION_MAX_RANGE_RATIO:
        return None

    far_rms = sqrt(weighted_error2(far_field_residuals, far_field_pairs) / max(1, len(far_field_pairs)))
    near_rms = sqrt(weighted_error2(position_residuals, position_pairs) / max(1, len(position_pairs)))
    if near_rms > min(inlier_threshold_m, far_rms * NEAR_FIELD_DIRECTION_IMPROVEMENT_RATIO):
        return None

    return _direction_from_position(microphone_array, position), tuple(position_pairs), list(position_residuals)


def _robust_solve(
    microphones: Sequence[Vec3],
    pair_delays: Sequence[PairDelay],
    speed_of_sound: float,
    mode: LocalizationMode,
    min_pairs: int,
    inlier_threshold_m: float,
    *,
    position_prior: Vec3 | None = None,
    position_prior_weight: float = 0.0,
) -> tuple[Vec3, tuple[PairDelay, ...], list[float]]:
    best_solution: Vec3 | None = None
    best_pairs: tuple[PairDelay, ...] = tuple()
    best_residuals: list[float] = []
    best_key: tuple[int, float] | None = None
    center = centroid(microphones)
    aperture = _array_aperture(microphones)

    max_subset_size = len(pair_delays)
    for subset_size in range(min_pairs, max_subset_size + 1):
        for subset in combinations(pair_delays, subset_size):
            try:
                candidate = (
                    solve_direction_tdoa(microphones, subset, speed_of_sound)
                    if mode == "direction"
                    else solve_position_tdoa(
                        microphones,
                        subset,
                        speed_of_sound,
                        position_prior=position_prior,
                        position_prior_weight=position_prior_weight,
                    )
                )
            except ValueError:
                continue
            residuals = (
                compute_direction_pair_residuals(candidate, microphones, pair_delays, speed_of_sound)
                if mode == "direction"
                else compute_position_pair_residuals(candidate, microphones, pair_delays, speed_of_sound)
            )
            inliers = tuple(item for item, residual in zip(pair_delays, residuals) if abs(residual) <= inlier_threshold_m)
            if len(inliers) < min_pairs:
                inliers = tuple(subset)
            score = huber_score(residuals, pair_delays, inlier_threshold_m)
            if mode == "position":
                score = _position_score(score, candidate, center, aperture, position_prior, position_prior_weight)
            key = (-len(inliers), score)
            if best_key is None or key < best_key:
                best_key = key
                best_solution = candidate
                best_pairs = inliers

    if best_solution is None or len(best_pairs) < min_pairs:
        raise ValueError("robust TDOA solve failed")

    final_solution = (
        solve_direction_tdoa(microphones, best_pairs, speed_of_sound)
        if mode == "direction"
        else solve_position_tdoa(
            microphones,
            best_pairs,
            speed_of_sound,
            position_prior=position_prior,
            position_prior_weight=position_prior_weight,
        )
    )
    final_residuals = (
        compute_direction_pair_residuals(final_solution, microphones, best_pairs, speed_of_sound)
        if mode == "direction"
        else compute_position_pair_residuals(final_solution, microphones, best_pairs, speed_of_sound)
    )
    return final_solution, best_pairs, final_residuals


def _direction_from_position(microphone_array: MicrophoneArray, position: Vec3) -> Vec3:
    center = microphone_array.center
    relative = position - center
    relative_norm = norm(relative)
    if relative_norm <= 0.0:
        raise ValueError("estimated source position coincides with array center")
    return relative * (1.0 / relative_norm)


def _angles_from_direction(direction: Vec3) -> tuple[float, float]:
    azimuth_deg = degrees(atan2(direction.y, direction.x))
    elevation_deg = degrees(atan2(direction.z, sqrt(direction.x * direction.x + direction.y * direction.y)))
    return azimuth_deg, elevation_deg


def _confidence(used_pairs: int, total_pairs: int, rms_m: float, threshold_m: float) -> float:
    pair_ratio = used_pairs / max(1, total_pairs)
    residual_factor = max(0.0, 1.0 - min(1.0, rms_m / max(threshold_m, 1e-12)))
    return pair_ratio * residual_factor


def _pair_weight(item: PairDelay) -> float:
    return max(1e-9, item.quality * item.weight)


def _array_plane_normal(microphones: Sequence[Vec3]) -> Vec3:
    for i in range(len(microphones)):
        for j in range(i + 1, len(microphones)):
            for k in range(j + 1, len(microphones)):
                normal = cross(microphones[j] - microphones[i], microphones[k] - microphones[i])
                normal_norm = norm(normal)
                if normal_norm > 1e-12:
                    return normal * (1.0 / normal_norm)
    raise ValueError("microphone geometry must contain at least three non-collinear points")


def _array_aperture(microphones: Sequence[Vec3]) -> float:
    max_distance = 0.0
    for i in range(len(microphones)):
        for j in range(i + 1, len(microphones)):
            max_distance = max(max_distance, norm(microphones[i] - microphones[j]))
    return max(max_distance, 1e-6)


def _position_score(
    error2: float,
    point: Vec3,
    center: Vec3,
    aperture: float,
    position_prior: Vec3 | None = None,
    position_prior_weight: float = 0.0,
) -> float:
    distance_ratio2 = (norm(point - center) / aperture) ** 2
    score = error2 + POSITION_PRIOR_WEIGHT * distance_ratio2
    if position_prior is not None and position_prior_weight > 0.0:
        prior_distance_ratio2 = (norm(point - position_prior) / aperture) ** 2
        score += position_prior_weight * prior_distance_ratio2
    return score
