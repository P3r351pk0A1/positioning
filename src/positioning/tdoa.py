"""TDOA pair validation and robust filtering helpers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

from .geometry import distance
from .models import MicrophoneArray, PairDelay, TdoaEstimate

TDOA_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


@dataclass(frozen=True)
class TransitivityError:
    triple: tuple[int, int, int]
    error_seconds: float


@dataclass(frozen=True)
class TdoaValidation:
    accepted: tuple[PairDelay, ...]
    rejected: tuple[PairDelay, ...]
    transitivity_errors: tuple[TransitivityError, ...]

    @property
    def accepted_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(item.pair for item in self.accepted)


def tdoa_from_arrival_times(arrival_time: Sequence[float], quality: float = 1.0, weight: float = 1.0) -> TdoaEstimate:
    if len(arrival_time) != 4:
        raise ValueError("exactly four arrival timestamps are required")
    return TdoaEstimate(
        tuple(
            PairDelay(pair=(i, j), delay_seconds=arrival_time[i] - arrival_time[j], quality=quality, weight=weight)
            for i, j in TDOA_PAIRS
        ),
        used_pairs=TDOA_PAIRS,
    )


def validate_tdoa(
    microphone_array: MicrophoneArray,
    estimate: TdoaEstimate,
    speed_of_sound: float,
    *,
    min_quality: float = 0.2,
    physical_tolerance_m: float = 1e-6,
    transitivity_tolerance_s: float = 5e-5,
) -> TdoaValidation:
    if speed_of_sound <= 0.0:
        raise ValueError("speed_of_sound must be positive")

    accepted: list[PairDelay] = []
    rejected: list[PairDelay] = []
    for item in estimate.pair_delays:
        i, j = item.pair
        max_delta_m = distance(microphone_array.microphones[i], microphone_array.microphones[j]) + physical_tolerance_m
        measured_delta_m = abs(item.delay_seconds) * speed_of_sound
        if item.quality < min_quality or measured_delta_m > max_delta_m:
            rejected.append(item)
        else:
            accepted.append(item)

    pair_map = _pair_map(accepted)
    transitivity_errors: list[TransitivityError] = []
    for i, j, k in combinations(range(4), 3):
        dij = _lookup_delay(pair_map, i, j)
        djk = _lookup_delay(pair_map, j, k)
        dik = _lookup_delay(pair_map, i, k)
        if dij is None or djk is None or dik is None:
            continue
        error = dij + djk - dik
        if abs(error) > transitivity_tolerance_s:
            transitivity_errors.append(TransitivityError((i, j, k), error))

    if transitivity_errors:
        suspicious_pairs = _suspicious_pairs_from_transitivity(transitivity_errors, pair_map)
        still_accepted = []
        for item in accepted:
            normalized = _normalize_pair(item.pair)
            if normalized in suspicious_pairs:
                rejected.append(item)
            else:
                still_accepted.append(item)
        accepted = still_accepted

    return TdoaValidation(tuple(accepted), tuple(rejected), tuple(transitivity_errors))


def _pair_map(items: Sequence[PairDelay]) -> dict[tuple[int, int], PairDelay]:
    result = {}
    for item in items:
        pair = _normalize_pair(item.pair)
        delay = item.delay_seconds if item.pair == pair else -item.delay_seconds
        result[pair] = PairDelay(pair, delay, item.quality, item.weight)
    return result


def _lookup_delay(pair_map: dict[tuple[int, int], PairDelay], i: int, j: int) -> float | None:
    if i == j:
        return 0.0
    if i < j:
        item = pair_map.get((i, j))
        return item.delay_seconds if item else None
    item = pair_map.get((j, i))
    return -item.delay_seconds if item else None


def _normalize_pair(pair: tuple[int, int]) -> tuple[int, int]:
    i, j = pair
    return pair if i < j else (j, i)


def _suspicious_pairs_from_transitivity(
    errors: Sequence[TransitivityError],
    pair_map: dict[tuple[int, int], PairDelay],
) -> set[tuple[int, int]]:
    counts: dict[tuple[int, int], int] = {}
    for error in errors:
        i, j, k = error.triple
        for pair in ((i, j), (j, k), (i, k)):
            normalized = _normalize_pair(pair)
            counts[normalized] = counts.get(normalized, 0) + 1
    if not counts:
        return set()
    max_count = max(counts.values())
    candidates = [pair for pair, count in counts.items() if count == max_count]
    candidates.sort(key=lambda pair: pair_map[pair].quality * pair_map[pair].weight)
    return {candidates[0]}
