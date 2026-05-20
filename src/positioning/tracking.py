"""Temporal smoothing and tracking for localization results."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, sin

from .models import LocalizationResult


@dataclass(frozen=True)
class TrackingConfig:
    alpha: float = 0.35
    max_jump_deg: float = 35.0
    min_confidence_for_jump: float = 0.4
    confidence_window: int = 5

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("tracking alpha must be in the range (0, 1]")
        if self.max_jump_deg <= 0.0:
            raise ValueError("max_jump_deg must be positive")
        if not 0.0 <= self.min_confidence_for_jump <= 1.0:
            raise ValueError("min_confidence_for_jump must be in the range [0, 1]")
        if self.confidence_window <= 0:
            raise ValueError("confidence_window must be positive")


@dataclass(frozen=True)
class TrackedResult:
    timestamp: float
    raw: LocalizationResult
    accepted: bool
    smoothed_azimuth_deg: float | None
    smoothed_elevation_deg: float | None
    azimuth_rate_deg_s: float | None
    elevation_rate_deg_s: float | None
    aggregated_confidence: float
    message: str = ""


class ResultTracker:
    def __init__(self, config: TrackingConfig | None = None) -> None:
        self.config = config or TrackingConfig()
        self.history: list[TrackedResult] = []
        self._azimuth: float | None = None
        self._elevation: float | None = None
        self._last_timestamp: float | None = None
        self._last_accepted_azimuth: float | None = None
        self._last_accepted_elevation: float | None = None

    def update(self, result: LocalizationResult, timestamp: float) -> TrackedResult:
        if result.azimuth_deg is None or result.elevation_deg is None:
            tracked = TrackedResult(
                timestamp=timestamp,
                raw=result,
                accepted=False,
                smoothed_azimuth_deg=self._azimuth,
                smoothed_elevation_deg=self._elevation,
                azimuth_rate_deg_s=None,
                elevation_rate_deg_s=None,
                aggregated_confidence=self._aggregated_confidence(result.confidence),
                message="result has no angular estimate",
            )
            self.history.append(tracked)
            return tracked

        accepted = result.status == "ok"
        message = ""
        if self._azimuth is not None:
            jump = abs(_angle_delta_deg(result.azimuth_deg, self._azimuth)) + abs(result.elevation_deg - (self._elevation or 0.0))
            if jump > self.config.max_jump_deg and result.confidence < self.config.min_confidence_for_jump:
                accepted = False
                message = "rejected abrupt low-confidence jump"

        previous_azimuth = self._azimuth
        previous_elevation = self._elevation
        previous_timestamp = self._last_timestamp

        if accepted:
            if self._azimuth is None:
                self._azimuth = _normalize_angle_deg(result.azimuth_deg)
                self._elevation = result.elevation_deg
            else:
                self._azimuth = _ema_angle_deg(self._azimuth, result.azimuth_deg, self.config.alpha)
                self._elevation = (1.0 - self.config.alpha) * (self._elevation or 0.0) + self.config.alpha * result.elevation_deg
            self._last_timestamp = timestamp
            self._last_accepted_azimuth = result.azimuth_deg
            self._last_accepted_elevation = result.elevation_deg

        azimuth_rate = None
        elevation_rate = None
        if accepted and previous_timestamp is not None and timestamp > previous_timestamp and previous_azimuth is not None and previous_elevation is not None:
            dt = timestamp - previous_timestamp
            azimuth_rate = _angle_delta_deg(self._azimuth or 0.0, previous_azimuth) / dt
            elevation_rate = ((self._elevation or 0.0) - previous_elevation) / dt

        tracked = TrackedResult(
            timestamp=timestamp,
            raw=result,
            accepted=accepted,
            smoothed_azimuth_deg=self._azimuth,
            smoothed_elevation_deg=self._elevation,
            azimuth_rate_deg_s=azimuth_rate,
            elevation_rate_deg_s=elevation_rate,
            aggregated_confidence=self._aggregated_confidence(result.confidence),
            message=message,
        )
        self.history.append(tracked)
        return tracked

    def _aggregated_confidence(self, current_confidence: float) -> float:
        recent = [item.raw.confidence for item in self.history[-self.config.confidence_window :]]
        recent.append(current_confidence)
        return sum(recent) / len(recent)


def _ema_angle_deg(previous: float, current: float, alpha: float) -> float:
    delta = _angle_delta_deg(current, previous)
    return _normalize_angle_deg(previous + alpha * delta)


def _angle_delta_deg(current: float, previous: float) -> float:
    return (current - previous + 180.0) % 360.0 - 180.0


def _normalize_angle_deg(angle: float) -> float:
    return atan2(sin(angle * 3.141592653589793 / 180.0), cos(angle * 3.141592653589793 / 180.0)) * 180.0 / 3.141592653589793
