"""Shared data models for positioning configuration and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

from .environment import speed_of_sound
from .geometry import Vec3, centroid, cross, dot

LocalizationMode = Literal["direction", "position"]
LocalizationStatus = Literal["ok", "low_confidence", "invalid_input", "insufficient_signal"]


@dataclass(frozen=True)
class MicrophoneArray:
    microphones: tuple[Vec3, Vec3, Vec3, Vec3]

    def __post_init__(self) -> None:
        if len(self.microphones) != 4:
            raise ValueError("MicrophoneArray requires exactly four microphones")
        if self.plane_normal is None:
            raise ValueError("microphone geometry must contain at least three non-collinear points")

    @property
    def center(self) -> Vec3:
        return centroid(self.microphones)

    @property
    def plane_normal(self) -> Vec3 | None:
        for i in range(4):
            for j in range(i + 1, 4):
                for k in range(j + 1, 4):
                    normal = cross(self.microphones[j] - self.microphones[i], self.microphones[k] - self.microphones[i])
                    length2 = dot(normal, normal)
                    if length2 > 1e-12:
                        length = length2**0.5
                        return Vec3(normal.x / length, normal.y / length, normal.z / length)
        return None

    @property
    def is_coplanar(self) -> bool:
        normal = self.plane_normal
        if normal is None:
            return False
        origin = self.microphones[0]
        return all(abs(dot(microphone - origin, normal)) < 1e-8 for microphone in self.microphones)


@dataclass(frozen=True)
class Environment:
    temperature_c: float = 20.0
    humidity_percent: float = 0.0

    @property
    def sound_speed(self) -> float:
        return speed_of_sound(self.temperature_c, self.humidity_percent)


@dataclass(frozen=True)
class AudioFrame:
    channels: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...], tuple[float, ...]]
    sample_rate: int
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if len(self.channels) != 4:
            raise ValueError("AudioFrame requires exactly four channels")
        lengths = {len(channel) for channel in self.channels}
        if len(lengths) != 1:
            raise ValueError("all AudioFrame channels must have the same length")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

    @property
    def sample_count(self) -> int:
        return len(self.channels[0])

    @property
    def duration(self) -> float:
        return self.sample_count / float(self.sample_rate)


@dataclass(frozen=True)
class PreprocessDiagnostics:
    rms_by_channel: tuple[float, float, float, float]
    mean_rms: float
    mean_abs_correlation: float
    informative: bool
    reason: str = ""


@dataclass(frozen=True)
class PairDelay:
    pair: tuple[int, int]
    delay_seconds: float
    quality: float = 1.0
    weight: float = 1.0


@dataclass(frozen=True)
class TdoaEstimate:
    pair_delays: tuple[PairDelay, ...]
    used_pairs: tuple[tuple[int, int], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for item in self.pair_delays:
            i, j = item.pair
            if not (0 <= i < 4 and 0 <= j < 4 and i != j):
                raise ValueError("TDOA pair indices must reference different channels in range [0, 3]")
            if item.quality < 0.0:
                raise ValueError("TDOA quality must be non-negative")
            if item.weight < 0.0:
                raise ValueError("TDOA weight must be non-negative")


@dataclass(frozen=True)
class LocalizationResult:
    mode: LocalizationMode
    status: LocalizationStatus
    position: Vec3 | None = None
    direction: Vec3 | None = None
    azimuth_deg: float | None = None
    elevation_deg: float | None = None
    residual_error2: float | None = None
    used_pairs: int = 0
    rejected_pairs: int = 0
    reliable: bool = False
    confidence: float = 0.0
    message: str = ""


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    window_size: int
    overlap: float
    frequency_band: tuple[float, float]

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError("overlap must be in the range [0, 1)")
        low, high = self.frequency_band
        if low < 0.0 or high <= low:
            raise ValueError("frequency_band must be [low, high] with 0 <= low < high")
        if high > self.sample_rate / 2.0:
            raise ValueError("frequency_band high value must not exceed Nyquist frequency")


@dataclass(frozen=True)
class LocalizationConfig:
    mode: LocalizationMode = "position"

    def __post_init__(self) -> None:
        if self.mode not in ("direction", "position"):
            raise ValueError("localization mode must be 'direction' or 'position'")


@dataclass(frozen=True)
class AppConfig:
    microphone_array: MicrophoneArray
    environment: Environment
    audio: AudioConfig
    localization: LocalizationConfig


def microphone_array_from_vectors(points: Sequence[Vec3]) -> MicrophoneArray:
    if len(points) != 4:
        raise ValueError("exactly four microphone points are required")
    return MicrophoneArray((points[0], points[1], points[2], points[3]))
