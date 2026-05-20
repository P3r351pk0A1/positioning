"""Passive acoustic localization prototype."""

from .environment import speed_of_sound
from .geometry import Vec3, centroid, distance, position_error
from .gcc_phat import GccPhatEstimate, GccPhatPairDiagnostics, estimate_tdoa_gcc_phat, gcc_phat_delay
from .localization import (
    LocalizationDiagnostics,
    estimate_localization_from_arrival_times,
    estimate_localization_from_tdoa,
    estimate_position_tdoa,
    solve_direction_tdoa,
)
from .models import AudioFrame, Environment, LocalizationResult, MicrophoneArray, PairDelay, PreprocessDiagnostics, TdoaEstimate
from .output import CsvWindowLogger, localization_result_to_dict, tracked_result_to_dict, write_json_result
from .preprocess import preprocess_frame, split_frames
from .simulation import generate_synthetic_signal, generate_synthetic_trajectory_signal
from .tracking import ResultTracker, TrackedResult, TrackingConfig

__all__ = [
    "LocalizationDiagnostics",
    "LocalizationResult",
    "GccPhatEstimate",
    "GccPhatPairDiagnostics",
    "MicrophoneArray",
    "PairDelay",
    "TdoaEstimate",
    "AudioFrame",
    "CsvWindowLogger",
    "Environment",
    "PreprocessDiagnostics",
    "ResultTracker",
    "TrackedResult",
    "TrackingConfig",
    "Vec3",
    "centroid",
    "distance",
    "estimate_tdoa_gcc_phat",
    "estimate_localization_from_arrival_times",
    "estimate_localization_from_tdoa",
    "estimate_position_tdoa",
    "generate_synthetic_signal",
    "generate_synthetic_trajectory_signal",
    "gcc_phat_delay",
    "localization_result_to_dict",
    "position_error",
    "preprocess_frame",
    "speed_of_sound",
    "split_frames",
    "solve_direction_tdoa",
    "tracked_result_to_dict",
    "write_json_result",
]
