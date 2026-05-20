import csv
import json

import pytest

from positioning.config import load_config
from positioning.geometry import Vec3
from positioning.models import LocalizationResult, PairDelay
from positioning.output import CsvWindowLogger, write_json_result
from positioning.tracking import ResultTracker, TrackingConfig


def _result(azimuth, confidence=1.0):
    return LocalizationResult(
        mode="direction",
        status="ok",
        direction=Vec3(1, 0, 0),
        azimuth_deg=azimuth,
        elevation_deg=20.0,
        used_pairs=6,
        reliable=confidence >= 0.5,
        confidence=confidence,
    )


def test_tracker_smooths_and_rejects_low_confidence_jump():
    tracker = ResultTracker(TrackingConfig(alpha=0.5, max_jump_deg=30.0, min_confidence_for_jump=0.4))

    first = tracker.update(_result(10.0), 0.0)
    second = tracker.update(_result(20.0), 1.0)
    jump = tracker.update(_result(120.0, confidence=0.1), 2.0)

    assert first.accepted
    assert second.accepted
    assert second.smoothed_azimuth_deg == pytest.approx(15.0)
    assert second.azimuth_rate_deg_s == pytest.approx(5.0)
    assert not jump.accepted
    assert jump.smoothed_azimuth_deg == second.smoothed_azimuth_deg
    assert jump.aggregated_confidence < 1.0


def test_json_and_csv_logging(tmp_path):
    config = load_config()
    tracker = ResultTracker()
    tracked = tracker.update(_result(15.0), 0.0)
    json_path = tmp_path / "result.json"
    csv_path = tmp_path / "windows.csv"

    write_json_result(tracked, json_path, config)
    with CsvWindowLogger(csv_path, config) as logger:
        logger.write(0, tracked, [PairDelay((0, 1), 0.001, quality=0.9)])

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["aggregated_confidence"] == 1.0

    rows = list(csv.DictReader(csv_path.open("r", encoding="utf-8")))
    assert rows[0]["status"] == "ok"
    assert rows[0]["tdoa_pairs_json"]
