from positioning.cli import main
import positioning.cli as cli
from positioning.config import load_config
from positioning.geometry import Vec3
from positioning.models import AudioFrame
from positioning.visualization import TrajectoryEstimate, TruthTrajectoryPoint


def test_demo_cli_smoke(capsys):
    code = main(["demo"])
    output = capsys.readouterr().out

    assert code == 0
    assert "Статус: ok" in output
    assert "Надежность: да" in output
    assert "Использовано пар TDOA:" in output
    assert "Скорость звука:" in output


def test_validate_cli_smoke(capsys):
    code = main(["validate"])
    output = capsys.readouterr().out

    assert code == 0
    assert "=== Planar direction tests ===" in output
    assert "max direction err" in output
    assert "Robust TDOA-pair outlier test" in output


def test_generate_and_preprocess_cli_smoke(tmp_path, capsys):
    signal_path = tmp_path / "signal.csv"
    processed_dir = tmp_path / "processed"

    generate_code = main(
        [
            "generate-synthetic",
            "--output",
            str(signal_path),
            "--duration",
            "0.1",
            "--noise-std",
            "0.001",
            "--reflection-delay",
            "0.002",
            "--reflection-gain",
            "0.2",
        ]
    )
    generate_output = capsys.readouterr().out

    preprocess_code = main(["preprocess", "--input", str(signal_path), "--output-dir", str(processed_dir)])
    preprocess_output = capsys.readouterr().out

    assert generate_code == 0
    assert "Сигнал сохранен" in generate_output
    assert signal_path.exists()
    assert signal_path.with_suffix(".csv.meta.json").exists()
    assert preprocess_code == 0
    assert "Окон обработано" in preprocess_output
    assert (processed_dir / "preprocess.meta.json").exists()
    assert any(processed_dir.glob("window_*.csv"))


def test_generate_wav_and_inspect_audio_cli_smoke(tmp_path, capsys):
    signal_path = tmp_path / "signal.wav"

    generate_code = main(
        [
            "generate-synthetic",
            "--format",
            "wav",
            "--output",
            str(signal_path),
            "--duration",
            "0.1",
            "--source-kind",
            "pulse",
        ]
    )
    capsys.readouterr()
    inspect_code = main(["inspect-audio", "--input", str(signal_path), "--require-metadata"])
    output = capsys.readouterr().out

    assert generate_code == 0
    assert inspect_code == 0
    assert "Статус: ok" in output
    assert "Формат: WAV" in output
    assert "Каналов: 4" in output
    assert "Метаданные: найдены" in output
    assert signal_path.with_suffix(".wav.meta.json").exists()


def test_generate_trajectory_cli_smoke(tmp_path, capsys):
    signal_path = tmp_path / "trajectory.csv"

    code = main(
        [
            "generate-trajectory",
            "--output",
            str(signal_path),
            "--duration",
            "1.0",
            "--point-count",
            "5",
        ]
    )
    output = capsys.readouterr().out

    assert code == 0
    assert "Сигнал траектории сохранен" in output
    assert "Точек траектории: 5" in output
    assert signal_path.exists()
    assert signal_path.with_suffix(".csv.meta.json").exists()


def test_visualize_trajectory_cli_smoke_without_opening_window(tmp_path, monkeypatch, capsys):
    signal_path = tmp_path / "trajectory.csv"
    main(["generate-trajectory", "--output", str(signal_path), "--duration", "1.0", "--point-count", "5"])
    capsys.readouterr()
    called = {}

    def fake_show(microphones, truth_points, estimates):
        called["microphones"] = len(microphones)
        called["truth_points"] = len(truth_points)
        called["estimates"] = len(estimates)

    monkeypatch.setattr(cli, "show_trajectory_window", fake_show)
    code = main(["visualize-trajectory", "--input", str(signal_path), "--trajectory-max-windows", "5"])
    output = capsys.readouterr().out

    assert code == 0
    assert "Открывается окно визуализации траектории" in output
    assert called["microphones"] == 4
    assert called["truth_points"] == 5


def test_demo_json_and_track_demo_logging(tmp_path, capsys):
    demo_json = tmp_path / "demo.json"
    csv_log = tmp_path / "tracking.csv"
    final_json = tmp_path / "final.json"

    demo_code = main(["demo", "--json-output", str(demo_json)])
    demo_output = capsys.readouterr().out
    track_code = main(
        [
            "track-demo",
            "--csv-log",
            str(csv_log),
            "--json-output",
            str(final_json),
            "--window-count",
            "6",
            "--inject-low-confidence-jump",
        ]
    )
    track_output = capsys.readouterr().out

    assert demo_code == 0
    assert "JSON-результат сохранен" in demo_output
    assert demo_json.exists()
    assert track_code == 0
    assert "CSV-журнал сохранен" in track_output
    assert csv_log.exists()
    assert final_json.exists()


def test_localize_audio_cli_uses_gcc_phat(tmp_path, capsys):
    signal_path = tmp_path / "signal.csv"
    csv_log = tmp_path / "audio_log.csv"
    final_json = tmp_path / "audio_final.json"

    generate_code = main(
        [
            "generate-synthetic",
            "--output",
            str(signal_path),
            "--duration",
            "0.12",
            "--source-kind",
            "pulse",
            "--noise-std",
            "0.001",
        ]
    )
    capsys.readouterr()
    localize_code = main(
        [
            "localize-audio",
            "--input",
            str(signal_path),
            "--mode",
            "direction",
            "--csv-log",
            str(csv_log),
            "--json-output",
            str(final_json),
            "--max-windows",
            "3",
        ]
    )
    output = capsys.readouterr().out

    assert generate_code == 0
    assert localize_code == 0
    assert "Окон со статусом ok:" in output
    assert "Режим локализации: direction" in output
    assert csv_log.exists()
    assert final_json.exists()


def test_localization_cli_can_visualize_single_target_without_opening_window(tmp_path, monkeypatch, capsys):
    signal_path = tmp_path / "point.csv"
    main(["generate-synthetic", "--output", str(signal_path), "--source-kind", "pulse", "--duration", "0.12"])
    capsys.readouterr()
    called = {}

    def fake_show(microphones, truth_points, estimates):
        called["microphones"] = len(microphones)
        called["truth_points"] = len(truth_points)
        called["estimates"] = len(estimates)

    monkeypatch.setattr(cli, "show_trajectory_window", fake_show)
    code = main(
        [
            "localization",
            "--input",
            str(signal_path),
            "--mode",
            "direction",
            "--max-windows",
            "1",
            "--show-visualization",
            "--trajectory-max-windows",
            "10",
        ]
    )
    output = capsys.readouterr().out

    assert code == 0
    assert "Открывается окно визуализации локализации" in output
    assert "исходных точек 1" in output
    assert called["microphones"] == 4
    assert called["truth_points"] == 1


def test_localization_cli_can_visualize_trajectory_without_opening_window(tmp_path, monkeypatch, capsys):
    signal_path = tmp_path / "trajectory.csv"
    main(["generate-trajectory", "--output", str(signal_path), "--duration", "1.0", "--point-count", "5"])
    capsys.readouterr()
    called = {}

    def fake_show(microphones, truth_points, estimates):
        called["microphones"] = len(microphones)
        called["truth_points"] = len(truth_points)
        called["estimates"] = len(estimates)

    monkeypatch.setattr(cli, "show_trajectory_window", fake_show)
    code = main(
        [
            "localization",
            "--input",
            str(signal_path),
            "--mode",
            "direction",
            "--show-visualization",
            "--trajectory-max-windows",
            "5",
        ]
    )
    output = capsys.readouterr().out

    assert code == 0
    assert "Открывается окно визуализации локализации" in output
    assert "исходных точек 5" in output
    assert called["microphones"] == 4
    assert called["truth_points"] == 5


def test_select_visualization_windows_keeps_monotonic_time_order():
    config = load_config()
    channel = (0.0, 0.0)
    windows = [
        AudioFrame((channel, channel, channel, channel), sample_rate=20, timestamp=index * 0.1)
        for index in range(6)
    ]
    center = config.microphone_array.center
    truth_samples = [
        TruthTrajectoryPoint(center, timestamp=0.25),
        TruthTrajectoryPoint(center, timestamp=0.05),
        TruthTrajectoryPoint(center, timestamp=0.15),
    ]

    selected = cli._select_visualization_windows(config, windows, truth_samples, limit=3)
    timestamps = [item.timestamp for item in selected]

    assert len(selected) == 3
    assert timestamps == sorted(timestamps)
    assert len(set(timestamps)) == len(timestamps)


def test_kinematic_filter_rejects_high_speed_outlier():
    estimates = [
        TrajectoryEstimate(Vec3(0.0, 0.0, 0.0), 0.0, 0.9, "ok"),
        TrajectoryEstimate(Vec3(0.1, 0.0, 0.0), 0.1, 0.9, "ok"),
        TrajectoryEstimate(Vec3(1.0, 0.0, 0.0), 0.2, 0.9, "ok"),
        TrajectoryEstimate(Vec3(0.2, 0.0, 0.0), 0.3, 0.9, "ok"),
    ]

    filtered, rejected = cli._filter_trajectory_by_kinematics(
        estimates,
        max_speed_m_s=2.0,
        max_acceleration_m_s2=30.0,
    )

    assert rejected == 1
    assert [item.timestamp for item in filtered] == [0.0, 0.1, 0.3]
