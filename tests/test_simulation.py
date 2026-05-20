from positioning.config import load_config
from positioning.geometry import Vec3
from positioning.simulation import generate_synthetic_signal, generate_synthetic_trajectory_signal


def test_generate_synthetic_signal_has_four_channels_and_metadata():
    config = load_config()

    frame, metadata = generate_synthetic_signal(
        config,
        Vec3(0.6, 0.4, 0.3),
        duration_s=0.05,
        noise_std=0.001,
        outlier_channel=2,
        outlier_time_s=0.03,
        outlier_amplitude=0.5,
        reflection_delay_s=0.002,
        reflection_gain=0.25,
    )

    assert len(frame.channels) == 4
    assert frame.sample_rate == config.audio.sample_rate
    assert frame.sample_count == 2400
    assert metadata["kind"] == "synthetic_signal"
    assert metadata["target"] == {"x": 0.6, "y": 0.4, "z": 0.3}
    assert metadata["reflection"] == {"delay_s": 0.002, "gain": 0.25}
    assert len(metadata["channels"]) == 4


def test_generate_synthetic_trajectory_signal_has_trajectory_metadata():
    config = load_config()

    frame, metadata = generate_synthetic_trajectory_signal(
        config,
        Vec3(0.35, 0.25, 0.3),
        Vec3(0.9, 0.55, 0.3),
        duration_s=1.2,
        point_count=6,
        noise_std=0.001,
    )

    assert len(frame.channels) == 4
    assert frame.sample_rate == config.audio.sample_rate
    assert frame.sample_count == 57600
    assert metadata["kind"] == "synthetic_trajectory_signal"
    assert len(metadata["trajectory"]) == 6
    assert len(metadata["events"]) == 6
