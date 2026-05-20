import pytest

from positioning.environment import speed_of_sound
from positioning.geometry import Vec3
from positioning.models import AudioFrame, Environment, MicrophoneArray


def test_speed_of_sound_uses_temperature_and_humidity():
    assert speed_of_sound(20.0, 0.0) == pytest.approx(343.42)
    assert speed_of_sound(25.0, 50.0) == pytest.approx(347.07)


def test_microphone_array_accepts_coplanar_geometry():
    array = MicrophoneArray(
        (
            Vec3(0, 0, 0),
            Vec3(1, 0, 0),
            Vec3(1, 1, 0),
            Vec3(0, 1, 0),
        )
    )

    assert array.is_coplanar
    assert array.plane_normal is not None


def test_microphone_array_rejects_collinear_geometry():
    with pytest.raises(ValueError, match="non-collinear"):
        MicrophoneArray(
            (
                Vec3(0, 0, 0),
                Vec3(1, 0, 0),
                Vec3(2, 0, 0),
                Vec3(3, 0, 0),
            )
        )


def test_environment_exposes_sound_speed():
    environment = Environment(temperature_c=20.0, humidity_percent=0.0)
    assert environment.sound_speed == pytest.approx(343.42)


def test_audio_frame_rejects_uneven_channels():
    with pytest.raises(ValueError, match="same length"):
        AudioFrame(
            channels=((0.0, 1.0), (0.0,), (0.0, 1.0), (0.0, 1.0)),
            sample_rate=48000,
        )


def test_audio_frame_exposes_sample_count_and_duration():
    frame = AudioFrame(
        channels=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        sample_rate=2,
    )

    assert frame.sample_count == 2
    assert frame.duration == 1.0
