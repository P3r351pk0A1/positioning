from positioning.models import AudioFrame
from positioning.offline_io import (
    metadata_path_for_signal,
    read_csv_audio,
    read_metadata,
    read_wav_audio,
    write_csv_audio,
    write_metadata,
    write_wav_audio,
)


def test_csv_audio_roundtrip(tmp_path):
    frame = AudioFrame(
        channels=((0.0, 0.1), (0.2, 0.3), (0.4, 0.5), (0.6, 0.7)),
        sample_rate=48000,
    )
    path = tmp_path / "signal.csv"

    write_csv_audio(frame, path)
    loaded = read_csv_audio(path, sample_rate=48000)

    assert loaded.sample_rate == 48000
    assert loaded.channels == frame.channels


def test_wav_audio_roundtrip(tmp_path):
    frame = AudioFrame(
        channels=((0.0, 0.25), (-0.25, 0.5), (0.75, -0.75), (1.0, -1.0)),
        sample_rate=48000,
    )
    path = tmp_path / "signal.wav"

    write_wav_audio(frame, path)
    loaded = read_wav_audio(path)

    assert loaded.sample_rate == 48000
    assert loaded.sample_count == 2
    assert abs(loaded.channels[0][1] - 0.25) < 1e-4
    assert abs(loaded.channels[3][1] + 1.0) < 1e-4


def test_metadata_roundtrip_and_default_path(tmp_path):
    signal_path = tmp_path / "signal.csv"
    metadata_path = metadata_path_for_signal(signal_path)
    metadata = {"kind": "test", "value": 3}

    write_metadata(metadata, metadata_path)

    assert metadata_path.name == "signal.csv.meta.json"
    assert read_metadata(metadata_path) == metadata
