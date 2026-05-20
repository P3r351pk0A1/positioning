from positioning.config import load_config
from positioning.geometry import Vec3
from positioning.preprocess import preprocess_frame, split_frames
from positioning.simulation import generate_synthetic_signal


def test_split_frames_uses_window_and_overlap():
    config = load_config()
    frame, _ = generate_synthetic_signal(config, Vec3(0.6, 0.4, 0.3), duration_s=0.1)

    windows = split_frames(frame, window_size=1000, overlap=0.5)

    assert len(windows) == 8
    assert all(window.sample_count == 1000 for window in windows)
    assert windows[1].timestamp == 500 / frame.sample_rate


def test_preprocess_frame_returns_diagnostics():
    config = load_config()
    frame, _ = generate_synthetic_signal(config, Vec3(0.6, 0.4, 0.3), duration_s=0.1, noise_std=0.001)
    window = split_frames(frame, config.audio.window_size, config.audio.overlap)[1]

    processed, diagnostics = preprocess_frame(window, config.audio)

    assert processed.sample_count == config.audio.window_size
    assert processed.sample_rate == config.audio.sample_rate
    assert len(diagnostics.rms_by_channel) == 4
    assert diagnostics.mean_rms > 0.0
    assert diagnostics.mean_abs_correlation >= 0.0
    assert diagnostics.informative
