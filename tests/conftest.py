"""
Pytest configuration and fixtures for testing.
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import os
from typing import Generator
from fastapi.testclient import TestClient


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_audio_mono() -> np.ndarray:
    """
    Create a synthetic mono audio signal for testing.

    Returns a 1-second sine wave at 440 Hz (A4 note) sampled at 16kHz.
    """
    duration = 1.0  # seconds
    sr = 16000
    frequency = 440.0  # Hz (A4)

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * frequency * t)

    return audio.astype(np.float32)


@pytest.fixture
def sample_audio_stereo() -> np.ndarray:
    """
    Create a synthetic stereo audio signal for testing.

    Returns a 1-second stereo signal with different frequencies in each channel.
    """
    duration = 1.0
    sr = 16000

    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Left channel: 440 Hz (A4)
    left = 0.5 * np.sin(2 * np.pi * 440.0 * t)

    # Right channel: 554.37 Hz (C#5)
    right = 0.5 * np.sin(2 * np.pi * 554.37 * t)

    # Stack to create stereo
    audio = np.vstack([left, right])

    return audio.astype(np.float32)


@pytest.fixture
def sample_audio_file(temp_dir: Path, sample_audio_mono: np.ndarray) -> Path:
    """
    Create a temporary WAV file with sample audio.

    Returns the path to the created file.
    """
    import soundfile as sf

    file_path = temp_dir / "test_audio.wav"
    sr = 16000

    sf.write(str(file_path), sample_audio_mono, sr)

    return file_path


@pytest.fixture
def sample_pitch_curve() -> np.ndarray:
    """
    Create a sample pitch curve for testing quantization.

    Returns an array of frequencies representing a simple melody.
    """
    # Simple melody: C4, D4, E4, F4, G4 (each held for 0.5 seconds)
    # MIDI: 60, 62, 64, 65, 67
    # Frequencies in Hz
    freqs = [261.63, 293.66, 329.63, 349.23, 392.00]

    # Repeat each frequency 50 times (simulating 50 time steps per note)
    pitch_curve = np.repeat(freqs, 50)

    return pitch_curve.astype(np.float32)


@pytest.fixture
def sample_confidence() -> np.ndarray:
    """
    Create sample confidence scores matching the pitch curve.
    """
    # High confidence (0.9) for all notes
    return np.full(250, 0.9, dtype=np.float32)


@pytest.fixture
def sample_times() -> np.ndarray:
    """
    Create sample time stamps for the pitch curve.

    250 time points over 2.5 seconds (0.01s intervals).
    """
    return np.linspace(0, 2.5, 250, dtype=np.float32)


@pytest.fixture
def mock_note_data() -> dict:
    """
    Create mock note data matching the expected schema.
    """
    return {
        "midi": 60,
        "note_name": "C4",
        "start_time": 0.0,
        "duration": 0.5,
        "velocity": 100,
        "is_rest": False
    }


@pytest.fixture
def mock_metadata() -> dict:
    """
    Create mock metadata matching the expected schema.
    """
    return {
        "title": "test_song.mp3",
        "bpm": 120.0,
        "key": "C Major",
        "time_signature": "4/4"
    }


@pytest.fixture
def mock_transcription_response(mock_metadata: dict, mock_note_data: dict) -> dict:
    """
    Create a mock complete transcription response.
    """
    return {
        "job_id": "test-job-id-12345",
        "metadata": mock_metadata,
        "notes": [mock_note_data]
    }


@pytest.fixture
def api_client() -> TestClient:
    """
    Create a FastAPI test client.

    Note: This will import the main app, which will initialize the audio engine.
    For unit tests, consider mocking the engine instead.
    """
    from main import app
    return TestClient(app)


@pytest.fixture
def mock_audio_bytes() -> bytes:
    """
    Create mock audio file bytes for upload testing.

    Creates a minimal valid WAV file.
    """
    import io
    import soundfile as sf

    # Create simple audio data
    sr = 16000
    duration = 0.1  # 100ms
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Write to bytes
    buffer = io.BytesIO()
    sf.write(buffer, audio, sr, format='WAV')
    buffer.seek(0)

    return buffer.read()


# Markers for slow tests
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "api: mark test as an API test")
    config.addinivalue_line("markers", "functional: mark test as a functional test")
