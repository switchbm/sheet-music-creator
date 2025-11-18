"""
Unit tests for audio loading utilities.
"""
import pytest
import numpy as np
from pathlib import Path
import soundfile as sf

from app.utils.audio_loader import AudioLoader, load_audio


class TestAudioLoader:
    """Tests for the AudioLoader class."""

    @pytest.mark.unit
    def test_audio_loader_initialization(self):
        """Test AudioLoader initialization with default parameters."""
        loader = AudioLoader()
        assert loader.target_sr == 16000

    @pytest.mark.unit
    def test_audio_loader_custom_sample_rate(self):
        """Test AudioLoader initialization with custom sample rate."""
        loader = AudioLoader(target_sr=22050)
        assert loader.target_sr == 22050

    @pytest.mark.unit
    def test_load_audio_file(self, sample_audio_file: Path):
        """Test loading a valid audio file."""
        loader = AudioLoader(target_sr=16000)
        audio, sr = loader.load(sample_audio_file)

        assert isinstance(audio, np.ndarray)
        assert audio.ndim == 1  # Mono audio
        assert sr == 16000
        assert len(audio) > 0

    @pytest.mark.unit
    def test_load_audio_with_custom_sr(self, sample_audio_file: Path):
        """Test loading audio with custom sample rate."""
        loader = AudioLoader(target_sr=16000)
        audio, sr = loader.load(sample_audio_file, sr=22050)

        assert sr == 22050  # Should use the specified sr, not target_sr

    @pytest.mark.unit
    def test_load_nonexistent_file(self):
        """Test loading a non-existent file raises FileNotFoundError."""
        loader = AudioLoader()

        with pytest.raises(FileNotFoundError, match="Audio file not found"):
            loader.load("nonexistent_file.wav")

    @pytest.mark.unit
    def test_load_stereo_preserved(self, temp_dir: Path, sample_audio_stereo: np.ndarray):
        """Test loading stereo audio without converting to mono."""
        # Create stereo audio file
        file_path = temp_dir / "stereo_test.wav"
        sr = 16000
        # sf.write expects (n_samples, n_channels) for stereo
        sf.write(str(file_path), sample_audio_stereo.T, sr)

        loader = AudioLoader(target_sr=sr)
        audio, returned_sr = loader.load(file_path, mono=False)

        assert returned_sr == sr
        # librosa returns stereo as (2, n_samples)
        assert audio.ndim == 2

    @pytest.mark.unit
    def test_load_stereo_to_mono(self, temp_dir: Path, sample_audio_stereo: np.ndarray):
        """Test converting stereo to mono during load."""
        # Create stereo audio file
        file_path = temp_dir / "stereo_to_mono.wav"
        sr = 16000
        sf.write(str(file_path), sample_audio_stereo.T, sr)

        loader = AudioLoader(target_sr=sr)
        audio, returned_sr = loader.load(file_path, mono=True)

        assert returned_sr == sr
        assert audio.ndim == 1  # Should be converted to mono

    @pytest.mark.unit
    def test_load_with_duration_limit(self, sample_audio_file: Path):
        """Test loading only a portion of an audio file."""
        loader = AudioLoader(target_sr=16000)

        # Load only 0.5 seconds
        audio, sr = loader.load(sample_audio_file, duration=0.5)

        expected_samples = int(0.5 * sr)
        assert len(audio) <= expected_samples + 100  # Small tolerance

    @pytest.mark.unit
    def test_load_stereo_method(self, temp_dir: Path, sample_audio_stereo: np.ndarray):
        """Test the load_stereo convenience method."""
        file_path = temp_dir / "stereo_method_test.wav"
        sr = 16000
        sf.write(str(file_path), sample_audio_stereo.T, sr)

        loader = AudioLoader(target_sr=sr)
        audio, returned_sr = loader.load_stereo(file_path)

        assert returned_sr == sr
        assert audio.ndim == 2  # Stereo

    @pytest.mark.unit
    def test_save_audio(self, temp_dir: Path, sample_audio_mono: np.ndarray):
        """Test saving audio to a file."""
        loader = AudioLoader()
        file_path = temp_dir / "saved_audio.wav"
        sr = 16000

        loader.save(sample_audio_mono, file_path, sr)

        # Verify file was created
        assert file_path.exists()

        # Load it back and verify
        loaded_audio, loaded_sr = loader.load(file_path)
        assert loaded_sr == sr
        assert len(loaded_audio) == len(sample_audio_mono)
        # Check similarity (allowing for small numerical differences from encoding/decoding)
        np.testing.assert_allclose(loaded_audio, sample_audio_mono, rtol=1e-2, atol=1e-4)

    @pytest.mark.unit
    def test_get_duration(self, sample_audio_file: Path):
        """Test getting audio file duration."""
        loader = AudioLoader()
        duration = loader.get_duration(sample_audio_file)

        assert isinstance(duration, float)
        assert duration > 0
        # Should be approximately 1 second (from fixture)
        assert 0.9 < duration < 1.1

    @pytest.mark.unit
    def test_load_with_path_object(self, sample_audio_file: Path):
        """Test loading audio using Path object instead of string."""
        loader = AudioLoader()
        audio, sr = loader.load(sample_audio_file)  # sample_audio_file is already a Path

        assert isinstance(audio, np.ndarray)
        assert sr == 16000

    @pytest.mark.unit
    def test_load_with_string_path(self, sample_audio_file: Path):
        """Test loading audio using string path."""
        loader = AudioLoader()
        audio, sr = loader.load(str(sample_audio_file))

        assert isinstance(audio, np.ndarray)
        assert sr == 16000


class TestLoadAudioConvenienceFunction:
    """Tests for the load_audio convenience function."""

    @pytest.mark.unit
    def test_load_audio_default_sr(self, sample_audio_file: Path):
        """Test convenience function with default sample rate."""
        audio, sr = load_audio(sample_audio_file)

        assert isinstance(audio, np.ndarray)
        assert sr == 16000

    @pytest.mark.unit
    def test_load_audio_custom_sr(self, sample_audio_file: Path):
        """Test convenience function with custom sample rate."""
        audio, sr = load_audio(sample_audio_file, sr=22050)

        assert sr == 22050

    @pytest.mark.unit
    def test_load_audio_string_path(self, sample_audio_file: Path):
        """Test convenience function with string path."""
        audio, sr = load_audio(str(sample_audio_file))

        assert isinstance(audio, np.ndarray)
        assert sr == 16000


class TestAudioLoaderEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.unit
    def test_load_empty_audio(self, temp_dir: Path):
        """Test handling of empty audio file."""
        # Create a very short audio file
        file_path = temp_dir / "empty.wav"
        sr = 16000
        empty_audio = np.array([], dtype=np.float32)

        sf.write(str(file_path), empty_audio, sr)

        loader = AudioLoader()
        audio, returned_sr = loader.load(file_path)

        assert len(audio) == 0

    @pytest.mark.unit
    def test_load_very_short_audio(self, temp_dir: Path):
        """Test loading very short audio file."""
        file_path = temp_dir / "very_short.wav"
        sr = 16000
        duration = 0.01  # 10ms

        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t)

        sf.write(str(file_path), audio, sr)

        loader = AudioLoader()
        loaded_audio, returned_sr = loader.load(file_path)

        assert returned_sr == sr
        assert len(loaded_audio) > 0

    @pytest.mark.unit
    def test_resample_downsampling(self, temp_dir: Path):
        """Test resampling to a lower sample rate."""
        # Create audio at 44100 Hz
        file_path = temp_dir / "high_sr.wav"
        original_sr = 44100
        duration = 1.0

        t = np.linspace(0, duration, int(original_sr * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        sf.write(str(file_path), audio, original_sr)

        # Load at 16000 Hz
        loader = AudioLoader(target_sr=16000)
        loaded_audio, returned_sr = loader.load(file_path)

        assert returned_sr == 16000
        # Should have fewer samples after downsampling
        assert len(loaded_audio) < len(audio)

    @pytest.mark.unit
    def test_resample_upsampling(self, temp_dir: Path):
        """Test resampling to a higher sample rate."""
        # Create audio at 8000 Hz
        file_path = temp_dir / "low_sr.wav"
        original_sr = 8000
        duration = 1.0

        t = np.linspace(0, duration, int(original_sr * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        sf.write(str(file_path), audio, original_sr)

        # Load at 16000 Hz
        loader = AudioLoader(target_sr=16000)
        loaded_audio, returned_sr = loader.load(file_path)

        assert returned_sr == 16000
        # Should have more samples after upsampling
        assert len(loaded_audio) > len(audio)
