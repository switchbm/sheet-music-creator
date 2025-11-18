"""
Unit tests for the audio processing engine.

Note: These tests mock the heavy ML models (Demucs, CREPE) to avoid
downloading models during testing.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from app.services.engine import AudioProcessingEngine


class TestAudioProcessingEngineInitialization:
    """Tests for AudioProcessingEngine initialization."""

    @pytest.mark.unit
    @patch('app.services.engine.get_model')
    def test_initialization_cpu(self, mock_get_model):
        """Test engine initialization on CPU."""
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        engine = AudioProcessingEngine(device='cpu')

        assert engine.device == 'cpu'
        assert engine.crepe_model == 'full'
        mock_get_model.assert_called_once_with(name='htdemucs')

    @pytest.mark.unit
    @patch('app.services.engine.get_model')
    @patch('app.services.engine.torch.cuda.is_available', return_value=False)
    def test_initialization_auto_detect_cpu(self, mock_cuda, mock_get_model):
        """Test engine auto-detects CPU when CUDA not available."""
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        engine = AudioProcessingEngine(device=None)

        assert engine.device == 'cpu'

    @pytest.mark.unit
    @patch('app.services.engine.get_model')
    def test_initialization_custom_models(self, mock_get_model):
        """Test initialization with custom model names."""
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        engine = AudioProcessingEngine(
            device='cpu',
            demucs_model='htdemucs',
            crepe_model='tiny'
        )

        assert engine.crepe_model == 'tiny'
        mock_get_model.assert_called_with(name='htdemucs')


class TestSeparateVocals:
    """Tests for vocal separation."""

    @pytest.mark.unit
    @patch('app.services.engine.get_model')
    @patch('app.services.engine.apply_model')
    @patch('app.services.engine.torch')
    def test_separate_vocals_mono_input(self, mock_torch, mock_apply_model, mock_get_model):
        """Test vocal separation with mono input."""
        # Setup mocks
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        # Mock the output of apply_model to return a torch-like tensor
        # Output shape: [batch, sources, channels, samples]
        output_data = np.random.randn(1, 4, 2, 16000).astype(np.float32)
        mock_tensor = MagicMock()
        mock_tensor.__getitem__ = lambda self, idx: MagicMock(cpu=lambda: MagicMock(numpy=lambda: output_data[idx]))
        mock_apply_model.return_value = mock_tensor

        engine = AudioProcessingEngine(device='cpu')

        # Create mono audio
        audio = np.random.randn(16000).astype(np.float32)
        sr = 16000

        vocals, returned_sr = engine.separate_vocals(audio, sr)

        assert isinstance(vocals, np.ndarray)
        assert returned_sr == sr
        assert vocals.ndim == 1  # Should be mono output

    @pytest.mark.unit
    @patch('app.services.engine.get_model')
    @patch('app.services.engine.apply_model')
    def test_separate_vocals_stereo_input(self, mock_apply_model, mock_get_model):
        """Test vocal separation with stereo input."""
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        # Mock stereo output
        mock_output = np.random.randn(1, 4, 2, 16000).astype(np.float32)
        mock_apply_model.return_value = mock_output

        engine = AudioProcessingEngine(device='cpu')

        # Create stereo audio (2, samples)
        audio = np.random.randn(2, 16000).astype(np.float32)
        sr = 16000

        vocals, returned_sr = engine.separate_vocals(audio, sr)

        assert isinstance(vocals, np.ndarray)
        assert vocals.ndim == 1  # Should be converted to mono


class TestDetectPitch:
    """Tests for pitch detection."""

    @pytest.mark.unit
    @patch('app.services.engine.get_model')
    @patch('app.services.engine.torchcrepe.predict')
    def test_detect_pitch_basic(self, mock_predict, mock_get_model):
        """Test basic pitch detection."""
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        # Mock CREPE output
        n_frames = 100
        mock_time = Mock()
        mock_time.cpu.return_value.numpy.return_value.squeeze.return_value = \
            np.linspace(0, 1, n_frames)

        mock_freq = Mock()
        mock_freq.cpu.return_value.numpy.return_value.squeeze.return_value = \
            np.full(n_frames, 440.0)  # A4

        mock_conf = Mock()
        mock_conf.cpu.return_value.numpy.return_value.squeeze.return_value = \
            np.full(n_frames, 0.9)

        mock_predict.return_value = (mock_time, mock_freq, mock_conf, None)

        engine = AudioProcessingEngine(device='cpu')

        vocals = np.random.randn(16000).astype(np.float32)
        sr = 16000

        pitch_curve, confidence, times = engine.detect_pitch(vocals, sr)

        assert isinstance(pitch_curve, np.ndarray)
        assert isinstance(confidence, np.ndarray)
        assert isinstance(times, np.ndarray)
        assert len(pitch_curve) == n_frames
        assert len(confidence) == n_frames
        assert len(times) == n_frames


class TestDetectTempo:
    """Tests for tempo detection."""

    @pytest.mark.unit
    @patch('app.services.engine.get_model')
    @patch('app.services.engine.librosa.beat.beat_track')
    def test_detect_tempo(self, mock_beat_track, mock_get_model):
        """Test tempo detection."""
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        # Mock librosa beat tracker
        mock_beat_track.return_value = (np.array([120.0]), None)

        engine = AudioProcessingEngine(device='cpu')

        audio = np.random.randn(44100).astype(np.float32)
        sr = 44100

        bpm = engine.detect_tempo(audio, sr)

        assert isinstance(bpm, float)
        assert bpm == 120.0

    @pytest.mark.unit
    @patch('app.services.engine.get_model')
    @patch('app.services.engine.librosa.beat.beat_track')
    def test_detect_tempo_fallback(self, mock_beat_track, mock_get_model):
        """Test tempo detection fallback to 120 BPM."""
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        # Mock librosa returning empty array
        mock_beat_track.return_value = (np.array([]), None)

        engine = AudioProcessingEngine(device='cpu')

        audio = np.random.randn(44100).astype(np.float32)
        sr = 44100

        bpm = engine.detect_tempo(audio, sr)

        assert bpm == 120.0  # Should fall back to default


class TestDetectKey:
    """Tests for key detection."""

    @pytest.mark.unit
    @patch('app.services.engine.get_model')
    @patch('app.services.engine.librosa.feature.chroma_cqt')
    def test_detect_key(self, mock_chroma, mock_get_model):
        """Test key detection."""
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        # Mock chroma features (12 pitch classes x time frames)
        # Create a chroma that favors C Major
        chroma = np.zeros((12, 100))
        chroma[0, :] = 1.0  # C
        chroma[4, :] = 0.8  # E
        chroma[7, :] = 0.8  # G

        mock_chroma.return_value = chroma

        engine = AudioProcessingEngine(device='cpu')

        audio = np.random.randn(44100).astype(np.float32)
        sr = 44100

        key = engine.detect_key(audio, sr)

        assert isinstance(key, str)
        assert "Major" in key or "Minor" in key


class TestProcessComplete:
    """Tests for the complete processing pipeline."""

    @pytest.mark.unit
    @patch('app.services.engine.get_model')
    @patch('app.services.engine.apply_model')
    @patch('app.services.engine.torchcrepe.predict')
    @patch('app.services.engine.librosa.beat.beat_track')
    @patch('app.services.engine.librosa.feature.chroma_cqt')
    def test_process_complete_pipeline(
        self,
        mock_chroma,
        mock_beat_track,
        mock_predict,
        mock_apply_model,
        mock_get_model,
        sample_audio_file: Path
    ):
        """Test the complete processing pipeline with mocks."""
        # Setup all mocks
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        # Mock Demucs output
        mock_output = np.random.randn(1, 4, 2, 16000).astype(np.float32)
        mock_apply_model.return_value = mock_output

        # Mock CREPE output
        n_frames = 100
        mock_time = Mock()
        mock_time.cpu.return_value.numpy.return_value.squeeze.return_value = \
            np.linspace(0, 1, n_frames)

        mock_freq = Mock()
        mock_freq.cpu.return_value.numpy.return_value.squeeze.return_value = \
            np.full(n_frames, 440.0)

        mock_conf = Mock()
        mock_conf.cpu.return_value.numpy.return_value.squeeze.return_value = \
            np.full(n_frames, 0.9)

        mock_predict.return_value = (mock_time, mock_freq, mock_conf, None)

        # Mock tempo detection
        mock_beat_track.return_value = (np.array([120.0]), None)

        # Mock key detection
        chroma = np.random.randn(12, 100)
        mock_chroma.return_value = chroma

        # Run the pipeline
        engine = AudioProcessingEngine(device='cpu')
        result = engine.process(str(sample_audio_file))

        # Verify output structure
        assert "metadata" in result
        assert "notes" in result

        metadata = result["metadata"]
        assert "title" in metadata
        assert "bpm" in metadata
        assert "key" in metadata
        assert "time_signature" in metadata

        notes = result["notes"]
        assert isinstance(notes, list)

    @pytest.mark.unit
    @patch('app.services.engine.get_model')
    def test_process_with_custom_title(self, mock_get_model, sample_audio_file: Path):
        """Test process with custom title."""
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        engine = AudioProcessingEngine(device='cpu')

        # We need to mock the entire pipeline to avoid actual processing
        with patch.object(engine, 'separate_vocals') as mock_sep, \
             patch.object(engine, 'detect_pitch') as mock_pitch, \
             patch.object(engine, 'detect_tempo') as mock_tempo, \
             patch.object(engine, 'detect_key') as mock_key, \
             patch.object(engine.quantizer, 'process') as mock_quant:

            mock_sep.return_value = (np.random.randn(16000), 44100)
            mock_pitch.return_value = (
                np.array([440.0]),
                np.array([0.9]),
                np.array([0.0])
            )
            mock_tempo.return_value = 120.0
            mock_key.return_value = "C Major"
            mock_quant.return_value = []

            result = engine.process(str(sample_audio_file), title="Custom Title")

            assert result["metadata"]["title"] == "Custom Title"


class TestAudioProcessingEngineEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.unit
    @patch('app.services.engine.get_model')
    def test_process_nonexistent_file(self, mock_get_model):
        """Test processing a non-existent file."""
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        engine = AudioProcessingEngine(device='cpu')

        # Should raise an exception when trying to load non-existent file
        with pytest.raises(Exception):
            engine.process("nonexistent_file.mp3")
