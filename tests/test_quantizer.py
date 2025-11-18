"""
Unit tests for note quantization and segmentation.
"""
import pytest
import numpy as np
from app.services.quantizer import NoteQuantizer


class TestNoteQuantizerInitialization:
    """Tests for NoteQuantizer initialization."""

    @pytest.mark.unit
    def test_default_initialization(self):
        """Test quantizer initialization with defaults."""
        quantizer = NoteQuantizer()

        assert quantizer.hop_length == 160
        assert quantizer.min_note_duration == 0.05
        assert quantizer.grid_resolution == 16

    @pytest.mark.unit
    def test_custom_initialization(self):
        """Test quantizer initialization with custom parameters."""
        quantizer = NoteQuantizer(
            hop_length=441,
            min_note_duration=0.1,
            grid_resolution=8
        )

        assert quantizer.hop_length == 441
        assert quantizer.min_note_duration == 0.1
        assert quantizer.grid_resolution == 8


class TestHzToMidi:
    """Tests for frequency to MIDI conversion."""

    @pytest.mark.unit
    def test_a4_440hz(self):
        """Test A4 (440 Hz) converts to MIDI 69."""
        quantizer = NoteQuantizer()
        midi = quantizer.hz_to_midi(440.0)

        assert midi == 69

    @pytest.mark.unit
    def test_c4_middle_c(self):
        """Test Middle C (261.63 Hz) converts to MIDI 60."""
        quantizer = NoteQuantizer()
        midi = quantizer.hz_to_midi(261.63)

        assert midi == 60

    @pytest.mark.unit
    def test_frequency_zero(self):
        """Test that 0 Hz returns MIDI 0."""
        quantizer = NoteQuantizer()
        midi = quantizer.hz_to_midi(0.0)

        assert midi == 0

    @pytest.mark.unit
    def test_negative_frequency(self):
        """Test that negative frequency returns MIDI 0."""
        quantizer = NoteQuantizer()
        midi = quantizer.hz_to_midi(-100.0)

        assert midi == 0

    @pytest.mark.unit
    def test_very_high_frequency(self):
        """Test very high frequency is clamped to MIDI 127."""
        quantizer = NoteQuantizer()
        midi = quantizer.hz_to_midi(20000.0)  # Very high frequency

        assert midi == 127  # Clamped to max MIDI value

    @pytest.mark.unit
    def test_octave_doubling(self):
        """Test that doubling frequency increases MIDI by 12."""
        quantizer = NoteQuantizer()

        midi_a4 = quantizer.hz_to_midi(440.0)  # A4
        midi_a5 = quantizer.hz_to_midi(880.0)  # A5 (one octave up)

        assert midi_a5 - midi_a4 == 12


class TestMidiToNoteName:
    """Tests for MIDI to note name conversion."""

    @pytest.mark.unit
    def test_middle_c(self):
        """Test MIDI 60 converts to C4."""
        quantizer = NoteQuantizer()
        name = quantizer.midi_to_note_name(60)

        assert name == "C4"

    @pytest.mark.unit
    def test_a4(self):
        """Test MIDI 69 converts to A4."""
        quantizer = NoteQuantizer()
        name = quantizer.midi_to_note_name(69)

        assert name == "A4"

    @pytest.mark.unit
    def test_c_sharp(self):
        """Test sharp notes."""
        quantizer = NoteQuantizer()
        name = quantizer.midi_to_note_name(61)  # C#4

        assert name == "C#4"

    @pytest.mark.unit
    def test_lowest_note(self):
        """Test MIDI 0 (C-1)."""
        quantizer = NoteQuantizer()
        name = quantizer.midi_to_note_name(0)

        assert name == "C-1"

    @pytest.mark.unit
    def test_highest_note(self):
        """Test MIDI 127 (G9)."""
        quantizer = NoteQuantizer()
        name = quantizer.midi_to_note_name(127)

        assert name == "G9"

    @pytest.mark.unit
    def test_invalid_midi_negative(self):
        """Test invalid negative MIDI number."""
        quantizer = NoteQuantizer()
        name = quantizer.midi_to_note_name(-1)

        assert name == "N/A"

    @pytest.mark.unit
    def test_invalid_midi_too_high(self):
        """Test invalid MIDI number > 127."""
        quantizer = NoteQuantizer()
        name = quantizer.midi_to_note_name(128)

        assert name == "N/A"


class TestSmoothPitchCurve:
    """Tests for pitch curve smoothing."""

    @pytest.mark.unit
    def test_smooth_pitch_curve_basic(self):
        """Test basic pitch curve smoothing."""
        quantizer = NoteQuantizer()

        # Create a simple pitch curve
        pitch_curve = np.array([440.0, 442.0, 440.0, 441.0, 440.0])
        confidence = np.array([0.9, 0.9, 0.9, 0.9, 0.9])

        smoothed = quantizer.smooth_pitch_curve(pitch_curve, confidence)

        assert len(smoothed) == len(pitch_curve)
        assert isinstance(smoothed, np.ndarray)

    @pytest.mark.unit
    def test_smooth_removes_low_confidence(self):
        """Test that low confidence regions are set to zero."""
        quantizer = NoteQuantizer()

        pitch_curve = np.array([440.0, 440.0, 440.0, 440.0, 440.0])
        confidence = np.array([0.9, 0.2, 0.9, 0.9, 0.9])  # Low confidence in middle

        smoothed = quantizer.smooth_pitch_curve(
            pitch_curve,
            confidence,
            confidence_threshold=0.5
        )

        assert smoothed[1] == 0.0  # Low confidence should be zeroed

    @pytest.mark.unit
    def test_smooth_all_zeros_with_low_confidence(self):
        """Test that all low confidence results in all zeros."""
        quantizer = NoteQuantizer()

        pitch_curve = np.array([440.0, 440.0, 440.0])
        confidence = np.array([0.1, 0.2, 0.1])  # All low confidence

        smoothed = quantizer.smooth_pitch_curve(
            pitch_curve,
            confidence,
            confidence_threshold=0.5
        )

        assert np.all(smoothed == 0.0)


class TestSegmentNotes:
    """Tests for note segmentation."""

    @pytest.mark.unit
    def test_segment_single_note(self):
        """Test segmenting a single sustained note."""
        quantizer = NoteQuantizer(min_note_duration=0.05)

        # Single frequency sustained for 1 second
        pitch_curve = np.full(100, 440.0)  # 100 time steps
        times = np.linspace(0, 1.0, 100)

        notes = quantizer.segment_notes(pitch_curve, times)

        assert len(notes) == 1  # Should detect one note
        start, duration, midi = notes[0]
        assert start == pytest.approx(0.0, abs=0.01)
        assert duration == pytest.approx(1.0, abs=0.1)
        assert midi == 69  # A4

    @pytest.mark.unit
    def test_segment_multiple_notes(self, sample_pitch_curve: np.ndarray, sample_times: np.ndarray):
        """Test segmenting multiple notes."""
        quantizer = NoteQuantizer(min_note_duration=0.05)

        notes = quantizer.segment_notes(sample_pitch_curve, sample_times)

        assert len(notes) == 5  # Should detect 5 notes (from fixture)

    @pytest.mark.unit
    def test_segment_with_silence(self):
        """Test segmentation with silence (zero frequency) gaps."""
        quantizer = NoteQuantizer(min_note_duration=0.05)

        # Note, silence, note
        pitch_curve = np.array([440.0] * 20 + [0.0] * 10 + [523.25] * 20)
        times = np.linspace(0, 0.5, 50)

        notes = quantizer.segment_notes(pitch_curve, times)

        assert len(notes) == 2  # Two notes separated by silence
        assert notes[0][2] == 69  # A4
        assert notes[1][2] == 72  # C5

    @pytest.mark.unit
    def test_segment_filters_short_notes(self):
        """Test that very short notes are filtered out."""
        quantizer = NoteQuantizer(min_note_duration=0.1)  # 100ms minimum

        # Very short note (20ms) followed by longer note
        pitch_curve = np.array([440.0] * 2 + [523.25] * 20)
        times = np.linspace(0, 0.22, 22)  # 10ms per sample

        notes = quantizer.segment_notes(pitch_curve, times)

        # First note is too short and should be filtered
        assert len(notes) == 1
        assert notes[0][2] == 72  # Only the C5 should remain


class TestQuantizeTime:
    """Tests for time quantization."""

    @pytest.mark.unit
    def test_quantize_on_beat(self):
        """Test quantizing time that's already on the beat."""
        quantizer = NoteQuantizer(grid_resolution=16)

        # At 120 BPM, each beat is 0.5 seconds
        # 1/16th note = 0.125 seconds
        time = 0.5  # Exactly on beat
        bpm = 120.0

        quantized = quantizer.quantize_time(time, bpm)

        assert quantized == pytest.approx(0.5, abs=0.001)

    @pytest.mark.unit
    def test_quantize_snaps_to_grid(self):
        """Test that off-grid times snap to nearest grid position."""
        quantizer = NoteQuantizer(grid_resolution=16)

        bpm = 120.0
        # At 120 BPM, 1/16th note = 0.125 seconds
        # Test a time slightly off (0.13) should snap to 0.125

        quantized = quantizer.quantize_time(0.13, bpm)

        assert quantized == pytest.approx(0.125, abs=0.01)

    @pytest.mark.unit
    def test_quantize_different_resolutions(self):
        """Test quantization with different grid resolutions."""
        bpm = 120.0
        time = 0.5

        # 1/8th notes
        quantizer_8 = NoteQuantizer(grid_resolution=8)
        q8 = quantizer_8.quantize_time(time, bpm)

        # 1/16th notes
        quantizer_16 = NoteQuantizer(grid_resolution=16)
        q16 = quantizer_16.quantize_time(time, bpm)

        # Both should quantize to 0.5 in this case
        assert q8 == pytest.approx(0.5, abs=0.01)
        assert q16 == pytest.approx(0.5, abs=0.01)


class TestQuantizeNotes:
    """Tests for note quantization."""

    @pytest.mark.unit
    def test_quantize_notes_basic(self):
        """Test basic note quantization."""
        quantizer = NoteQuantizer(grid_resolution=16)

        notes = [
            (0.0, 0.5, 60),  # C4
            (0.5, 0.5, 64),  # E4
        ]

        bpm = 120.0

        quantized = quantizer.quantize_notes(notes, bpm)

        assert len(quantized) == 2
        assert quantized[0]["midi"] == 60
        assert quantized[0]["note_name"] == "C4"
        assert quantized[0]["start_time"] == pytest.approx(0.0, abs=0.01)
        assert quantized[0]["velocity"] == 100
        assert quantized[0]["is_rest"] is False

    @pytest.mark.unit
    def test_quantize_notes_formats_correctly(self):
        """Test that quantized notes have all required fields."""
        quantizer = NoteQuantizer()

        notes = [(0.0, 0.5, 67)]  # G4
        bpm = 120.0

        quantized = quantizer.quantize_notes(notes, bpm)

        note = quantized[0]
        assert "midi" in note
        assert "note_name" in note
        assert "start_time" in note
        assert "duration" in note
        assert "velocity" in note
        assert "is_rest" in note


class TestProcessComplete:
    """Tests for the complete processing pipeline."""

    @pytest.mark.unit
    def test_process_end_to_end(
        self,
        sample_pitch_curve: np.ndarray,
        sample_confidence: np.ndarray,
        sample_times: np.ndarray
    ):
        """Test the complete processing pipeline."""
        quantizer = NoteQuantizer()

        bpm = 120.0

        result = quantizer.process(
            sample_pitch_curve,
            sample_confidence,
            sample_times,
            bpm
        )

        assert isinstance(result, list)
        assert len(result) > 0  # Should detect notes
        assert all("midi" in note for note in result)
        assert all("note_name" in note for note in result)
        assert all("start_time" in note for note in result)
        assert all("duration" in note for note in result)

    @pytest.mark.unit
    def test_process_with_low_confidence(self):
        """Test processing with mixed confidence scores."""
        quantizer = NoteQuantizer()

        pitch_curve = np.array([440.0] * 50 + [523.25] * 50)
        confidence = np.array([0.9] * 50 + [0.3] * 50)  # Second half has low confidence
        times = np.linspace(0, 1.0, 100)
        bpm = 120.0

        result = quantizer.process(pitch_curve, confidence, times, bpm)

        # Should only detect the first note (high confidence)
        # Second note should be filtered due to low confidence
        assert len(result) >= 1

    @pytest.mark.unit
    def test_process_empty_input(self):
        """Test processing empty arrays."""
        quantizer = NoteQuantizer()

        pitch_curve = np.array([])
        confidence = np.array([])
        times = np.array([])
        bpm = 120.0

        result = quantizer.process(pitch_curve, confidence, times, bpm)

        assert isinstance(result, list)
        assert len(result) == 0
