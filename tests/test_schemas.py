"""
Unit tests for Pydantic data models and schemas.
"""
import pytest
from pydantic import ValidationError
from app.models.schemas import (
    NoteSchema,
    MetadataSchema,
    TranscriptionResponse,
    ErrorResponse
)


class TestNoteSchema:
    """Tests for NoteSchema model."""

    @pytest.mark.unit
    def test_valid_note_creation(self):
        """Test creating a valid note."""
        note = NoteSchema(
            midi=60,
            note_name="C4",
            start_time=0.0,
            duration=0.5,
            velocity=100,
            is_rest=False
        )

        assert note.midi == 60
        assert note.note_name == "C4"
        assert note.start_time == 0.0
        assert note.duration == 0.5
        assert note.velocity == 100
        assert note.is_rest is False

    @pytest.mark.unit
    def test_note_defaults(self):
        """Test note creation with default values."""
        note = NoteSchema(
            midi=64,
            note_name="E4",
            start_time=1.0,
            duration=0.25
        )

        assert note.velocity == 100  # Default value
        assert note.is_rest is False  # Default value

    @pytest.mark.unit
    def test_note_midi_range(self):
        """Test MIDI note numbers at boundaries."""
        # Valid MIDI values
        note_low = NoteSchema(midi=0, note_name="C-1", start_time=0.0, duration=0.5)
        assert note_low.midi == 0

        note_high = NoteSchema(midi=127, note_name="G9", start_time=0.0, duration=0.5)
        assert note_high.midi == 127

    @pytest.mark.unit
    def test_note_rest(self):
        """Test creating a rest (silent note)."""
        rest = NoteSchema(
            midi=0,
            note_name="Rest",
            start_time=2.0,
            duration=1.0,
            velocity=0,
            is_rest=True
        )

        assert rest.is_rest is True
        assert rest.velocity == 0

    @pytest.mark.unit
    def test_note_json_serialization(self):
        """Test note serialization to JSON."""
        note = NoteSchema(
            midi=67,
            note_name="G4",
            start_time=0.5,
            duration=0.25,
            velocity=90,
            is_rest=False
        )

        note_dict = note.model_dump()

        assert note_dict == {
            "midi": 67,
            "note_name": "G4",
            "start_time": 0.5,
            "duration": 0.25,
            "velocity": 90,
            "is_rest": False
        }


class TestMetadataSchema:
    """Tests for MetadataSchema model."""

    @pytest.mark.unit
    def test_valid_metadata_creation(self):
        """Test creating valid metadata."""
        metadata = MetadataSchema(
            title="Test Song",
            bpm=120.0,
            key="C Major",
            time_signature="4/4"
        )

        assert metadata.title == "Test Song"
        assert metadata.bpm == 120.0
        assert metadata.key == "C Major"
        assert metadata.time_signature == "4/4"

    @pytest.mark.unit
    def test_metadata_default_time_signature(self):
        """Test metadata with default time signature."""
        metadata = MetadataSchema(
            title="Song",
            bpm=140.0,
            key="A Minor"
        )

        assert metadata.time_signature == "4/4"  # Default value

    @pytest.mark.unit
    def test_metadata_various_keys(self):
        """Test metadata with various musical keys."""
        keys = ["C Major", "D Minor", "F# Major", "Bb Minor", "G Major"]

        for key in keys:
            metadata = MetadataSchema(
                title="Test",
                bpm=120.0,
                key=key
            )
            assert metadata.key == key

    @pytest.mark.unit
    def test_metadata_various_time_signatures(self):
        """Test metadata with various time signatures."""
        time_sigs = ["3/4", "6/8", "5/4", "7/8", "4/4"]

        for time_sig in time_sigs:
            metadata = MetadataSchema(
                title="Test",
                bpm=120.0,
                key="C Major",
                time_signature=time_sig
            )
            assert metadata.time_signature == time_sig

    @pytest.mark.unit
    def test_metadata_bpm_range(self):
        """Test metadata with various BPM values."""
        # Very slow
        slow = MetadataSchema(title="Slow", bpm=40.0, key="C Major")
        assert slow.bpm == 40.0

        # Normal
        normal = MetadataSchema(title="Normal", bpm=120.0, key="C Major")
        assert normal.bpm == 120.0

        # Fast
        fast = MetadataSchema(title="Fast", bpm=200.0, key="C Major")
        assert fast.bpm == 200.0


class TestTranscriptionResponse:
    """Tests for TranscriptionResponse model."""

    @pytest.mark.unit
    def test_valid_transcription_response(self):
        """Test creating a valid transcription response."""
        metadata = MetadataSchema(
            title="Test Song",
            bpm=120.0,
            key="C Major"
        )

        notes = [
            NoteSchema(midi=60, note_name="C4", start_time=0.0, duration=0.5),
            NoteSchema(midi=64, note_name="E4", start_time=0.5, duration=0.5),
            NoteSchema(midi=67, note_name="G4", start_time=1.0, duration=0.5)
        ]

        response = TranscriptionResponse(
            job_id="test-job-123",
            metadata=metadata,
            notes=notes
        )

        assert response.job_id == "test-job-123"
        assert response.metadata.title == "Test Song"
        assert len(response.notes) == 3
        assert response.notes[0].midi == 60

    @pytest.mark.unit
    def test_transcription_response_empty_notes(self):
        """Test transcription response with no notes."""
        metadata = MetadataSchema(
            title="Silent Track",
            bpm=120.0,
            key="C Major"
        )

        response = TranscriptionResponse(
            job_id="test-job-456",
            metadata=metadata,
            notes=[]
        )

        assert response.job_id == "test-job-456"
        assert len(response.notes) == 0

    @pytest.mark.unit
    def test_transcription_response_json_serialization(self):
        """Test full transcription response serialization."""
        metadata = MetadataSchema(
            title="Test",
            bpm=120.0,
            key="C Major",
            time_signature="4/4"
        )

        notes = [
            NoteSchema(midi=60, note_name="C4", start_time=0.0, duration=0.5)
        ]

        response = TranscriptionResponse(
            job_id="job-789",
            metadata=metadata,
            notes=notes
        )

        response_dict = response.model_dump()

        assert response_dict["job_id"] == "job-789"
        assert response_dict["metadata"]["title"] == "Test"
        assert response_dict["metadata"]["bpm"] == 120.0
        assert response_dict["metadata"]["key"] == "C Major"
        assert len(response_dict["notes"]) == 1
        assert response_dict["notes"][0]["midi"] == 60

    @pytest.mark.unit
    def test_transcription_response_nested_validation(self):
        """Test that nested models are properly validated."""
        # This should work
        valid_response = TranscriptionResponse(
            job_id="test",
            metadata={
                "title": "Test",
                "bpm": 120.0,
                "key": "C Major"
            },
            notes=[
                {
                    "midi": 60,
                    "note_name": "C4",
                    "start_time": 0.0,
                    "duration": 0.5
                }
            ]
        )

        assert valid_response.job_id == "test"
        assert isinstance(valid_response.metadata, MetadataSchema)
        assert isinstance(valid_response.notes[0], NoteSchema)


class TestErrorResponse:
    """Tests for ErrorResponse model."""

    @pytest.mark.unit
    def test_error_response_with_message(self):
        """Test creating error response with message."""
        error = ErrorResponse(
            error="File not found"
        )

        assert error.error == "File not found"
        assert error.detail is None

    @pytest.mark.unit
    def test_error_response_with_detail(self):
        """Test creating error response with detail."""
        error = ErrorResponse(
            error="Processing failed",
            detail="Audio file format not supported"
        )

        assert error.error == "Processing failed"
        assert error.detail == "Audio file format not supported"

    @pytest.mark.unit
    def test_error_response_json_serialization(self):
        """Test error response serialization."""
        error = ErrorResponse(
            error="Invalid input",
            detail="File size exceeds 10MB limit"
        )

        error_dict = error.model_dump()

        assert error_dict == {
            "error": "Invalid input",
            "detail": "File size exceeds 10MB limit"
        }
