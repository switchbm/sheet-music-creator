"""
Integration tests for the FastAPI application.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import io

from main import app


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app)


@pytest.fixture
def mock_engine():
    """Create a mock audio processing engine."""
    mock = MagicMock()
    mock.process.return_value = {
        "metadata": {
            "title": "test.mp3",
            "bpm": 120.0,
            "key": "C Major",
            "time_signature": "4/4"
        },
        "notes": [
            {
                "midi": 60,
                "note_name": "C4",
                "start_time": 0.0,
                "duration": 0.5,
                "velocity": 100,
                "is_rest": False
            }
        ]
    }
    return mock


class TestRootEndpoint:
    """Tests for the root endpoint."""

    @pytest.mark.integration
    def test_root_endpoint(self, client: TestClient):
        """Test GET / returns correct response."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Melody Extraction API"
        assert data["status"] == "running"
        assert data["version"] == "1.0.0"


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    @pytest.mark.integration
    def test_health_endpoint(self, client: TestClient):
        """Test GET /health returns health status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "engine_initialized" in data


class TestTranscribeEndpoint:
    """Tests for the /transcribe endpoint."""

    @pytest.mark.integration
    @patch('main.engine')
    def test_transcribe_valid_audio(
        self,
        mock_engine_module,
        client: TestClient,
        mock_engine,
        mock_audio_bytes: bytes
    ):
        """Test POST /transcribe with valid audio file."""
        mock_engine_module.process = mock_engine.process

        # Make the request
        files = {"file": ("test.mp3", io.BytesIO(mock_audio_bytes), "audio/mpeg")}
        response = client.post("/transcribe", files=files)

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert "job_id" in data
        assert "metadata" in data
        assert "notes" in data

        # Verify metadata
        assert data["metadata"]["title"] == "test.mp3"
        assert data["metadata"]["bpm"] == 120.0
        assert data["metadata"]["key"] == "C Major"

        # Verify notes
        assert len(data["notes"]) >= 1

    @pytest.mark.integration
    def test_transcribe_missing_file(self, client: TestClient):
        """Test POST /transcribe without file returns error."""
        response = client.post("/transcribe")

        assert response.status_code == 422  # Validation error

    @pytest.mark.integration
    def test_transcribe_invalid_file_type(self, client: TestClient):
        """Test POST /transcribe with invalid file type."""
        # Create a fake text file
        files = {"file": ("test.txt", io.BytesIO(b"not an audio file"), "text/plain")}
        response = client.post("/transcribe", files=files)

        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]

    @pytest.mark.integration
    @patch('main.engine')
    def test_transcribe_file_too_large(
        self,
        mock_engine_module,
        client: TestClient,
        mock_engine
    ):
        """Test POST /transcribe with file exceeding size limit."""
        mock_engine_module.process = mock_engine.process

        # Create a file that's too large (> 10MB)
        large_file = b"0" * (11 * 1024 * 1024)  # 11MB
        files = {"file": ("large.mp3", io.BytesIO(large_file), "audio/mpeg")}

        response = client.post("/transcribe", files=files)

        assert response.status_code == 413
        assert "File too large" in response.json()["detail"]

    @pytest.mark.integration
    def test_transcribe_no_filename(self, client: TestClient):
        """Test POST /transcribe with no filename."""
        files = {"file": ("", io.BytesIO(b"fake audio"), "audio/mpeg")}
        response = client.post("/transcribe", files=files)

        assert response.status_code == 400
        assert "No filename provided" in response.json()["detail"]

    @pytest.mark.integration
    @patch('main.engine', None)  # Simulate engine not initialized
    def test_transcribe_engine_not_initialized(
        self,
        client: TestClient,
        mock_audio_bytes: bytes
    ):
        """Test POST /transcribe when engine is not initialized."""
        files = {"file": ("test.mp3", io.BytesIO(mock_audio_bytes), "audio/mpeg")}
        response = client.post("/transcribe", files=files)

        assert response.status_code == 503
        assert "Audio processing engine not initialized" in response.json()["detail"]

    @pytest.mark.integration
    @patch('main.engine')
    def test_transcribe_processing_error(
        self,
        mock_engine_module,
        client: TestClient,
        mock_audio_bytes: bytes
    ):
        """Test POST /transcribe when processing fails."""
        # Mock engine to raise an exception
        mock_engine_module.process.side_effect = Exception("Processing failed")

        files = {"file": ("test.mp3", io.BytesIO(mock_audio_bytes), "audio/mpeg")}
        response = client.post("/transcribe", files=files)

        assert response.status_code == 500
        assert "Error processing audio" in response.json()["detail"]


class TestFileValidation:
    """Tests for file validation logic."""

    @pytest.mark.integration
    @patch('main.engine')
    def test_accept_mp3(
        self,
        mock_engine_module,
        client: TestClient,
        mock_engine,
        mock_audio_bytes: bytes
    ):
        """Test that MP3 files are accepted."""
        mock_engine_module.process = mock_engine.process

        files = {"file": ("song.mp3", io.BytesIO(mock_audio_bytes), "audio/mpeg")}
        response = client.post("/transcribe", files=files)

        assert response.status_code == 200

    @pytest.mark.integration
    @patch('main.engine')
    def test_accept_wav(
        self,
        mock_engine_module,
        client: TestClient,
        mock_engine,
        mock_audio_bytes: bytes
    ):
        """Test that WAV files are accepted."""
        mock_engine_module.process = mock_engine.process

        files = {"file": ("song.wav", io.BytesIO(mock_audio_bytes), "audio/wav")}
        response = client.post("/transcribe", files=files)

        assert response.status_code == 200

    @pytest.mark.integration
    @patch('main.engine')
    def test_accept_flac(
        self,
        mock_engine_module,
        client: TestClient,
        mock_engine,
        mock_audio_bytes: bytes
    ):
        """Test that FLAC files are accepted."""
        mock_engine_module.process = mock_engine.process

        files = {"file": ("song.flac", io.BytesIO(mock_audio_bytes), "audio/flac")}
        response = client.post("/transcribe", files=files)

        assert response.status_code == 200


class TestBackgroundTasks:
    """Tests for background file cleanup."""

    @pytest.mark.integration
    @patch('main.engine')
    @patch('main.os.path.exists')
    @patch('main.os.remove')
    def test_file_cleanup_scheduled(
        self,
        mock_remove,
        mock_exists,
        mock_engine_module,
        client: TestClient,
        mock_engine,
        mock_audio_bytes: bytes
    ):
        """Test that file cleanup is scheduled as background task."""
        mock_engine_module.process = mock_engine.process
        mock_exists.return_value = True

        files = {"file": ("test.mp3", io.BytesIO(mock_audio_bytes), "audio/mpeg")}
        response = client.post("/transcribe", files=files)

        assert response.status_code == 200

        # Background task should have been scheduled
        # (Note: In a real test, we'd need to wait for the background task to complete)


class TestCORS:
    """Tests for CORS middleware."""

    @pytest.mark.integration
    def test_cors_headers_present(self, client: TestClient):
        """Test that CORS headers are present."""
        response = client.options("/")

        # CORS headers should allow all origins in development
        assert "access-control-allow-origin" in response.headers


class TestResponseFormats:
    """Tests for response format validation."""

    @pytest.mark.integration
    @patch('main.engine')
    def test_response_matches_schema(
        self,
        mock_engine_module,
        client: TestClient,
        mock_engine,
        mock_audio_bytes: bytes
    ):
        """Test that response matches the TranscriptionResponse schema."""
        mock_engine_module.process = mock_engine.process

        files = {"file": ("test.mp3", io.BytesIO(mock_audio_bytes), "audio/mpeg")}
        response = client.post("/transcribe", files=files)

        assert response.status_code == 200
        data = response.json()

        # Verify all required fields are present
        assert "job_id" in data
        assert isinstance(data["job_id"], str)

        assert "metadata" in data
        metadata = data["metadata"]
        assert "title" in metadata
        assert "bpm" in metadata
        assert "key" in metadata
        assert "time_signature" in metadata

        assert "notes" in data
        assert isinstance(data["notes"], list)

        if len(data["notes"]) > 0:
            note = data["notes"][0]
            assert "midi" in note
            assert "note_name" in note
            assert "start_time" in note
            assert "duration" in note
            assert "velocity" in note
            assert "is_rest" in note
