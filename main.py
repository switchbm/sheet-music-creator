"""
FastAPI application for polyphonic melody extraction.

POST /transcribe - Upload an audio file and get back a JSON with extracted melody.
"""
import uuid
import os
from pathlib import Path
from typing import Optional
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.models.schemas import TranscriptionResponse, ErrorResponse
from app.services.engine import AudioProcessingEngine

# Initialize FastAPI app
app = FastAPI(
    title="Melody Extraction API",
    description="Polyphonic melody extraction from audio files",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the audio processing engine (singleton)
# This loads the models once at startup
engine: Optional[AudioProcessingEngine] = None

# Temporary directory for uploaded files
TEMP_DIR = Path("temp_audio")
TEMP_DIR.mkdir(exist_ok=True)

# Maximum file size (10 MB for MVP)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB in bytes


@app.on_event("startup")
async def startup_event():
    """Initialize the audio processing engine on startup."""
    global engine
    print("Initializing audio processing engine...")
    engine = AudioProcessingEngine(device=None)  # Auto-detect device
    print("Engine initialized successfully!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("Shutting down...")


def cleanup_file(file_path: str):
    """
    Background task to delete temporary files.

    Args:
        file_path: Path to the file to delete
    """
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        print(f"Error cleaning up file {file_path}: {e}")


def validate_audio_file(filename: str) -> bool:
    """
    Validate that the uploaded file is an audio file.

    Args:
        filename: Name of the uploaded file

    Returns:
        True if valid audio file, False otherwise
    """
    # Allowed audio extensions
    allowed_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'}
    file_ext = Path(filename).suffix.lower()
    return file_ext in allowed_extensions


@app.get("/")
async def root():
    """Root endpoint - health check."""
    return {
        "message": "Melody Extraction API",
        "status": "running",
        "version": "1.0.0"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "engine_initialized": engine is not None
    }


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Transcribe an audio file to extract the melody.

    Args:
        file: Audio file (MP3, WAV, FLAC, etc.)

    Returns:
        TranscriptionResponse with metadata and notes

    Raises:
        HTTPException: If file is invalid or processing fails
    """
    # Generate unique job ID
    job_id = str(uuid.uuid4())

    # Validate engine is initialized
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Audio processing engine not initialized"
        )

    # Validate file type
    if not validate_audio_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: MP3, WAV, FLAC, OGG, M4A, AAC"
        )

    # Read file content
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error reading file: {str(e)}"
        )

    # Check file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024):.1f} MB"
        )

    # Save to temporary file
    file_ext = Path(file.filename).suffix
    temp_file_path = TEMP_DIR / f"{job_id}{file_ext}"

    try:
        with open(temp_file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error saving file: {str(e)}"
        )

    # Schedule file cleanup
    background_tasks.add_task(cleanup_file, str(temp_file_path))

    # Process the audio file
    try:
        result = engine.process(
            file_path=str(temp_file_path),
            title=file.filename
        )

        # Add job_id to result
        result["job_id"] = job_id

        return result

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=f"Audio file not found: {str(e)}"
        )
    except Exception as e:
        # Log the error
        print(f"Error processing file {job_id}: {str(e)}")
        import traceback
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    print(f"Unhandled exception: {exc}")
    import traceback
    traceback.print_exc()

    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Enable auto-reload for development
    )
