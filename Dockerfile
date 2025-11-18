# Polyphonic Melody Extraction API - Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# ffmpeg: Required for audio file decoding (librosa, soundfile)
# libsndfile1: Required for soundfile
# build-essential: Required for building some Python packages
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Install Python dependencies using uv
RUN uv sync --frozen --no-dev

# Copy application code
COPY app/ ./app/
COPY main.py ./

# Create directory for temporary audio files
RUN mkdir -p temp_audio

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models
ENV UPLOAD_FOLDER=/app/temp_audio

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application using uv
# Note: We use uv run to execute the app within the virtual environment
CMD ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
