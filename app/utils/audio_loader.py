"""
Audio loading utilities for the melody extraction system.
"""
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional, Union


class AudioLoader:
    """Utility class for loading and preprocessing audio files."""

    def __init__(self, target_sr: int = 16000):
        """
        Initialize the audio loader.

        Args:
            target_sr: Target sample rate for resampling (default: 16000 Hz)
                      16kHz is required for most deep learning models
        """
        self.target_sr = target_sr

    def load(
        self,
        file_path: Union[str, Path],
        sr: Optional[int] = None,
        mono: bool = True,
        duration: Optional[float] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Load an audio file and resample to target sample rate.

        Args:
            file_path: Path to the audio file (MP3, WAV, FLAC, etc.)
            sr: Sample rate to use. If None, uses target_sr
            mono: Convert to mono if True
            duration: Maximum duration to load in seconds

        Returns:
            Tuple of (audio_data, sample_rate)
            - audio_data: numpy array of audio samples
            - sample_rate: sample rate of the loaded audio

        Raises:
            FileNotFoundError: If the audio file doesn't exist
            librosa.exceptions.LibrosaError: If the file cannot be loaded
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        # Use target_sr if sr not specified
        if sr is None:
            sr = self.target_sr

        try:
            # Load audio file
            # librosa automatically resamples to the specified sr
            audio, sample_rate = librosa.load(
                str(path),
                sr=sr,
                mono=mono,
                duration=duration
            )

            return audio, int(sample_rate)

        except Exception as e:
            raise Exception(f"Error loading audio file {path}: {str(e)}")

    def load_stereo(
        self,
        file_path: Union[str, Path],
        sr: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Load an audio file in stereo (preserving both channels).

        Args:
            file_path: Path to the audio file
            sr: Sample rate to use. If None, uses target_sr

        Returns:
            Tuple of (audio_data, sample_rate)
            - audio_data: numpy array of shape (2, n_samples) for stereo
            - sample_rate: sample rate of the loaded audio
        """
        return self.load(file_path, sr=sr, mono=False)

    def save(
        self,
        audio: np.ndarray,
        file_path: Union[str, Path],
        sr: int
    ) -> None:
        """
        Save audio data to a file.

        Args:
            audio: Audio data as numpy array
            file_path: Path where to save the audio
            sr: Sample rate of the audio
        """
        sf.write(file_path, audio, sr)

    def get_duration(self, file_path: Union[str, Path]) -> float:
        """
        Get the duration of an audio file without loading it entirely.

        Args:
            file_path: Path to the audio file

        Returns:
            Duration in seconds
        """
        return librosa.get_duration(path=file_path)


# Convenience function for quick loading
def load_audio(file_path: Union[str, Path], sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Quick utility to load an audio file.

    Args:
        file_path: Path to the audio file
        sr: Target sample rate (default: 16000 Hz)

    Returns:
        Tuple of (audio_data, sample_rate)
    """
    loader = AudioLoader(target_sr=sr)
    return loader.load(file_path)
