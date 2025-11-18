"""
Main audio processing engine for melody extraction.
Handles source separation, pitch detection, tempo/key detection, and quantization.
"""
import numpy as np
import librosa
import torch
import torchcrepe
from pathlib import Path
from typing import Tuple, Optional, Dict

from demucs.pretrained import get_model
from demucs.apply import apply_model

from app.utils.audio_loader import AudioLoader
from app.services.quantizer import NoteQuantizer


class AudioProcessingEngine:
    """
    Main pipeline for polyphonic melody extraction.

    Pipeline:
    1. Load audio
    2. Source separation (extract vocals using Demucs)
    3. Pitch detection on vocals (using TorchCREPE)
    4. BPM detection on original mix (better transient info)
    5. Key detection on original mix (harmonic context)
    6. Note segmentation and quantization
    """

    def __init__(
        self,
        device: Optional[str] = None,
        demucs_model: str = "htdemucs",
        crepe_model: str = "full"
    ):
        """
        Initialize the audio processing engine.

        Args:
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            demucs_model: Demucs model name (htdemucs is the latest and best)
            crepe_model: CREPE model size ('full', 'large', 'medium', 'small', 'tiny')
        """
        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Initializing AudioProcessingEngine on device: {self.device}")

        # Initialize audio loader
        self.audio_loader = AudioLoader(target_sr=44100)  # Demucs uses 44.1kHz

        # Initialize quantizer
        self.quantizer = NoteQuantizer(
            hop_length=441,  # ~10ms at 44.1kHz
            min_note_duration=0.05,
            grid_resolution=16
        )

        # Load Demucs model (singleton pattern - load once)
        print(f"Loading Demucs model: {demucs_model}...")
        self.demucs_model = get_model(name=demucs_model)
        self.demucs_model.to(self.device)
        self.demucs_model.eval()

        # CREPE configuration
        self.crepe_model = crepe_model
        self.crepe_sr = 16000  # CREPE works best at 16kHz
        self.crepe_hop_length = 160  # 10ms at 16kHz

        print("AudioProcessingEngine initialized successfully!")

    def separate_vocals(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Tuple[np.ndarray, int]:
        """
        Separate vocals from the audio mix using Demucs.

        Args:
            audio: Audio waveform (mono or stereo)
            sr: Sample rate

        Returns:
            Tuple of (vocals_audio, sample_rate)
        """
        print("Separating vocals from mix...")

        # Demucs expects stereo input
        if audio.ndim == 1:
            # Convert mono to stereo
            audio = np.stack([audio, audio])
        elif audio.ndim == 2 and audio.shape[0] != 2:
            # If it's (n_samples, 2), transpose to (2, n_samples)
            audio = audio.T

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float().to(self.device)

        # Add batch dimension: (batch, channels, samples)
        if audio_tensor.ndim == 2:
            audio_tensor = audio_tensor.unsqueeze(0)

        # Apply Demucs
        with torch.no_grad():
            sources = apply_model(
                self.demucs_model,
                audio_tensor,
                device=self.device,
                split=True,
                overlap=0.25
            )

        # Demucs outputs: [batch, sources, channels, samples]
        # Sources order for htdemucs: drums, bass, other, vocals
        vocals_idx = 3  # vocals is the 4th source (index 3)

        vocals = sources[0, vocals_idx].cpu().numpy()

        # Convert stereo to mono by averaging channels
        if vocals.shape[0] == 2:
            vocals = vocals.mean(axis=0)

        print(f"Vocals separated successfully. Shape: {vocals.shape}")
        return vocals, sr

    def detect_pitch(
        self,
        vocals: np.ndarray,
        sr: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect pitch using TorchCREPE.

        Args:
            vocals: Vocal audio waveform (mono)
            sr: Sample rate

        Returns:
            Tuple of (pitch_curve, confidence, times)
            - pitch_curve: Frequency in Hz at each time step
            - confidence: Confidence score (0-1) at each time step
            - times: Time in seconds for each estimate
        """
        print("Detecting pitch using TorchCREPE...")

        # Resample to CREPE's expected sample rate if needed
        if sr != self.crepe_sr:
            vocals_resampled = librosa.resample(
                vocals,
                orig_sr=sr,
                target_sr=self.crepe_sr
            )
        else:
            vocals_resampled = vocals

        # Convert to torch tensor
        audio_tensor = torch.from_numpy(vocals_resampled).float().to(self.device)

        # Run CREPE
        # Returns: time, frequency, confidence, activation
        with torch.no_grad():
            time, frequency, confidence, _ = torchcrepe.predict(
                audio_tensor,
                sample_rate=self.crepe_sr,
                hop_length=self.crepe_hop_length,
                fmin=50,  # Minimum frequency (low male voice)
                fmax=2000,  # Maximum frequency (high female voice)
                model=self.crepe_model,
                device=self.device,
                return_periodicity=True
            )

        # Convert to numpy
        pitch_curve = frequency.cpu().numpy().squeeze()
        confidence = confidence.cpu().numpy().squeeze()
        times = time.cpu().numpy().squeeze()

        print(f"Pitch detection complete. Found {len(pitch_curve)} estimates.")
        return pitch_curve, confidence, times

    def detect_tempo(
        self,
        audio: np.ndarray,
        sr: int
    ) -> float:
        """
        Detect tempo (BPM) from the audio.

        Args:
            audio: Audio waveform
            sr: Sample rate

        Returns:
            BPM (beats per minute)
        """
        print("Detecting tempo...")

        # Use librosa's beat tracker
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)

        # tempo is returned as a numpy array with one element
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo = float(tempo)

        print(f"Detected tempo: {tempo:.1f} BPM")
        return tempo

    def detect_key(
        self,
        audio: np.ndarray,
        sr: int
    ) -> str:
        """
        Detect the musical key using the Krumhansl-Schmuckler algorithm.

        Args:
            audio: Audio waveform
            sr: Sample rate

        Returns:
            Key signature string (e.g., 'C Major', 'A Minor')
        """
        print("Detecting key signature...")

        # Compute chroma features
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)

        # Average over time to get overall pitch class distribution
        chroma_mean = chroma.mean(axis=1)

        # Krumhansl-Schmuckler key profiles
        # Major key profile
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                  2.52, 5.19, 2.39, 3.66, 2.29, 2.88])

        # Minor key profile
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                  2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

        # Note names
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                      'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Calculate correlation for each key
        max_corr = -1
        detected_key = 'C Major'

        for i in range(12):
            # Rotate profiles to test each key
            major_rotated = np.roll(major_profile, i)
            minor_rotated = np.roll(minor_profile, i)

            # Calculate correlation
            major_corr = np.corrcoef(chroma_mean, major_rotated)[0, 1]
            minor_corr = np.corrcoef(chroma_mean, minor_rotated)[0, 1]

            # Check if this is the best match
            if major_corr > max_corr:
                max_corr = major_corr
                detected_key = f"{note_names[i]} Major"

            if minor_corr > max_corr:
                max_corr = minor_corr
                detected_key = f"{note_names[i]} Minor"

        print(f"Detected key: {detected_key}")
        return detected_key

    def process(
        self,
        file_path: str,
        title: Optional[str] = None
    ) -> Dict:
        """
        Complete processing pipeline.

        Args:
            file_path: Path to the audio file
            title: Song title (uses filename if not provided)

        Returns:
            Dictionary with complete transcription data
        """
        print(f"\n{'='*60}")
        print(f"Processing: {file_path}")
        print(f"{'='*60}\n")

        # Get title from filename if not provided
        if title is None:
            title = Path(file_path).name

        # Step 1: Load audio
        print("Step 1: Loading audio...")
        audio, sr = self.audio_loader.load(file_path, sr=44100, mono=False)

        # Convert to mono for some operations
        if audio.ndim == 2:
            audio_mono = audio.mean(axis=0)
        else:
            audio_mono = audio

        # Step 2: Separate vocals
        print("\nStep 2: Source separation...")
        vocals, sr = self.separate_vocals(audio, sr)

        # Step 3: Pitch detection on vocals
        print("\nStep 3: Pitch detection...")
        pitch_curve, confidence, times = self.detect_pitch(vocals, sr)

        # Step 4: Detect tempo on original mix
        print("\nStep 4: Tempo detection...")
        bpm = self.detect_tempo(audio_mono, sr)

        # Step 5: Detect key on original mix
        print("\nStep 5: Key detection...")
        key = self.detect_key(audio_mono, sr)

        # Step 6: Quantize notes
        print("\nStep 6: Note quantization...")
        notes = self.quantizer.process(pitch_curve, confidence, times, bpm)

        print(f"\nExtracted {len(notes)} notes")
        print(f"{'='*60}\n")

        # Prepare result
        result = {
            "metadata": {
                "title": title,
                "bpm": float(bpm),
                "key": key,
                "time_signature": "4/4"
            },
            "notes": notes
        }

        return result
