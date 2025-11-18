"""
Note quantization and segmentation module.
Converts continuous pitch curves (Hz over time) into discrete musical events.
"""
import numpy as np
from typing import List, Tuple
from scipy.ndimage import median_filter


class NoteQuantizer:
    """
    Converts continuous pitch detection output into quantized musical notes.
    """

    def __init__(
        self,
        hop_length: int = 160,  # 10ms at 16kHz
        min_note_duration: float = 0.05,  # 50ms minimum note duration
        grid_resolution: int = 16  # Quantize to 1/16th notes
    ):
        """
        Initialize the note quantizer.

        Args:
            hop_length: Number of samples between pitch estimates
            min_note_duration: Minimum duration for a note in seconds (vibrato suppression)
            grid_resolution: Quantization grid resolution (16 = 1/16th notes)
        """
        self.hop_length = hop_length
        self.min_note_duration = min_note_duration
        self.grid_resolution = grid_resolution

    def hz_to_midi(self, frequency_hz: float) -> int:
        """
        Convert frequency in Hz to MIDI note number.

        Formula: m = 69 + 12 * log2(f / 440)

        Args:
            frequency_hz: Frequency in Hz

        Returns:
            MIDI note number (0-127)
        """
        if frequency_hz <= 0:
            return 0

        midi_float = 69 + 12 * np.log2(frequency_hz / 440.0)
        midi_note = int(np.round(midi_float))

        # Clamp to valid MIDI range
        return max(0, min(127, midi_note))

    def midi_to_note_name(self, midi: int) -> str:
        """
        Convert MIDI note number to note name with octave.

        Args:
            midi: MIDI note number (0-127)

        Returns:
            Note name (e.g., 'C4', 'F#5')
        """
        if midi < 0 or midi > 127:
            return "N/A"

        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi // 12) - 1
        note = note_names[midi % 12]

        return f"{note}{octave}"

    def smooth_pitch_curve(
        self,
        pitch_curve: np.ndarray,
        confidence: np.ndarray,
        confidence_threshold: float = 0.5,
        median_filter_size: int = 5
    ) -> np.ndarray:
        """
        Smooth the pitch curve and remove low-confidence regions.

        Args:
            pitch_curve: Array of frequency values in Hz
            confidence: Confidence scores for each pitch estimate
            confidence_threshold: Minimum confidence to keep a pitch estimate
            median_filter_size: Size of median filter for smoothing

        Returns:
            Smoothed pitch curve
        """
        # Set low-confidence pitches to 0 (silence/unvoiced)
        smoothed = pitch_curve.copy()
        smoothed[confidence < confidence_threshold] = 0

        # Apply median filter to remove jitter
        # Only filter where we have voiced segments
        voiced_mask = smoothed > 0
        if np.any(voiced_mask):
            # Median filter helps remove octave errors and vibrato
            smoothed = median_filter(smoothed, size=median_filter_size)
            # Re-apply the voiced mask to keep silence regions as 0
            smoothed[~voiced_mask] = 0

        return smoothed

    def segment_notes(
        self,
        pitch_curve: np.ndarray,
        times: np.ndarray,
        min_duration: float = None
    ) -> List[Tuple[float, float, int]]:
        """
        Segment continuous pitch curve into discrete note events.

        Args:
            pitch_curve: Smoothed pitch curve in Hz
            times: Time points corresponding to each pitch estimate
            min_duration: Minimum note duration (uses self.min_note_duration if None)

        Returns:
            List of (start_time, duration, midi_note) tuples
        """
        if min_duration is None:
            min_duration = self.min_note_duration

        # Convert to MIDI
        midi_curve = np.array([self.hz_to_midi(f) for f in pitch_curve])

        notes = []
        current_midi = None
        start_time = None

        for i, (midi, time) in enumerate(zip(midi_curve, times)):
            # Skip if it's silence (midi == 0)
            if midi == 0:
                # End current note if there was one
                if current_midi is not None and start_time is not None:
                    duration = time - start_time
                    if duration >= min_duration:
                        notes.append((start_time, duration, current_midi))
                current_midi = None
                start_time = None
                continue

            # Start new note or continue current
            if current_midi is None:
                # Start new note
                current_midi = midi
                start_time = time
            elif midi != current_midi:
                # Pitch changed - end current note and start new one
                duration = time - start_time
                if duration >= min_duration:
                    notes.append((start_time, duration, current_midi))

                current_midi = midi
                start_time = time

        # Don't forget the last note
        if current_midi is not None and start_time is not None:
            duration = times[-1] - start_time
            if duration >= min_duration:
                notes.append((start_time, duration, current_midi))

        return notes

    def quantize_time(self, time: float, bpm: float, resolution: int = None) -> float:
        """
        Quantize time to the nearest grid position.

        Args:
            time: Time in seconds
            bpm: Beats per minute
            resolution: Grid resolution (uses self.grid_resolution if None)

        Returns:
            Quantized time in seconds
        """
        if resolution is None:
            resolution = self.grid_resolution

        # Calculate the duration of one grid unit
        seconds_per_beat = 60.0 / bpm
        seconds_per_grid = seconds_per_beat / (resolution / 4.0)

        # Snap to nearest grid position
        grid_position = np.round(time / seconds_per_grid)
        quantized_time = grid_position * seconds_per_grid

        return quantized_time

    def quantize_notes(
        self,
        notes: List[Tuple[float, float, int]],
        bpm: float
    ) -> List[dict]:
        """
        Quantize note timings to musical grid and format as dictionaries.

        Args:
            notes: List of (start_time, duration, midi_note) tuples
            bpm: Beats per minute for grid quantization

        Returns:
            List of note dictionaries with all required fields
        """
        quantized_notes = []

        for start_time, duration, midi_note in notes:
            # Quantize start time and duration
            q_start = self.quantize_time(start_time, bpm)
            q_end = self.quantize_time(start_time + duration, bpm)
            q_duration = max(q_end - q_start, 60.0 / bpm / self.grid_resolution)

            # Create note dictionary
            note_dict = {
                "midi": int(midi_note),
                "note_name": self.midi_to_note_name(int(midi_note)),
                "start_time": float(q_start),
                "duration": float(q_duration),
                "velocity": 100,  # Default velocity
                "is_rest": False
            }

            quantized_notes.append(note_dict)

        return quantized_notes

    def process(
        self,
        pitch_curve: np.ndarray,
        confidence: np.ndarray,
        times: np.ndarray,
        bpm: float
    ) -> List[dict]:
        """
        Complete pipeline: smooth, segment, and quantize pitch curve into notes.

        Args:
            pitch_curve: Array of frequency values in Hz
            confidence: Confidence scores for each pitch estimate
            times: Time points corresponding to each pitch estimate
            bpm: Beats per minute for quantization

        Returns:
            List of quantized note dictionaries
        """
        # Step 1: Smooth the pitch curve
        smoothed = self.smooth_pitch_curve(pitch_curve, confidence)

        # Step 2: Segment into notes
        notes = self.segment_notes(smoothed, times)

        # Step 3: Quantize to musical grid
        quantized = self.quantize_notes(notes, bpm)

        return quantized
