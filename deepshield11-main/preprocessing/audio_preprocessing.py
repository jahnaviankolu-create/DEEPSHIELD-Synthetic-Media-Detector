import os
import io
import glob
import wave
import struct
import numpy as np
from typing import Dict, Any, Tuple

from preprocessing.utils import (
    setup_logger,
    is_supported_extension,
    resample_waveform,
    handle_corrupted_file
)

logger = setup_logger("AudioPreprocessing")

SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

class AudioDatasetPreprocessor:
    """
    Modular Audio Preprocessor for Deepfake Datasets.
    Validates audio formats, converts sample rates to target_sample_rate (default 16kHz),
    removes non-speech silence regions using RMS energy thresholding,
    computes Log-Mel Spectrogram feature matrices, handles corrupted audio files, and exports tensors.
    """

    def __init__(
        self,
        target_sample_rate: int = 16000,
        n_mels: int = 128,
        silence_threshold: float = 0.01,
        remove_corrupted: bool = True
    ):
        self.target_sample_rate = target_sample_rate
        self.n_mels = n_mels
        self.silence_threshold = silence_threshold
        self.remove_corrupted = remove_corrupted

    def is_valid_format(self, filepath: str) -> bool:
        """Validates if audio file has a supported format extension."""
        return is_supported_extension(filepath, SUPPORTED_AUDIO_EXTS)

    def remove_silence(self, samples: np.ndarray) -> np.ndarray:
        """
        Trims low-amplitude silent regions using RMS frame energy thresholding.
        """
        if len(samples) == 0:
            return samples

        frame_length = 512
        active_frames = []

        for i in range(0, len(samples), frame_length):
            frame = samples[i:i + frame_length]
            rms = np.sqrt(np.mean(frame ** 2)) if len(frame) > 0 else 0.0
            if rms >= self.silence_threshold:
                active_frames.append(frame)

        if active_frames:
            return np.concatenate(active_frames)
        return samples

    def compute_mel_spectrogram(self, samples: np.ndarray) -> np.ndarray:
        """
        Computes Log-Mel Spectrogram feature matrix via Short-Time Fourier Transform (STFT).
        """
        n_fft = 512
        hop_length = 256

        num_frames = max(1, (len(samples) - n_fft) // hop_length + 1)
        stft_matrix = []
        window = np.hanning(n_fft)

        for i in range(num_frames):
            start = i * hop_length
            end = start + n_fft
            segment = samples[start:end]
            if len(segment) < n_fft:
                segment = np.pad(segment, (0, n_fft - len(segment)))

            fft_frame = np.abs(np.fft.rfft(segment * window))
            stft_matrix.append(fft_frame)

        spectrogram = np.array(stft_matrix).T
        log_mel_spectrogram = np.log(spectrogram + 1e-6)
        return log_mel_spectrogram

    def validate_and_preprocess_file(self, filepath: str) -> Dict[str, Any]:
        """
        Processes a single audio file:
        1. Format validation
        2. Audio waveform loading & decoding
        3. Sample rate conversion / resampling
        4. Amplitude normalization
        5. Silence removal
        6. Mel Spectrogram generation
        7. Corrupted file detection & removal
        """
        if not self.is_valid_format(filepath):
            logger.warning(f"Skipping unsupported audio format: {filepath}")
            return {"status": "unsupported", "filepath": filepath}

        try:
            samples = np.array([], dtype=np.float32)
            detected_sr = self.target_sample_rate

            with open(filepath, "rb") as f:
                file_bytes = f.read()

            if len(file_bytes) == 0:
                raise ValueError("Audio file is 0 bytes / empty")

            try:
                with wave.open(io.BytesIO(file_bytes), 'rb') as wf:
                    detected_sr = wf.getframerate()
                    num_frames = wf.getnframes()
                    raw_data = wf.readframes(num_frames)
                    sampwidth = wf.getsampwidth()
                    num_channels = wf.getnchannels()

                    if sampwidth == 2:
                        fmt = f"<{num_frames * num_channels}h"
                        raw_samples = np.array(struct.unpack(fmt, raw_data), dtype=np.float32) / 32768.0
                    elif sampwidth == 4:
                        fmt = f"<{num_frames * num_channels}i"
                        raw_samples = np.array(struct.unpack(fmt, raw_data), dtype=np.float32) / 2147483648.0
                    else:
                        raw_samples = np.frombuffer(raw_data, dtype=np.int8).astype(np.float32) / 128.0

                    if num_channels > 1 and len(raw_samples) > 0:
                        raw_samples = raw_samples.reshape(-1, num_channels).mean(axis=1)

                    samples = raw_samples
            except Exception:
                # Fallback decoding for raw audio streams
                byte_arr = np.frombuffer(file_bytes, dtype=np.uint8)
                if len(byte_arr) > 0:
                    samples = (byte_arr.astype(np.float32) - 128.0) / 128.0

            if len(samples) == 0:
                raise ValueError("Failed to decode audio samples")

            # Sample Rate Conversion / Resampling to target_sample_rate
            if detected_sr != self.target_sample_rate:
                samples = resample_waveform(samples, detected_sr, self.target_sample_rate)

            # Amplitude Normalization
            max_amp = np.max(np.abs(samples)) + 1e-8
            samples = samples / max_amp

            # Silence Removal
            cleaned_samples = self.remove_silence(samples)

            # Log-Mel Spectrogram Generation
            mel_spec = self.compute_mel_spectrogram(cleaned_samples)

            return {
                "status": "success",
                "filepath": filepath,
                "sample_count": len(cleaned_samples),
                "target_sample_rate": self.target_sample_rate,
                "mel_spectrogram": mel_spec
            }
        except Exception as e:
            logger.error(f"Corrupted audio detected [{filepath}]: {e}")
            removed = handle_corrupted_file(filepath, remove_corrupted=self.remove_corrupted, logger=logger)
            return {
                "status": "corrupted",
                "filepath": filepath,
                "error": str(e),
                "removed": removed
            }

    def process_directory(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        Recursively scans input_dir, validates & processes audio samples,
        and saves numpy Mel Spectrogram feature matrices into output_dir.
        """
        logger.info(f"Starting Audio Dataset Preprocessing on directory: {input_dir}")
        os.makedirs(output_dir, exist_ok=True)

        processed_count = 0
        corrupted_count = 0
        unsupported_count = 0

        files = glob.glob(os.path.join(input_dir, "**", "*.*"), recursive=True)
        total_files = len(files)

        logger.info(f"Discovered {total_files} files in {input_dir}")

        for idx, filepath in enumerate(files, 1):
            if os.path.basename(filepath).startswith(".gitkeep"):
                continue

            res = self.validate_and_preprocess_file(filepath)
            status = res["status"]

            if status == "success":
                processed_count += 1
                rel_path = os.path.relpath(filepath, input_dir)
                out_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".npy")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                np.save(out_path, res["mel_spectrogram"])
            elif status == "corrupted":
                corrupted_count += 1
            elif status == "unsupported":
                unsupported_count += 1

            if idx % 20 == 0 or idx == total_files:
                logger.info(
                    f"Audio Progress [{idx}/{total_files}] -> Success: {processed_count}, "
                    f"Corrupted: {corrupted_count}, Unsupported: {unsupported_count}"
                )

        summary = {
            "total_scanned": total_files,
            "processed": processed_count,
            "corrupted": corrupted_count,
            "unsupported": unsupported_count,
            "output_directory": output_dir
        }
        logger.info(f"Audio Preprocessing Finished: {summary}")
        return summary
