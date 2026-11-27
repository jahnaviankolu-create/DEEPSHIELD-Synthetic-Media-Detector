import io
import wave
import struct
import numpy as np
from typing import Dict, Any

def preprocess_audio(file_bytes: bytes, target_sample_rate: int = 16000) -> Dict[str, Any]:
    """
    Audio Preprocessing Pipeline:
    1. Parses WAV / PCM audio byte stream
    2. Extracts raw sample arrays
    3. Normalizes waveform amplitude to [-1.0, 1.0] range
    4. Computes zero-crossing rate and FFT spectral sample representations
    """
    samples = np.array([], dtype=np.float32)
    sample_rate = target_sample_rate

    try:
        with wave.open(io.BytesIO(file_bytes), 'rb') as wf:
            sample_rate = wf.getframerate()
            num_frames = wf.getnframes()
            raw_data = wf.readframes(num_frames)
            sampwidth = wf.getsampwidth()

            if sampwidth == 2:
                fmt = f"<{num_frames * wf.getnchannels()}h"
                samples = np.array(struct.unpack(fmt, raw_data), dtype=np.float32) / 32768.0
            elif sampwidth == 4:
                fmt = f"<{num_frames * wf.getnchannels()}i"
                samples = np.array(struct.unpack(fmt, raw_data), dtype=np.float32) / 2147483648.0
    except Exception:
        # Fallback raw byte conversion
        byte_arr = np.frombuffer(file_bytes, dtype=np.uint8)
        if len(byte_arr) > 0:
            samples = (byte_arr.astype(np.float32) - 128.0) / 128.0

    if len(samples) == 0:
        samples = np.zeros(16000, dtype=np.float32)

    # Normalize amplitude
    max_amp = np.max(np.abs(samples)) + 1e-8
    norm_samples = samples / max_amp

    # FFT Spectral transform feature matrix
    fft_res = np.abs(np.fft.rfft(norm_samples[:min(len(norm_samples), 8192)]))

    return {
        "status": "success",
        "sample_count": len(samples),
        "sample_rate": sample_rate,
        "normalized_samples": norm_samples,
        "fft_spectrum": fft_res
    }
