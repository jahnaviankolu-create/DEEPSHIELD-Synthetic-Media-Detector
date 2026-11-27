# Purpose: Preprocesses audio input data before passing it to Hugging Face models for analysis.

import os
import sys
import wave
import numpy as np

# Add project root directory to python path for package resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from backend.ai_models.audio.model_loader import load_audio_model

def _load_and_resample_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    Helper function to load an audio file, convert it to mono, and resample to 16,000 Hz.
    Uses librosa/torchaudio if available, with standard library wave + numpy as a robust fallback.
    """
    # 1. Try loading with librosa if available
    try:
        import librosa
        speech, _ = librosa.load(audio_path, sr=target_sr, mono=True)
        return speech.astype(np.float32)
    except ImportError:
        pass

    # 2. Try loading with torchaudio if available
    try:
        import torchaudio
        import torch
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
            waveform = resampler(waveform)
        return waveform.squeeze().numpy().astype(np.float32)
    except ImportError:
        pass

    # 3. Native fallback using Python standard library wave + numpy for WAV files
    with wave.open(audio_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()

        raw_bytes = wf.readframes(n_frames)

        if sampwidth == 2:
            dtype = np.int16
        elif sampwidth == 4:
            dtype = np.int32
        else:
            dtype = np.uint8

        audio_arr = np.frombuffer(raw_bytes, dtype=dtype).astype(np.float32)

        # Normalize audio signal to range [-1.0, 1.0]
        if sampwidth == 2:
            audio_arr /= 32768.0
        elif sampwidth == 4:
            audio_arr /= 2147483648.0

        # Convert stereo to mono by averaging channels
        if n_channels > 1:
            audio_arr = audio_arr.reshape(-1, n_channels).mean(axis=1)

        # Resample audio to 16,000 Hz if necessary using linear interpolation
        if framerate != target_sr:
            num_target_samples = int(len(audio_arr) * target_sr / framerate)
            audio_arr = np.interp(
                np.linspace(0, len(audio_arr), num_target_samples, endpoint=False),
                np.arange(len(audio_arr)),
                audio_arr
            ).astype(np.float32)

        return audio_arr

def preprocess_audio(audio_path: str, feature_extractor=None):
    """
    Preprocesses an input audio file for Wav2Vec2 model inference.

    Steps:
    1. Loads the audio file from disk.
    2. Converts multi-channel (stereo) audio to mono.
    3. Resamples audio signal to 16,000 Hz.
    4. Uses the Hugging Face Wav2Vec2 Feature Extractor from model_loader.py.
    5. Returns PyTorch tensors ready for model inference.

    Args:
        audio_path (str): File path to the audio file to be analyzed.
        feature_extractor (optional): Hugging Face Wav2Vec2 feature extractor instance. Loaded if None.

    Returns:
        dict: Preprocessed tensor inputs dictionary (containing 'input_values') ready for model prediction.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found at path: '{audio_path}'")

    # Step 1-3: Load, convert to mono, and resample to 16,000 Hz
    speech_array = _load_and_resample_audio(audio_path, target_sr=16000)

    # Step 4: Load Wav2Vec2 feature extractor if not provided
    if feature_extractor is None:
        feature_extractor, _ = load_audio_model()

    # Step 5: Extract 1D raw waveform features into PyTorch tensors
    inputs = feature_extractor(
        speech_array,
        sampling_rate=16000,
        return_tensors="pt"
    )

    return inputs
