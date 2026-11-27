import os
import logging
import numpy as np
from typing import Set, Optional, Tuple

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def setup_logger(name: str) -> logging.Logger:
    """Configures a standardized logger with timestamp formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def is_supported_extension(filepath: str, allowed_extensions: Set[str]) -> bool:
    """Validates if file extension matches supported extensions set."""
    ext = os.path.splitext(filepath)[1].lower()
    return ext in allowed_extensions

def normalize_imagenet(array: np.ndarray) -> np.ndarray:
    """
    Normalizes a float32 RGB image/frame array [0, 1] using standard ImageNet mean and std.
    Input shape: (H, W, 3) or (N, H, W, 3)
    """
    return (array - IMAGENET_MEAN) / IMAGENET_STD

def resample_waveform(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resamples 1D audio sample array from orig_sr to target_sr using linear interpolation.
    """
    if orig_sr == target_sr or len(samples) == 0:
        return samples

    duration = len(samples) / float(orig_sr)
    target_num_samples = int(duration * target_sr)
    if target_num_samples <= 0:
        return np.array([], dtype=np.float32)

    orig_indices = np.linspace(0, len(samples) - 1, num=len(samples))
    target_indices = np.linspace(0, len(samples) - 1, num=target_num_samples)
    resampled = np.interp(target_indices, orig_indices, samples)
    return resampled.astype(np.float32)

def handle_corrupted_file(filepath: str, remove_corrupted: bool = True, logger: Optional[logging.Logger] = None) -> bool:
    """
    Logs and optionally removes a corrupted dataset file from disk.
    Returns True if file was successfully removed or isolated.
    """
    msg = f"Corrupted file detected: {filepath}"
    if logger:
        logger.warning(msg)
    else:
        print(f"[Warning] {msg}")

    if remove_corrupted and os.path.exists(filepath):
        try:
            os.remove(filepath)
            if logger:
                logger.info(f"Successfully removed corrupted file: {filepath}")
            return True
        except Exception as err:
            if logger:
                logger.error(f"Failed to remove corrupted file {filepath}: {err}")
            return False
    return False
