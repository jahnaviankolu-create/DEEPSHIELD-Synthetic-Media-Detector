import hashlib
from typing import Tuple

def compute_sha256(content: bytes) -> str:
    """Computes SHA-256 cryptographic hash of byte buffer."""
    return hashlib.sha256(content).hexdigest()

def get_media_type(filename: str, content_type: str) -> str:
    """
    Classifies uploaded file into 'video', 'audio', or 'image'
    based on MIME type and file extension.
    """
    mime = (content_type or "").lower()
    ext = (filename.split('.')[-1] if '.' in filename else "").lower()

    video_exts = {"mp4", "avi", "mov", "mkv", "webm"}
    audio_exts = {"mp3", "wav", "aac", "flac", "ogg", "m4a"}

    if "video" in mime or ext in video_exts:
        return "video"
    elif "audio" in mime or ext in audio_exts:
        return "audio"
    else:
        return "image"
