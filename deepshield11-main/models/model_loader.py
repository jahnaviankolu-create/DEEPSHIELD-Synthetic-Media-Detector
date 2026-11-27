from typing import Dict, Any
from models.base_model import BaseAIModel
from models.image_model import ImageEfficientNetModel
from models.video_model import VideoOpenCVEfficientNetModel
from models.audio_model import AudioSpectrogramModel

class ModelLoader:
    """
    Model Loader Singleton Manager.
    Loads all AI deepfake detection models into memory ONCE during application startup.
    Dispatches media payloads to their respective modular model instances.
    """

    def __init__(self):
        self.image_model: BaseAIModel = None
        self.video_model: BaseAIModel = None
        self.audio_model: BaseAIModel = None
        self._is_initialized = False

    def load_all_models(self) -> None:
        """Loads all AI deepfake detection model pipelines into memory once at startup."""
        if self._is_initialized:
            print("[DeepShield ModelLoader] Models already loaded in memory.")
            return

        print("\n[DeepShield ModelLoader] Initializing AI Models Package...")
        self.image_model = ImageEfficientNetModel()
        self.video_model = VideoOpenCVEfficientNetModel()
        self.audio_model = AudioSpectrogramModel()
        self._is_initialized = True
        print("[DeepShield ModelLoader] All 3 AI Models Loaded Successfully ONCE at Startup.\n")

    def get_model_for_media(self, filename: str, content_type: str) -> BaseAIModel:
        """
        Dispatches media file payload to the corresponding modular model engine.
        - Video -> VideoOpenCVEfficientNetModel
        - Audio -> AudioSpectrogramModel
        - Image -> ImageEfficientNetModel
        """
        if not self._is_initialized:
            self.load_all_models()

        mime = (content_type or "").lower()
        ext = (filename.split('.')[-1] if '.' in filename else "").lower()

        video_exts = {"mp4", "avi", "mov", "mkv", "webm"}
        audio_exts = {"mp3", "wav", "aac", "flac", "ogg", "m4a"}

        if "video" in mime or ext in video_exts:
            return self.video_model
        elif "audio" in mime or ext in audio_exts:
            return self.audio_model
        else:
            return self.image_model

# Global Singleton Instance
model_loader = ModelLoader()
