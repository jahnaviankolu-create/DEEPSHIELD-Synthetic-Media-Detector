from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple

class BaseAIModel(ABC):
    """
    Abstract Base Class for all DeepShield AI Deepfake Detection Models.
    Ensures modular design so models for image, video, and audio can be updated independently.
    """

    def __init__(self, model_name: str, version: str = "1.0.0"):
        self.model_name = model_name
        self.version = version
        self.is_loaded = False

    @abstractmethod
    def load_weights(self) -> None:
        """
        Loads model weights, network architecture, or pipeline parameters into memory.
        Must be executed only ONCE during application startup.
        """
        pass

    @abstractmethod
    def preprocess(self, file_bytes: bytes, filename: str) -> Any:
        """
        Executes media-specific preprocessing pipeline:
        - Images: Resizing, normalization, color channel extraction
        - Videos: Frame extraction, sampling, intensity scaling
        - Audio: Waveform decoding, normalization, spectral transforms
        """
        pass

    @abstractmethod
    def predict(self, file_bytes: bytes, file_hash: str, filename: str, content_type: str) -> Dict[str, Any]:
        """
        Runs model inference on preprocessed media payload and returns structured JSON predictions:
        - prediction ("Real" or "Fake")
        - confidence (float percentage)
        - risk_level ("High", "Medium", or "Low")
        - factors (dict breakdown of facial, voice, GAN, and metadata scores)
        - explainability_type (description of feature heatmap or analysis method)
        """
        pass
