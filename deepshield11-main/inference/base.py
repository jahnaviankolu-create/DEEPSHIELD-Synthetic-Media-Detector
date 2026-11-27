from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseInferenceEngine(ABC):
    """Abstract Base Class for DeepShield Media Inference Engines."""

    def __init__(self, engine_name: str):
        self.engine_name = engine_name

    @abstractmethod
    def run_inference(self, file_bytes: bytes, filename: str, content_type: str) -> Dict[str, Any]:
        """
        Executes media preprocessing and model evaluation.
        If trained model weights are missing/unavailable, returns 'Model not yet trained.'.
        """
        pass
