import os
from typing import Optional, Any

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")

def load_pytorch_model(weights_path: str) -> Optional[Any]:
    """
    Attempts to load PyTorch weights (.pth / .pt) from the specified path.
    Returns model object if successfully loaded, or None if weights file is missing.
    """
    if not os.path.exists(weights_path):
        return None

    try:
        import torch
        print(f"[DeepShield ModelLoader] Loading PyTorch model weights from: {weights_path}")
        model = torch.load(weights_path, map_location="cpu")
        model.eval()
        return model
    except Exception as e:
        print(f"[DeepShield ModelLoader] PyTorch load error: {e}")
        return None

def load_onnx_model(onnx_path: str) -> Optional[Any]:
    """
    Attempts to load ONNX inference session (.onnx) from the specified path.
    Returns session object if successfully loaded, or None if ONNX file is missing.
    """
    if not os.path.exists(onnx_path):
        return None

    try:
        import onnxruntime as ort
        print(f"[DeepShield ModelLoader] Loading ONNX session from: {onnx_path}")
        session = ort.InferenceSession(onnx_path)
        return session
    except Exception as e:
        print(f"[DeepShield ModelLoader] ONNX load error: {e}")
        return None

def get_trained_model(media_type: str) -> Optional[Any]:
    """
    Returns the loaded trained/pretrained model for the given media type ('image', 'video', 'audio').
    Checks local models/weights/ directory first; if absent, loads the integrated pretrained Hugging Face model.
    """
    if not os.path.exists(WEIGHTS_DIR):
        os.makedirs(WEIGHTS_DIR, exist_ok=True)

    possible_files = [
        os.path.join(WEIGHTS_DIR, f"{media_type}_model.onnx"),
        os.path.join(WEIGHTS_DIR, f"{media_type}_model.pt"),
        os.path.join(WEIGHTS_DIR, f"{media_type}_model.pth")
    ]

    for path in possible_files:
        if os.path.exists(path):
            if path.endswith(".onnx"):
                model = load_onnx_model(path)
                if model is not None:
                    return model
            elif path.endswith(".pt") or path.endswith(".pth"):
                model = load_pytorch_model(path)
                if model is not None:
                    return model

    # Fallback to integrated Hugging Face pretrained models if no local custom weight file exists
    try:
        if media_type == "image":
            from backend.ai_models.image.model_loader import load_model
            _, model = load_model()
            return model
        elif media_type == "video":
            from backend.ai_models.video.model_loader import load_video_model
            _, model = load_video_model()
            return model
        elif media_type == "audio":
            from backend.ai_models.audio.model_loader import load_audio_model
            _, model = load_audio_model()
            return model
    except Exception as err:
        print(f"[DeepShield ModelLoader] Error loading pretrained Hugging Face model for {media_type}: {err}")

    return None
