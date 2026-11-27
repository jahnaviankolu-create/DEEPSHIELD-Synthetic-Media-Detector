import os
import tempfile
import time
from typing import Dict, Any
from inference.base import BaseInferenceEngine
from backend.ai_models.audio.predictor import predict_audio

class AudioInferenceEngine(BaseInferenceEngine):
    """Audio Deepfake Inference Engine."""

    def __init__(self):
        super().__init__(engine_name="Audio Deepfake Detection Engine")

    def run_inference(self, file_bytes: bytes, filename: str, content_type: str) -> Dict[str, Any]:
        start_time = time.time()

        temp_dir = tempfile.gettempdir()
        file_ext = os.path.splitext(filename)[1].lower() or ".wav"
        temp_path = os.path.join(temp_dir, f"deepshield_aud_engine_{os.urandom(8).hex()}{file_ext}")

        try:
            with open(temp_path, "wb") as f:
                f.write(file_bytes)

            prediction_result = predict_audio(temp_path)
            proc_time = round(time.time() - start_time, 2)

            prediction_label = prediction_result["prediction"]
            confidence = prediction_result["confidence"]
            risk_level = prediction_result.get("risk_level", "High" if prediction_label != "Real" and confidence > 75 else ("Medium" if prediction_label != "Real" else "Low"))
            msg = prediction_result.get("message", f"Synthetic voice clone detected with {confidence}% confidence." if prediction_label != "Real" else f"Authentic audio verified with {confidence}% confidence.")
            is_fake = prediction_label != "Real"
            spectrogram_b64 = prediction_result.get("spectrogram", "")

            return {
                "status": "success",
                "prediction": prediction_label,
                "message": msg,
                "confidence": confidence,
                "risk_level": risk_level,
                "processing_time": proc_time,
                "spectrogram": spectrogram_b64,
                "explainability_images": [spectrogram_b64] if spectrogram_b64 else [],
                "model": prediction_result.get("model", "Wav2Vec2 Voice Detector"),
                "media_type": "audio",
                "filename": filename,
                "factors": {
                    "face": 0,
                    "voice": round(confidence * 0.95, 1) if is_fake else round((100 - confidence) * 0.2, 1),
                    "gan": round(confidence * 0.6, 1) if is_fake else 0.0,
                    "meta": 10.0
                }
            }
        except Exception as e:
            proc_time = round(time.time() - start_time, 2)
            return {
                "status": "error",
                "prediction": "Model evaluation error.",
                "message": str(e),
                "confidence": None,
                "risk_level": "Error",
                "processing_time": proc_time,
                "model": self.engine_name,
                "media_type": "audio",
                "filename": filename,
                "factors": { "face": 0, "voice": 0, "gan": 0, "meta": 0 }
            }
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
