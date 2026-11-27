import os
import tempfile
import time
from typing import Dict, Any
from inference.base import BaseInferenceEngine
from backend.ai_models.image.predictor import predict_image

class ImageInferenceEngine(BaseInferenceEngine):
    """Image Deepfake Inference Engine."""

    def __init__(self):
        super().__init__(engine_name="Image Deepfake Detection Engine")

    def run_inference(self, file_bytes: bytes, filename: str, content_type: str) -> Dict[str, Any]:
        start_time = time.time()

        # Write bytes to temporary file for model inference
        temp_dir = tempfile.gettempdir()
        file_ext = os.path.splitext(filename)[1].lower() or ".jpg"
        temp_path = os.path.join(temp_dir, f"deepshield_img_engine_{os.urandom(8).hex()}{file_ext}")

        try:
            with open(temp_path, "wb") as f:
                f.write(file_bytes)

            # Perform real model inference using predict_image()
            prediction_result = predict_image(temp_path)
            proc_time = round(time.time() - start_time, 2)

            prediction_label = prediction_result["prediction"]
            confidence = prediction_result["confidence"]
            risk_level = prediction_result.get("risk_level", "High" if prediction_label != "Real" and confidence > 75 else ("Medium" if prediction_label != "Real" else "Low"))
            msg = prediction_result.get("message", f"Synthetic/manipulated image detected with {confidence}% confidence." if prediction_label != "Real" else f"Authentic image verified with {confidence}% confidence.")
            is_fake = prediction_label != "Real"

            return {
                "status": "success",
                "prediction": prediction_label,
                "message": msg,
                "confidence": confidence,
                "risk_level": risk_level,
                "processing_time": proc_time,
                "model": prediction_result.get("model", "Vision Transformer"),
                "media_type": "image",
                "filename": filename,
                "factors": prediction_result.get("factors") or {
                    "face": round(confidence * 0.9, 1) if is_fake else round((100 - confidence) * 0.2, 1),
                    "voice": 0,
                    "gan": round(confidence * 0.95, 1) if is_fake else round((100 - confidence) * 0.1, 1),
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
                "media_type": "image",
                "filename": filename,
                "factors": { "face": 0, "voice": 0, "gan": 0, "meta": 0 }
            }
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
