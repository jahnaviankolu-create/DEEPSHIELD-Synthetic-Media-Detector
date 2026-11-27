import os
import tempfile
import time
from typing import Dict, Any
from inference.base import BaseInferenceEngine
from backend.ai_models.video.predictor import predict_video

class VideoInferenceEngine(BaseInferenceEngine):
    """Video Deepfake Inference Engine."""

    def __init__(self):
        super().__init__(engine_name="Video Deepfake Detection Engine")

    def run_inference(self, file_bytes: bytes, filename: str, content_type: str) -> Dict[str, Any]:
        start_time = time.time()

        temp_dir = tempfile.gettempdir()
        file_ext = os.path.splitext(filename)[1].lower() or ".mp4"
        temp_path = os.path.join(temp_dir, f"deepshield_vid_engine_{os.urandom(8).hex()}{file_ext}")

        try:
            with open(temp_path, "wb") as f:
                f.write(file_bytes)

            prediction_result = predict_video(temp_path)
            proc_time = round(time.time() - start_time, 2)

            prediction_label = prediction_result["prediction"]
            confidence = prediction_result["confidence"]
            risk_level = prediction_result.get("risk_level", "High" if prediction_label != "Real" and confidence > 75 else ("Medium" if prediction_label != "Real" else "Low"))
            reason = prediction_result.get("reason", f"Deepfake video detected with {confidence}% confidence." if prediction_label != "Real" else f"Authentic video verified with {confidence}% confidence.")
            is_fake = prediction_label != "Real"

            return {
                "status": "success",
                "prediction": prediction_label,
                "message": reason,
                "reason": reason,
                "confidence": confidence,
                "risk_level": risk_level,
                "processing_time": proc_time,
                "video_prediction": prediction_result.get("video_prediction", prediction_label),
                "video_confidence": prediction_result.get("video_confidence", confidence),
                "audio_prediction": prediction_result.get("audio_prediction", "N/A"),
                "audio_confidence": prediction_result.get("audio_confidence", 0.0),
                "model": prediction_result.get("model", "VideoMAE + Wav2Vec2 Pipeline"),
                "media_type": "video",
                "filename": filename,
                "factors": {
                    "face": round(confidence * 0.85, 1) if is_fake else round((100 - confidence) * 0.2, 1),
                    "voice": round(confidence * 0.7, 1) if is_fake else 0.0,
                    "gan": round(confidence * 0.9, 1) if is_fake else round((100 - confidence) * 0.1, 1),
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
                "media_type": "video",
                "filename": filename,
                "factors": { "face": 0, "voice": 0, "gan": 0, "meta": 0 }
            }
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
