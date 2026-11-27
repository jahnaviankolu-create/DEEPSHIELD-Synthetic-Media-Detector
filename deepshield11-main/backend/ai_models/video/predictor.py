# Purpose: Handles inference and predictions for video deepfake detection using Hugging Face models.

import os
import sys

# Add project root directory to python path for package resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import torch
import torch.nn.functional as F
from backend.ai_models.video.model_loader import load_video_model
from backend.ai_models.video.preprocess import preprocess_video

def extract_audio_from_video(video_path: str) -> str:
    """Extracts 16kHz mono audio track from input video using FFmpeg."""
    import tempfile
    import subprocess
    import shutil

    candidate_bins = [
        shutil.which("ffmpeg"),
        shutil.which("ffmpeg.exe"),
        r"C:\ffmpeg\bin\ffmpeg.exe",
        r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
        r"C:\tools\ffmpeg\bin\ffmpeg.exe",
        "ffmpeg"
    ]
    ffmpeg_bin = next((b for b in candidate_bins if b and (os.path.isabs(b) and os.path.exists(b) or not os.path.isabs(b))), "ffmpeg")

    temp_dir = tempfile.gettempdir()
    audio_out_path = os.path.join(temp_dir, f"extracted_audio_{os.urandom(6).hex()}.wav")

    try:
        cmd = [
            ffmpeg_bin, "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            audio_out_path
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if os.path.exists(audio_out_path) and os.path.getsize(audio_out_path) > 1024:
            return audio_out_path
    except Exception as err:
        print(f"[VIDEO PREDICTOR] Audio extraction warning: {err}")

    if os.path.exists(audio_out_path):
        try: os.remove(audio_out_path)
        except Exception: pass
    return None

def predict_video(video_path: str):
    """
    Runs multimodal deepfake detection analyzing both Video Frames AND Audio Track independently.
    """
    from backend.ai_models.audio.predictor import predict_audio

    # 1. Visual Stream Analysis using VideoMAE (16 frames)
    processor, model = load_video_model()
    num_frames = 16
    inputs = preprocess_video(video_path, processor=processor, num_frames=num_frames)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probabilities = F.softmax(logits, dim=-1)
    predicted_class_id = torch.argmax(probabilities, dim=-1).item()
    video_conf = round(probabilities[0][predicted_class_id].item() * 100.0, 1)
    raw_label = model.config.id2label.get(predicted_class_id, f"LABEL_{predicted_class_id}")
    video_pred = "Real" if "real" in str(raw_label).lower() else "Fake"

    # 2. Audio Stream Extraction and Analysis
    extracted_audio_path = extract_audio_from_video(video_path)
    audio_pred = "Unavailable"
    audio_conf = 0.0

    if extracted_audio_path:
        try:
            audio_result = predict_audio(extracted_audio_path)
            aud_p = audio_result.get("prediction", "Real")
            audio_pred = "Real" if aud_p.lower() == "real" else "Fake"
            audio_conf = round(float(audio_result.get("confidence", 85.0)), 1)
        except Exception as aud_err:
            print(f"[VIDEO PREDICTOR] Audio analysis exception: {aud_err}")
            audio_pred = "Unavailable"
            audio_conf = 0.0
        finally:
            if os.path.exists(extracted_audio_path):
                try: os.remove(extracted_audio_path)
                except Exception: pass

    # 3. Multimodal Overall Decision & Reasoning
    is_vid_fake = video_pred.lower() == "fake"
    is_aud_fake = audio_pred.lower() == "fake"
    is_aud_available = audio_pred.lower() != "unavailable"

    if is_vid_fake and is_aud_fake:
        overall_pred = "Fake"
        overall_conf = round(max(video_conf, audio_conf), 1)
        reason = "Both video frames and audio track contain synthetic manipulations."
    elif is_vid_fake:
        overall_pred = "Fake"
        overall_conf = round(video_conf, 1)
        reason = "Fake video frames detected while audio appears authentic." if is_aud_available else "Fake video frames detected."
    elif is_aud_fake:
        overall_pred = "Fake"
        overall_conf = round(audio_conf, 1)
        reason = "Real video frames with synthetic AI voice overlay."
    else:
        overall_pred = "Real"
        if is_aud_available:
            overall_conf = round((video_conf + audio_conf) / 2.0, 1)
            reason = "Both video frames and audio track verified authentic."
        else:
            overall_conf = round(video_conf, 1)
            reason = "Video frames verified authentic. Audio track unavailable."

    risk_level = "High" if overall_pred == "Fake" and overall_conf >= 85.0 else ("Medium" if overall_pred == "Fake" else "Low")

    print("\n" + "=" * 65)
    print(f"[MULTIMODAL VIDEO DEBUG] Analyzing file: '{os.path.basename(video_path)}'")
    print(f"  Visual Frame Stream : {video_pred} ({video_conf}%)")
    print(f"  Audio Track Stream  : {audio_pred} ({audio_conf}%)")
    print(f"  Overall Verdict     : {overall_pred} ({overall_conf}%) [{risk_level} Risk]")
    print(f"  Reasoning           : {reason}")
    print("=" * 65 + "\n")

    video_dict = {
        "prediction": video_pred,
        "confidence": video_conf,
        "model": "VideoMAE",
        "frames_analyzed": num_frames
    }

    audio_dict = {
        "prediction": audio_pred,
        "confidence": audio_conf,
        "model": "Wav2Vec2"
    }

    overall_dict = {
        "prediction": overall_pred,
        "confidence": overall_conf,
        "risk_level": risk_level,
        "reason": reason
    }

    return {
        "status": "success",
        "video": video_dict,
        "audio": audio_dict,
        "overall": overall_dict,
        "prediction": overall_pred,
        "confidence": overall_conf,
        "risk_level": risk_level,
        "reason": reason,
        "message": reason,
        "model": "VideoMAE + Wav2Vec2 Pipeline"
    }

if __name__ == "__main__":
    import cv2
    import numpy as np

    # Generate a temporary synthetic test video using OpenCV
    sample_video_path = "temp_test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(sample_video_path, fourcc, 30.0, (224, 224))

    # Generate 45 dummy frames
    for i in range(45):
        frame = np.full((224, 224, 3), (i * 5 % 255, 120, 180), dtype=np.uint8)
        out.write(frame)
    out.release()

    try:
        print(f"Running video inference test on sample video: '{sample_video_path}'...")
        result = predict_video(sample_video_path)
        print("\n--- Video Test Results ---")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence Score: {result['confidence']}%")
        print("--------------------------")
    finally:
        if os.path.exists(sample_video_path):
            os.remove(sample_video_path)
