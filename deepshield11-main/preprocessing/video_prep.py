import os
import tempfile
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple

def extract_video_frames(
    file_bytes: bytes, 
    filename: str, 
    num_frames: int = 8, 
    target_size: Tuple[int, int] = (224, 224)
) -> Dict[str, Any]:
    """
    Video Frame Extraction & Preprocessing Pipeline:
    1. Saves video byte buffer to a temporary file stream
    2. Reads video stream using OpenCV cv2.VideoCapture
    3. Extracts keyframes sampled evenly across video duration
    4. Resizes frames to 224x224 RGB and normalizes pixel matrices
    """
    temp_file_path = None
    frames: List[np.ndarray] = []

    try:
        ext = os.path.splitext(filename)[1] if '.' in filename else '.mp4'
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name

        cap = cv2.VideoCapture(temp_file_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

        step = max(1, total_frames // num_frames)

        for i in range(0, total_frames, step):
            if len(frames) >= num_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            resized = cv2.resize(frame, target_size)
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            norm_frame = rgb_frame.astype(np.float32) / 255.0
            frames.append(norm_frame)

        cap.release()

        return {
            "status": "success",
            "frame_count": len(frames),
            "total_frames_in_video": total_frames,
            "fps": fps,
            "frames_tensor": np.array(frames) if frames else None
        }

    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "frame_count": 0,
            "total_frames_in_video": 0,
            "fps": 0.0,
            "frames_tensor": None
        }
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass
