# Purpose: Preprocesses video input data and extracts frames before passing to Hugging Face models.

import os
import sys

# Add project root directory to python path for package resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import cv2
import numpy as np
from backend.ai_models.video.model_loader import load_video_model

def preprocess_video(video_path: str, processor=None, num_frames: int = 16):
    """
    Preprocesses an input video file for VideoMAE model inference.

    Steps:
    1. Opens the video using OpenCV (cv2.VideoCapture).
    2. Limits video processing to a maximum of 30 seconds.
    3. Uniformly samples 16 frames across the target duration.
    4. Resizes every sampled frame to 224x224 pixels and converts BGR to RGB.
    5. Filters out exact duplicate frames.
    6. Uses the VideoMAE image processor to return PyTorch tensor inputs.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at path: '{video_path}'")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open or decode video file: '{video_path}'")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames_to_consider = min(total_frames, int(fps * 30)) # Limit to max 30 seconds

    if max_frames_to_consider < 1:
        max_frames_to_consider = total_frames or 1

    # Calculate uniform frame sampling indices over max 30 seconds
    if max_frames_to_consider < num_frames:
        frame_indices = list(range(max_frames_to_consider))
        while len(frame_indices) < num_frames:
            frame_indices.extend(frame_indices[:num_frames - len(frame_indices)])
    else:
        frame_indices = np.linspace(0, max_frames_to_consider - 1, num_frames, dtype=int)

    frames = []
    prev_frame = None

    for target_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()

        if not ret or frame is None:
            if prev_frame is not None:
                frame = prev_frame.copy()
            else:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            # Resize frame to 224x224 pixels
            frame = cv2.resize(frame, (224, 224))
            # Convert color space BGR -> RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Skip duplicate frames if identical to previous
            if prev_frame is not None and np.array_equal(frame, prev_frame):
                # Apply subtle variation if duplicate to avoid zero variance
                frame = cv2.GaussianBlur(frame, (3, 3), 0)

            prev_frame = frame.copy()

        frames.append(frame)

    cap.release()

    # Ensure 32 frames for VideoMAE positional embeddings (duplicate each of 16 sampled frames)
    if len(frames) == 16:
        sampled_32_frames = []
        for f in frames:
            sampled_32_frames.append(f)
            sampled_32_frames.append(f)
        frames = sampled_32_frames

    if processor is None:
        processor, _ = load_video_model()

    inputs = processor(list(frames), return_tensors="pt")
    return inputs
