import os
import glob
import cv2
import numpy as np
from typing import Dict, Any, List, Tuple

from preprocessing.utils import (
    setup_logger,
    is_supported_extension,
    normalize_imagenet,
    handle_corrupted_file
)

logger = setup_logger("VideoPreprocessing")

SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

class VideoDatasetPreprocessor:
    """
    Modular Video Preprocessor for Deepfake Datasets.
    Validates video formats, extracts keyframes at configurable sample intervals,
    converts BGR frames to RGB, resizes frames to target resolution, normalizes frames,
    detects/removes corrupted video files, and saves frame tensor arrays.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        frame_sample_count: int = 8,
        remove_corrupted: bool = True
    ):
        self.target_size = target_size
        self.frame_sample_count = frame_sample_count
        self.remove_corrupted = remove_corrupted

    def is_valid_format(self, filepath: str) -> bool:
        """Validates if video file has a supported format extension."""
        return is_supported_extension(filepath, SUPPORTED_VIDEO_EXTS)

    def validate_and_extract_frames(self, filepath: str) -> Dict[str, Any]:
        """
        Processes a single video file:
        1. Format validation
        2. Video stream opening & integrity check
        3. Frame sampling at configurable intervals
        4. BGR -> RGB conversion & resizing
        5. Pixel normalization
        6. Corrupted file detection & removal
        """
        if not self.is_valid_format(filepath):
            logger.warning(f"Skipping unsupported video format: {filepath}")
            return {"status": "unsupported", "filepath": filepath}

        frames: List[np.ndarray] = []
        try:
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                raise ValueError("OpenCV failed to open video file stream")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

            if total_frames <= 0:
                cap.release()
                raise ValueError("Video contains no readable frames")

            step = max(1, total_frames // self.frame_sample_count)

            for i in range(0, total_frames, step):
                if len(frames) >= self.frame_sample_count:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                resized = cv2.resize(frame, self.target_size)
                rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                norm_frame = rgb_frame.astype(np.float32) / 255.0
                norm_tensor = normalize_imagenet(norm_frame)
                frames.append(norm_tensor)

            cap.release()

            if len(frames) == 0:
                raise ValueError("Zero valid frames extracted from video")

            frames_array = np.array(frames, dtype=np.float32)

            return {
                "status": "success",
                "filepath": filepath,
                "frame_count": len(frames),
                "total_frames": total_frames,
                "fps": fps,
                "frames_tensor": frames_array
            }
        except Exception as e:
            logger.error(f"Corrupted video detected [{filepath}]: {e}")
            removed = handle_corrupted_file(filepath, remove_corrupted=self.remove_corrupted, logger=logger)
            return {
                "status": "corrupted",
                "filepath": filepath,
                "error": str(e),
                "removed": removed
            }

    def process_directory(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        Recursively scans input_dir, validates & extracts video frames,
        and saves numpy frame sequence arrays into output_dir.
        """
        logger.info(f"Starting Video Dataset Preprocessing on directory: {input_dir}")
        os.makedirs(output_dir, exist_ok=True)

        processed_count = 0
        corrupted_count = 0
        unsupported_count = 0

        files = glob.glob(os.path.join(input_dir, "**", "*.*"), recursive=True)
        total_files = len(files)

        logger.info(f"Discovered {total_files} files in {input_dir}")

        for idx, filepath in enumerate(files, 1):
            if os.path.basename(filepath).startswith(".gitkeep"):
                continue

            res = self.validate_and_extract_frames(filepath)
            status = res["status"]

            if status == "success":
                processed_count += 1
                rel_path = os.path.relpath(filepath, input_dir)
                out_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".npy")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                np.save(out_path, res["frames_tensor"])
            elif status == "corrupted":
                corrupted_count += 1
            elif status == "unsupported":
                unsupported_count += 1

            if idx % 10 == 0 or idx == total_files:
                logger.info(
                    f"Video Progress [{idx}/{total_files}] -> Success: {processed_count}, "
                    f"Corrupted: {corrupted_count}, Unsupported: {unsupported_count}"
                )

        summary = {
            "total_scanned": total_files,
            "processed": processed_count,
            "corrupted": corrupted_count,
            "unsupported": unsupported_count,
            "output_directory": output_dir
        }
        logger.info(f"Video Preprocessing Finished: {summary}")
        return summary
