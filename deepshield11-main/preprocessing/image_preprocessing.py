import os
import glob
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple

from preprocessing.utils import (
    setup_logger,
    is_supported_extension,
    normalize_imagenet,
    handle_corrupted_file
)

logger = setup_logger("ImagePreprocessing")

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

class ImageDatasetPreprocessor:
    """
    Modular Image Preprocessor for Deepfake Datasets.
    Validates image formats, isolates/removes corrupted files, converts color modes to RGB,
    resizes to target resolution, normalizes pixel matrices, and exports preprocessed tensors.
    """

    def __init__(self, target_size: Tuple[int, int] = (224, 224), remove_corrupted: bool = True):
        self.target_size = target_size
        self.remove_corrupted = remove_corrupted

    def is_valid_format(self, filepath: str) -> bool:
        """Validates if image file has a supported format extension."""
        return is_supported_extension(filepath, SUPPORTED_IMAGE_EXTS)

    def validate_and_preprocess_file(self, filepath: str) -> Dict[str, Any]:
        """
        Processes a single image file:
        1. Format validation
        2. Corrupted file verification & removal
        3. Color space conversion to RGB
        4. Image resizing
        5. Pixel normalization
        """
        if not self.is_valid_format(filepath):
            logger.warning(f"Skipping unsupported image format: {filepath}")
            return {"status": "unsupported", "filepath": filepath}

        try:
            # First pass: Integrity verification
            with Image.open(filepath) as img:
                img.verify()

            # Second pass: Data extraction & processing
            with Image.open(filepath) as img:
                rgb_img = img.convert("RGB")
                resized_img = rgb_img.resize(self.target_size, Image.Resampling.BILINEAR)
                raw_np = np.array(resized_img, dtype=np.float32) / 255.0

                # ImageNet Pixel Normalization
                norm_tensor = normalize_imagenet(raw_np)

                return {
                    "status": "success",
                    "filepath": filepath,
                    "tensor": norm_tensor,
                    "shape": norm_tensor.shape
                }
        except Exception as e:
            logger.error(f"Corrupted image detected [{filepath}]: {e}")
            removed = handle_corrupted_file(filepath, remove_corrupted=self.remove_corrupted, logger=logger)
            return {
                "status": "corrupted",
                "filepath": filepath,
                "error": str(e),
                "removed": removed
            }

    def process_directory(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """
        Recursively scans input_dir, validates & preprocesses image samples,
        and saves normalized numpy tensors into output_dir.
        """
        logger.info(f"Starting Image Dataset Preprocessing on directory: {input_dir}")
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

            res = self.validate_and_preprocess_file(filepath)
            status = res["status"]

            if status == "success":
                processed_count += 1
                rel_path = os.path.relpath(filepath, input_dir)
                out_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + ".npy")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                np.save(out_path, res["tensor"])
            elif status == "corrupted":
                corrupted_count += 1
            elif status == "unsupported":
                unsupported_count += 1

            if idx % 50 == 0 or idx == total_files:
                logger.info(
                    f"Image Progress [{idx}/{total_files}] -> Success: {processed_count}, "
                    f"Corrupted: {corrupted_count}, Unsupported: {unsupported_count}"
                )

        summary = {
            "total_scanned": total_files,
            "processed": processed_count,
            "corrupted": corrupted_count,
            "unsupported": unsupported_count,
            "output_directory": output_dir
        }
        logger.info(f"Image Preprocessing Finished: {summary}")
        return summary
