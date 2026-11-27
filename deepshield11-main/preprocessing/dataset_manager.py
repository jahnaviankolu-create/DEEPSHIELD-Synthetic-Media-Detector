import os
from typing import Dict, Any

from preprocessing.image_preprocessing import ImageDatasetPreprocessor
from preprocessing.video_preprocessing import VideoDatasetPreprocessor
from preprocessing.audio_preprocessing import AudioDatasetPreprocessor
from preprocessing.utils import setup_logger

logger = setup_logger("DatasetManager")

BASE_DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets")
PROCESSED_DATASET_DIR = os.path.join(BASE_DATASET_DIR, "processed")

class DatasetManager:
    """
    Master Dataset Pipeline Manager.
    Orchestrates batch dataset validation, preprocessing, corrupted file removal,
    and feature extraction across Image, Video, and Audio modalities.
    """

    def __init__(self, base_dir: str = BASE_DATASET_DIR, remove_corrupted: bool = True):
        self.base_dir = base_dir
        self.processed_dir = os.path.join(base_dir, "processed")
        self.image_preprocessor = ImageDatasetPreprocessor(remove_corrupted=remove_corrupted)
        self.video_preprocessor = VideoDatasetPreprocessor(remove_corrupted=remove_corrupted)
        self.audio_preprocessor = AudioDatasetPreprocessor(remove_corrupted=remove_corrupted)

    def process_image_dataset(self) -> Dict[str, Any]:
        """Preprocesses real and fake image dataset directories."""
        logger.info("\n==================================================")
        logger.info("  Starting Image Dataset Preprocessing Pipeline  ")
        logger.info("==================================================")
        results = {}
        for category in ["real", "fake"]:
            input_path = os.path.join(self.base_dir, "images", category)
            output_path = os.path.join(self.processed_dir, "images", category)
            results[category] = self.image_preprocessor.process_directory(input_path, output_path)
        return results

    def process_video_dataset(self) -> Dict[str, Any]:
        """Preprocesses real and fake video dataset directories."""
        logger.info("\n==================================================")
        logger.info("  Starting Video Dataset Preprocessing Pipeline  ")
        logger.info("==================================================")
        results = {}
        for category in ["real", "fake"]:
            input_path = os.path.join(self.base_dir, "videos", category)
            output_path = os.path.join(self.processed_dir, "videos", category)
            results[category] = self.video_preprocessor.process_directory(input_path, output_path)
        return results

    def process_audio_dataset(self) -> Dict[str, Any]:
        """Preprocesses real and fake audio dataset directories."""
        logger.info("\n==================================================")
        logger.info("  Starting Audio Dataset Preprocessing Pipeline  ")
        logger.info("==================================================")
        results = {}
        for category in ["real", "fake"]:
            input_path = os.path.join(self.base_dir, "audio", category)
            output_path = os.path.join(self.processed_dir, "audio", category)
            results[category] = self.audio_preprocessor.process_directory(input_path, output_path)
        return results

    def run_all(self) -> Dict[str, Any]:
        """Runs full dataset preprocessing pipeline across all modalities."""
        logger.info("\n==========================================================")
        logger.info("  DeepShield Master Dataset Preprocessing Pipeline Start  ")
        logger.info("==========================================================\n")

        summary = {
            "images": self.process_image_dataset(),
            "videos": self.process_video_dataset(),
            "audio": self.process_audio_dataset()
        }

        logger.info("\n==========================================================")
        logger.info("  DeepShield Master Dataset Preprocessing Pipeline Done   ")
        logger.info("==========================================================\n")
        return summary

if __name__ == "__main__":
    manager = DatasetManager()
    manager.run_all()
