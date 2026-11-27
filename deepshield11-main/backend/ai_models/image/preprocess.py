# Purpose: Preprocesses image input data before passing it to Hugging Face models for analysis.

import os
import sys

# Add project root directory to python path for package resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from PIL import Image
from backend.ai_models.image.model_loader import load_model

def preprocess_image(image_path: str, processor=None):
    """
    Preprocesses an input image for Hugging Face Vision Transformer model inference.

    Steps:
    1. Validates file existence and non-zero size.
    2. Opens the image using Pillow and verifies image integrity.
    3. Converts RGBA, CMYK, P (palette), or L (grayscale) images to 3-channel RGB.
    4. Applies model processor resizing, normalization, and tensor extraction without compression loss.

    Args:
        image_path (str): Path to the image file to be analyzed.
        processor (optional): Hugging Face image processor instance.

    Returns:
        dict: Processed PyTorch tensors dictionary (containing 'pixel_values').
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: '{image_path}'")

    if os.path.getsize(image_path) == 0:
        raise ValueError(f"Image file is empty (0 bytes): '{image_path}'")

    # Step 1: Open and validate image integrity
    try:
        with Image.open(image_path) as img_check:
            img_check.verify()
    except Exception as img_err:
        raise ValueError(f"Invalid or corrupted image file '{image_path}': {str(img_err)}")

    # Step 2: Load clean uncompressed image for RGB conversion
    image = Image.open(image_path)

    # Step 3: Preserve 3-channel RGB color space across all image formats
    if image.mode != "RGB":
        image_rgb = image.convert("RGB")
    else:
        image_rgb = image.copy()

    # Step 4: Load default Hugging Face image processor if not provided
    if processor is None:
        processor, _ = load_model()

    # Step 5: Normalize and extract PyTorch tensors (224x224 input resolution)
    inputs = processor(images=image_rgb, return_tensors="pt")

    return inputs
