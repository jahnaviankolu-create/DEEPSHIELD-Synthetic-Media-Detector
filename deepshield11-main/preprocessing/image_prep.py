import io
import numpy as np
from PIL import Image
import cv2
from typing import Dict, Any, Tuple

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def preprocess_image(file_bytes: bytes, target_size: Tuple[int, int] = (224, 224)) -> Dict[str, Any]:
    """
    Image Preprocessing Pipeline:
    1. Decodes image bytes using PIL / OpenCV
    2. Converts color space to RGB
    3. Resizes image to target dimensions (224x224)
    4. Normalizes pixel values using ImageNet mean & standard deviation
    """
    try:
        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        orig_w, orig_h = pil_img.size

        # Resize image
        resized_img = pil_img.resize(target_size, Image.Resampling.BILINEAR)
        img_np = np.array(resized_img, dtype=np.float32) / 255.0

        # Normalize pixel values
        normalized_tensor = (img_np - IMAGENET_MEAN) / IMAGENET_STD

        return {
            "status": "success",
            "tensor": normalized_tensor,
            "orig_size": (orig_w, orig_h),
            "processed_shape": normalized_tensor.shape
        }
    except Exception as e:
        return {
            "status": "error",
            "error_message": str(e),
            "tensor": None,
            "orig_size": (0, 0),
            "processed_shape": (0, 0, 0)
        }
