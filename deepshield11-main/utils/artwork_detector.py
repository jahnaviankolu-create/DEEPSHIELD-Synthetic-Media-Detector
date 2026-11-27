# Purpose: Detects whether an input image is an Artwork (painting, drawing, illustration, digital art, AI art) or a natural camera photograph.

import os
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any

ARTWORK_KEYWORDS = [
    "art", "artwork", "painting", "drawing", "illustration", "sketch",
    "anime", "digital_art", "dalle", "midjourney", "stable_diffusion",
    "synth", "ai_art", "oil_painting", "watercolor", "water_color",
    "canvas", "manga", "toon", "vector", "portrait_art", "design", "graphic"
]

def detect_artwork(image_path: str, filename: str = "") -> Dict[str, Any]:
    """
    Analyzes an input image to determine if it is an Artwork (drawing, painting, illustration, digital art, AI art)
    or a photographic camera image.

    Returns:
        dict: {
            "is_artwork": bool,
            "media_category": "Artwork" | "Photo",
            "artwork_style": str,
            "artwork_confidence": float
        }
    """
    fn_lower = (filename or os.path.basename(image_path)).lower()
    has_keyword = any(kw in fn_lower for kw in ARTWORK_KEYWORDS)

    is_artwork = False
    style = "Photographic Image"
    artwork_conf = 85.0

    try:
        pil_img = Image.open(image_path).convert("RGB")
        img_np = np.array(pil_img)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

        # 1. Color Saturation analysis
        sat = hsv[:, :, 1]
        mean_sat = float(np.mean(sat))

        # 2. Unique Colors ratio (Artworks/illustrations usually have distinct palette clustering)
        small = pil_img.resize((100, 100))
        unique_colors = len(set(small.getdata()))
        color_uniqueness_ratio = unique_colors / 10000.0

        # 3. Laplacian edge characteristics
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # Heuristic scoring for artwork vs photograph
        art_score = 0.0

        if has_keyword:
            art_score += 0.55

        # Artworks & illustrations often have high color saturation (mean_sat > 85/255)
        if mean_sat > 85:
            art_score += 0.25
        elif mean_sat > 60:
            art_score += 0.15

        # Artworks & digital art often have distinct/quantized color palettes (< 3500 unique colors in 100x100)
        if color_uniqueness_ratio < 0.35:
            art_score += 0.20
        elif color_uniqueness_ratio < 0.50:
            art_score += 0.10

        if laplacian_var > 450:
            art_score += 0.15

        if art_score >= 0.40:
            is_artwork = True
            artwork_conf = min(99.5, round(85.0 + art_score * 14.0, 1))
            if has_keyword or color_uniqueness_ratio < 0.30:
                style = "Digital Art / Illustration"
            else:
                style = "Fine Art / Painting"
        else:
            is_artwork = False
            style = "Photographic Image"

    except Exception as e:
        print(f"[Artwork Detector Warning] {e}")
        if has_keyword:
            is_artwork = True
            style = "Digital Artwork"
            artwork_conf = 90.0

    return {
        "is_artwork": is_artwork,
        "media_category": "Artwork" if is_artwork else "Photo",
        "artwork_style": style,
        "artwork_confidence": artwork_conf
    }
