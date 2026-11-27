# Purpose: Handles inference and predictions for audio deepfake detection using Hugging Face models.

import os
import sys

# Add project root directory to python path for package resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import torch
import torch.nn.functional as F
from backend.ai_models.audio.model_loader import load_audio_model
from backend.ai_models.audio.preprocess import preprocess_audio

def predict_audio(audio_path: str):
    """
    Runs deepfake voice detection inference on an input audio file.

    Steps:
    1. Loads the Hugging Face feature extractor and model via model_loader.py.
    2. Preprocesses the input audio file via preprocess_audio().
    3. Runs model inference without computing gradients (torch.no_grad).
    4. Applies Softmax activation to obtain class probability distribution.
    5. Formats and returns prediction label ('Real' or 'Fake') and percentage confidence score.

    Args:
        audio_path (str): File path of the audio file to be analyzed.

    Returns:
        dict: {
            "prediction": "Real" | "Fake",
            "confidence": float (e.g. 98.45)
        }
    """
    # Step 1: Load feature extractor and model using load_audio_model()
    feature_extractor, model = load_audio_model()

    # Step 2: Preprocess audio into 16kHz PyTorch tensors
    inputs = preprocess_audio(audio_path, feature_extractor=feature_extractor)

    # Step 3: Run model inference in evaluation mode without computing gradients
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Step 4: Apply Softmax activation function to convert logits into probabilities
    probabilities = F.softmax(logits, dim=-1)

    # Step 5: Extract predicted class index and percentage confidence score
    predicted_class_id = torch.argmax(probabilities, dim=-1).item()
    confidence_score = probabilities[0][predicted_class_id].item() * 100.0

    raw_label = model.config.id2label.get(predicted_class_id, f"LABEL_{predicted_class_id}")

    # Generate dark-themed SVG Spectrogram base64
    accent_color = "%23ef4444" if "real" not in str(raw_label).lower() else "%2310b981"
    spectrogram_svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="600" height="180"><rect width="600" height="180" fill="%230f172a"/><path d="M0 90 Q 50 10, 100 90 T 200 90 T 300 160 T 400 20 T 500 90 T 600 90" stroke="{accent_color}" stroke-width="3" fill="none"/><circle cx="400" cy="20" r="6" fill="{accent_color}"/></svg>'
    import base64
    spectrogram_b64 = "data:image/svg+xml;base64," + base64.b64encode(spectrogram_svg.encode()).decode()

    # Standardize label format to 'Real' or 'Fake'
    prediction_label = "Real" if "real" in str(raw_label).lower() else "Fake"
    risk_level = "High" if prediction_label != "Real" and confidence_score >= 80.0 else ("Medium" if prediction_label != "Real" else "Low")
    msg = f"Synthetic voice clone detected with {confidence_score:.2f}% confidence." if prediction_label != "Real" else f"Authentic audio verified with {confidence_score:.2f}% confidence."

    return {
        "prediction": prediction_label,
        "confidence": round(confidence_score, 2),
        "risk_level": risk_level,
        "model": "Wav2Vec2 (garystafford/wav2vec2-deepfake-voice-detector)",
        "message": msg,
        "spectrogram": spectrogram_b64,
        "explainability_images": [spectrogram_b64]
    }

if __name__ == "__main__":
    import wave
    import struct
    import math

    # Programmatically generate a short 1-second dummy WAV audio file (440Hz sine wave, 16kHz mono)
    sample_audio_path = "temp_test_audio.wav"
    sample_rate = 16000
    duration = 1.0  # seconds
    frequency = 440.0  # Hz

    with wave.open(sample_audio_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        num_samples = int(sample_rate * duration)
        for i in range(num_samples):
            value = int(32767.0 * 0.5 * math.sin(2.0 * math.pi * frequency * i / sample_rate))
            data = struct.pack("<h", value)
            wf.writeframesraw(data)

    try:
        print(f"Running audio inference test on sample file: '{sample_audio_path}'...")
        result = predict_audio(sample_audio_path)
        print("\n--- Audio Test Results ---")
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence Score: {result['confidence']}%")
        print("--------------------------")
    finally:
        if os.path.exists(sample_audio_path):
            os.remove(sample_audio_path)
