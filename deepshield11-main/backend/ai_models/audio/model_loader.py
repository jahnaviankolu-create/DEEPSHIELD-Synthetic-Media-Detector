# Purpose: Handles loading and initializing Hugging Face models for audio deepfake detection.

from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

AUDIO_MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"

def load_audio_model(model_name: str = AUDIO_MODEL_NAME):
    """
    Loads and returns the Hugging Face feature extractor and model for audio deepfake detection.

    Args:
        model_name (str): Hugging Face audio model repository ID.

    Returns:
        tuple: (feature_extractor, model)
    """
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModelForAudioClassification.from_pretrained(model_name)
    return feature_extractor, model

if __name__ == "__main__":
    print(f"Downloading/loading audio model '{AUDIO_MODEL_NAME}' from Hugging Face...")
    feature_extractor, model = load_audio_model()
    print("Audio model and feature extractor loaded successfully!")
