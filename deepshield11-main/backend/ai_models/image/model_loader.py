# Purpose: Handles loading and initializing Hugging Face pretrained models for image deepfake & AI generation detection.

from transformers import AutoImageProcessor, AutoModelForImageClassification

MODEL_FACE_DEEPFAKE = "dima806/deepfake_vs_real_image_detection"
MODEL_AI_GENERIC = "umm-maybe/AI-image-detector"

_model_cache = {}

def load_model(model_name: str = MODEL_FACE_DEEPFAKE):
    """
    Loads and returns the Hugging Face processor and model with memory caching.
    """
    if model_name in _model_cache:
        return _model_cache[model_name]

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name)
    _model_cache[model_name] = (processor, model)
    return processor, model

def load_ensemble_models():
    """
    Loads both face deepfake detector and general AI image detector.
    """
    p_face, m_face = load_model(MODEL_FACE_DEEPFAKE)
    p_gen, m_gen = load_model(MODEL_AI_GENERIC)
    return (p_face, m_face), (p_gen, m_gen)

if __name__ == "__main__":
    print("Loading ensemble image models...")
    (p_face, m_face), (p_gen, m_gen) = load_ensemble_models()
    print("Ensemble image models loaded successfully!")
