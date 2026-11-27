# Purpose: Handles loading and initializing Hugging Face models for video deepfake detection.

from transformers import AutoImageProcessor, AutoModelForVideoClassification

VIDEO_MODEL_NAME = "shylhy/videomae-large-finetuned-deepfake-subset"

def load_video_model(model_name: str = VIDEO_MODEL_NAME):
    """
    Loads and returns the Hugging Face processor and model for video deepfake detection.

    Args:
        model_name (str): Hugging Face video model repository ID.

    Returns:
        tuple: (processor, model)
    """
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForVideoClassification.from_pretrained(model_name)
    return processor, model

if __name__ == "__main__":
    processor, model = load_video_model()
    print("Processor loaded successfully.")
    print("Model loaded successfully.")
    print(f"Model class name: {model.__class__.__name__}")

    num_labels = getattr(model.config, "num_labels", len(model.config.id2label) if hasattr(model.config, "id2label") else "N/A")
    print(f"Number of labels: {num_labels}")

    id2label = getattr(model.config, "id2label", None)
    if id2label:
        print(f"Label names: {id2label}")
    else:
        print("Label names: Not available")
