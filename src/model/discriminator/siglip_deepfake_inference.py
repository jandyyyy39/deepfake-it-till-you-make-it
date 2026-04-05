"""
SigLIP Deepfake Detector (Ateeqq, HuggingFace)
------------------------------------------------
Fine-tuned SigLIP for AI-generated vs human image classification.
Model weights auto-download from HuggingFace Hub on first run (~370 MB).

Accuracy: 99.23% on test set (120k images: 60k AI + 60k human).

Usage:
    python siglip_deepfake_inference.py <image_path>
"""

import sys

import torch
from PIL import Image

MODEL_ID = "Ateeqq/ai-vs-human-image-detector"

# Label mapping: 0 = ai, 1 = hum (human)
FAKE_CLASS = 0


def load_model():
    """
    Loads the SigLIP deepfake detector from HuggingFace Hub.

    Returns:
        (model, processor), device
    """
    from transformers import AutoImageProcessor, SiglipForImageClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = SiglipForImageClassification.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()

    return (model, processor), device


def get_fakeness_score(image_path: str, model_tuple, device) -> float:
    """
    Computes a fakeness score for a single image.

    Returns:
        float in [0, 1] — higher means more likely AI-generated.
    """
    model, processor = model_tuple

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        score = probs[0, FAKE_CLASS].item()

    return score


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python siglip_deepfake_inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    print("Loading SigLIP Deepfake Detector...")
    model_tuple, device = load_model()

    score = get_fakeness_score(image_path, model_tuple, device)
    print(f"Fakeness Score : {score:.4f}  ({'likely AI-generated' if score > 0.5 else 'likely real'})")
