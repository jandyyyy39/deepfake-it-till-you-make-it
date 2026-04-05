"""
Swin SDXL Detector (Organika, HuggingFace)
-------------------------------------------
Fine-tuned Swin Transformer for detecting SDXL-generated images vs real photos.
Model weights auto-download from HuggingFace Hub on first run (~330 MB).

Accuracy: 98.1% | AUC: 0.998

Usage:
    python swin_sdxl_inference.py <image_path>
"""

import sys

import torch
from PIL import Image

MODEL_ID = "Organika/sdxl-detector"

# Label mapping: 0 = artificial, 1 = human
FAKE_CLASS = 0


def load_model():
    """
    Loads the Swin SDXL detector from HuggingFace Hub.

    Returns:
        (model, processor), device
    """
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
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
        print("Usage: python drct_inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    print("Loading Swin SDXL Detector...")
    model_tuple, device = load_model()

    score = get_fakeness_score(image_path, model_tuple, device)
    print(f"Fakeness Score : {score:.4f}  ({'likely AI-generated' if score > 0.5 else 'likely real'})")
