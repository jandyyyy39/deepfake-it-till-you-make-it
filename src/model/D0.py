"""
D0.py - Naive discriminator using ClassificationCLIP (untrained classifier head).

Architecture credit: team's CLIP classification notebooks.

The CLIP Vision Encoder is loaded with pretrained weights.
The linear classification head is randomly initialised (untrained) — making this a naive baseline.

Outputs a fakeness score in [0, 1] for a given image.
    - Score close to 1.0 → likely AI-generated
    - Score close to 0.0 → likely real
"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor
from PIL import Image
from torchvision import transforms


# --- CLIP preprocessing (must match training params) ---
CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])


class ClassificationCLIP(nn.Module):
    """
    CLIP Vision Encoder + Linear classification head.
    Identical architecture to the team's fine-tuned model,
    but here the classifier head is left randomly initialised (untrained).
    """
    def __init__(self, model_path: str):
        super(ClassificationCLIP, self).__init__()

        print("Loading CLIP Vision Encoder...")
        self.vision_encoder = CLIPVisionModel.from_pretrained(model_path)
        hidden_size = self.vision_encoder.config.hidden_size

        print("Attaching untrained Classification Head...")
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, pixel_values):
        outputs = self.vision_encoder(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        logits = self.classifier(pooled_output)
        return logits


def load_model(model_path: str = None):
    """
    Loads ClassificationCLIP with a randomly initialised classifier head (D0 baseline).

    Args:
        model_path: Path to the saved CLIP Vision Encoder. Defaults to ./clip_model.

    Returns:
        model, device
    """
    import os
    if model_path is None:
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clip_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClassificationCLIP(model_path).to(device)
    model.eval()
    return model, device


def get_fakeness_score(image_path: str, model, device) -> float:
    """
    Computes a fakeness score for a single image.

    Args:
        image_path: Path to the image file.
        model: Loaded ClassificationCLIP model.
        device: torch device.

    Returns:
        float in [0, 1] — higher means more likely AI-generated.
    """
    image = Image.open(image_path).convert("RGB")
    tensor = CLIP_TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(tensor)
        score = torch.sigmoid(logit).item()

    return score


if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) < 2:
        print("Usage: python D0.py <image_path>")
        sys.exit(1)

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clip_model")
    image_path = sys.argv[1]

    print("Loading D0 (untrained baseline)...")
    model, device = load_model(model_path)

    score = get_fakeness_score(image_path, model, device)
    print(f"Fakeness Score : {score:.4f}  ({'likely AI-generated' if score > 0.5 else 'likely real'})")
    print("Note: Score is expected to be near-random as the classifier head is untrained.")