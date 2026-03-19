"""
Community Forensics Detector (CVPR 2025)
-----------------------------------------
ViT-Small trained on 4,803 different generators (2.7M images).
Model weights auto-download from HuggingFace Hub on first run (~88 MB).

94.6% accuracy on GenImage. #1 ranked detector across 291 generators.

Usage:
    python community_forensics_inference.py <image_path>
"""

import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

MODEL_REPO = "OwensLab/commfor-model-384"
INPUT_SIZE = 384
RESIZE_SIZE = 440


def _build_model_class():
    """Build the ViTClassifier class with HuggingFace Hub integration."""
    import timm
    from huggingface_hub import PyTorchModelHubMixin

    class ViTClassifier(nn.Module, PyTorchModelHubMixin):
        def __init__(self, model_size="small", input_size=384, patch_size=16,
                     freeze_backbone=False, device="cuda", dtype=torch.float32):
            super().__init__()
            model_name = f"vit_{model_size}_patch{patch_size}_{input_size}.augreg_in21k_ft_in1k"
            self.vit = timm.create_model(model_name, pretrained=False)
            embed_dim = {"small": 384, "tiny": 192}[model_size]
            self.vit.head = nn.Linear(embed_dim, 1)

        def forward(self, x):
            return self.vit(x)

    return ViTClassifier


def load_model():
    """
    Loads the Community Forensics detector from HuggingFace Hub.

    Returns:
        (model, transform), device
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ViTClassifier = _build_model_class()
    model = ViTClassifier.from_pretrained(MODEL_REPO)
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(RESIZE_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return (model, transform), device


@torch.no_grad()
def get_fakeness_score(image_path: str, model_tuple, device) -> float:
    """
    Computes a fakeness score for a single image.

    Returns:
        float in [0, 1] — higher means more likely AI-generated.
    """
    model, transform = model_tuple
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    logit = model(tensor)
    return torch.sigmoid(logit).item()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python community_forensics_inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    print("Loading Community Forensics Detector...")
    model_tuple, device = load_model()

    score = get_fakeness_score(image_path, model_tuple, device)
    print(f"Fakeness Score : {score:.4f}  ({'likely AI-generated' if score > 0.5 else 'likely real'})")
