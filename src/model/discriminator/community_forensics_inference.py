"""
Community Forensics Detector (CVPR 2025)
-----------------------------------------
ViT-Small trained on 4,803 different generators (2.7M images).

94.6% accuracy on GenImage. #1 ranked detector across 291 generators.

Usage:
    python community_forensics_inference.py <image_path>
"""

import io
import sys
import tarfile
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

WEIGHTS_TAR = Path(__file__).parent / "weights" / "commfor_weights.tar"
CKPT_NAME = "pretrained_weights/model_v11_ViT_384_base_ckpt.pt"
INPUT_SIZE = 384
RESIZE_SIZE = 440


def _build_model():
    """Build the ViTClassifier and return it (uninitialized weights)."""
    import timm

    class ViTClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.vit = timm.create_model(
                "vit_small_patch16_384.augreg_in21k_ft_in1k", pretrained=False
            )
            self.vit.head = nn.Linear(384, 1)

        def forward(self, x):
            return self.vit(x)

    return ViTClassifier()


def load_model(weights_tar=None):
    """
    Loads the Community Forensics detector from local checkpoint.

    Args:
        weights_tar: Path to commfor_weights.tar. Defaults to weights/commfor_weights.tar.

    Returns:
        (model, transform), device
    """
    if weights_tar is None:
        weights_tar = WEIGHTS_TAR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _build_model()

    # Extract checkpoint from tar and load state dict
    with tarfile.open(weights_tar) as tar:
        f = tar.extractfile(CKPT_NAME)
        checkpoint = torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=False)

    state_dict = checkpoint["model"]

    # Strip _orig_mod. prefix added by torch.compile() if present
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
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
