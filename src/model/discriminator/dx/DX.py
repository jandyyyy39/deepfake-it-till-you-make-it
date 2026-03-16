"""
DX.py - Trained discriminator pipeline.

Pipeline:
    1. ClassificationCLIP (fine-tuned) → is_fake
    2. Segformer (artifact detector)   → binary_mask
    3. CLIP artifact classifier        → inpainting_prompt

Outputs (as one dict):
    - is_fake (bool)
    - artifact (str | None)
    - inpainting_prompt (str | None)
    - binary_mask (np.ndarray | None): HxW uint8, 255=fake region
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModel, SegformerForSemanticSegmentation, SegformerImageProcessor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "artifact_classifier"))

from artifact_classifier import load_clip, classify_artifact
# --- Paths (relative to this file) ---
_DIR = os.path.dirname(os.path.abspath(__file__))
CLIP_MODEL_PATH       = os.path.join(_DIR, "../../clip_model")
CLIP_WEIGHTS_PATH     = os.path.join(_DIR, "../../clip_model/best_classifier_weights.pth")
SEGFORMER_WEIGHTS     = os.path.join(_DIR, "../artifact_detector/checkpoints/ad_richhf_baseline_model.bin")

# --- CLIP preprocessing ---
CLIP_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])


# ClassificationCLIP
class ClassificationCLIP(nn.Module):
    def __init__(self, model_path: str):
        super(ClassificationCLIP, self).__init__()
        self.vision_encoder = CLIPVisionModel.from_pretrained(model_path)
        hidden_size = self.vision_encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, pixel_values):
        outputs = self.vision_encoder(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)


# Model loading
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ClassificationCLIP
    print("Loading fine-tuned ClassificationCLIP...")
    clip_model = ClassificationCLIP(CLIP_MODEL_PATH).to(device)
    clip_model.load_state_dict(
        torch.load(CLIP_WEIGHTS_PATH, map_location=device, weights_only=False)
    )
    clip_model.eval()
    print("ClassificationCLIP loaded.")

    # Segformer
    print("Loading Segformer artifact detector...")
    seg_processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b5")
    # Correct order (must match ad_inference.py exactly)
    seg_model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b5")
    in_channels = seg_model.decode_head.classifier.in_channels
    seg_model.decode_head.classifier = nn.Conv2d(in_channels, 1, kernel_size=1)  # replace FIRST
    seg_model.load_state_dict(torch.load(SEGFORMER_WEIGHTS, map_location=device, weights_only=False), strict=False)
    seg_model.to(device).eval()
    print("Segformer loaded.")

    # CLIP artifact classifier (reuses same clip_model folder)
    print("Loading CLIP artifact classifier...")
    art_clip_model, art_clip_processor = load_clip(CLIP_MODEL_PATH)
    print("CLIP artifact classifier loaded.")

    return clip_model, seg_model, seg_processor, art_clip_model, art_clip_processor, device


# Inference helpers
def _classify(image_path: str, clip_model, device) -> bool:
    """Returns True if the image is fake (score < 0.5 = fake per label convention)."""
    image = Image.open(image_path).convert("RGB")
    tensor = CLIP_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = clip_model(tensor)
        score = torch.sigmoid(logit).item()
    return score < 0.5


def _segment(image_path: str, seg_model, seg_processor, device) -> np.ndarray:
    """Returns a dilated binary mask (HxW uint8, 255=fake region) from Segformer."""
    import cv2
    raw = cv2.imread(image_path)
    resized = cv2.resize(raw, (512, 512))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    tensor = transforms.ToTensor()(rgb).to(device)
    inputs = seg_processor(tensor, return_tensors="pt", do_rescale=False)
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        output = seg_model(pixel_values)
        upsampled = nn.functional.interpolate(
            output.logits,
            size=pixel_values.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        preds = torch.sigmoid(upsampled)

    mask = (preds[0].detach().cpu().numpy().transpose(-2, -1, -3) * 255).astype(np.uint8)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((9, 9), np.uint8)
    return cv2.dilate(binary, kernel, iterations=1)


# Main entry point
def analyse(image_path: str, clip_model, seg_model, seg_processor, art_clip_model, art_clip_processor, device) -> dict:
    """
    Full DX pipeline for a single image.

    Returns:
        {
            "is_fake": bool,
            "artifact": str | None,
            "inpainting_prompt": str | None,
            "binary_mask": np.ndarray | None
        }
    """
    is_fake = _classify(image_path, clip_model, device)

    if not is_fake:
        return {"is_fake": False, "artifact": None, "inpainting_prompt": None, "binary_mask": None}

    binary_mask = _segment(image_path, seg_model, seg_processor, device)

    image = Image.open(image_path).convert("RGB")
    artifact, inpainting_prompt = classify_artifact(
        image, binary_mask, art_clip_model, art_clip_processor, device
    )

    return {
        "is_fake": True,
        "artifact": artifact,
        "inpainting_prompt": inpainting_prompt,
        "binary_mask": binary_mask
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python DX.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    clip_model, seg_model, seg_processor, art_clip_model, art_clip_processor, device = load_models()

    print(f"\nAnalysing: {image_path}\n")
    result = analyse(image_path, clip_model, seg_model, seg_processor, art_clip_model, art_clip_processor, device)

    print(f"Is Fake          : {result['is_fake']}")
    print(f"Artifact         : {result['artifact']}")
    print(f"Inpainting Prompt: {result['inpainting_prompt']}")
    print(f"Binary Mask      : {result['binary_mask'].shape if result['binary_mask'] is not None else None}")
