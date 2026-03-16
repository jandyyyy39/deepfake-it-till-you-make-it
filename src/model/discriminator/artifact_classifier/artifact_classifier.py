"""
artifact_classifier.py - Classifies the type of AI artifact in a masked region using CLIP.

Given an image and a binary mask (from Segformer), crops the flagged region and scores it
against known artifact descriptions using CLIP similarity matching.
"""

import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


CLIP_MODEL_ID = "openai/clip-vit-large-patch14"

# Artifact types → inpainting prompt mappings
ARTIFACT_MAP = {
    "blurry and out of focus texture":     "sharp, highly detailed texture with clear focus",
    "unnatural smooth skin texture":        "natural skin with realistic pores and subtle texture variation",
    "inconsistent lighting and shadows":    "consistent natural lighting with accurate shadows",
    "distorted or warped geometry":         "correct geometry with natural proportions and straight edges",
    "noisy or grainy texture":              "clean, smooth surface with natural detail and no noise",
    "oversmoothed background":              "realistic background with natural depth, texture and detail",
}

ARTIFACT_DESCRIPTIONS = list(ARTIFACT_MAP.keys())


def load_clip(model_path: str):
    """Loads CLIP model and processor from local path."""
    processor = CLIPProcessor.from_pretrained(model_path)
    model = CLIPModel.from_pretrained(model_path)
    model.eval()
    return model, processor


def _crop_masked_region(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """
    Crops the bounding box of the masked region from the image.
    Falls back to the full image if the mask is empty.

    Args:
        image: PIL RGB image.
        mask: HxW uint8 numpy array (255 = fake region, 0 = real).

    Returns:
        Cropped PIL image of the fake region.
    """
    # Resize mask to match image size if needed
    if mask.shape[:2] != (image.height, image.width):
        mask = np.array(
            Image.fromarray(mask).resize((image.width, image.height), Image.NEAREST)
        )

    coords = np.argwhere(mask > 127)
    if len(coords) == 0:
        return image  # fallback: use full image

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return image.crop((x_min, y_min, x_max, y_max))


def classify_artifact(
    image: Image.Image,
    mask: np.ndarray,
    clip_model,
    clip_processor,
    device: torch.device,
) -> tuple[str, str]:
    """
    Classifies the artifact type in the masked region using CLIP.

    Args:
        image: Full PIL RGB image.
        mask: HxW uint8 numpy array from Segformer (255 = fake region).
        clip_model: Loaded CLIP model.
        clip_processor: Loaded CLIP processor.
        device: torch device.

    Returns:
        (artifact_description, inpainting_prompt)
    """
    cropped = _crop_masked_region(image, mask)

    inputs = clip_processor(
        text=ARTIFACT_DESCRIPTIONS,
        images=cropped,
        return_tensors="pt",
        padding=True
    ).to(device)

    clip_model = clip_model.to(device)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits = outputs.logits_per_image  # (1, num_artifacts)
        probs = logits.softmax(dim=-1).squeeze(0)

    best_idx = probs.argmax().item()
    best_artifact = ARTIFACT_DESCRIPTIONS[best_idx]
    inpainting_prompt = ARTIFACT_MAP[best_artifact]

    return best_artifact, inpainting_prompt


if __name__ == "__main__":
    import sys
    import os

    if len(sys.argv) < 3:
        print("Usage: python artifact_classifier.py <image_path> <mask_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    mask_path = sys.argv[2]

    clip_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../clip_model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading CLIP...")
    clip_model, clip_processor = load_clip(clip_path)

    image = Image.open(image_path).convert("RGB")
    mask = np.array(Image.open(mask_path).convert("L"))

    artifact, prompt = classify_artifact(image, mask, clip_model, clip_processor, device)

    print(f"Detected Artifact  : {artifact}")
    print(f"Inpainting Prompt  : {prompt}")