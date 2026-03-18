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
    "blurry and out of focus texture":      "sharp, highly detailed texture with clear focus",
    "unnatural smooth skin texture":        "natural skin with realistic pores and subtle texture variation",
    "inconsistent lighting and shadows":    "consistent natural lighting with accurate shadows",
    "distorted or warped geometry":         "correct geometry with natural proportions and straight edges",
    "noisy or grainy texture":              "clean, smooth surface with natural detail and no noise",
    "oversmoothed background":              "realistic background with natural depth, texture and detail",
    "mangled hands and extra fingers":       "anatomically correct hands with exactly five distinct, well-proportioned fingers",
    "asymmetrical or mismatched eyes":       "symmetrical, anatomically correct facial features with perfectly aligned eyes, matching pupils",
    "gibberish text and alien runes":        "clear, legible, correctly spelled english text with perfect typography and straight alignment",
    "melted or fused object boundaries":     "distinct, sharp edges separating objects with natural depth of field and clear separation",
    "checkerboard grid artifacts":           "smooth, continuous texture with no repeating grid, pure spatial consistency",
    "floating or disconnected elements":     "physically grounded objects with natural structural connections, proper gravity and weight",
    "plastic or waxy surface sheen":         "physically accurate surface material with natural light absorption and organic roughness",
    "color bleeding and chromatic shifting": "clean color separation with accurate, natural color boundaries and zero chromatic aberration",
    "repetitive cloned background patterns": "diverse, organic background with non-repeating, natural variations",
    "structurally impossible architecture":  "structurally sound building with coherent perspective, straight architectural lines, and logical physics",
    "hallucinated phantom limbs":            "clean negative space, natural body proportions with exact human anatomical constraints",
    "mutated or duplicate facial features":  "single mathematically perfect face with exactly two eyes, one nose, and one mouth",
    "blending background into foreground":   "sharp, distinct depth of field strictly separating the solid foreground subject from the background",
    "impossible reflections or shadows":     "accurate physical reflections and logically cast shadows matching a single primary light source",
    "deep dream hallucinatory fractal noise": "photorealistic, natural color palette with zero psychedelic, repeating, or fractal patterns",
    "AI watermark or signature artifacts":   "clean, continuous background texture extending to the edges with absolutely no text or watermarks",
    "asymmetrical or impossible clothing":   "logically constructed garments with natural, gravity-based fabric draping and symmetrical seams",
    "mismatched biological animal features": "anatomically correct animal with biologically consistent species traits, proper skeletal structure, and uniform fur",
    "teeth blending into lips or gums":      "distinct, individual human teeth with clear separation from natural gums and lips",
    "pupils bleeding into the iris":         "sharp, perfectly round pupils centered inside distinct, natural irises with realistic specular catchlights",
    "spaghetti-like chaotic hair strands":   "natural hair flow with logical strand behavior, clear directional styling, and organic volume",
    "jewelry merging into skin":             "solid, metallic jewelry sitting physically on top of the skin with clear contact shadows",
    "background objects floating in mid-air": "physically grounded objects resting firmly on surfaces with correct spatial perspective",
    "recursive or infinite object generation": "a single, clearly defined object with definitive boundaries and no recursive nesting",
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