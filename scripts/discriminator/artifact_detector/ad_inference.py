"""Inference script for the Segformer-based artifact detector."""

import logging
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)

np.set_printoptions(threshold=sys.maxsize)


def initialize_segformer(
    model_name_or_path: str | Path, out_channels: int = 1
) -> tuple[SegformerImageProcessor, SegformerForSemanticSegmentation]:
    """Loads and modifies a pre-trained Segformer model for artifact detection."""

    logger.info(f"Loading Segformer model from: {model_name_or_path}")
    
    # load a pretrained Segformer model
    preprocessor = SegformerImageProcessor.from_pretrained(model_name_or_path)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name_or_path)

    # change the number of output channels
    in_channels = model.decode_head.classifier.in_channels
    model.decode_head.classifier = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    return preprocessor, model


def process_images(
    input_dir: Path,
    output_heatmap_dir: Path,
    output_mask_dir: Path,
    output_mask_txt_dir: Path,
    model_weights_path: Path,
    device: torch.device,
) -> None:
    """Runs artifact detection inference on a directory of images."""
    
    output_heatmap_dir.mkdir(parents=True, exist_ok=True)
    output_mask_dir.mkdir(parents=True, exist_ok=True)
    output_mask_txt_dir.mkdir(parents=True, exist_ok=True)

    seg_preprocessor, artifact_detector = initialize_segformer("nvidia/mit-b5", out_channels=1)

    logger.info(f"Loading weights from {model_weights_path}")
    state_dict = torch.load(model_weights_path, weights_only=True)
    artifact_detector.load_state_dict(state_dict)
    
    artifact_detector.to(device)
    artifact_detector.eval()

    image_paths: list[Path] = [
        p for p in input_dir.iterdir() 
        if p.is_file() and p.suffix.lower() in (".png", ".jpeg", ".jpg")
    ]
    
    if not image_paths:
        logger.warning(f"No valid images found in {input_dir}")
        return

    logger.info(f"Starting inference on {len(image_paths)} images...")

    for image_path in tqdm(image_paths, desc="Processing Images"):
        image_name: str = image_path.stem
        
        raw_image: np.ndarray = cv2.imread(str(image_path))
        if raw_image is None:
            logger.error(f"Failed to read image: {image_path}")
            continue

        resized_image: np.ndarray = cv2.resize(raw_image, (512, 512))
        rgb_image: np.ndarray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        visualizable_image: np.ndarray = resized_image.copy()

        with torch.no_grad():
            tensor_image: torch.Tensor = transforms.ToTensor()(rgb_image).to(device)
            processed_inputs = seg_preprocessor(
                tensor_image, return_tensors="pt", do_rescale=False
            )
            pixel_values: torch.Tensor = processed_inputs["pixel_values"].to(device)

            model_output = artifact_detector(pixel_values)
            
            upsampled_logits: torch.Tensor = nn.functional.interpolate(
                model_output.logits,
                size=pixel_values.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            normalized_predictions: torch.Tensor = torch.sigmoid(upsampled_logits)

        # reformat the tensor output back into a valid cv2 image format
        mask_array: np.ndarray = (
            normalized_predictions[0].detach().cpu().numpy().transpose(-2, -1, -3) * 255
        ).astype(np.uint8)

        # save the raw numpy array as txt file
        mask_txt_save_path: Path = output_mask_txt_dir / f"mask_{image_name}.txt"
        with mask_txt_save_path.open("w") as f:
            f.write(str(mask_array))

        heatmap_color: np.ndarray = cv2.applyColorMap(mask_array, cv2.COLORMAP_JET)
        heatmap_alpha: float = 0.6 # the transparency of the heatmap
        blended_heatmap: np.ndarray = cv2.addWeighted(
            heatmap_color, heatmap_alpha, visualizable_image, 1 - heatmap_alpha, 0
        )

        heatmap_save_path: Path = output_heatmap_dir / f"result_{image_name}.png"
        cv2.imwrite(str(heatmap_save_path), blended_heatmap)

        _, binary_mask = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)

        # Dilation: Expands the white areas slightly so the downstream inpainting 
        # model has a clean edge buffer to blend the synthesized textures with.
        dilation_kernel: np.ndarray = np.ones((9, 9), np.uint8)
        dilated_mask: np.ndarray = cv2.dilate(binary_mask, dilation_kernel, iterations=1)

        mask_save_path: Path = output_mask_dir / f"mask_{image_name}.png"
        cv2.imwrite(str(mask_save_path), dilated_mask)

    logger.info("Done!")


if __name__ == "__main__":
    # configure paths
    project_root: Path = Path.cwd()

    cfg_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_weights = project_root / "scripts" / "discriminator"/ "artifact_detector" / "checkpoints" / "ad_richhf_baseline_model.bin"
    cfg_input_dir = project_root / "scripts" / "discriminator"/ "artifact_detector" / "asset" / "input"
    cfg_heatmap_dir = project_root / "scripts" / "discriminator"/ "artifact_detector" / "asset" / "output" / "heatmap"
    cfg_mask_dir = project_root /  "scripts" / "discriminator"/ "artifact_detector" / "asset" / "output" / "mask"
    cfg_mask_txt_dir = project_root /  "scripts" / "discriminator"/ "artifact_detector" / "asset" / "output" / "mask_txt"

    process_images(
        input_dir=cfg_input_dir,
        output_heatmap_dir=cfg_heatmap_dir,
        output_mask_dir=cfg_mask_dir,
        output_mask_txt_dir=cfg_mask_txt_dir,
        model_weights_path=cfg_weights,
        device=cfg_device,
    )