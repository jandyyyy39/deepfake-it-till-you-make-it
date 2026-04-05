import torch
from pathlib import Path
import os
import re
import numpy as np

from PIL import Image, ImageFilter
import torchvision.transforms as T

from diffusers import KandinskyV22PriorPipeline, KandinskyV22InpaintPipeline, AutoPipelineForInpainting
from diffusers.utils import load_image

import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

token = ""

# SDXL inpainting pipeline
pipe_sdxl = AutoPipelineForInpainting.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
    variant="fp16",
    token=token,
).to("cuda")

# Kandinsky Impainting pipeline
prior = KandinskyV22PriorPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-prior",
    torch_dtype=torch.float16,
    token=token,
).to("cuda")

pipe_kandinsky = KandinskyV22InpaintPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder-inpaint",
    torch_dtype=torch.float16,
    token=token,
).to("cuda")

def txt_mask_to_binary_image(txt_path, output_dir, shape=(512, 512), threshold=50, max_value=None, invert=False):
    """
    Convert a txt mask containing numeric values into a binary PNG mask.

    Parameters
      txt_path: Path to the input txt file.
      output_dir: Directory where the output PNG will be saved.
      shape: Expected mask shape as (height, width).
      threshold: Values >= THRESHOLD become white (255), below become black (0).
      max_value: Optional upper clamp for input values. If None, uses the max in the file.
      invert: If True, swap black/white output.
      feather_radius: Optional slight blur radius for softer mask edges.
    """
    txt_path = Path(txt_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    content = txt_path.read_text()

    # Extract integers
    numbers = np.array(re.findall(r'-?\d+', content), dtype=np.int32)

    expected_size = shape[0] * shape[1]
    if numbers.size != expected_size:
        raise ValueError(
            f"Expected {expected_size} values for shape {shape}, "
            f"but found {numbers.size}."
        )

    arr = numbers.reshape(shape)

    if max_value is None:
        max_value = int(arr.max())

    arr = np.clip(arr, 0, max_value)

    # Binary conversion
    binary = (arr >= threshold).astype(np.uint8) * 255

    if invert:
        binary = 255 - binary

    mask_img = Image.fromarray(binary, mode="L")

    output_path = output_dir / f"{txt_path.stem}_binary_t{threshold}.png"
    mask_img.save(output_path)

    return output_path

def maskImageSDXL(name, mask, prompt):
    """
    Inpainting with SDXL
    Parameters:
    name: Path to the image to be inpainted
    mask: Path to the mask to be used
    prompt: Guided inpainting prompt (optional)
    """
    image = load_image(str(name)).convert("RGB")
    mask_image = load_image(str(mask)).convert("L")

    print(f"DEBUG - Loading SDXL image: {name}")

    out = pipe_sdxl(
        prompt=prompt,
        image=image,
        mask_image=mask_image,
        num_inference_steps=60,
        guidance_scale=5.0,
    ).images[0]

    stem = Path(name).stem
    out_dir = Path("/content/drive/MyDrive/output_inpainting_sdxl")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{stem}_out_SDXL.png"
    out.save(out_path)

    print(f"DEBUG - SDXL image saved: {out_path}")
    return out_path

def maskImageKandinsky(name, mask, prompt):
    """
    Inpainting with Kandinsky
    Parameters:
    name: Path to the image to be inpainted
    mask: Path to the mask to be used
    prompt: Guided inpainting prompt (optional)
    """
    print("DEBUG 1 - loading images")
    image = load_image(str(name)).convert("RGB")
    mask = load_image(str(mask)).convert("L")

    print("DEBUG 2 - running prior")
    prior_output = prior(
        prompt=prompt,
        negative_prompt="",
    )

    print("DEBUG 3 - running inpainting pipe")
    out = pipe_kandinsky(
        image=image,
        mask_image=mask,
        image_embeds=prior_output.image_embeds,
        negative_image_embeds=prior_output.negative_image_embeds,
        num_inference_steps=60,
        guidance_scale=5.0,
    ).images[0]

    print("DEBUG 4 - creating output dir")
    stem = Path(name).stem
    out_dir = Path("/content/drive/MyDrive/output_inpainting_kandinsky")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("DEBUG 5 - saving image")
    out_path = out_dir / f"{stem}_out_Kandinsky.png"
    out.save(out_path)

    print(f"DEBUG 6 - saved: {out_path}")
    return out_path

def convert_txt_to_mask_multiple(base_dir):
    """
    Converts multiple mask.txt files into png
    Parameters
    base_dir: Path to the directory containing the main directory with images and mask
    """
    base_dir = Path(base_dir)

    mask_dir = base_dir / "output" / "mask_txt"
    output_dir = base_dir / "output" / "mask_redone"

    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in mask_dir.glob("*.txt"):
        print(f"Processing: {input_path}")
        txt_mask_to_binary_image(input_path, output_dir)

def process_multiple(base_dir, model="sdxl"):
    """
    Quick function to load multiple images and use the inpainting model
    Parameters:
      base_dir: Path to the directory containing the images
      model: Which model to use either Kandinsky or SDXL
    """
    base_dir = Path(base_dir)

    input_dir = base_dir / "input"
    mask_dir = base_dir / "output" / "mask"
    prompt_dir = base_dir / "output" / "prompt"   # optional
    prompt = ""

    for input_path in input_dir.glob("*.*"):
        mask_path = mask_dir / f"mask_{input_path.stem}.png"
        prompt_path = prompt_dir / f"prompt_{input_path.stem}.txt"

        if prompt_path.exists():
            prompt = prompt_path.read_text(encoding="utf-8").strip()
        else:
            prompt = ""

        if not mask_path.exists():
            print(f"Mask not found for {input_path.name}")
            continue

        try:
            if model.lower() == "kandinsky":
                maskImageKandinsky(input_path, mask_path, prompt)
            elif model.lower() == "sdxl":
                maskImageSDXL(input_path, mask_path, prompt)
            else:
                print(f"Unknown model '{model}' for {input_path.name}")
        except Exception as e:
            print(f"Failed on {input_path.name}: {e}")

def main():
    # Example of how to use the process_multiple method
    process_multiple(Path("/content/drive/MyDrive/impainting_input"), "kandinsky")
    process_multiple(Path(("/content/drive/MyDrive/impainting_input"), "sdxl"))

    # Sample of how impainting is done with SDXL - we used this image for the pptx
    base_dir = Path("/content/drive/MyDrive/inpainting_input")
    mask_dir = base_dir / "output" / "mask_redone" / "mask_802_sdv5_00195_binary_t50.png"
    output_dir = base_dir / "input" / "802_sdv5_00195.png"

    maskImageSDXL(output_dir, mask_dir, "")

if __name__ == "__main__":
    main()