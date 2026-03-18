import torch
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
from PIL import Image
from pathlib import Path

from PIL import Image
import torchvision.transforms as T

import numpy as np
import re

MODEL_LIST = ["diffusers/stable-diffusion-xl-1.0-inpainting-0.1", "stable-diffusion-v1-5/stable-diffusion-inpainting", "kandinsky-community/kandinsky-2-2-decoder-inpaint"]
pipe = AutoPipelineForInpainting.from_pretrained(
    MODEL_LIST[0],
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

def txt_mask_to_image(txt_path, output_path):
    with open(txt_path, "r") as f:
        content = f.read()
    
    # Extract all integers directly — bypasses ALL numpy string formatting issues
    numbers = np.array(re.findall(r'\d+', content), dtype=np.uint8)
    
    arr = numbers.reshape(512, 512) # may need to change depending on what is the final size of the mask we need
    
    mask_img = Image.fromarray(arr, mode="L")
    mask_img.save(output_path)
    return output_path

def maskImage(name, mask, prompt):
    image = load_image(str(name)).convert("RGB")
    mask  = load_image(str(mask)).convert("L")
    
    out = pipe(
        prompt=prompt, # May need change
        image=image,
        mask_image=mask,
        num_inference_steps=60, # may need increase for better results
        guidance_scale=5.0, # how closely it sticks to the prompt, higher it more closely follows prompt
    ).images[0]
    
    stem = Path(name).stem

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    out_path = out_dir / f"{stem}_out.png"
    out.save(out_path)

    return out_path

def process_single(mask_path, input_path, prompt):
    mask_path = Path(mask_path)
    maskImage(input_path, mask_path, prompt)

def process_multiple(base_dir):
    input_dir = base_dir / "input"
    mask_dir = base_dir / "output" / "mask"
    prompt_dir = base_dir / "output" / "prompt"

    for input_path in input_dir.glob("*.*"):
        
        mask_path = mask_dir / ("mask_"+ input_path.stem + ".png")
        prompt_path = prompt_dir / ("prompt_"+ input_path.stem + ".txt")

        if not prompt_path.exists():
            print(f"Prompt not found for {input_path.name}")
            continue

        prompt = prompt_path.read_text().strip()

        if not mask_path.exists():
            print(f"Mask not found for {input_path.name}")
            continue

        maskImage(input_path, mask_path, prompt)

def main():
    # Example Usage - if mask is already a png then add the png instead of the txt
    process_single(txt_mask_to_image("mask.txt", "mask.png"), Path("test_img.png"), "")

    """
    Takes a directory path with an 'input' sub-folder with the images in png format
    and an 'output', 'mask', and 'prompt' sub-folder with the mask in png format

    for example:
    svd5 > input > 'image.png'
         > output > mask > 'mask_image.png' - naming convention is critical
                  > prompt > 'prompt_image.txt'
    
    """
    process_multiple(Path("./datasets/svd5"))

if __name__ == "__main__":
    main()