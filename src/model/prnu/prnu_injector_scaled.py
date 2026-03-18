import cv2
import numpy as np
import random
import sys
import shutil
from pathlib import Path
import argparse

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
PRNU_DIR = BASE_DIR / "datasets" / "prnu_fingerprints"
OUTPUT_DIR = BASE_DIR / "datasets" / "prnu_injected"

def crop_center(img_array, cropx=512, cropy=512):
    """Strictly crops the center of an array to the exact dimensions."""
    y, x = img_array.shape[:2]
    
    # Pad if the array is smaller than the target crop
    if y < cropy or x < cropx:
        if img_array.ndim == 3:
            padded = np.zeros((max(y, cropy), max(x, cropx), img_array.shape[2]), dtype=img_array.dtype)
            padded[:y, :x, :] = img_array
        else:
            padded = np.zeros((max(y, cropy), max(x, cropx)), dtype=img_array.dtype)
            padded[:y, :x] = img_array
        img_array = padded
        y, x = img_array.shape[:2]
        
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    
    if img_array.ndim == 3:
        return img_array[starty:starty+cropy, startx:startx+cropx, :]
    else:
        return img_array[starty:starty+cropy, startx:startx+cropx]

def process_batch(image_paths):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load all available real PRNU files
    prnu_files = list(PRNU_DIR.rglob("*_fingerprint.npy"))
    if not prnu_files:
        print(f"Error: No PRNU .npy files found in {PRNU_DIR}")
        sys.exit(1)
        
    print(f"Found {len(prnu_files)} camera fingerprints. Starting batch injection...")
    
    success_count = 0
    
    for img_path in image_paths:
        img_path = Path(img_path)
        if not img_path.exists():
            print(f"Skipping {img_path.name}: File not found.")
            continue
            
        # Load and prepare the Image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Skipping {img_path.name}: Unreadable image format.")
            continue
            
        img_cropped = crop_center(img_bgr, 512, 512)
        img_float = img_cropped.astype(np.float32) / 255.0  # normalize to [0, 1]

        # Select a random hardware fingerprint
        selected_prnu_path = random.choice(prnu_files)
        prnu_array = np.load(selected_prnu_path)

        # Prepare the PRNU (Crop to match, expand channels)
        prnu_cropped = crop_center(prnu_array, 512, 512)

        if prnu_cropped.ndim == 2:
            prnu_cropped = np.stack([prnu_cropped]*3, axis=-1)

        # The Mathematical Injection (Spoofing) — Multiplicative PRNU model: I = (1 + K) * Y
        poisoned_float = img_float * (1.0 + prnu_cropped)
        alpha = 3  # tunable strength
        prnu_normalised = prnu_cropped / (np.std(prnu_cropped) + 1e-8)
        poisoned_float = img_float + (alpha * prnu_normalised) / 255.0
        poisoned_img = (np.clip(poisoned_float, 0, 1.0) * 255).astype(np.uint8)
        
        # Output Generation
        base_name = img_path.stem
        camera_name = selected_prnu_path.stem.replace("_fingerprint", "")
        
        out_img_name = OUTPUT_DIR / f"{base_name}_spoofed_by_{camera_name}.png"
        out_npy_name = OUTPUT_DIR / f"{base_name}_spoofed_by_{camera_name}.npy"
        
        # Save the poisoned image
        cv2.imwrite(str(out_img_name), poisoned_img)
        
        # Copy the exact mathematical array used so your team can verify it
        shutil.copy(selected_prnu_path, out_npy_name)
        
        print(f"Injected: {img_path.name} -> Used {camera_name}")
        success_count += 1
        
    print(f"\nBatch complete. {success_count} images successfully poisoned.")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    # python batch_inject.py image1.png image2.jpg image3.png
    if len(sys.argv) > 1:
        input_images = sys.argv[1:]
        process_batch(input_images)
    else:
        # Fallback for testing: explicitly define a list of files here
        print("No arguments provided. Please provide image paths.")
        print("Usage: python batch_inject.py <path_to_image1> <path_to_image2> ...")