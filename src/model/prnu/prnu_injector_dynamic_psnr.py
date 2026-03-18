import cv2
import numpy as np
import random
import sys
import shutil
import argparse
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
PRNU_DIR = BASE_DIR / "datasets" / "prnu_fingerprints"
OUTPUT_DIR = BASE_DIR / "datasets" / "prnu_injected_dynamic"

# The Forensic Sweet Spot. Do not touch this unless D1 completely fails.
TARGET_PSNR = 45.0 

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
    
    prnu_files = list(PRNU_DIR.rglob("*_fingerprint.npy"))
    if not prnu_files:
        print(f"Error: No PRNU .npy files found in {PRNU_DIR}")
        sys.exit(1)
        
    print(f"Found {len(prnu_files)} camera fingerprints.")
    print(f"Targeting strict {TARGET_PSNR} dB PSNR. Starting batch injection...\n")
    
    success_count = 0
    
    for img_path in image_paths:
        img_path = Path(img_path)
        if not img_path.exists():
            print(f" [Skipping] {img_path.name}: File not found.")
            continue
            
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f" [Skipping] {img_path.name}: Unreadable image format.")
            continue
            
        # Image Prep
        img_cropped = crop_center(img_bgr, 512, 512)
        img_float = img_cropped.astype(np.float32)

        # PRNU Selection & Prep
        selected_prnu_path = random.choice(prnu_files)
        prnu_array = np.load(selected_prnu_path)
        prnu_cropped = crop_center(prnu_array, 512, 512)

        # Broadcasting Fix: Force to (H, W, 1) so it multiplies across BGR channels evenly
        if prnu_cropped.ndim == 2:
            prnu_cropped = np.expand_dims(prnu_cropped, axis=-1)

        # The Dynamic PSNR Injection Math
        raw_noise = img_float * prnu_cropped
        mse_raw = np.mean(raw_noise ** 2)
        
        if mse_raw <= 1e-8:
            print(f" [Warning] {selected_prnu_path.stem} or image is practically blank. Skipping.")
            continue
            
        mse_target = (255.0 ** 2) / (10 ** (TARGET_PSNR / 10.0))
        dynamic_alpha = np.sqrt(mse_target / mse_raw)
        
        poisoned_float = img_float + (dynamic_alpha * raw_noise)
        
        # Strict Rounding & 8-Bit Cast (The Quantization Fix)
        poisoned_img = np.clip(np.round(poisoned_float), 0, 255).astype(np.uint8)
        
        # Output Generation
        base_name = img_path.stem
        camera_name = selected_prnu_path.stem.replace("_fingerprint", "")
        
        out_img_name = OUTPUT_DIR / f"{base_name}_spoofed_by_{camera_name}.png"
        out_npy_name = OUTPUT_DIR / f"{base_name}_spoofed_by_{camera_name}.npy"
        
        # Save strictly as PNG to preserve high-frequency noise
        cv2.imwrite(str(out_img_name), poisoned_img)
        shutil.copy(selected_prnu_path, out_npy_name)
        
        print(f" [Success] {img_path.name} -> Poisoned with {camera_name}")
        success_count += 1
        
    print(f"\nBatch complete. {success_count} images successfully weaponized.")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inject Dynamic PSNR hardware fingerprints into images.")
    parser.add_argument("images", nargs="+", help="Paths to the images you want to spoof.")
    args = parser.parse_args()
    
    process_batch(args.images)