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
# Default fallback if no output is provided
DEFAULT_OUTPUT = BASE_DIR / "datasets" / "prnu_injected_dynamic"

# The Forensic Sweet Spot
TARGET_PSNR = 45.0 

def extract_matching_prnu(prnu_array, target_h, target_w):
    """Crops the absolute center of the massive PRNU array to match the image dimensions."""
    h, w = prnu_array.shape[:2]
    
    if target_h > h or target_w > w:
        raise ValueError(f"Image ({target_h}x{target_w}) is larger than PRNU array ({h}x{w}). Cannot inject.")
        
    start_y = h // 2 - (target_h // 2)
    start_x = w // 2 - (target_w // 2)
    
    return prnu_array[start_y:start_y+target_h, start_x:start_x+target_w]


def process_batch(image_paths, output_dir, target_psnr=TARGET_PSNR, limit=None):
    out_path = Path(output_dir)
    
    # Logic unchanged: Clean and recreate the target folder
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    
    prnu_files = list(PRNU_DIR.glob("*_fingerprint.npy"))
    if not prnu_files:
        print(f"Error: No PRNU .npy files found in {PRNU_DIR}")
        sys.exit(1)
        
    print(f"Found {len(prnu_files)} camera fingerprints.")
    print(f"Targeting strict {target_psnr} dB PSNR. Starting batch injection...\n")
    
    success_count = 0

    if not limit:
        limit = len(image_paths)
    
    for i in range(min(limit, len(image_paths))):
        img_path = Path(image_paths[i])
        
        if not img_path.exists():
            print(f" [Skipping] {img_path.name}: File not found.")
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f" [Skipping] {img_path.name}: Unreadable image format.")
            continue
            
        img_float = img_bgr.astype(np.float32)
        h, w = img_float.shape[:2]

        selected_prnu_path = random.choice(prnu_files)
        prnu_array = np.load(selected_prnu_path)
        
        try:
            prnu_cropped = extract_matching_prnu(prnu_array, h, w)
        except ValueError as e:
            print(f" [Skipping] {img_path.name}: {e}")
            continue

        if prnu_cropped.ndim == 2:
            prnu_cropped = np.expand_dims(prnu_cropped, axis=-1)

        # ── CORE INJECTION MATH (UNTOUCHED) ──
        raw_noise = img_float * prnu_cropped
        mse_raw = np.mean(raw_noise ** 2)
        
        if mse_raw <= 1e-8:
            print(f" [Warning] {selected_prnu_path.stem} or image is practically blank. Skipping.")
            continue
            
        # PSNR Scaling Math
        mse_target = (255.0 ** 2) / (10 ** (target_psnr / 10.0))
        dynamic_alpha = np.sqrt(mse_target / mse_raw)
        
        poisoned_float = img_float + (dynamic_alpha * raw_noise)
        
        # Quantization Logic
        poisoned_img = np.clip(np.round(poisoned_float), 0, 255).astype(np.uint8)
        # ──────────────────────────────────────
        
        base_name = img_path.stem
        camera_name = selected_prnu_path.stem.replace("_fingerprint", "")
        
        out_img_name = out_path / f"{base_name}_spoofed_by_{camera_name}.png"
        
        cv2.imwrite(str(out_img_name), poisoned_img)
        
        print(f" [Success] {img_path.name} -> Poisoned with {camera_name}")
        success_count += 1
        
    print(f"\nBatch complete. {success_count} images successfully weaponized.")
    print(f"Output directory: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inject Dynamic PSNR hardware fingerprints.")
    parser.add_argument("path", help="Path to a single image file OR a directory.")
    parser.add_argument("--output", "-o", type=str, default=str(DEFAULT_OUTPUT), help="Target output directory")
    parser.add_argument("--psnr", type=float, default=TARGET_PSNR, help="Target PSNR (default: 45.0 dB)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images")
    
    args = parser.parse_args()
    
    input_path = Path(args.path)
    
    if input_path.is_file():
        images = [input_path]
    elif input_path.is_dir():
        # Respecting your folder globbing
        images = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")) + list(input_path.glob("*.PNG"))
    else:
        print(f"Error: {args.path} is not a valid file or directory.")
        sys.exit(1)
        
    process_batch(images, args.output, args.psnr, args.limit)