import cv2
import numpy as np
import random
import sys
import shutil
from pathlib import Path
import argparse

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
PRNU_DIR = BASE_DIR / "datasets" / "prnu_fingerprints"
# Default fallback if no output is provided
DEFAULT_OUTPUT = BASE_DIR / "datasets" / "prnu_injected_scaled"

def extract_matching_prnu(prnu_array, target_h, target_w):
    """Crops the center of the PRNU array to match the native image dimensions."""
    h, w = prnu_array.shape[:2]
    
    if target_h > h or target_w > w:
        raise ValueError(f"Image ({target_h}x{target_w}) is larger than PRNU array ({h}x{w}).")
        
    start_y = h // 2 - (target_h // 2)
    start_x = w // 2 - (target_w // 2)
    
    return prnu_array[start_y:start_y+target_h, start_x:start_x+target_w]

def process_batch(image_paths, output_dir, limit=None):
    out_path = Path(output_dir)
    
    # Logic unchanged: Clean and recreate the target folder
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    
    prnu_files = list(PRNU_DIR.glob("*_fingerprint.npy"))
    if not prnu_files:
        print(f"Error: No PRNU .npy files found in {PRNU_DIR}")
        sys.exit(1)
        
    print(f"Found {len(prnu_files)} camera fingerprints. Starting batch scaled injection...")
    
    success_count = 0
    if not limit:
        limit = len(image_paths)
    
    for i in range(min(limit, len(image_paths))):
        img_path = Path(image_paths[i])
        if not img_path.exists():
            print(f"Skipping {img_path.name}: File not found.")
            continue
            
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Skipping {img_path.name}: Unreadable image format.")
            continue
            
        # 1. Native Image Prep (NO PADDING)
        img_float = img_bgr.astype(np.float32) / 255.0
        h, w = img_float.shape[:2]

        # 2. Select PRNU and crop to match native resolution
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
        # Scaling math (calculated but overwritten below as per original)
        poisoned_float = img_float * (1.0 + prnu_cropped)
        alpha = 3  
        prnu_normalised = prnu_cropped / (np.std(prnu_cropped) + 1e-8)
        
        # Additive override
        poisoned_float = img_float + (alpha * prnu_normalised) / 255.0
        
        # Save logic
        poisoned_img = (np.clip(poisoned_float, 0, 1.0) * 255).astype(np.uint8)
        # ──────────────────────────────────────
        
        base_name = img_path.stem
        camera_name = selected_prnu_path.stem.replace("_fingerprint", "")
        
        out_img_name = out_path / f"{base_name}_spoofed_by_{camera_name}.png"
        cv2.imwrite(str(out_img_name), poisoned_img)
        
        print(f"Injected: {img_path.name} -> Used {camera_name}")
        success_count += 1
        
    print(f"\nBatch complete. {success_count} images successfully poisoned.")
    print(f"Output directory: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inject hardware fingerprints (Scaled/Normalized).")
    parser.add_argument("path", help="Path to a single image file OR a directory.")
    parser.add_argument("--output", "-o", type=str, default=str(DEFAULT_OUTPUT), help="Target output directory")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images processed")
    
    args = parser.parse_args()
    
    input_path = Path(args.path)
    
    if input_path.is_file():
        images = [input_path]
    elif input_path.is_dir():
        # Standardize across all scripts: look for png and jpg
        images = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")) + list(input_path.glob("*.PNG"))
    else:
        print(f"Error: {args.path} is not a valid file or directory.")
        sys.exit(1)
        
    process_batch(images, args.output, args.limit)