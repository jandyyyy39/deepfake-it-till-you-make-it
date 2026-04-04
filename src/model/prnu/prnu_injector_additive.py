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
DEFAULT_OUTPUT = BASE_DIR / "datasets" / "prnu_injected_additive"

def extract_matching_prnu(prnu_array, target_h, target_w):
    """Crops the absolute center of the massive PRNU array to match the native image dimensions."""
    h, w = prnu_array.shape[:2]
    
    if target_h > h or target_w > w:
        raise ValueError(f"Image ({target_h}x{target_w}) is larger than PRNU array ({h}x{w}). Cannot inject.")
        
    start_y = h // 2 - (target_h // 2)
    start_x = w // 2 - (target_w // 2)
    
    return prnu_array[start_y:start_y+target_h, start_x:start_x+target_w]

def process_batch(image_paths, output_dir):
    out_path = Path(output_dir)
    
    # Logic unchanged: Clean and recreate the target folder
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Load all available real PRNU files
    prnu_files = list(PRNU_DIR.glob("*_fingerprint.npy"))
    if not prnu_files:
        print(f"Error: No PRNU .npy files found in {PRNU_DIR}")
        sys.exit(1)
        
    print(f"Found {len(prnu_files)} camera fingerprints. Starting batch additive injection...")
    
    success_count = 0
    
    for img_path_str in image_paths:
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"Skipping {img_path.name}: File not found.")
            continue
            
        # Load and prepare the Image (Native Resolution)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Skipping {img_path.name}: Unreadable image format.")
            continue
            
        img_float = img_bgr.astype(np.float32)
        h, w = img_float.shape[:2]

        # Select a random hardware fingerprint
        selected_prnu_path = random.choice(prnu_files)
        prnu_array = np.load(selected_prnu_path)

        # Prepare the PRNU (Crop to match image, expand channels)
        try:
            prnu_cropped = extract_matching_prnu(prnu_array, h, w)
        except ValueError as e:
            print(f" [Skipping] {img_path.name}: {e}")
            continue

        if prnu_cropped.ndim == 2:
            prnu_cropped = np.expand_dims(prnu_cropped, axis=-1)

        # ── CORE INJECTION MATH (UNTOUCHED) ──
        # The Mathematical Injection (Spoofing) — Additive
        poisoned_float = img_float + prnu_cropped
        poisoned_img = np.clip(np.round(poisoned_float), 0, 255).astype(np.uint8)
        # ──────────────────────────────────────
        
        # Output Generation
        base_name = img_path.stem
        camera_name = selected_prnu_path.stem.replace("_fingerprint", "")
        
        out_img_name = out_path / f"{base_name}_spoofed_by_{camera_name}.png"
        
        # Save the poisoned image
        cv2.imwrite(str(out_img_name), poisoned_img)
        
        print(f"Injected: {img_path.name} -> Used {camera_name}")
        success_count += 1
        
    print(f"\nBatch complete. {success_count} images successfully poisoned.")
    print(f"Output directory: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inject hardware fingerprints (Additive).")
    parser.add_argument("path", help="Path to a single image file OR a directory.")
    parser.add_argument("--output", "-o", type=str, default=str(DEFAULT_OUTPUT), help="Target output directory")
    
    args = parser.parse_args()
    
    input_path = Path(args.path)
    
    if input_path.is_file():
        # If it's a single file, just put it in a list
        images = [input_path]
    elif input_path.is_dir():
        # Glob all PNGs and JPGs just like the Dynamic script
        images = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")) + list(input_path.glob("*.PNG"))
    else:
        print(f"Error: {args.path} is not a valid file or directory.")
        sys.exit(1)
        
    process_batch(images, args.output)