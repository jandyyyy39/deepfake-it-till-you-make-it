import cv2
import numpy as np
import random
import sys
import shutil
from pathlib import Path
import argparse

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
PRNU_DIR = BASE_DIR / "datasets" / "prnu_fingerprints"
OUTPUT_DIR = BASE_DIR / "datasets" / "prnu_injected_additive"

def extract_matching_prnu(prnu_array, target_h, target_w):
    """Crops the absolute center of the massive PRNU array to match the native image dimensions."""
    h, w = prnu_array.shape[:2]
    
    if target_h > h or target_w > w:
        raise ValueError(f"Image ({target_h}x{target_w}) is larger than PRNU array ({h}x{w}). Cannot inject.")
        
    start_y = h // 2 - (target_h // 2)
    start_x = w // 2 - (target_w // 2)
    
    return prnu_array[start_y:start_y+target_h, start_x:start_x+target_w]

def process_batch(image_paths):
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load all available real PRNU files
    prnu_files = list(PRNU_DIR.rglob("*_fingerprint.npy"))
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

        # The Mathematical Injection (Spoofing) — Additive
        poisoned_float = img_float + prnu_cropped
        poisoned_img = np.clip(np.round(poisoned_float), 0, 255).astype(np.uint8)
        
        # Output Generation
        base_name = img_path.stem
        camera_name = selected_prnu_path.stem.replace("_fingerprint", "")
        
        out_img_name = OUTPUT_DIR / f"{base_name}_spoofed_by_{camera_name}.png"
        out_npy_name = OUTPUT_DIR / f"{base_name}_spoofed_by_{camera_name}.npy"
        
        # Save the poisoned image
        cv2.imwrite(str(out_img_name), poisoned_img)
        
        # Copy the exact mathematical array used so your team can verify it
        # shutil.copy(selected_prnu_path, out_npy_name)
        
        print(f"Injected: {img_path.name} -> Used {camera_name}")
        success_count += 1
        
    print(f"\nBatch complete. {success_count} images successfully poisoned.")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inject hardware fingerprints.")
    parser.add_argument("images", nargs="+", help="Paths to the images you want to spoof.")
    args = parser.parse_args()
    
    process_batch(args.images)