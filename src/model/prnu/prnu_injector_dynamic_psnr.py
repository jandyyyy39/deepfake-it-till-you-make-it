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
# OUTPUT_DIR = BASE_DIR / "datasets" / "prnu_injected_dynamic"
OUTPUT_DIR = BASE_DIR / "datasets" / "prnu_injected_train_vaccine"

# The Forensic Sweet Spot
TARGET_PSNR = 45.0 

def extract_matching_prnu(prnu_array, target_h, target_w):
    """Crops the absolute center of the massive PRNU array to match the image dimensions."""
    h, w = prnu_array.shape[:2]
    
    # If the image is somehow larger than the sensor (rare), we must abort
    if target_h > h or target_w > w:
        raise ValueError(f"Image ({target_h}x{target_w}) is larger than PRNU array ({h}x{w}). Cannot inject.")
        
    start_y = h // 2 - (target_h // 2)
    start_x = w // 2 - (target_w // 2)
    
    return prnu_array[start_y:start_y+target_h, start_x:start_x+target_w]


def process_batch(image_paths, target_psnr=TARGET_PSNR, limit=None):
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    prnu_files = list(PRNU_DIR.rglob("*_fingerprint.npy"))
    if not prnu_files:
        print(f"Error: No PRNU .npy files found in {PRNU_DIR}")
        sys.exit(1)
        
    print(f"Found {len(prnu_files)} camera fingerprints.")
    print(f"Targeting strict {TARGET_PSNR} dB PSNR. Starting batch injection...\n")
    
    success_count = 0

    if not limit:
        limit = len(image_paths)
    
    for i in range(min(limit, len(image_paths))):
        img_path = Path(image_paths[i])
        
        # 2. Check if the file actually exists before trying to read it
        if not img_path.exists():
            print(f" [Skipping] {img_path.name}: File not found.")
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f" [Skipping] {img_path.name}: Unreadable image format.")
            continue
            
        # 1. Image Prep (We DO NOT force 512x512 anymore. We respect native resolution.)
        img_float = img_bgr.astype(np.float32)
        h, w = img_float.shape[:2]

        # 2. PRNU Selection & Prep
        selected_prnu_path = random.choice(prnu_files)
        prnu_array = np.load(selected_prnu_path)
        
        # 3. Crop the PRNU to seamlessly fit the native image
        try:
            prnu_cropped = extract_matching_prnu(prnu_array, h, w)
        except ValueError as e:
            print(f" [Skipping] {img_path.name}: {e}")
            continue

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
        
        # Save strictly as PNG to preserve high-frequency noise
        cv2.imwrite(str(out_img_name), poisoned_img)
        
        print(f" [Success] {img_path.name} -> Poisoned with {camera_name}")
        success_count += 1
        
    print(f"\nBatch complete. {success_count} images successfully weaponized.")
    print(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch inject Dynamic PSNR hardware fingerprints into images.")
    parser.add_argument("--psnr", type=float, default=TARGET_PSNR, help="Target PSNR value for the injected noise (default: 28.0 dB)")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on the number of images to process from the provided list.")
    parser.add_argument("images", nargs="+", help="Paths to the images you want to spoof.")
    args = parser.parse_args()
    img_dir = Path(args.images[0])
    images = list(img_dir.glob("*.png"))
    
    process_batch(images, args.psnr, args.limit)