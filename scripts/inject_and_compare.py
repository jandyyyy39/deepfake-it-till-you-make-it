import cv2
import numpy as np
import random
import sys
from pathlib import Path

# --- CONFIGURATION ---
AI_IMAGES_DIR = Path("../datasets/genimage/imagenet_ai_0424_sdv5/train/ai")
REAL_PRNU_DIR = Path("../datasets/prnu_fingerprints")
FAKE_PRNU_PATH = Path("../datasets/failed_gan_output/failed_synthetic_prnu.npy")
OUTPUT_DIR = Path("../datasets/injected")

def crop_center(img, cropx=512, cropy=512):
    """Takes a center crop of the image to perfectly match the PRNU dimensions."""
    y, x, _ = img.shape
    if y < cropy or x < cropx:
        # If the AI image is smaller than 512x512, we pad it instead of resizing
        padded = np.zeros((max(y, cropy), max(x, cropx), 3), dtype=img.dtype)
        padded[:y, :x, :] = img
        img = padded
        y, x, _ = img.shape
        
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty+cropy, startx:startx+cropx]

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Select a random GenImage SDv5 image
    ai_images = list(AI_IMAGES_DIR.rglob("*.png")) + list(AI_IMAGES_DIR.rglob("*.jpg"))
    if not ai_images:
        print(f"Error: No AI images found in {AI_IMAGES_DIR}")
        sys.exit(1)
        
    original_img_path = random.choice(ai_images)
    print(f"Selected AI Image: {original_img_path.name}")
    
    # Load and crop to strictly 512x512
    img_bgr = cv2.imread(str(original_img_path))
    img_cropped = crop_center(img_bgr)
    img_float = img_cropped.astype(np.float32)
    
    # Save the baseline crop for comparison
    cv2.imwrite(str(OUTPUT_DIR / "0_baseline_crop.png"), img_cropped)

    # 2. Inject REAL PRNU
    real_prnu_files = list(REAL_PRNU_DIR.rglob("*_fingerprint.npy"))
    if not real_prnu_files:
        print("Error: No real PRNU .npy files found.")
        sys.exit(1)
        
    real_prnu_path = random.choice(real_prnu_files)
    print(f"Selected Real PRNU: {real_prnu_path.name}")
    
    real_prnu = np.load(real_prnu_path)
    
    # If the PRNU is 2D (grayscale), expand it to 3 channels to match the image
    if real_prnu.ndim == 2:
        real_prnu = np.stack([real_prnu]*3, axis=-1)
        
    # ---> ADD THIS FIX <---
    # Crop the high-resolution PRNU array down to 512x512 to match the AI image
    real_prnu = crop_center(real_prnu)
        
    # Additive Injection: Original Image + Sensor Noise
    real_injected_float = img_float + real_prnu
    real_injected = np.clip(real_injected_float, 0, 255).astype(np.uint8)
    
    cv2.imwrite(str(OUTPUT_DIR / "1_real_prnu_injected.png"), real_injected)

    # 3. Inject FAILED GAN PRNU
    if not FAKE_PRNU_PATH.exists():
        print(f"Error: Failed GAN output not found at {FAKE_PRNU_PATH}")
        sys.exit(1)
        
    fake_prnu = np.load(FAKE_PRNU_PATH)
    if fake_prnu.ndim == 2:
        fake_prnu = np.stack([fake_prnu]*3, axis=-1)
        
    # The GAN failed and likely collapsed to max Tanh values [-1, 1].
    # To make the structural failure of the GAN visually obvious to your team, 
    # we inject it directly. If we reverse the 1000x scale we used during training, 
    # the collapsed noise will just be invisible +/- 0.001 pixel shifts.
    # This directly applies the mathematical garbage the generator thought was "correct".
    fake_injected_float = img_float + (fake_prnu * 20.0) # Multiplier makes the artifacting visible to human eyes
    fake_injected = np.clip(fake_injected_float, 0, 255).astype(np.uint8)
    
    cv2.imwrite(str(OUTPUT_DIR / "2_failed_gan_injected.png"), fake_injected)

    print("\n--- INJECTION COMPLETE ---")
    print(f"Check the {OUTPUT_DIR.name} folder.")
    print("Send the 3 images (Baseline, Real PRNU, Failed GAN) to your team.")

if __name__ == "__main__":
    main()