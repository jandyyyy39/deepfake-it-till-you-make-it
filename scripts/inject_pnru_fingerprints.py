import cv2
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
# Point this to the specific .npy file you just generated
FINGERPRINT_PATH = Path(__file__).parent / "../datasets/fingerprint/D01_Samsung_GalaxyS3Mini_fingerprint.npy"

# Point this to your GenImage sdv5 directory
GENIMAGE_ROOT = Path(__file__).parent / "../datasets/genimage/sdv5_img"
OUTPUT_DIR = (Path(__file__).parent / "../datasets/spoofed/sdv5").resolve()

# Alpha multiplier for the PRNU strength. 
# Finding the balance is key: Too low = discriminator catches it. Too high = visual artifacts.
INJECTION_STRENGTH = 0.05 

def inject_prnu(img_path, prnu_payload, output_dir):
    # 1. Load the synthetic AI image
    img = cv2.imread(str(img_path))
    if img is None:
        return False
        
    img_float = img.astype(np.float32)
    h, w = img_float.shape[:2]
    fh, fw = prnu_payload.shape
    
    # 2. Match Dimensions (Center Crop the AI image to match the 512x512 fingerprint)
    if h < fh or w < fw:
        print(f"  [SKIP] {img_path.name} is smaller than the fingerprint.")
        return False
        
    start_y = h // 2 - fh // 2
    start_x = w // 2 - fw // 2
    img_crop = img_float[start_y:start_y + fh, start_x:start_x + fw]

    # 3. Apply Multiplicative Math
    # The PRNU is 2D (grayscale). We expand it to 3D to multiply across BGR channels.
    prnu_3d = np.expand_dims(prnu_payload, axis=2)
    
    # Spoofed = Synthetic + (Synthetic * PRNU * Strength)
    spoofed_img = img_crop + (img_crop * prnu_3d * INJECTION_STRENGTH)
    
    # 4. Clip to valid 8-bit image range and convert
    spoofed_img = np.clip(spoofed_img, 0, 255).astype(np.uint8)
    
    # Save the output
    save_path = output_dir / f"spoofed_{img_path.name}"
    cv2.imwrite(str(save_path), spoofed_img)
    return True

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load the payload once
    if not FINGERPRINT_PATH.exists():
        print(f"Error: Fingerprint payload not found at {FINGERPRINT_PATH}")
        return
        
    print(f"Loading payload: {FINGERPRINT_PATH.name}")
    prnu_payload = np.load(FINGERPRINT_PATH)
    
    # Grab the synthetic images
    # Adjust the glob pattern if your images are in subfolders
    synthetic_images = list(GENIMAGE_ROOT.rglob("*.png"))[:50] # Testing on first 50
    
    if not synthetic_images:
        print(f"No synthetic images found in {GENIMAGE_ROOT}")
        return
        
    print(f"\nInjecting PRNU into {len(synthetic_images)} synthetic images...")
    
    success_count = 0
    for i, img_path in enumerate(synthetic_images):
        print(f"  ({i+1}/{len(synthetic_images)}) Spoofing: {img_path.name}", end='\r')
        if inject_prnu(img_path, prnu_payload, OUTPUT_DIR):
            success_count += 1
            
    print(f"\n\n[COMPLETE] Successfully spoofed {success_count} images.")
    print(f"Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()