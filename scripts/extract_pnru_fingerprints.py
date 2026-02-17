import cv2
import numpy as np
from pathlib import Path

# --- CONFIGURATION ---
TARGET_SIZE = (512, 512)
NUM_IMAGES_PER_DEVICE = 10  # Your current download limit
VISION_ROOT = Path(__file__).parent / "../datasets/vision"
OUTPUT_DIR = (Path(__file__).parent / "../datasets/fingerprint").resolve()

def extract_residual(img_path):
    """Core logic: Original - Denoised = Noise Residual."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = cv2.resize(img, TARGET_SIZE)
    
    # Use standard forensic denoising (Non-Local Means)
    denoised = cv2.fastNlMeansDenoisingColored(img.astype(np.uint8), None, 10, 10, 7, 21)
    residual = img - denoised.astype(np.float32)
    
    # Remove scene content bias
    residual -= np.mean(residual, axis=(0, 1), keepdims=True)
    return residual

def main():
    # Ensure the output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Iterate through each camera folder in vision
    # Expected: datasets/vision/D04_LG_D290/
    for device_folder in VISION_ROOT.iterdir():
        if not device_folder.is_dir():
            continue
            
        print(f"\n[PROCESSING] {device_folder.name}")
        
        # Look for images inside the device folder (or a 'flat' subfolder if you used one)
        # Using rglob to find all .jpg files recursively within the device folder
        image_files = list(device_folder.rglob("*.jpg"))[:NUM_IMAGES_PER_DEVICE]
        
        if len(image_files) < 1:
            print(f"  [SKIP] No images found in {device_folder.name}")
            continue

        residuals = []
        for i, f in enumerate(image_files):
            print(f"  ({i+1}/{len(image_files)}) Extracting: {f.name}", end='\r')
            res = extract_residual(f)
            if res is not None:
                residuals.append(res)

        if residuals:
            # Average the 10 residuals to isolate the fixed PRNU fingerprint
            fingerprint = np.mean(residuals, axis=0)
            
            # Save using the device folder name
            save_name = f"{device_folder.name}_fingerprint.npy"
            save_path = OUTPUT_DIR / save_name
            
            np.save(save_path, fingerprint)
            print(f"\n  [SUCCESS] Saved to: {save_path.name}")
        else:
            print(f"\n  [FAILED] Could not extract any residuals for {device_folder.name}")

if __name__ == "__main__":
    main()