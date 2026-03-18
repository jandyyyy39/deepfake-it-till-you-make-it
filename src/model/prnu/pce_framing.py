import numpy as np
from PIL import Image
import sys
import os
from pathlib import Path

# Bring in the library you already have
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'scripts', 'prnu-python')))
import prnu
import cv2

def crop_center_2d(array, cropx=512, cropy=512):
    """Strictly crops the center of a 2D array to match the injection geometry."""
    y, x = array.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return array[starty:starty+cropy, startx:startx+cropx]

def calculate_framing_success(spoofed_image_path: str, target_fingerprint_path: str) -> float:
    try:
        # 1. Load the 512x512 spoofed image
        img = cv2.imread(spoofed_image_path, 0) # Read as grayscale for noise extraction
        if img is None:
            return float('nan')
            
        # Extract the noise residual (W) from the spoofed image
        W = prnu.extract_single(img)
        
        # 2. Load the massive, full-resolution hardware fingerprint (K)
        K_full = np.load(target_fingerprint_path)
        
        # 3. Apply the exact same center crop used during the injection phase
        K_cropped = crop_center_2d(K_full, 512, 512)
        
        # 4. Pure Mathematical Cross-Correlation
        cc2d = prnu.crosscorr_2d(W, K_cropped)
        pce_dict = prnu.pce(cc2d)
        
        return pce_dict['pce']
        
    except Exception as e:
        print(f"Error calculating PCE for {Path(spoofed_image_path).name}: {e}")
        return float('nan')

# --- Example Usage ---
# stats = calculate_framing_success(
#     "poisoned_fakes/my_fake_D23.jpg", 
#     "datasets/prnu_fingerprints/D23_Asus_Zenfone2Laser_fingerprint.npy"
# )