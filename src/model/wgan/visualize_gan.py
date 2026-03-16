import torch
import numpy as np
import cv2
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model.prnu_generator import PRNUGenerator

MODEL_WEIGHTS = "synthetic_prnu_generator.pth"
LATENT_DIM = 128
OUTPUT_DIR = Path("../datasets/failed_gan_output")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading failed generator weights from {MODEL_WEIGHTS}...")
    
    gen = PRNUGenerator(latent_dim=LATENT_DIM).to(device)
    try:
        gen.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return
    
    gen.eval()
    
    z = torch.randn(1, LATENT_DIM, device=device)
    
    with torch.no_grad():
        fake_prnu_tensor = gen(z)

    fake_prnu_raw = fake_prnu_tensor.squeeze().cpu().numpy()
    fake_prnu_raw = np.transpose(fake_prnu_raw, (1, 2, 0))
    
    print("\n--- SYNTHETIC OUTPUT STATISTICS ---")
    print(f"Min value:  {fake_prnu_raw.min():.6f}")
    print(f"Max value:  {fake_prnu_raw.max():.6f}")
    print(f"Mean value: {fake_prnu_raw.mean():.6f}")
    print(f"Std Dev:    {fake_prnu_raw.std():.6f}")

    # --- NEW: SAVE THE RAW MATH ---
    npy_save_path = OUTPUT_DIR / "failed_synthetic_prnu.npy"
    np.save(npy_save_path, fake_prnu_raw)
    print(f"\nSaved raw math evidence to: {npy_save_path}")

    # --- SAVE THE VISUAL ---
    visual_fp = ((fake_prnu_raw + 1.0) / 2.0) * 255.0
    visual_fp = visual_fp.astype(np.uint8)
    
    if visual_fp.shape[2] == 3:
        visual_fp = cv2.cvtColor(visual_fp, cv2.COLOR_RGB2BGR)
        
    img_save_path = OUTPUT_DIR / "failed_synthetic_prnu.png"
    cv2.imwrite(str(img_save_path), visual_fp)
    
    print(f"Saved visual evidence to:   {img_save_path}")
    print("\nSend the PNG, the NPY, and these terminal stats to the team.")

if __name__ == "__main__":
    main()