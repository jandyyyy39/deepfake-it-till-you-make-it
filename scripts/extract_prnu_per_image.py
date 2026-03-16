"""
PRNU Fingerprint Extractor
--------------------------
Walks through <root>/datasets/vision/<camera_x>/ directories,
extracts the PRNU noise residual from each image, and saves:
  - <image>_prnu.npy   : raw float32 noise residual
  - <image>_prnu.png   : (Optional) contrast-stretched grayscale visualisation

Usage:
    python extract_prnu.py                  # Saves only .npy
    python extract_prnu.py --visual         # Saves .npy AND .png
"""

import argparse
import numpy as np
import sys
sys.path.append("scripts/prnu-python")
import prnu
from PIL import Image
from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
CLIP_PERCENTILE = 2

def noise_to_image(noise: np.ndarray) -> Image.Image:
    """Convert float PRNU noise into 8-bit grayscale PNG."""
    if noise.ndim == 3:
        gray = noise.mean(axis=2)
    else:
        gray = noise.copy()

    lo, hi = np.percentile(gray, (CLIP_PERCENTILE, 100 - CLIP_PERCENTILE))
    stretched = np.clip((gray - lo) / (hi - lo + 1e-8) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(stretched, mode="L")

def extract_and_save(root: Path, save_visual: bool) -> None:
    vision_dir = root / "datasets" / "vision"
    prnu_dir   = root / "datasets" / "prnu"

    if not vision_dir.exists():
        raise FileNotFoundError(f"Vision dataset directory not found: {vision_dir}")

    camera_dirs = sorted([d for d in vision_dir.iterdir() if d.is_dir()])
    if not camera_dirs:
        print("No camera directories found.")
        return

    print(f"Found {len(camera_dirs)} camera(s) in: {vision_dir}\n")

    for camera_dir in camera_dirs:
        image_paths = sorted([
            p for p in camera_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ])

        if not image_paths:
            print(f"  [{camera_dir.name}] No supported images found, skipping.")
            continue

        out_dir = prnu_dir / camera_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [{camera_dir.name}] Processing {len(image_paths)} image(s)...")

        for img_path in image_paths:
            npy_path = out_dir / (img_path.stem + "_prnu.npy")
            vis_path = out_dir / (img_path.stem + "_prnu.png")

            # Condition: Skip only if .npy exists (and if --visual is on, .png must exist too)
            if npy_path.exists() and (not save_visual or vis_path.exists()):
                print(f"    Skipping {img_path.name} (already exists)")
                continue

            try:
                img = np.array(Image.open(img_path).convert("RGB"))
                noise = prnu.extract_single(img)

                # Always save raw math
                np.save(npy_path, noise)
                print(f"    Saved: {npy_path.relative_to(root)}")

                # Save visual only if flag is passed
                if save_visual:
                    noise_to_image(noise).save(vis_path)
                    print(f"    Saved: {vis_path.relative_to(root)}")

            except Exception as e:
                print(f"    ERROR processing {img_path.name}: {e}")

    print("\nDone.")

def main() -> None:
    default_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Extract PRNU fingerprints from camera image sets.")
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=f"Project root directory (default: {default_root})"
    )
    # Added Boolean flag
    parser.add_argument(
        "--visual", "-v",
        action="store_true",
        help="Generate contrast-stretched PNG visualisations of the noise"
    )
    
    args = parser.parse_args()

    print(f"Root: {args.root}")
    print(f"Visual Mode: {'Enabled' if args.visual else 'Disabled'}\n")
    
    extract_and_save(args.root, args.visual)

if __name__ == "__main__":
    main()