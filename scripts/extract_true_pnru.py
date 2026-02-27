"""
Camera PRNU Fingerprint Extractor (True Fingerprint)
-----------------------------------------------------
For each camera in <root>/datasets/vision/<camera_x>/, loads all images
and extracts the true PRNU fingerprint using extract_multiple_aligned().

Outputs per camera:
  - <camera_x>_fingerprint.npy        : always
  - <camera_x>_fingerprint.png        : only with --visual

Results saved to <root>/datasets/prnu_fingerprints/

Usage:
    python extract_camera_fingerprint.py                # .npy only
    python extract_camera_fingerprint.py --visual       # .npy + .png
    python extract_camera_fingerprint.py --root /path/to/root
"""

import argparse
import numpy as np
import prnu
from PIL import Image
from pathlib import Path

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
CLIP_PERCENTILE = 2


def noise_to_image(noise: np.ndarray) -> Image.Image:
    """Convert float PRNU noise into 8-bit grayscale PNG."""
    gray = noise.copy() if noise.ndim == 2 else noise.mean(axis=2)
    lo, hi = np.percentile(gray, (CLIP_PERCENTILE, 100 - CLIP_PERCENTILE))
    stretched = np.clip((gray - lo) / (hi - lo + 1e-8) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(stretched, mode="L")


def extract_fingerprints(root: Path, save_visual: bool) -> None:
    vision_dir = root / "datasets" / "vision"
    out_dir    = root / "datasets" / "prnu_fingerprints"

    if not vision_dir.exists():
        raise FileNotFoundError(f"Vision dataset directory not found: {vision_dir}")

    camera_dirs = sorted([d for d in vision_dir.iterdir() if d.is_dir()])
    if not camera_dirs:
        print("No camera directories found.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Found {len(camera_dirs)} camera(s) in: {vision_dir}\n")

    for camera_dir in camera_dirs:
        npy_path = out_dir / (camera_dir.name + "_fingerprint.npy")
        vis_path = out_dir / (camera_dir.name + "_fingerprint.png")

        # Skip if already extracted
        if npy_path.exists() and (not save_visual or vis_path.exists()):
            print(f"  [{camera_dir.name}] Already extracted, skipping.")
            continue

        # Load existing .npy and just generate missing .png
        if npy_path.exists() and save_visual and not vis_path.exists():
            print(f"  [{camera_dir.name}] Generating missing visual...")
            fingerprint = np.load(npy_path)
            noise_to_image(fingerprint).save(vis_path)
            print(f"    Saved: {vis_path.relative_to(root)}")
            continue

        # Load all images for this camera
        image_paths = sorted([
            p for p in camera_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        ])

        if len(image_paths) < 2:
            print(f"  [{camera_dir.name}] Need at least 2 images for extract_multiple_aligned(), skipping.")
            continue

        print(f"  [{camera_dir.name}] Loading {len(image_paths)} image(s)...")
        imgs = []
        for img_path in image_paths:
            try:
                imgs.append(np.clip(np.array(Image.open(img_path).convert("RGB")), 0, 255).astype(np.uint8))
            except Exception as e:
                print(f"    WARNING: Could not load {img_path.name}: {e}")

        if len(imgs) < 2:
            print(f"  [{camera_dir.name}] Not enough loadable images, skipping.")
            continue

        # Check all images are the same size (required by extract_multiple_aligned)
        shapes = set(img.shape for img in imgs)
        if len(shapes) > 1:
            print(f"  [{camera_dir.name}] WARNING: Images have mixed sizes {shapes}.")
            print(f"    extract_multiple_aligned() requires identical dimensions.")
            print(f"    Skipping — consider resizing images to a common resolution first.")
            continue

        try:
            print(f"  [{camera_dir.name}] Extracting true PRNU fingerprint...")
            fingerprint = prnu.extract_multiple_aligned(
                imgs,
                processes=1,
                tqdm_str=f"  [{camera_dir.name}]"
            )

            np.save(npy_path, fingerprint)
            print(f"    Saved: {npy_path.relative_to(root)}")

            if save_visual:
                noise_to_image(fingerprint).save(vis_path)
                print(f"    Saved: {vis_path.relative_to(root)}")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [{camera_dir.name}] ERROR during extraction: {e}")

    print("\nDone.")


def main() -> None:
    default_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Extract true PRNU camera fingerprints using extract_multiple_aligned().")
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=f"Project root directory (default: {default_root})"
    )
    parser.add_argument(
        "--visual", "-v",
        action="store_true",
        help="Generate contrast-stretched PNG visualisations of the fingerprint"
    )
    args = parser.parse_args()

    print(f"Root        : {args.root}")
    print(f"Visual Mode : {'Enabled' if args.visual else 'Disabled'}\n")

    extract_fingerprints(args.root, args.visual)


if __name__ == "__main__":
    main()