"""
Inter-Camera PCE Analysis (Exhaustive)
----------------------------------------
For every unique camera pair (A, B), picks one random image from each
and computes the PCE. Proves that inter-camera PCE is low, confirming
that PRNU fingerprints are unique per camera.

Output: <root>/datasets/pce_inter_camera.txt

Usage:
    python pce_inter_camera.py
    python pce_inter_camera.py --root /path/to/root
"""

import argparse
import sys
import random
import numpy as np
from itertools import combinations
from pathlib import Path
import os

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(os.path.join(os.path.dirname(__file__), "prnu-python"))

import prnu

def load_fingerprint(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def compute_pce(fp1: np.ndarray, fp2: np.ndarray) -> float:
    cc_map = prnu.crosscorr_2d(fp1.copy(), fp2.copy())
    return float(prnu.pce(cc_map)['pce'])


def analyse(root: Path, out_path: Path) -> None:
    prnu_dir = root / "datasets" / "prnu"

    if not prnu_dir.exists():
        raise FileNotFoundError(f"PRNU directory not found: {prnu_dir}")

    # Build camera -> [fingerprint paths] map
    camera_map = {
        d.name: sorted(d.glob("*.npy"))
        for d in sorted(prnu_dir.iterdir())
        if d.is_dir() and list(d.glob("*.npy"))
    }

    if len(camera_map) < 2:
        print("Need at least 2 cameras for inter-camera comparison.")
        return

    camera_pairs = list(combinations(sorted(camera_map.keys()), 2))
    print(f"Found {len(camera_map)} cameras — {len(camera_pairs)} unique pairs.\n")

    lines = []
    pce_values = []

    # Header
    lines.append("")
    lines.append("╔" + "═" * 62 + "╗")
    lines.append("║  Inter-Camera PCE Analysis (Exhaustive){:<22} ║".format(""))
    lines.append("║  Cameras : {:<51} ║".format(len(camera_map)))
    lines.append("║  Pairs   : {:<51} ║".format(len(camera_pairs)))
    lines.append("╚" + "═" * 62 + "╝")
    lines.append("")

    for cam_a, cam_b in camera_pairs:
        img_a_path = random.choice(camera_map[cam_a])
        img_b_path = random.choice(camera_map[cam_b])

        print(f"  {cam_a}  ×  {cam_b}")

        fp_a = load_fingerprint(img_a_path)
        fp_b = load_fingerprint(img_b_path)
        pce_val = compute_pce(fp_a, fp_b)
        pce_values.append(pce_val)

        lines.append("  ┌" + "─" * 60 + "┐")
        lines.append("  │  Camera A : {:<48}│".format(cam_a))
        lines.append("  │  Image  A : {:<48}│".format(img_a_path.name))
        lines.append("  │  Camera B : {:<48}│".format(cam_b))
        lines.append("  │  Image  B : {:<48}│".format(img_b_path.name))
        lines.append("  │  PCE      : {:<48}│".format(f"{pce_val:.4f}"))
        lines.append("  └" + "─" * 60 + "┘")

    # Summary
    lines.append("")
    lines.append("╔" + "═" * 62 + "╗")
    lines.append("║  Overall Summary{:<45} ║".format(""))
    lines.append("╠" + "═" * 62 + "╣")
    lines.append("║  Mean PCE : {:<50} ║".format(f"{np.mean(pce_values):.4f}"))
    lines.append("║  Max  PCE : {:<50} ║".format(f"{np.max(pce_values):.4f}"))
    lines.append("║  Min  PCE : {:<50} ║".format(f"{np.min(pce_values):.4f}"))
    lines.append("║  Std  PCE : {:<50} ║".format(f"{np.std(pce_values):.4f}"))
    lines.append("║{:<62} ║".format(""))
    lines.append("║  (Low PCE confirms fingerprints are unique per camera)║")
    lines.append("╚" + "═" * 62 + "╝")
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nResults written to: {out_path}")
    print(f"Mean PCE: {np.mean(pce_values):.4f}  |  Std: {np.std(pce_values):.4f}")
    print("(Compare against pce_intra_camera.txt — intra should be significantly higher)")


def main():
    default_root = Path(__file__).resolve().parent.parent
    out_path = default_root / "datasets" / "pce_inter_camera.txt"

    parser = argparse.ArgumentParser(description="Exhaustive inter-camera PCE comparison.")
    parser.add_argument("--root", type=Path, default=default_root,
                        help=f"Project root directory (default: {default_root})")
    args = parser.parse_args()

    print(f"Root  : {args.root}")
    print(f"Output: {out_path}\n")

    analyse(args.root, out_path)


if __name__ == "__main__":
    main()