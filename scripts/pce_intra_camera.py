"""
Intra-Camera PCE Analysis
--------------------------
For each camera folder in <root>/datasets/prnu/, computes the PCE between
every pair of PRNU fingerprints within that camera and writes results to
<root>/datasets/pce_intra_camera.txt.

Usage:
    python pce_intra_camera.py                  # auto-detects root from script location
    python pce_intra_camera.py --root /path/to/root
"""

import argparse
import sys
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

    camera_dirs = sorted([d for d in prnu_dir.iterdir() if d.is_dir()])
    if not camera_dirs:
        print("No camera directories found.")
        return

    lines = []

    for camera_dir in camera_dirs:
        fingerprints = sorted(camera_dir.glob("*.npy"))

        if len(fingerprints) < 2:
            print(f"  [{camera_dir.name}] Fewer than 2 fingerprints, skipping.")
            continue

        pairs = list(combinations(fingerprints, 2))

        # ── Camera header ──────────────────────────────────────────────────
        lines.append("")
        lines.append("╔" + "═" * 62 + "╗")
        lines.append("║  Camera: {:<52} ║".format(camera_dir.name))
        lines.append("║  Pairs : {:<52} ║".format(f"{len(pairs)} comparisons ({len(fingerprints)} fingerprints)"))
        lines.append("╚" + "═" * 62 + "╝")

        pce_values = []

        print(f"  [{camera_dir.name}] Computing {len(pairs)} pair(s)...")

        for fp1_path, fp2_path in pairs:
            fp1 = load_fingerprint(fp1_path)
            fp2 = load_fingerprint(fp2_path)
            pce_val = compute_pce(fp1, fp2)
            pce_values.append(pce_val)

            # ── Pair block ─────────────────────────────────────────────────
            lines.append("  ┌" + "─" * 60 + "┐")
            lines.append("  │  Image 1 : {:<49}│".format(fp1_path.name))
            lines.append("  │  Image 2 : {:<49}│".format(fp2_path.name))
            lines.append("  │  PCE     : {:<49}│".format(f"{pce_val:.4f}"))
            lines.append("  └" + "─" * 60 + "┘")

        # ── Camera summary ─────────────────────────────────────────────────
        lines.append("")
        lines.append("  Summary:")
        lines.append("    Mean PCE : {:.4f}".format(np.mean(pce_values)))
        lines.append("    Max  PCE : {:.4f}".format(np.max(pce_values)))
        lines.append("    Min  PCE : {:.4f}".format(np.min(pce_values)))
        lines.append("    Std  PCE : {:.4f}".format(np.std(pce_values)))
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResults written to: {out_path}")


def main():
    default_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(description="Compute intra-camera PCE for all PRNU fingerprint pairs.")
    parser.add_argument("--root", type=Path, default=default_root,
                        help=f"Project root directory (default: {default_root})")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output .txt path (default: <root>/datasets/pce_intra_camera.txt)")
    args = parser.parse_args()

    out_path = args.out or (args.root / "datasets" / "pce_intra_camera.txt")

    print(f"Root : {args.root}")
    print(f"Output: {out_path}\n")
    analyse(args.root, out_path)


if __name__ == "__main__":
    main()