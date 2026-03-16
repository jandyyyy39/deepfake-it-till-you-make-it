"""
PRNU Fingerprint Comparator
----------------------------
Takes two PRNU fingerprints (.npy files) as input and computes
multiple distance/similarity metrics between them.

Standard PRNU literature uses PCE as the primary metric, but we
also compute NCC and cosine similarity for a fuller picture.

Usage:
    python compare_prnu.py <fingerprint_1.npy> <fingerprint_2.npy>

Examples:
    # Same camera (expect low distance / high similarity)
    python compare_prnu.py datasets/prnu/camera_1/img1_prnu.npy datasets/prnu/camera_1/img2_prnu.npy

    # Different cameras (expect high distance / low similarity)
    python compare_prnu.py datasets/prnu/camera_1/img1_prnu.npy datasets/prnu/camera_2/img1_prnu.npy
"""

import argparse
import sys
import numpy as np
from pathlib import Path
import os

# Allow running from <root>/script/ with local prnu package
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.append(os.path.join(os.path.dirname(__file__), "prnu-python"))
import prnu

def load_fingerprint(path: Path) -> np.ndarray:
    fp = np.load(path)
    if fp.ndim != 2:
        raise ValueError(f"Expected a 2D fingerprint array, got shape {fp.shape} in {path}")
    return fp.astype(np.float32)


def compare(fp1: np.ndarray, fp2: np.ndarray) -> dict:
    """
    Compute all relevant distance/similarity metrics between two PRNU fingerprints.

    Metrics:
      PCE   - Peak to Correlation Energy. The gold standard for PRNU matching.
              Higher = more likely same camera. Typically >60 is considered a match.

      NCC   - Normalized Cross-Correlation [-1, 1].
              Closer to 1 = more similar. Rotation/scale invariant.

      CC    - Raw cross-correlation value at the peak.

      Cosine similarity [-1, 1].
              Treats the 2D fingerprint as a flat vector.
              Closer to 1 = more similar.

      Euclidean distance.
              Straight L2 norm of the difference. Lower = more similar.
              Sensitive to fingerprint magnitude, so less reliable alone.
    """

    # --- PCE and CC via 2D cross-correlation (primary PRNU metric) ---
    cc_map = prnu.crosscorr_2d(fp1.copy(), fp2.copy())
    pce_result = prnu.pce(cc_map)

    # --- NCC via aligned_cc (treats fingerprints as flat vectors) ---
    ncc_result = prnu.aligned_cc(fp1[np.newaxis], fp2[np.newaxis])

    # --- Cosine similarity (manual, same idea as NCC but explicit) ---
    flat1 = fp1.flatten()
    flat2 = fp2.flatten()
    cosine_sim = np.dot(flat1, flat2) / (np.linalg.norm(flat1) * np.linalg.norm(flat2) + 1e-8)

    # --- Euclidean distance ---
    euclidean = np.linalg.norm(flat1 - flat2)

    return {
        'pce':        float(pce_result['pce']),
        'cc':         float(pce_result['cc']),
        'ncc':        float(ncc_result['ncc'][0, 0]),
        'cosine_sim': float(cosine_sim),
        'euclidean':  float(euclidean),
    }


def interpret(metrics: dict) -> str:
    """
    Provide a plain-English interpretation based on the PCE score,
    which is the most reliable single indicator.
    """
    pce = metrics['pce']
    if pce > 60:
        return "LIKELY SAME CAMERA  ✓  (PCE > 60)"
    elif pce > 20:
        return "UNCERTAIN           ?  (PCE 20–60, inconclusive)"
    else:
        return "LIKELY DIFFERENT CAMERA  ✗  (PCE < 20)"


def main():
    parser = argparse.ArgumentParser(description="Compare two PRNU fingerprints.")
    parser.add_argument("fingerprint_1", type=Path, help="Path to first .npy PRNU fingerprint")
    parser.add_argument("fingerprint_2", type=Path, help="Path to second .npy PRNU fingerprint")
    args = parser.parse_args()

    for p in [args.fingerprint_1, args.fingerprint_2]:
        if not p.exists():
            print(f"ERROR: File not found: {p}")
            sys.exit(1)

    print(f"\nFingerprint 1 : {args.fingerprint_1}")
    print(f"Fingerprint 2 : {args.fingerprint_2}\n")

    fp1 = load_fingerprint(args.fingerprint_1)
    fp2 = load_fingerprint(args.fingerprint_2)

    if fp1.shape != fp2.shape:
        print(f"WARNING: Fingerprints have different shapes ({fp1.shape} vs {fp2.shape}).")
        print("They will be padded to the same size by crosscorr_2d, but this may affect accuracy.")
        print("For best results, compare fingerprints from images of the same resolution.\n")

    metrics = compare(fp1, fp2)

    print("=" * 45)
    print(f"  PCE              : {metrics['pce']:>12.4f}   (higher = more likely same camera)")
    print(f"  CC               : {metrics['cc']:>12.4f}   (raw cross-correlation peak)")
    print(f"  NCC              : {metrics['ncc']:>12.6f}   (normalised, range -1 to 1)")
    print(f"  Cosine Similarity: {metrics['cosine_sim']:>12.6f}   (range -1 to 1)")
    print(f"  Euclidean Dist.  : {metrics['euclidean']:>12.4f}   (lower = more similar)")
    print("=" * 45)
    print(f"\n  Verdict: {interpret(metrics)}\n")


if __name__ == "__main__":
    main()