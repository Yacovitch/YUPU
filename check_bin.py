import os
import sys
import glob
import numpy as np
import argparse

def pick_file(p):
    if os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, "*.bin")))
        if not files:
            raise FileNotFoundError(f"No .bin files in {p}")
        return files[0]
    return p

def analyze_bin(file_path):
    arr = np.fromfile(file_path, dtype=np.float32)
    n = arr.size
    print(f"\nFile: {file_path}")
    print(f"- float32 count: {n} (bytes: {n*4})")

    candidates = []
    for c in (4, 6):
        if n % c == 0:
            pts = arr.reshape(-1, c)
            tail_var = None
            if c > 3:
                tail = pts[:, 3:]
                tail_var = tail.var(axis=0)
            candidates.append((c, pts.shape[0], tail_var, pts[:3]))

    if not candidates:
        print("- Not divisible by 4 or 6 floats per point.")
        return

    for c, num_pts, tail_var, head in candidates:
        print(f"- Reshape {c} floats/pt → points: {num_pts}")
        if tail_var is not None:
            print(f"  tail var (cols 3..): {np.round(tail_var, 6)}")
        print(f"  sample (first 3 rows):\n{head}")

    # Heuristic suggestion
    suggestion = None
    if len(candidates) == 1:
        suggestion = candidates[0][0]
    else:
        # Prefer the layout with nontrivial tail variance
        scored = []
        for c, _, tail_var, _ in candidates:
            score = 0.0 if tail_var is None else float(np.mean(np.abs(tail_var)))
            scored.append((score, c))
        # If both divisible, pick the one with larger tail variance
        suggestion = max(scored)[1]

    print(f"=> Suggested layout: {suggestion} floats per point")

def main():
    ap = argparse.ArgumentParser(description="Check LiDAR .bin layout (4f vs 6f).")
    ap.add_argument("paths", nargs="+", help="File or directory paths to check")
    args = ap.parse_args()

    for p in args.paths:
        f = pick_file(p)
        analyze_bin(f)

if __name__ == "__main__":
    main()