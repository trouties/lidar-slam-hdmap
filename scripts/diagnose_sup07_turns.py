#!/usr/bin/env python3
"""SUP-07 diagnostic: are Seq 00 cond spikes confounded by sharp turns?

Hypothesis (from PR review): scan-to-scan NN correspondence on turn frames
is biased — the rotation between consecutive frames causes normal
distributions to skew, which inflates cond_number without the scene being
actually degenerate.

Test: correlate per-frame heading delta (yaw between consecutive poses)
with cond_number. If the top-cond frames cluster at high yaw delta, the
confounder is real.

Usage::

    python -m scripts.diagnose_sup07_turns --sequence 00
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_pipeline import load_config  # noqa: E402
from src.cache import LayeredCache  # noqa: E402


def _yaw_from_rotation(R: np.ndarray) -> float:
    """Extract yaw (rotation around z) from a 3x3 rotation matrix."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _load_cond(csv_path: Path) -> np.ndarray:
    cond = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = float(row["cond_number"])
            cond.append(v)
    return np.asarray(cond, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--threshold", type=float, default=5.514)
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Path to degeneracy CSV; defaults to benchmarks/sup07/degeneracy_{seq}_baseline.csv",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["sequence"] = args.sequence
    cache = LayeredCache(cfg.get("cache", {}).get("root", "cache/kitti"), args.sequence)
    loaded = cache.load_odometry(cfg)
    if loaded is None:
        sys.exit(f"No odometry cache for seq {args.sequence}")
    poses_arr, _ = loaded
    n = poses_arr.shape[0]

    # Yaw delta between consecutive poses
    yaw = np.array([_yaw_from_rotation(poses_arr[i, :3, :3]) for i in range(n)])
    dyaw = np.zeros(n)
    for i in range(1, n):
        dyaw[i] = abs(_wrap_to_pi(yaw[i] - yaw[i - 1]))

    # Load cond
    csv_path = (
        Path(args.csv_path)
        if args.csv_path
        else Path(f"benchmarks/sup07/degeneracy_{args.sequence}_baseline.csv")
    )
    cond = _load_cond(csv_path)
    if cond.shape[0] != n:
        # Trim to the shorter length (CSV may have 1 extra null row from frame 0)
        m = min(n, cond.shape[0])
        cond = cond[:m]
        dyaw = dyaw[:m]
        n = m

    finite = np.isfinite(cond)
    cond_f = cond[finite]
    dyaw_f = dyaw[finite]

    print(f"Seq {args.sequence}: {cond_f.size} frames with finite cond")
    print()

    # Correlation between yaw delta and cond
    corr = float(np.corrcoef(dyaw_f, cond_f)[0, 1])
    print(f"Pearson correlation(|dyaw|, cond) = {corr:+.4f}")
    print()

    # Turn detector: top 5% of yaw delta
    turn_bar = float(np.percentile(dyaw_f, 95))
    turn_mask = dyaw_f > turn_bar
    print(f"Turn frames (|dyaw| > p95 = {np.degrees(turn_bar):.2f}°/frame): {int(turn_mask.sum())}")
    print(f"  cond p50 on turns:   {np.percentile(cond_f[turn_mask], 50):.3f}")
    print(f"  cond p95 on turns:   {np.percentile(cond_f[turn_mask], 95):.3f}")
    print(f"  cond p50 off turns:  {np.percentile(cond_f[~turn_mask], 50):.3f}")
    print(f"  cond p95 off turns:  {np.percentile(cond_f[~turn_mask], 95):.3f}")
    print()

    # Above-threshold subset
    above = cond_f > args.threshold
    print(f"Frames above cond threshold {args.threshold}: {int(above.sum())} / {cond_f.size}")
    if above.any():
        above_yaw = dyaw_f[above]
        print(f"  |dyaw| p50 on above:  {np.degrees(np.percentile(above_yaw, 50)):.3f}°/frame")
        print(f"  |dyaw| p95 on above:  {np.degrees(np.percentile(above_yaw, 95)):.3f}°/frame")
        print(f"  mean |dyaw| all:      {np.degrees(dyaw_f.mean()):.3f}°/frame")
        # Fraction of above-threshold that are also turns
        turn_and_above = above & turn_mask
        print(
            f"  above ∩ top-5% turn:  {int(turn_and_above.sum())}/{int(above.sum())} "
            f"= {100 * turn_and_above.sum() / max(above.sum(), 1):.1f}%"
        )
        # Expected: 5% if independent
        expected_pct = 100 * turn_mask.mean()
        print(f"  (expected by chance: {expected_pct:.1f}%)")

    print()
    print("Interpretation:")
    print("  If above∩turn fraction >> expected, turns are a confounder.")
    print("  If correlation > 0.3, there's a linear relationship.")

    # Clustering analysis: where do above-threshold frames sit?
    # If they cluster into a handful of long runs, detection is real.
    print()
    print("Above-threshold frame clustering:")
    above_idx = np.where(above)[0]
    if above_idx.size > 0:
        runs = []
        start = above_idx[0]
        for i in range(1, len(above_idx)):
            if above_idx[i] - above_idx[i - 1] > 3:
                runs.append((start, above_idx[i - 1]))
                start = above_idx[i]
        runs.append((start, above_idx[-1]))
        runs = [(s, e) for s, e in runs if e - s >= 1]
        print(f"  {len(runs)} runs (merged with gap<=3):")
        for rs, re in sorted(runs, key=lambda r: r[0] - r[1])[:15]:
            length = re - rs + 1
            avg_cond = float(cond_f[rs : re + 1].mean())
            print(f"    frames [{rs:4d}..{re:4d}] len={length:3d} mean_cond={avg_cond:.2f}")
        lens = np.array([e - s + 1 for s, e in runs])
        p50 = np.percentile(lens, 50)
        p95 = np.percentile(lens, 95)
        print(f"  run-length p50={p50:.0f} p95={p95:.0f} max={lens.max()}")
        sustained = int((lens >= 5).sum())
        print(f"  runs >= 5 frames: {sustained}/{len(runs)}")


if __name__ == "__main__":
    main()
