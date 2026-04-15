#!/usr/bin/env python3
"""SUP-07 fast analyzer tuning loop.

Bypasses Stage 2 by loading cached KISS-ICP poses and re-iterating the
KITTI point cloud stream to recompute degeneracy scores with different
analyzer parameters. One full pass takes ~6 min on Seq 00 + Seq 01 (vs
~45 min for a full eval_sup07 rerun).

Usage::

    python -m scripts.tune_sup07 \
        --sequences 00,01 --voxel-size 0.3 --min-quality 0.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_pipeline import _scores_to_array, load_config  # noqa: E402
from src.cache import LayeredCache  # noqa: E402
from src.data.kitti_loader import KITTIDataset  # noqa: E402
from src.odometry.degeneracy import DegeneracyAnalyzer  # noqa: E402


def _run_seq(
    cfg: dict,
    sequence: str,
    analyzer: DegeneracyAnalyzer,
) -> np.ndarray:
    cache_root = cfg.get("cache", {}).get("root", "cache/kitti")
    cache = LayeredCache(cache_root, sequence)
    # Force-load odometry even if degeneracy sidecar is stale/missing.
    cfg_seq = dict(cfg)
    cfg_seq["data"] = {**cfg.get("data", {}), "sequence": sequence}
    odom = cache.load_odometry(cfg_seq)
    if odom is None:
        raise RuntimeError(
            f"Odometry cache miss for seq {sequence}. "
            "Run eval_sup07.py or run_pipeline.py first to populate it."
        )
    poses_arr, _ = odom
    n_frames = poses_arr.shape[0]
    print(f"  Seq {sequence}: {n_frames} cached poses")

    dataset = KITTIDataset(
        root_path=cfg_seq["data"]["kitti_root"],
        sequence=sequence,
    )
    assert len(dataset) >= n_frames

    scores: list = []
    prev_world: np.ndarray | None = None
    for idx in range(n_frames):
        xyz = dataset[idx][0][:, :3]
        pose = poses_arr[idx]
        curr_world = xyz @ pose[:3, :3].T + pose[:3, 3]
        if prev_world is None:
            from src.odometry.degeneracy import DegeneracyScore

            scores.append(DegeneracyScore.null())
        else:
            scores.append(analyzer.analyze(curr_world, prev_world))
        prev_world = curr_world
        if (idx + 1) % 500 == 0:
            print(f"    {idx + 1}/{n_frames}")
    return _scores_to_array(scores)


def _stats(arr: np.ndarray, label: str) -> dict:
    cond = arr[:, 0]
    finite = cond[np.isfinite(cond)]
    if finite.size == 0:
        return {"label": label, "count": 0}
    lmin = arr[:, 1][np.isfinite(arr[:, 1])]
    return {
        "label": label,
        "count": int(finite.size),
        "cond_p10": float(np.percentile(finite, 10)),
        "cond_p50": float(np.percentile(finite, 50)),
        "cond_p95": float(np.percentile(finite, 95)),
        "cond_p99": float(np.percentile(finite, 99)),
        "cond_max": float(finite.max()),
        "lmin_p50": float(np.percentile(lmin, 50)) if lmin.size else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sequences", default="00,01")
    parser.add_argument("--max-correspondences", type=int, default=5000)
    parser.add_argument("--normal-k", type=int, default=10)
    parser.add_argument("--max-nn-dist", type=float, default=1.0)
    parser.add_argument("--voxel-size", type=float, default=0.5)
    parser.add_argument("--min-quality", type=float, default=0.0)
    parser.add_argument("--mode", choices=("3x3", "6x6"), default="3x3")
    parser.add_argument("--length-scale", type=float, default=10.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    analyzer = DegeneracyAnalyzer(
        max_correspondences=args.max_correspondences,
        normal_k=args.normal_k,
        max_nn_dist=args.max_nn_dist,
        voxel_size=args.voxel_size,
        min_quality=args.min_quality,
        mode=args.mode,
        length_scale=args.length_scale,
    )
    print(
        f"Analyzer: mode={args.mode} max_corr={args.max_correspondences} k={args.normal_k} "
        f"max_nn={args.max_nn_dist} voxel={args.voxel_size} min_q={args.min_quality} "
        f"L={args.length_scale}"
    )
    print()

    results: dict[str, dict] = {}
    for seq in args.sequences.split(","):
        seq = seq.strip()
        arr = _run_seq(cfg, seq, analyzer)
        results[seq] = _stats(arr, f"seq{seq}")
        print(f"  Seq {seq} stats: {results[seq]}")
        print()

    print("=" * 72)
    print("Tuning report")
    print("=" * 72)
    for seq, stats in results.items():
        if stats.get("count", 0) == 0:
            continue
        print(
            f"Seq {seq}: cond p50={stats['cond_p50']:.3f} "
            f"p95={stats['cond_p95']:.3f} p99={stats['cond_p99']:.3f} "
            f"max={stats['cond_max']:.3f} lmin_p50={stats['lmin_p50']:.1f}"
        )
    if "00" in results and "01" in results:
        bar = results["00"]["cond_p95"] * 2.0
        med01 = results["01"]["cond_p50"]
        gap = med01 / bar if bar else float("nan")
        passed = med01 >= bar
        print(
            f"\nCriterion 1: Seq 01 cond_p50={med01:.3f} vs 2*Seq 00 cond_p95={bar:.3f} "
            f"-> gap={gap:.3f}×  {'PASS' if passed else 'FAIL'}"
        )


if __name__ == "__main__":
    main()
