"""SUP-05: nuScenes Cross-Dataset Evaluation.

Runs Stage 1-3 of the SLAM pipeline on all nuScenes mini scenes and
writes benchmarks/nuscenes_ape.csv with APE metrics per scene.

Usage:
    python scripts/eval_nuscenes.py --dataroot ~/data/nuscenes
    python scripts/eval_nuscenes.py --dataroot ~/data/nuscenes --max-frames 20
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

# Ensure project root is on path when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.nuscenes_loader import NuScenesDataset
from src.odometry.kiss_icp_wrapper import KissICPOdometry, evaluate_odometry
from src.optimization.pose_graph import PoseGraphOptimizer


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_scene(
    nusc,
    scene: dict,
    kiss_cfg: dict,
    gtsam_cfg: dict,
    max_frames: int | None = None,
) -> dict:
    """Run Stage 1-3 on a single nuScenes scene.

    Args:
        nusc: Initialised NuScenes instance.
        scene: nuScenes scene record dict.
        kiss_cfg: kiss_icp config section from default.yaml.
        gtsam_cfg: gtsam config section from default.yaml.
        max_frames: Optional frame cap for quick smoke tests.

    Returns:
        Dict with scene_token, scene_name, n_frames, stage2/3 APE metrics.
    """
    scene_token = scene["token"]
    scene_name = scene["name"]

    # --- Stage 1: load dataset (sweep mode ~20 Hz for reliable ICP registration) ---
    # nuScenes keyframes are at 2 Hz: ~4 m/frame at city speed, too sparse for
    # KISS-ICP with 32-beam LiDAR.  Sweeps at ~20 Hz give ~0.4 m/frame.
    dataset = NuScenesDataset(nusc, scene_token)
    n_frames = len(dataset)
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)
        # Slice internal list so KissICPOdometry sees the capped length.
        dataset._filepaths = dataset._filepaths[:n_frames]
        dataset.poses = dataset.poses[:n_frames]
        dataset.timestamps = dataset.timestamps[:n_frames]

    print(f"  Scene {scene_name}: {n_frames} sweeps")

    gt_poses = [dataset.poses[i] for i in range(n_frames)]

    # --- Stage 2: KISS-ICP odometry ---
    # Use smaller voxel_size (0.5 vs 1.0) and min_range (3.0 vs 5.0) for
    # nuScenes 32-beam LiDAR: fewer points require finer voxels to preserve
    # enough geometric features for reliable ICP registration.
    odom = KissICPOdometry(
        max_range=kiss_cfg.get("max_range", 100.0),
        min_range=3.0,
        voxel_size=0.5,
    )
    odom_poses = odom.run(dataset)

    n_eval = min(len(odom_poses), len(gt_poses))
    s2_metrics = evaluate_odometry(odom_poses[:n_eval], gt_poses[:n_eval])
    s2_ape_mean = float(s2_metrics["ape"]["mean"])
    s2_ape_rmse = float(s2_metrics["ape"]["rmse"])

    # --- Stage 3: pose graph optimisation (no loop closure for short scenes) ---
    optimizer = PoseGraphOptimizer(
        odom_sigmas=gtsam_cfg.get("odom_sigmas"),
        prior_sigmas=gtsam_cfg.get("prior_sigmas"),
    )
    optimizer.build_graph(odom_poses)
    opt_poses = optimizer.optimize()

    n_eval_opt = min(len(opt_poses), len(gt_poses))
    s3_metrics = evaluate_odometry(opt_poses[:n_eval_opt], gt_poses[:n_eval_opt])
    s3_ape_mean = float(s3_metrics["ape"]["mean"])
    s3_ape_rmse = float(s3_metrics["ape"]["rmse"])

    return {
        "scene_token": scene_token,
        "scene_name": scene_name,
        "n_frames": n_frames,
        "stage2_ape_mean": s2_ape_mean,
        "stage2_ape_rmse": s2_ape_rmse,
        "stage3_ape_mean": s3_ape_mean,
        "stage3_ape_rmse": s3_ape_rmse,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="SUP-05 nuScenes evaluation")
    parser.add_argument(
        "--dataroot",
        default="~/data/nuscenes",
        help="Path to nuScenes mini dataset root",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Pipeline config YAML",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Cap frames per scene (for quick smoke tests)",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/nuscenes_ape.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError:
        print("ERROR: nuscenes-devkit not installed. Run: pip install nuscenes-devkit")
        sys.exit(1)

    cfg = load_config(args.config)
    kiss_cfg = cfg.get("kiss_icp", {})
    gtsam_cfg = cfg.get("gtsam", {})

    dataroot = str(Path(args.dataroot).expanduser())
    print(f"Loading nuScenes mini from {dataroot} ...")
    nusc = NuScenes(version="v1.0-mini", dataroot=dataroot, verbose=False)
    print(f"Found {len(nusc.scene)} scenes\n")

    rows: list[dict] = []
    for i, scene in enumerate(nusc.scene):
        print(f"[{i + 1}/{len(nusc.scene)}] Processing {scene['name']} ...")
        try:
            row = run_scene(nusc, scene, kiss_cfg, gtsam_cfg, args.max_frames)
            rows.append(row)
            print(
                f"  Stage2 APE mean={row['stage2_ape_mean']:.3f}m  "
                f"Stage3 APE mean={row['stage3_ape_mean']:.3f}m"
            )
        except Exception as e:
            print(f"  WARNING: scene {scene['name']} failed: {e}")
            continue

    # Write CSV
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scene_token",
        "scene_name",
        "n_frames",
        "stage2_ape_mean",
        "stage2_ape_rmse",
        "stage3_ape_mean",
        "stage3_ape_rmse",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_path}")

    # Verify acceptance criteria
    if rows:
        tokens = [r["scene_token"] for r in rows]
        unique_tokens = len(set(tokens)) == len(tokens)
        ape_ok = all(r["stage2_ape_mean"] < 10.0 for r in rows)
        print("\n=== Acceptance check ===")
        print(f"  Scenes: {len(rows)} >= 5: {'PASS' if len(rows) >= 5 else 'FAIL'}")
        print(f"  Unique tokens: {'PASS' if unique_tokens else 'FAIL'}")
        print(f"  All Stage2 APE < 10m: {'PASS' if ape_ok else 'FAIL'}")
        if not ape_ok:
            for r in rows:
                if r["stage2_ape_mean"] >= 10.0:
                    print(f"    FAIL: {r['scene_name']} APE={r['stage2_ape_mean']:.3f}m")

    # Print summary table
    print("\nScene summary:")
    print(f"{'Scene':<45} {'Frames':>6} {'S2 APE mean':>12} {'S3 APE mean':>12}")
    print("-" * 78)
    for r in rows:
        print(
            f"{r['scene_name']:<45} {r['n_frames']:>6} "
            f"{r['stage2_ape_mean']:>11.3f}m {r['stage3_ape_mean']:>11.3f}m"
        )


if __name__ == "__main__":
    main()
