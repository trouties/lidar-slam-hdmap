"""Main entry point for the LiDAR SLAM HD Map pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.data.kitti_loader import KITTIDataset
from src.odometry.kiss_icp_wrapper import KissICPOdometry, evaluate_odometry


def load_config(config_path: str | Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="LiDAR SLAM HD Map Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config.get("output", {}).get("dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Stage 1: Data Loading ---
    print("=== Stage 1: Loading KITTI dataset ===")
    dataset = KITTIDataset(
        root_path=config["data"]["kitti_root"],
        sequence=config["data"]["sequence"],
    )
    print(f"  Sequence: {dataset.sequence}")
    print(f"  Frames: {len(dataset)}")
    if len(dataset) == 0:
        print("  No scans found. Check data path and run scripts/verify_kitti.py")
        return

    # --- Stage 2: LiDAR Odometry ---
    print("\n=== Stage 2: KISS-ICP Odometry ===")
    kiss_cfg = config.get("kiss_icp", {})
    odom = KissICPOdometry(
        max_range=kiss_cfg.get("max_range", 100.0),
        min_range=kiss_cfg.get("min_range", 5.0),
        voxel_size=kiss_cfg.get("voxel_size", 1.0),
    )
    poses = odom.run(dataset)

    # Save poses
    poses_path = output_dir / f"poses_{dataset.sequence}.txt"
    KissICPOdometry.save_poses_kitti_format(poses, poses_path)
    print(f"  Saved {len(poses)} poses to {poses_path}")

    # Evaluate against ground truth if available
    if dataset.poses is not None:
        print("\n=== Evaluation ===")
        n = min(len(poses), len(dataset.poses))
        result = evaluate_odometry(poses[:n], dataset.poses[:n])
        print(f"  APE RMSE: {result['ape']['rmse']:.4f} m")
        print(f"  APE Mean: {result['ape']['mean']:.4f} m")
        print(f"  RPE RMSE: {result['rpe']['rmse']:.4f} m")
    else:
        print(f"\n  No ground truth for sequence {dataset.sequence} (only 00-10 have GT)")

    print("\nDone.")


if __name__ == "__main__":
    main()
