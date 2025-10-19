"""Evaluate odometry results: compute APE/RPE and generate trajectory plots.

Usage:
    python scripts/evaluate_odometry.py --est results/poses.txt --gt poses/00.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from evo.core.trajectory import PosePath3D
from evo.tools.plot import PlotMode, traj

from src.data.kitti_loader import load_poses
from src.odometry.kiss_icp_wrapper import evaluate_odometry


def plot_trajectories(
    est_poses: list,
    gt_poses: list,
    output_path: Path,
) -> None:
    """Plot estimated vs ground truth trajectories and save as PNG."""
    traj_est = PosePath3D(poses_se3=est_poses)
    traj_ref = PosePath3D(poses_se3=gt_poses)

    fig, ax = plt.subplots(figsize=(10, 10))
    traj(ax, PlotMode.xy, traj_ref, style="-", color="gray", label="Ground Truth")
    traj(ax, PlotMode.xy, traj_est, style="-", color="blue", label="KISS-ICP")
    ax.legend()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Trajectory Comparison")
    ax.set_aspect("equal")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved trajectory plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate odometry results")
    parser.add_argument("--est", type=str, required=True, help="Estimated poses (KITTI format)")
    parser.add_argument("--gt", type=str, required=True, help="Ground truth poses (KITTI format)")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load poses
    est_array = load_poses(Path(args.est))
    gt_array = load_poses(Path(args.gt))

    # Truncate to same length
    n = min(len(est_array), len(gt_array))
    est_list = [est_array[i] for i in range(n)]
    gt_list = [gt_array[i] for i in range(n)]

    # Evaluate
    result = evaluate_odometry(est_list, gt_list)

    print(f"\n{'=' * 50}")
    print(f"Evaluated {n} poses")
    print("\nAPE (Absolute Pose Error) [m]:")
    for key in ["rmse", "mean", "median", "std", "min", "max"]:
        print(f"  {key:>8s}: {result['ape'][key]:.4f}")
    print("\nRPE (Relative Pose Error) [m]:")
    for key in ["rmse", "mean", "median", "std", "min", "max"]:
        print(f"  {key:>8s}: {result['rpe'][key]:.4f}")
    print(f"{'=' * 50}")

    # Plot
    plot_trajectories(est_list, gt_list, output_dir / "trajectory_comparison.png")


if __name__ == "__main__":
    main()
