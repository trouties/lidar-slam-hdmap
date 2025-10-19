"""KISS-ICP odometry wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from evo.core import units
from evo.core.metrics import APE, RPE, PoseRelation
from evo.core.trajectory import PosePath3D
from kiss_icp.config import KISSConfig
from kiss_icp.config.config import DataConfig, MappingConfig
from kiss_icp.kiss_icp import KissICP


class KissICPOdometry:
    """Wrapper around the KISS-ICP pipeline for LiDAR odometry.

    Encapsulates KISS-ICP configuration and provides a simple interface
    for running odometry on a dataset.
    """

    def __init__(
        self,
        max_range: float = 100.0,
        min_range: float = 5.0,
        voxel_size: float = 1.0,
    ) -> None:
        """Initialize KISS-ICP odometry.

        Args:
            max_range: Maximum point range in meters.
            min_range: Minimum point range in meters.
            voxel_size: Voxel size for downsampling in meters.
        """
        self.max_range = max_range
        self.min_range = min_range
        self.voxel_size = voxel_size

    def run(self, dataset) -> list[np.ndarray]:
        """Run KISS-ICP odometry on a dataset.

        Iterates through all frames, registers each point cloud, and
        collects the cumulative SE(3) poses.

        Args:
            dataset: Indexable object with len() support. Each item should be
                a tuple where the first element is an (N, 4) or (N, 3) point cloud.

        Returns:
            List of 4x4 pose matrices (one per frame).
        """
        config = KISSConfig(
            data=DataConfig(max_range=self.max_range, min_range=self.min_range, deskew=False),
            mapping=MappingConfig(voxel_size=self.voxel_size),
        )
        icp = KissICP(config)
        poses = []
        n_frames = len(dataset)
        for idx in range(n_frames):
            pointcloud = dataset[idx][0]
            xyz = pointcloud[:, :3]  # strip reflectance if (N, 4)
            timestamps = np.zeros(len(xyz))  # no per-point timestamps in KITTI Odometry
            icp.register_frame(xyz, timestamps)
            poses.append(icp.last_pose.copy())
            if (idx + 1) % 100 == 0 or idx == n_frames - 1:
                print(f"  Frame {idx + 1}/{n_frames}")
        return poses

    @staticmethod
    def save_poses_kitti_format(poses: list[np.ndarray], path: str | Path) -> None:
        """Save poses in KITTI format (12 values per line, row-major 3x4).

        Args:
            poses: List of 4x4 pose matrices.
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for pose in poses:
                row = pose[:3, :].flatten()
                f.write(" ".join(f"{v:.6e}" for v in row) + "\n")


def evaluate_odometry(
    est_poses: list[np.ndarray],
    gt_poses: list[np.ndarray] | np.ndarray,
) -> dict[str, dict]:
    """Compute APE and RPE metrics using the evo toolkit.

    Args:
        est_poses: Estimated poses as list of 4x4 arrays.
        gt_poses: Ground truth poses as list of 4x4 arrays or (M, 4, 4) array.

    Returns:
        Dictionary with 'ape' and 'rpe' keys, each mapping to a statistics dict
        containing rmse, mean, median, std, min, max.
    """
    if isinstance(gt_poses, np.ndarray) and gt_poses.ndim == 3:
        gt_list = [gt_poses[i] for i in range(gt_poses.shape[0])]
    else:
        gt_list = list(gt_poses)

    traj_est = PosePath3D(poses_se3=list(est_poses))
    traj_ref = PosePath3D(poses_se3=gt_list)

    ape = APE(PoseRelation.translation_part)
    ape.process_data((traj_ref, traj_est))
    ape_stats = ape.get_all_statistics()

    rpe = RPE(PoseRelation.translation_part, delta=1.0, delta_unit=units.Unit.frames)
    rpe.process_data((traj_ref, traj_est))
    rpe_stats = rpe.get_all_statistics()

    return {"ape": ape_stats, "rpe": rpe_stats}
