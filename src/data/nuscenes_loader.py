"""nuScenes dataset loader for LiDAR point clouds and ego poses.

Provides a NuScenesDataset class with the same interface as KITTIDataset,
enabling Stage 1-3 of the SLAM pipeline to run on nuScenes mini without
modification to the odometry or optimization modules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def _quat_trans_to_se3(rotation: list[float], translation: list[float]) -> np.ndarray:
    """Build a 4x4 SE(3) matrix from nuScenes quaternion and translation.

    Args:
        rotation: Quaternion [w, x, y, z] (nuScenes convention).
        translation: [tx, ty, tz] in metres.

    Returns:
        4x4 SE(3) transformation matrix.
    """
    from pyquaternion import Quaternion

    T = np.eye(4)
    T[:3, :3] = Quaternion(rotation).rotation_matrix
    T[:3, 3] = translation
    return T


class NuScenesDataset:
    """nuScenes dataset wrapper for a single scene.

    By default, iterates over ALL LIDAR_TOP sweeps (~20 Hz) for reliable
    KISS-ICP registration.  Set ``keyframes_only=True`` to use only the
    annotated 2 Hz keyframes (useful for debugging or annotation tasks).

    Implements the same interface as KITTIDataset so that KissICPOdometry
    and PoseGraphOptimizer can be used without modification.

    Ground truth poses are expressed in the LIDAR_TOP sensor frame,
    relative to the first frame (pose 0 is identity).  This matches
    the convention used by KITTIDataset.

    Attributes:
        scene_token: nuScenes scene token string.
        poses: (M, 4, 4) array of GT SE(3) matrices (relative to frame 0).
        timestamps: 1D float64 array of seconds from first frame.
        calibration: Dict with 'Tr' key set to np.eye(4).  Allows
            run_pipeline.py to call transform_poses_to_camera_frame with
            an identity matrix (no frame conversion needed for nuScenes).
    """

    def __init__(self, nusc, scene_token: str, *, keyframes_only: bool = False) -> None:
        """Initialize dataset for one nuScenes scene.

        Args:
            nusc: Initialised NuScenes instance (shared across scenes).
            scene_token: Token for the scene to load.
            keyframes_only: If True, load only 2 Hz annotated keyframes.
                Default False loads all sweeps (~20 Hz) for better
                KISS-ICP frame-to-frame registration quality.
        """
        self.scene_token = scene_token
        self._dataroot = Path(nusc.dataroot)

        scene = nusc.get("scene", scene_token)

        if keyframes_only:
            filepaths, ego_pose_tokens, cs_tokens, timestamps_us = self._walk_keyframes(nusc, scene)
        else:
            filepaths, ego_pose_tokens, cs_tokens, timestamps_us = self._walk_sweeps(nusc, scene)

        self._filepaths = filepaths

        # Compute sensor extrinsic T_ego_lidar from the first frame
        # (constant across frames within a scene).
        cs = nusc.get("calibrated_sensor", cs_tokens[0])
        T_ego_lidar = _quat_trans_to_se3(cs["rotation"], cs["translation"])

        # Build GT poses: T_global_lidar_i = T_global_ego_i @ T_ego_lidar
        # then relativise to first frame.
        global_poses: list[np.ndarray] = []
        for ep_token in ego_pose_tokens:
            ep = nusc.get("ego_pose", ep_token)
            T_global_ego = _quat_trans_to_se3(ep["rotation"], ep["translation"])
            global_poses.append(T_global_ego @ T_ego_lidar)

        T0_inv = np.linalg.inv(global_poses[0])
        self.poses: np.ndarray = np.stack([T0_inv @ T for T in global_poses])

        # Timestamps in seconds from first frame.
        t0 = timestamps_us[0]
        self.timestamps: np.ndarray = np.array(
            [(t - t0) / 1e6 for t in timestamps_us], dtype=np.float64
        )

        # Identity Tr: nuScenes poses already in LiDAR frame, no conversion needed.
        self.calibration: dict[str, np.ndarray] = {"Tr": np.eye(4)}

    @staticmethod
    def _walk_keyframes(nusc, scene: dict) -> tuple:
        """Walk 2 Hz annotated keyframes for the scene."""
        filepaths: list[Path] = []
        ego_pose_tokens: list[str] = []
        cs_tokens: list[str] = []
        timestamps_us: list[int] = []

        dataroot = Path(nusc.dataroot)
        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = nusc.get("sample", sample_token)
            sd_token = sample["data"]["LIDAR_TOP"]
            sd = nusc.get("sample_data", sd_token)
            filepaths.append(dataroot / sd["filename"])
            ego_pose_tokens.append(sd["ego_pose_token"])
            cs_tokens.append(sd["calibrated_sensor_token"])
            timestamps_us.append(sd["timestamp"])
            sample_token = sample["next"]

        return filepaths, ego_pose_tokens, cs_tokens, timestamps_us

    @staticmethod
    def _walk_sweeps(nusc, scene: dict) -> tuple:
        """Walk all LIDAR_TOP sweeps (~20 Hz) for the scene."""
        filepaths: list[Path] = []
        ego_pose_tokens: list[str] = []
        cs_tokens: list[str] = []
        timestamps_us: list[int] = []

        dataroot = Path(nusc.dataroot)

        # Start from the first sample's LIDAR_TOP sample_data token and follow
        # the sample_data 'next' chain which covers ALL sweeps at ~20 Hz.
        first_sample = nusc.get("sample", scene["first_sample_token"])
        sd_token = first_sample["data"]["LIDAR_TOP"]

        while sd_token:
            sd = nusc.get("sample_data", sd_token)
            filepaths.append(dataroot / sd["filename"])
            ego_pose_tokens.append(sd["ego_pose_token"])
            cs_tokens.append(sd["calibrated_sensor_token"])
            timestamps_us.append(sd["timestamp"])
            sd_token = sd["next"]  # empty string at last sweep

        return filepaths, ego_pose_tokens, cs_tokens, timestamps_us

    def __len__(self) -> int:
        return len(self._filepaths)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray | None, float | None]:
        """Get a single frame.

        Args:
            idx: Frame index.

        Returns:
            Tuple of (pointcloud, pose, timestamp).
            - pointcloud: (N, 4) float32 [x, y, z, intensity] where intensity
              is normalised from nuScenes range [0, 255] to [0, 1].
            - pose: (4, 4) GT SE(3) matrix relative to first frame.
            - timestamp: seconds from first frame.
        """
        raw = np.fromfile(self._filepaths[idx], dtype=np.float32).reshape(-1, 5)
        pointcloud = raw[:, :4].copy()
        pointcloud[:, 3] /= 255.0  # normalise intensity to [0, 1]

        pose = self.poses[idx] if idx < len(self.poses) else None
        timestamp = float(self.timestamps[idx]) if idx < len(self.timestamps) else None
        return pointcloud, pose, timestamp
