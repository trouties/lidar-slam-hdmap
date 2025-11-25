"""Loop closure detection.

Detects revisited places using distance-based candidate search,
then optionally verifies with ICP point cloud registration.
"""

from __future__ import annotations

import numpy as np
import open3d as o3d


class LoopClosureDetector:
    """Distance-based loop closure detector with optional ICP verification.

    v1 implementation: detects candidates by comparing pose translations,
    then verifies with Open3D ICP if point clouds are available.
    """

    def __init__(
        self,
        distance_threshold: float = 15.0,
        min_frame_gap: int = 100,
        icp_fitness_threshold: float = 0.3,
    ) -> None:
        """Initialize detector.

        Args:
            distance_threshold: Maximum distance (m) between poses to be a candidate.
            min_frame_gap: Minimum frame index gap to avoid detecting neighbors.
            icp_fitness_threshold: Minimum ICP fitness to accept a match.
        """
        self.distance_threshold = distance_threshold
        self.min_frame_gap = min_frame_gap
        self.icp_fitness_threshold = icp_fitness_threshold

    def detect_candidates(self, poses: list[np.ndarray]) -> list[tuple[int, int]]:
        """Find loop closure candidates based on pose distance.

        For each frame j, finds the single closest earlier frame i
        (with sufficient gap) below the distance threshold.

        Args:
            poses: List of 4x4 SE(3) poses.

        Returns:
            List of (i, j) pairs where i < j and poses are close.
        """
        n = len(poses)
        translations = np.array([p[:3, 3] for p in poses])
        candidates = []

        for j in range(self.min_frame_gap, n):
            # Compare against all earlier poses with sufficient gap
            search_end = j - self.min_frame_gap + 1
            diffs = translations[:search_end] - translations[j]
            dists = np.linalg.norm(diffs, axis=1)
            min_idx = int(np.argmin(dists))
            if dists[min_idx] < self.distance_threshold:
                candidates.append((min_idx, j))

        return candidates

    def verify_with_icp(
        self,
        source_points: np.ndarray,
        target_points: np.ndarray,
        initial_transform: np.ndarray,
        max_correspondence_distance: float = 2.0,
    ) -> tuple[np.ndarray, float] | None:
        """Verify a loop closure candidate using ICP.

        Args:
            source_points: (N, 3) source point cloud.
            target_points: (M, 3) target point cloud.
            initial_transform: 4x4 initial alignment guess.
            max_correspondence_distance: ICP max correspondence distance.

        Returns:
            (relative_pose, fitness) if fitness exceeds threshold, else None.
        """
        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(source_points[:, :3])

        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(target_points[:, :3])

        # Downsample for speed
        src_pcd = src_pcd.voxel_down_sample(voxel_size=1.0)
        tgt_pcd = tgt_pcd.voxel_down_sample(voxel_size=1.0)

        result = o3d.pipelines.registration.registration_icp(
            src_pcd,
            tgt_pcd,
            max_correspondence_distance,
            initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

        if result.fitness >= self.icp_fitness_threshold:
            return result.transformation, result.fitness
        return None

    def detect(
        self,
        poses: list[np.ndarray],
        dataset=None,
    ) -> list[tuple[int, int, np.ndarray]]:
        """Run full loop closure detection.

        Args:
            poses: List of 4x4 estimated poses.
            dataset: Optional dataset for ICP verification. If None, uses
                pose-derived relative transforms without ICP verification.

        Returns:
            List of (i, j, relative_pose_4x4) loop closure constraints.
        """
        candidates = self.detect_candidates(poses)
        if not candidates:
            return []

        closures = []
        for i, j in candidates:
            if dataset is not None:
                # ICP verification
                source_cloud = dataset[j][0][:, :3]
                target_cloud = dataset[i][0][:, :3]
                initial = np.linalg.inv(poses[i]) @ poses[j]

                result = self.verify_with_icp(source_cloud, target_cloud, initial)
                if result is not None:
                    relative_pose, fitness = result
                    closures.append((i, j, relative_pose))
                    print(f"  Loop closure: {i} ↔ {j} (fitness={fitness:.3f})")
            else:
                # Without point clouds, use pose-derived relative transform
                relative_pose = np.linalg.inv(poses[i]) @ poses[j]
                closures.append((i, j, relative_pose))

        return closures
