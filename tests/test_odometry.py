"""Tests for KISS-ICP odometry wrapper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from src.data.kitti_loader import load_poses
from src.odometry.kiss_icp_wrapper import KissICPOdometry, evaluate_odometry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ring(center_x: float = 0.0, radius: float = 20.0, n_points: int = 10000) -> np.ndarray:
    """Create a ring-shaped point cloud (N, 4) with reflectance."""
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = center_x + radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros(n_points)
    reflectance = np.ones(n_points) * 0.5
    return np.stack([x, y, z, reflectance], axis=1).astype(np.float32)


class FakeDataset:
    """Minimal dataset for testing: list of (pointcloud, pose, timestamp) tuples."""

    def __init__(self, frames: list[np.ndarray]):
        self._frames = frames

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, idx):
        return self._frames[idx], None, None


# ---------------------------------------------------------------------------
# Tests: run() with mocked KissICP
# ---------------------------------------------------------------------------


@patch("src.odometry.kiss_icp_wrapper.KissICP")
@patch("src.odometry.kiss_icp_wrapper.KISSConfig")
def test_run_collects_poses(mock_config_cls, mock_icp_cls):
    """run() should return one pose per frame."""
    mock_icp = MagicMock()
    mock_icp.last_pose = np.eye(4)
    mock_icp_cls.return_value = mock_icp

    frames = [_make_ring(i * 1.0) for i in range(5)]
    dataset = FakeDataset(frames)
    odom = KissICPOdometry()
    poses = odom.run(dataset)

    assert len(poses) == 5
    assert mock_icp.register_frame.call_count == 5
    for p in poses:
        assert p.shape == (4, 4)


@patch("src.odometry.kiss_icp_wrapper.KissICP")
@patch("src.odometry.kiss_icp_wrapper.KISSConfig")
def test_run_strips_reflectance(mock_config_cls, mock_icp_cls):
    """run() should pass (N, 3) xyz to register_frame, not (N, 4)."""
    mock_icp = MagicMock()
    mock_icp.last_pose = np.eye(4)
    mock_icp_cls.return_value = mock_icp

    frame = _make_ring()
    dataset = FakeDataset([frame])
    odom = KissICPOdometry()
    odom.run(dataset)

    call_args = mock_icp.register_frame.call_args
    xyz_arg = call_args[0][0]
    assert xyz_arg.shape[1] == 3  # must be (N, 3), not (N, 4)


@patch("src.odometry.kiss_icp_wrapper.KissICP")
@patch("src.odometry.kiss_icp_wrapper.KISSConfig")
def test_run_copies_poses(mock_config_cls, mock_icp_cls):
    """run() should copy each pose (not store references to mutable array)."""
    mock_icp = MagicMock()
    pose_array = np.eye(4)
    mock_icp.last_pose = pose_array
    mock_icp_cls.return_value = mock_icp

    # Mutate last_pose between frames
    def update_pose(*args, **kwargs):
        mock_icp.last_pose = mock_icp.last_pose.copy()
        mock_icp.last_pose[0, 3] += 1.0

    mock_icp.register_frame.side_effect = update_pose

    dataset = FakeDataset([_make_ring(), _make_ring()])
    poses = KissICPOdometry().run(dataset)
    # Poses should have different translations
    assert poses[0][0, 3] != poses[1][0, 3]


# ---------------------------------------------------------------------------
# Tests: integration with real KISS-ICP (structured point clouds)
# ---------------------------------------------------------------------------


def test_integration_ring_translation():
    """Real KISS-ICP: ring shifted along x should produce x-translation in pose."""
    frames = [_make_ring(center_x=i * 1.0) for i in range(3)]
    dataset = FakeDataset(frames)
    odom = KissICPOdometry(max_range=100.0, min_range=0.0, voxel_size=1.0)
    poses = odom.run(dataset)

    assert len(poses) == 3
    # First frame should be near identity
    np.testing.assert_array_almost_equal(poses[0], np.eye(4), decimal=3)
    # Last frame should have moved in x direction (negative because KISS-ICP tracks world frame)
    x_translation = abs(poses[2][0, 3])
    assert x_translation > 0.5, f"Expected significant x-translation, got {x_translation}"


# ---------------------------------------------------------------------------
# Tests: save_poses_kitti_format
# ---------------------------------------------------------------------------


def test_save_and_load_poses(tmp_path):
    """Saved KITTI-format poses should round-trip through load_poses."""
    original = [np.eye(4), np.eye(4)]
    original[1][0, 3] = 5.0  # translate x by 5

    path = tmp_path / "poses.txt"
    KissICPOdometry.save_poses_kitti_format(original, path)

    loaded = load_poses(path)
    assert loaded.shape == (2, 4, 4)
    np.testing.assert_array_almost_equal(loaded[0], original[0], decimal=5)
    np.testing.assert_array_almost_equal(loaded[1], original[1], decimal=5)


def test_save_creates_parent_dirs(tmp_path):
    """save_poses_kitti_format should create parent directories."""
    path = tmp_path / "sub" / "dir" / "poses.txt"
    KissICPOdometry.save_poses_kitti_format([np.eye(4)], path)
    assert path.exists()


# ---------------------------------------------------------------------------
# Tests: evaluate_odometry
# ---------------------------------------------------------------------------


def test_evaluate_identical_trajectories():
    """Identical trajectories should have APE ≈ 0."""
    poses = [np.eye(4)]
    for i in range(1, 10):
        p = np.eye(4)
        p[0, 3] = float(i)
        poses.append(p)

    result = evaluate_odometry(poses, poses)
    assert "ape" in result
    assert "rpe" in result
    assert result["ape"]["rmse"] < 1e-10


def test_evaluate_odometry_stats_structure():
    """evaluate_odometry should return dicts with standard evo stat keys."""
    gt_poses = []
    est_poses = []
    for i in range(10):
        gt = np.eye(4)
        gt[0, 3] = float(i)
        gt_poses.append(gt)

        est = np.eye(4)
        est[0, 3] = float(i) + 0.1
        est_poses.append(est)

    result = evaluate_odometry(est_poses, gt_poses)
    expected_keys = {"rmse", "mean", "median", "std", "min", "max", "sse"}
    assert expected_keys.issubset(result["ape"].keys())
    assert expected_keys.issubset(result["rpe"].keys())
    assert result["ape"]["rmse"] > 0


def test_evaluate_accepts_ndarray_gt():
    """evaluate_odometry should accept (M, 4, 4) ndarray for gt_poses."""
    poses = [np.eye(4) for _ in range(5)]
    gt_array = np.array(poses)  # (5, 4, 4)
    result = evaluate_odometry(poses, gt_array)
    assert result["ape"]["rmse"] < 1e-10
