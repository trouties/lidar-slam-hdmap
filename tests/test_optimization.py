"""Tests for pose graph optimization and loop closure detection."""

from __future__ import annotations

import numpy as np

from src.optimization.loop_closure import LoopClosureDetector
from src.optimization.pose_graph import PoseGraphOptimizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_straight_trajectory(n: int, step: float = 1.0) -> list[np.ndarray]:
    """Create a straight-line trajectory along x-axis."""
    poses = []
    for i in range(n):
        T = np.eye(4)
        T[0, 3] = i * step
        poses.append(T)
    return poses


def _make_loop_trajectory(n: int = 40, radius: float = 20.0) -> list[np.ndarray]:
    """Create a circular trajectory that revisits the start."""
    poses = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        T = np.eye(4)
        T[0, 3] = radius * np.cos(angle)
        T[1, 3] = radius * np.sin(angle)
        # Rotation around z-axis to face forward
        T[0, 0] = np.cos(angle + np.pi / 2)
        T[0, 1] = -np.sin(angle + np.pi / 2)
        T[1, 0] = np.sin(angle + np.pi / 2)
        T[1, 1] = np.cos(angle + np.pi / 2)
        poses.append(T)
    return poses


def _add_drift(poses: list[np.ndarray], drift_per_frame: float = 0.01) -> list[np.ndarray]:
    """Add cumulative drift to a trajectory."""
    drifted = []
    for i, pose in enumerate(poses):
        p = pose.copy()
        p[0, 3] += i * drift_per_frame
        p[1, 3] += i * drift_per_frame * 0.5
        drifted.append(p)
    return drifted


# ---------------------------------------------------------------------------
# Tests: PoseGraphOptimizer
# ---------------------------------------------------------------------------


def test_build_graph_size():
    """3 poses → 1 prior + 2 between = 3 factors."""
    poses = _make_straight_trajectory(3)
    opt = PoseGraphOptimizer()
    opt.build_graph(poses)
    assert opt.graph_size == 3
    assert opt.n_poses == 3


def test_optimize_straight_line():
    """Straight trajectory should remain roughly straight after optimization."""
    poses = _make_straight_trajectory(5)
    opt = PoseGraphOptimizer()
    opt.build_graph(poses)
    result = opt.optimize()

    assert len(result) == 5
    for i, pose in enumerate(result):
        assert pose.shape == (4, 4)
        np.testing.assert_array_almost_equal(pose[:3, 3], [float(i), 0, 0], decimal=2)


def test_optimize_reduces_error():
    """Optimization should bring noisy poses closer to the prior."""
    poses = _make_straight_trajectory(5)
    # Add noise to initial poses (but keep the factors from the clean poses)
    noisy = [p.copy() for p in poses]
    noisy[2][0, 3] += 0.5  # perturb middle pose
    noisy[3][1, 3] += 0.3

    opt = PoseGraphOptimizer()
    opt.build_graph(poses)  # build graph from clean relative transforms

    # Replace initial values with noisy ones
    import gtsam

    opt.initial_values = gtsam.Values()
    for i, p in enumerate(noisy):
        opt.initial_values.insert(i, gtsam.Pose3(p))

    result = opt.optimize()
    # Optimized poses should be closer to clean trajectory
    err_before = sum(np.linalg.norm(noisy[i][:3, 3] - poses[i][:3, 3]) for i in range(5))
    err_after = sum(np.linalg.norm(result[i][:3, 3] - poses[i][:3, 3]) for i in range(5))
    assert err_after < err_before


def test_add_loop_closure_increases_graph():
    """Adding a loop closure should increase graph size by 1."""
    poses = _make_straight_trajectory(5)
    opt = PoseGraphOptimizer()
    opt.build_graph(poses)
    size_before = opt.graph_size

    opt.add_loop_closure(0, 4, np.eye(4))
    assert opt.graph_size == size_before + 1


def test_loop_closure_reduces_drift():
    """Loop closure should pull drifted initial values toward the constraint."""
    import gtsam

    n = 20
    poses_clean = _make_loop_trajectory(n, radius=20.0)
    poses_drifted = _add_drift(poses_clean, drift_per_frame=0.1)

    # Build graph with CLEAN between factors (correct local motion)
    opt = PoseGraphOptimizer()
    opt.build_graph(poses_clean)

    # Replace initial values with drifted estimates
    opt.initial_values = gtsam.Values()
    for i, p in enumerate(poses_drifted):
        opt.initial_values.insert(i, gtsam.Pose3(p))

    # Add loop closure: last pose should be near first (near-identity)
    relative = np.linalg.inv(poses_clean[0]) @ poses_clean[-1]
    opt.add_loop_closure(0, n - 1, relative)

    result = opt.optimize()

    # After optimization, poses should be closer to clean trajectory
    err_before = sum(
        np.linalg.norm(poses_drifted[k][:3, 3] - poses_clean[k][:3, 3]) for k in range(n)
    )
    err_after = sum(np.linalg.norm(result[k][:3, 3] - poses_clean[k][:3, 3]) for k in range(n))
    assert err_after < err_before, (
        f"Loop closure should reduce total error: {err_before:.3f} → {err_after:.3f}"
    )


# ---------------------------------------------------------------------------
# Tests: LoopClosureDetector
# ---------------------------------------------------------------------------


def test_detect_candidates_finds_revisit():
    """Circular trajectory should have candidates near the closure point."""
    poses = _make_loop_trajectory(40, radius=10.0)
    detector = LoopClosureDetector(distance_threshold=5.0, min_frame_gap=10)
    candidates = detector.detect_candidates(poses)
    assert len(candidates) > 0
    # At least one candidate should involve early and late frames
    has_closure = any(i < 10 and j > 30 for i, j in candidates)
    assert has_closure, f"Expected early-late closure, got: {candidates}"


def test_detect_candidates_no_loops():
    """Straight trajectory should have no candidates."""
    poses = _make_straight_trajectory(50, step=5.0)
    detector = LoopClosureDetector(distance_threshold=15.0, min_frame_gap=10)
    candidates = detector.detect_candidates(poses)
    assert len(candidates) == 0


def test_detect_candidates_respects_min_gap():
    """Candidates must have at least min_frame_gap separation."""
    poses = _make_loop_trajectory(40, radius=10.0)
    detector = LoopClosureDetector(distance_threshold=50.0, min_frame_gap=20)
    candidates = detector.detect_candidates(poses)
    for i, j in candidates:
        assert j - i >= 20


def test_detect_without_dataset():
    """detect() without dataset should return pose-derived closures."""
    poses = _make_loop_trajectory(40, radius=10.0)
    detector = LoopClosureDetector(distance_threshold=5.0, min_frame_gap=10)
    closures = detector.detect(poses, dataset=None)
    assert len(closures) > 0
    for i, j, rel_pose in closures:
        assert rel_pose.shape == (4, 4)
        assert j > i


# ---------------------------------------------------------------------------
# Tests: Integration (pose graph + loop closure)
# ---------------------------------------------------------------------------


def test_full_pipeline_synthetic():
    """Full pipeline: drifted loop → detect closure → optimize → reduced drift."""
    poses_clean = _make_loop_trajectory(40, radius=20.0)
    poses_drifted = _add_drift(poses_clean, drift_per_frame=0.1)

    # Detect loop closures
    detector = LoopClosureDetector(distance_threshold=10.0, min_frame_gap=15)
    closures = detector.detect(poses_drifted, dataset=None)

    # Build and optimize pose graph
    opt = PoseGraphOptimizer()
    opt.build_graph(poses_drifted)
    for i, j, rel_pose in closures:
        opt.add_loop_closure(i, j, rel_pose)

    result = opt.optimize()
    assert len(result) == 40

    # Overall trajectory error should decrease
    err_before = sum(
        np.linalg.norm(poses_drifted[k][:3, 3] - poses_clean[k][:3, 3]) for k in range(40)
    )
    err_after = sum(np.linalg.norm(result[k][:3, 3] - poses_clean[k][:3, 3]) for k in range(40))
    assert err_after < err_before
