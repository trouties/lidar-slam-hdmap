"""Tests for SUP-07 degeneracy analyzer."""

from __future__ import annotations

import numpy as np

from src.odometry.degeneracy import (
    DegeneracyAnalyzer,
    DegeneracyScore,
    _pca_normals,
    _voxel_downsample,
)

# ---------------------------------------------------------------------------
# Synthetic point cloud generators
# ---------------------------------------------------------------------------


def _corridor(length: float = 30.0, width: float = 4.0, n_per_side: int = 800) -> np.ndarray:
    """Two parallel walls + a ground plane, no forward features.

    The x-axis is the unobserved (forward) direction: sliding the source
    along +x produces zero residual for all correspondences, so the
    translation Hessian must have a near-zero eigenvalue along x.
    """
    rng = np.random.default_rng(0)
    ys_left = np.full(n_per_side, -width / 2)
    ys_right = np.full(n_per_side, width / 2)
    zs = rng.uniform(-1.0, 2.0, size=n_per_side)
    xs = rng.uniform(0.0, length, size=n_per_side)
    left_wall = np.stack([xs, ys_left, zs], axis=1)
    right_wall = np.stack([xs, ys_right, zs], axis=1)

    n_floor = n_per_side
    xf = rng.uniform(0.0, length, size=n_floor)
    yf = rng.uniform(-width / 2, width / 2, size=n_floor)
    zf = np.full(n_floor, -1.5)
    floor = np.stack([xf, yf, zf], axis=1)
    return np.concatenate([left_wall, right_wall, floor], axis=0)


def _intersection(size: float = 20.0, n_per_plane: int = 800) -> np.ndarray:
    """Three mutually orthogonal planes (floor + 2 walls) — fully constrained."""
    rng = np.random.default_rng(1)
    xs1 = rng.uniform(-size / 2, size / 2, n_per_plane)
    ys1 = rng.uniform(-size / 2, size / 2, n_per_plane)
    zs1 = np.full(n_per_plane, -1.5)
    floor = np.stack([xs1, ys1, zs1], axis=1)

    xs2 = np.full(n_per_plane, -size / 2)
    ys2 = rng.uniform(-size / 2, size / 2, n_per_plane)
    zs2 = rng.uniform(-1.0, 2.0, n_per_plane)
    wall_x = np.stack([xs2, ys2, zs2], axis=1)

    xs3 = rng.uniform(-size / 2, size / 2, n_per_plane)
    ys3 = np.full(n_per_plane, -size / 2)
    zs3 = rng.uniform(-1.0, 2.0, n_per_plane)
    wall_y = np.stack([xs3, ys3, zs3], axis=1)
    return np.concatenate([floor, wall_x, wall_y], axis=0)


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------


def test_voxel_downsample_reduces_points():
    pts = np.random.default_rng(0).uniform(-1, 1, size=(1000, 3))
    ds = _voxel_downsample(pts, voxel_size=0.5)
    assert ds.shape[0] < pts.shape[0]
    assert ds.shape[1] == 3


def test_voxel_downsample_empty_input():
    assert _voxel_downsample(np.zeros((0, 3)), 0.5).shape == (0, 3)


def test_pca_normals_shapes_and_unit_length():
    pts = _intersection()
    normals, quality = _pca_normals(pts, k=10)
    assert normals.shape == pts.shape
    assert quality.shape == (pts.shape[0],)
    norm_mag = np.linalg.norm(normals, axis=1)
    np.testing.assert_allclose(norm_mag, 1.0, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Unit tests: DegeneracyAnalyzer core behavior
# ---------------------------------------------------------------------------


def test_corridor_is_degenerate():
    """Long corridor should produce a very small lambda_min along x."""
    cloud = _corridor()
    analyzer = DegeneracyAnalyzer(voxel_size=0.3, normal_k=10)
    score = analyzer.analyze(cloud, cloud)
    assert not score.is_null
    # x direction should dominate the smallest eigenvector
    assert abs(score.eig_direction[0]) > 0.9, (
        f"expected eig_direction aligned with x, got {score.eig_direction}"
    )
    # lambda_min << lambda_max for a corridor
    assert score.cond_number > 50.0, f"expected corridor cond > 50, got {score.cond_number}"


def test_intersection_is_well_constrained():
    """Three orthogonal planes should yield a cond number close to 1."""
    cloud = _intersection()
    analyzer = DegeneracyAnalyzer(voxel_size=0.3, normal_k=10)
    score = analyzer.analyze(cloud, cloud)
    assert not score.is_null
    assert score.cond_number < 5.0, f"expected intersection cond < 5, got {score.cond_number}"
    assert score.lambda_min > 0.0


def test_corridor_cond_vastly_exceeds_intersection_cond():
    """Monotonicity: corridor degeneracy must be strictly larger."""
    a = DegeneracyAnalyzer(voxel_size=0.3, normal_k=10)
    corr = a.analyze(_corridor(), _corridor())
    inter = a.analyze(_intersection(), _intersection())
    assert corr.cond_number > inter.cond_number * 10.0


def test_max_nn_dist_rejects_far_points():
    """All correspondences beyond max_nn_dist should be filtered out."""
    source = np.random.default_rng(2).uniform(-1, 1, size=(500, 3))
    target = source + np.array([100.0, 0.0, 0.0])  # translated out of range
    analyzer = DegeneracyAnalyzer(max_nn_dist=0.5, voxel_size=0.1, normal_k=10)
    score = analyzer.analyze(source, target)
    assert score.is_null


def test_empty_inputs_return_null():
    analyzer = DegeneracyAnalyzer()
    score = analyzer.analyze(np.zeros((0, 3)), np.zeros((0, 3)))
    assert score.is_null


def test_score_null_constructor():
    s = DegeneracyScore.null()
    assert s.is_null
    assert np.isnan(s.cond_number)
    assert s.n_corr == 0


# ---------------------------------------------------------------------------
# Hysteresis / EMA tests (SUP-07 post-review fixes)
# ---------------------------------------------------------------------------


def test_hysteresis_rejects_single_spike():
    """A 1-frame spike above threshold must not trigger."""
    from scripts.run_pipeline import _apply_hysteresis

    cond = np.array([1.0, 1.0, 1.0, 10.0, 1.0, 1.0, 1.0, 1.0])
    sustained = _apply_hysteresis(cond, threshold=5.0, ema_alpha=1.0, min_consecutive=5)
    assert not sustained.any()


def test_hysteresis_triggers_on_sustained_run():
    """A run of 6 frames above threshold should trigger those frames."""
    from scripts.run_pipeline import _apply_hysteresis

    cond = np.array([1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 1.0, 1.0])
    sustained = _apply_hysteresis(cond, threshold=5.0, ema_alpha=1.0, min_consecutive=5)
    # Indices 2-7 are above threshold for 6 consecutive frames; erode/dilate
    # with a 5-wide structuring element should keep all 6.
    assert sustained[2:8].all()
    assert not sustained[0:2].any()
    assert not sustained[8:].any()


def test_hysteresis_ema_suppresses_noise():
    """EMA smoothing should let alternating noise below threshold not trigger."""
    from scripts.run_pipeline import _apply_hysteresis

    # Alternating 2 and 8 around threshold 5: raw above=[F,T,F,T,F,T,F,T]
    # EMA with alpha=0.3 pulls toward midpoint ~5, stays below 5 on dip frames.
    cond = np.tile([2.0, 8.0], 10)
    sustained = _apply_hysteresis(cond, threshold=5.0, ema_alpha=0.3, min_consecutive=5)
    # No 5-consecutive run should form even though half the raw frames exceed.
    runs = sustained.astype(int)
    max_run = 0
    current = 0
    for v in runs:
        if v:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    assert max_run < 5, f"EMA should break the alternating pattern, max_run={max_run}"


def test_build_edge_sigmas_hysteresis_integration():
    """End-to-end: single spike is suppressed, sustained run is downgraded."""
    from scripts.run_pipeline import _build_edge_sigmas

    n = 20
    scores = np.zeros((n, 7), dtype=np.float64)
    scores[:, 0] = 1.0  # all below threshold
    scores[5, 0] = 100.0  # single spike — should NOT trigger
    scores[10:18, 0] = 50.0  # 8-frame sustained run — SHOULD trigger

    edges = _build_edge_sigmas(
        scores,
        base_sigmas=[0.1, 0.1, 0.1, 0.01, 0.01, 0.01],
        threshold=5.0,
        inflation_factor=10.0,
        ema_alpha=1.0,
        min_consecutive=5,
    )
    assert edges[5] is None, "single spike should NOT trigger"
    assert all(edges[i] is not None for i in range(10, 18)), "sustained run should trigger"
    assert edges[0] is None and edges[9] is None
