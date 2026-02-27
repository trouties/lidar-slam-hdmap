"""Tests for the benchmarks infrastructure."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.benchmarks.evaluator import load_poses_kitti_format
from src.benchmarks.git_info import get_git_sha
from src.benchmarks.gnss_denial import make_denial_window, make_prior_indices, score_denial_drift
from src.benchmarks.manifest import BenchmarkManifest
from src.benchmarks.timing import StageTimer


def test_get_git_sha():
    sha = get_git_sha()
    assert sha != "unknown"
    assert len(sha) == 40


def test_get_git_sha_short():
    sha = get_git_sha(short=True)
    assert sha != "unknown"
    assert len(sha) == 7


class TestStageTimer:
    def test_single_shot(self):
        with StageTimer("test") as t:
            time.sleep(0.01)
        s = t.summary()
        assert s["n"] == 1
        assert s["p50"] > 0

    def test_multi_lap(self):
        t = StageTimer("test")
        for _ in range(5):
            with t:
                time.sleep(0.001)
        s = t.summary()
        assert s["n"] == 5
        assert s["total_ms"] > 0

    def test_empty(self):
        t = StageTimer("test")
        s = t.summary()
        assert s["n"] == 0


class TestManifest:
    def test_append_and_read(self, tmp_path: Path):
        path = tmp_path / "manifest.json"
        m = BenchmarkManifest(path)
        rec = m.append(
            task="SUP-TEST",
            config={"foo": 1},
            sequences=["00"],
            artifacts=["a.csv"],
            metrics={"ape": 1.0},
        )
        assert rec["task"] == "SUP-TEST"
        assert "git_sha" in rec
        assert "config_hash" in rec

        data = json.loads(path.read_text())
        assert len(data) == 1

        m.append(
            task="SUP-TEST2",
            config={"foo": 2},
            sequences=["01"],
            artifacts=[],
            metrics={},
        )
        data = json.loads(path.read_text())
        assert len(data) == 2


class TestLoadPoses:
    def test_roundtrip(self, tmp_path: Path):
        poses = [np.eye(4), np.eye(4)]
        poses[1][:3, 3] = [1.0, 2.0, 3.0]
        path = tmp_path / "poses.txt"
        with path.open("w") as f:
            for p in poses:
                row = p[:3, :].flatten()
                f.write(" ".join(f"{v:.6e}" for v in row) + "\n")
        loaded = load_poses_kitti_format(path)
        assert len(loaded) == 2
        np.testing.assert_allclose(loaded[1][:3, 3], [1.0, 2.0, 3.0])


def _make_straight_poses(n: int, step: float = 1.0) -> list[np.ndarray]:
    """Create a straight-line trajectory along x-axis."""
    poses = []
    for i in range(n):
        T = np.eye(4)
        T[0, 3] = i * step
        poses.append(T)
    return poses


class TestValidateTables:
    """Tests for the _validate_tables acceptance checker."""

    def test_valid_tables_pass(self, tmp_path: Path, monkeypatch):
        from scripts.run_baseline_compare import _validate_tables

        monkeypatch.setattr(
            "scripts.run_baseline_compare.BENCHMARKS_DIR", tmp_path
        )
        # Write valid accuracy table
        (tmp_path / "accuracy_table.csv").write_text(
            "system,sequence,stage,ape_rmse,ape_mean,rpe_rmse,source\n"
            "own,00,fused,11.1912,9.8391,0.1036,own_run\n"
        )
        (tmp_path / "latency_table.csv").write_text(
            "system,sequence,stage,p50_ms,p95_ms,max_ms,mean_ms\n"
            "own,00,stage2,1200.0,1200.0,1200.0,1200.0\n"
        )
        (tmp_path / "robustness_gnss_denied.csv").write_text(
            "system,sequence,denial_start_frame,denial_end_frame,"
            "denial_distance_m,ape_in_window_m,drift_per_meter\n"
            "own,00,2270,2464,150.3000,0.4114,0.002738\n"
        )
        assert _validate_tables() == []

    def test_catches_missing_file(self, tmp_path: Path, monkeypatch):
        from scripts.run_baseline_compare import _validate_tables

        monkeypatch.setattr(
            "scripts.run_baseline_compare.BENCHMARKS_DIR", tmp_path
        )
        violations = _validate_tables()
        assert any("MISSING" in v for v in violations)

    def test_catches_empty_value(self, tmp_path: Path, monkeypatch):
        from scripts.run_baseline_compare import _validate_tables

        monkeypatch.setattr(
            "scripts.run_baseline_compare.BENCHMARKS_DIR", tmp_path
        )
        (tmp_path / "accuracy_table.csv").write_text(
            "system,sequence,ape_rmse\n"
            "own,00,\n"
        )
        (tmp_path / "latency_table.csv").write_text(
            "system,sequence,p50_ms\nown,00,1.0\n"
        )
        (tmp_path / "robustness_gnss_denied.csv").write_text(
            "system,sequence,drift_per_meter\nown,00,0.001\n"
        )
        violations = _validate_tables()
        assert any("EMPTY" in v for v in violations)

    def test_catches_nan_value(self, tmp_path: Path, monkeypatch):
        from scripts.run_baseline_compare import _validate_tables

        monkeypatch.setattr(
            "scripts.run_baseline_compare.BENCHMARKS_DIR", tmp_path
        )
        (tmp_path / "accuracy_table.csv").write_text(
            "system,sequence,ape_rmse\nown,00,nan\n"
        )
        (tmp_path / "latency_table.csv").write_text(
            "system,sequence,p50_ms\nown,00,1.0\n"
        )
        (tmp_path / "robustness_gnss_denied.csv").write_text(
            "system,sequence,drift_per_meter\nown,00,0.001\n"
        )
        violations = _validate_tables()
        assert any("NAN" in v for v in violations)


class TestCacheTimingRoundtrip:
    """Test that timing survives cache save/load cycle."""

    def test_get_stage_metrics(self, tmp_path: Path):
        from src.cache.layered_cache import LayeredCache

        cache = LayeredCache(root=str(tmp_path / "cache"), sequence="test")
        cfg = {"data": {"kitti_root": "/tmp"}, "kiss_icp": {"voxel_size": 1.0}}

        # Save odometry with timing in metrics
        poses = np.eye(4).reshape(1, 4, 4)
        ts = np.array([0.0])
        timing = {
            "n": 1, "p50": 100.0, "p95": 100.0,
            "max": 100.0, "mean": 100.0, "total_ms": 100.0,
        }
        cache.save_odometry(poses, ts, cfg, metrics={"timing": timing})

        # Load metrics back
        m = cache.get_stage_metrics("odometry", cfg)
        assert m is not None
        assert m["timing"]["p50"] == 100.0

    def test_get_stage_metrics_stale(self, tmp_path: Path):
        from src.cache.layered_cache import LayeredCache

        cache = LayeredCache(root=str(tmp_path / "cache"), sequence="test")
        cfg1 = {"data": {"kitti_root": "/tmp"}, "kiss_icp": {"voxel_size": 1.0}}
        cfg2 = {"data": {"kitti_root": "/tmp"}, "kiss_icp": {"voxel_size": 2.0}}

        poses = np.eye(4).reshape(1, 4, 4)
        ts = np.array([0.0])
        cache.save_odometry(poses, ts, cfg1, metrics={"timing": {"p50": 50.0}})

        # Different config → stale → returns None
        assert cache.get_stage_metrics("odometry", cfg2) is None


class TestGnssDenial:
    def test_make_denial_window(self):
        poses = _make_straight_poses(500, step=1.0)
        start, end = make_denial_window(poses, target_distance=150.0)
        assert start == 250  # midpoint
        assert end > start

        # Check arc-length ≥ 150m
        trans = np.array([p[:3, 3] for p in poses[start : end + 1]])
        arc = float(np.sum(np.linalg.norm(np.diff(trans, axis=0), axis=1)))
        assert arc >= 150.0

    def test_make_prior_indices(self):
        indices = make_prior_indices(500, 100, 200, prior_stride=50)
        assert 0 in indices
        assert all(i < 100 or i > 200 for i in indices if i != 0)

    def test_score_denial_drift(self):
        gt = _make_straight_poses(100)
        est = [p.copy() for p in gt]
        est[50][:3, 3] += [0.5, 0, 0]
        result = score_denial_drift(est, gt, 40, 60)
        assert result["ape_mean"] > 0
        assert result["window_length_m"] > 0
