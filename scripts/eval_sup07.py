#!/usr/bin/env python3
"""SUP-07: Degeneracy detection + downgrade evaluation.

Runs only Stage 2 (KISS-ICP + degeneracy probe) and Stage 3 (pose graph +
loop closure) on the requested sequences. Downstream stages (ESKF, master
map, Stage 5, Stage 6) are skipped because they are irrelevant to the
SUP-07 acceptance criteria and add ~15 min / sequence to a cold run.

Two-pass evaluation:
  - Pass A (baseline):  sup07 scores computed but no downgrade applied.
  - Pass B (downgrade): edge sigmas inflated on frames above cond threshold.

Acceptance criteria (from refs/backlog.md SUP-07):
  1. Seq 01 cond_number median >= Seq 00 cond_number p95 * 2
  2. Seq 01 APE(downgrade) / APE(baseline) <= 1.0
  3. At least one visualization exists (plot_sup07.py run separately).

Outputs under ``benchmarks/sup07/``:
  - degeneracy_{seq}_baseline.csv    (per-frame cond scores)
  - degeneracy_summary.csv           (per-sequence stats)
  - ape_compare.csv                  (baseline vs downgrade APE per seq)
  - benchmarks/benchmark_manifest.json  (appended SUP-07 record)
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_pipeline import (  # noqa: E402
    _build_edge_sigmas,
    _degeneracy_stats,
    _scores_to_array,
    _write_degeneracy_csv,
    load_config,
)
from src.benchmarks import BenchmarkManifest  # noqa: E402
from src.cache import LayeredCache  # noqa: E402
from src.data.kitti_loader import KITTIDataset  # noqa: E402
from src.odometry.degeneracy import DegeneracyAnalyzer  # noqa: E402
from src.odometry.kiss_icp_wrapper import (  # noqa: E402
    KissICPOdometry,
    evaluate_odometry,
    transform_poses_to_camera_frame,
)
from src.optimization.loop_closure import LoopClosureDetector  # noqa: E402
from src.optimization.pose_graph import PoseGraphOptimizer  # noqa: E402

OUT_DIR = Path("benchmarks/sup07")
DEFAULT_SEQUENCES = ["00", "01"]


# ---------------------------------------------------------------------------
# Stage 2 + Stage 3 runner (cache-aware, skips stages 4-6)
# ---------------------------------------------------------------------------


def _stage2(
    config: dict,
    cache: LayeredCache | None,
    dataset: KITTIDataset,
    analyzer: DegeneracyAnalyzer,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return (poses, scores_arr). Uses odometry + degeneracy sidecar cache."""
    cached_scores: np.ndarray | None = None
    odom_cached = cache.load_odometry(config) if cache else None
    if cache is not None and odom_cached is not None:
        cached_scores = cache.load_degeneracy(config)
        if cached_scores is None:
            print("  [sup07] degeneracy sidecar missing/stale — re-running Stage 2")
            odom_cached = None

    if odom_cached is not None and cached_scores is not None:
        poses_arr, _ = odom_cached
        poses = [poses_arr[i] for i in range(poses_arr.shape[0])]
        print(f"  [cache hit] {len(poses)} poses + degeneracy")
        return poses, cached_scores

    kiss_cfg = config.get("kiss_icp", {})
    odom = KissICPOdometry(
        max_range=kiss_cfg.get("max_range", 100.0),
        min_range=kiss_cfg.get("min_range", 5.0),
        voxel_size=kiss_cfg.get("voxel_size", 1.0),
    )
    result = odom.run(dataset, degeneracy_analyzer=analyzer)
    poses, scores = result  # type: ignore[assignment]
    scores_arr = _scores_to_array(scores)
    if dataset.timestamps is not None:
        timestamps = np.asarray(dataset.timestamps[: len(poses)], dtype=np.float64)
    else:
        timestamps = np.arange(len(poses), dtype=np.float64) * 0.1
    if cache is not None:
        cache.save_odometry(np.asarray(poses), timestamps, config)
        cache.save_degeneracy(scores_arr, config)
    return poses, scores_arr


def _stage3(
    config: dict,
    dataset: KITTIDataset,
    poses: list[np.ndarray],
    edge_sigmas: list[list[float] | None] | None,
    closures: list | None = None,
) -> tuple[list[np.ndarray], list]:
    """Run pose graph optimization with loop closure. No caching.

    If ``closures`` is provided the loop-closure detection step is skipped and
    the supplied list is reused directly (useful for Pass B which shares the
    same closures as Pass A).
    """
    if closures is None:
        lc_cfg = config.get("loop_closure", {})
        sc_cfg = lc_cfg.get("scan_context", {})
        detector = LoopClosureDetector(
            distance_threshold=lc_cfg.get("distance_threshold", 15.0),
            min_frame_gap=lc_cfg.get("min_frame_gap", 100),
            icp_fitness_threshold=lc_cfg.get("icp_fitness_threshold", 0.9),
            mode=lc_cfg.get("mode", "v1"),
            sc_num_rings=sc_cfg.get("num_rings", 20),
            sc_num_sectors=sc_cfg.get("num_sectors", 60),
            sc_max_range=sc_cfg.get("max_range", 80.0),
            sc_distance_threshold=sc_cfg.get("distance_threshold", 0.4),
            sc_top_k=sc_cfg.get("top_k", 10),
            sc_query_stride=sc_cfg.get("query_stride", 1),
            sc_max_matches_per_query=sc_cfg.get("max_matches_per_query", 0),
            icp_downsample_voxel=lc_cfg.get("icp_downsample_voxel", 1.0),
        )
        closures = detector.detect(poses, dataset=dataset)
        print(f"  detected {len(closures)} loop closure(s)")
    else:
        print(f"  reusing {len(closures)} cached loop closure(s)")

    gtsam_cfg = config.get("gtsam", {})
    optimizer = PoseGraphOptimizer(
        odom_sigmas=gtsam_cfg.get("odom_sigmas"),
        prior_sigmas=gtsam_cfg.get("prior_sigmas"),
    )
    optimizer.build_graph(poses, edge_sigmas=edge_sigmas)
    for i, j, rel_pose in closures:
        optimizer.add_loop_closure(i, j, rel_pose)
    return optimizer.optimize(), closures


def _evaluate_ape(est: list[np.ndarray], gt: list[np.ndarray] | None) -> float:
    if gt is None:
        return float("nan")
    n = min(len(est), len(gt))
    result = evaluate_odometry(est[:n], gt[:n])
    return float(result["ape"]["rmse"])


def _run_pass(
    cfg: dict,
    sequence: str,
    label: str,
    analyzer: DegeneracyAnalyzer,
    inflation_factor: float,
    cond_threshold: float,
    cache: LayeredCache | None,
    max_frames: int | None,
    out_dir: Path,
    closures_cache: dict[str, list] | None = None,
    sigma_mode: str = "uniform",
) -> tuple[np.ndarray, float]:
    """Return (scores_arr, ape_rmse) for one pass."""
    cfg_seq = copy.deepcopy(cfg)
    cfg_seq.setdefault("data", {})["sequence"] = sequence
    dataset = KITTIDataset(
        root_path=cfg_seq["data"]["kitti_root"],
        sequence=sequence,
    )
    if max_frames is not None and max_frames < len(dataset):
        dataset.scan_files = dataset.scan_files[:max_frames]
    print(f"  Seq {sequence}: {len(dataset)} frames")

    Tr = dataset.calibration["Tr"] if dataset.calibration else np.eye(4)
    Tr_inv = np.linalg.inv(Tr)
    gt_velo: list[np.ndarray] | None = None
    if dataset.poses is not None:
        gt_velo = [Tr_inv @ dataset.poses[i] @ Tr for i in range(len(dataset.poses))]

    print(f"  [{label}] Stage 2 (KISS-ICP + degeneracy)")
    poses, scores_arr = _stage2(cfg_seq, cache, dataset, analyzer)

    sup07_cfg = cfg_seq.get("sup07", {}) or {}
    edge_sigmas = _build_edge_sigmas(
        scores_arr,
        base_sigmas=cfg_seq.get("gtsam", {}).get("odom_sigmas")
        or [0.1, 0.1, 0.1, 0.01, 0.01, 0.01],
        threshold=cond_threshold,
        inflation_factor=inflation_factor,
        ema_alpha=float(sup07_cfg.get("ema_alpha", 0.3)),
        min_consecutive=int(sup07_cfg.get("min_consecutive", 5)),
        sigma_mode=sigma_mode,
    )
    n_downgraded = sum(1 for e in edge_sigmas if e is not None)
    print(f"  [{label}] downgrading {n_downgraded}/{len(poses)} edges (sigma_mode={sigma_mode})")

    print(f"  [{label}] Stage 3 (pose graph + loop closure)")
    cached_closures = closures_cache.get(sequence) if closures_cache is not None else None
    optimized, detected = _stage3(cfg_seq, dataset, poses, edge_sigmas, closures=cached_closures)
    if closures_cache is not None and cached_closures is None:
        closures_cache[sequence] = detected
    ape = _evaluate_ape(optimized, gt_velo)
    print(f"  [{label}] APE RMSE = {ape:.4f} m")

    # Write optimized poses for plot_sup07 BEV use (only on baseline).
    if label == "baseline":
        baseline_dir = out_dir / "baseline" / sequence
        baseline_dir.mkdir(parents=True, exist_ok=True)
        opt_cam = transform_poses_to_camera_frame(optimized, Tr)
        KissICPOdometry.save_poses_kitti_format(
            opt_cam, baseline_dir / f"poses_optimized_{sequence}.txt"
        )
    return scores_arr, ape


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _make_analyzer(cfg: dict) -> DegeneracyAnalyzer:
    sup07 = cfg.get("sup07", {}) or {}
    return DegeneracyAnalyzer(
        max_correspondences=int(sup07.get("max_correspondences", 5000)),
        normal_k=int(sup07.get("normal_k", 10)),
        max_nn_dist=float(sup07.get("max_nn_dist", 1.0)),
        voxel_size=float(sup07.get("voxel_size", 0.5)),
        min_quality=float(sup07.get("min_quality", 0.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SUP-07 degeneracy eval")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sequences", default=",".join(DEFAULT_SEQUENCES))
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Frame cap for quick smoke runs; omit for full sequences.",
    )
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    parser.add_argument("--inflation-factor", type=float, default=10.0)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--baseline-sequence", default="00")
    parser.add_argument("--threshold-multiplier", type=float, default=1.0)
    parser.add_argument("--threshold-override", type=float, default=None)
    parser.add_argument(
        "--sigma-mode",
        choices=["uniform", "directional"],
        default=None,
        help=(
            "SUP-07 inflation strategy: 'uniform' (legacy) scales all of tx/ty/tz by "
            "inflation_factor; 'directional' (Zhang 2016 ICRA-faithful) inflates "
            "only the variance along the least-observed eigenvector. Overrides "
            "configs/default.yaml sup07.sigma_mode when set."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable layered cache entirely (forces Stage 2 rerun)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = [s.strip() for s in args.sequences.split(",") if s.strip()]
    cfg = load_config(args.config)
    # SUP-07 must be "active" so analyzer params get into the cache hash.
    cfg.setdefault("sup07", {})["enabled"] = True
    if args.sigma_mode is not None:
        cfg["sup07"]["sigma_mode"] = args.sigma_mode
    sigma_mode = str(cfg["sup07"].get("sigma_mode", "uniform"))
    print(f"sigma_mode = {sigma_mode}")

    analyzer = _make_analyzer(cfg)
    cache_root = cfg.get("cache", {}).get("root", "cache/kitti")
    use_cache = (
        cfg.get("cache", {}).get("enabled", True) and not args.no_cache and args.max_frames is None
    )

    def _cache_for(seq: str) -> LayeredCache | None:
        return LayeredCache(cache_root, seq) if use_cache else None

    # Shared loop-closure cache: Pass A detects closures, Pass B reuses them.
    closures_cache: dict[str, list] = {}

    # ------------------------------------------------------------------
    # Pass A: baseline (inflation=1.0 => no downgrade)
    # ------------------------------------------------------------------
    print("=" * 72)
    print("PASS A: baseline (inflation=1.0)")
    print("=" * 72)
    baseline_scores: dict[str, np.ndarray] = {}
    baseline_ape: dict[str, float] = {}
    for seq in sequences:
        cache = _cache_for(seq)
        scores, ape = _run_pass(
            cfg,
            seq,
            label="baseline",
            analyzer=analyzer,
            inflation_factor=1.0,
            cond_threshold=1e30,
            cache=cache,
            max_frames=args.max_frames,
            out_dir=out_dir,
            closures_cache=closures_cache,
            sigma_mode=sigma_mode,
        )
        baseline_scores[seq] = scores
        baseline_ape[seq] = ape
        _write_degeneracy_csv(
            scores,
            out_dir / f"degeneracy_{seq}_baseline.csv",
            threshold=1e30,
        )

    # ------------------------------------------------------------------
    # Threshold fit
    # ------------------------------------------------------------------
    if args.threshold_override is not None:
        cond_threshold = float(args.threshold_override)
        print(f"\nThreshold override = {cond_threshold:.4f}")
    else:
        base_seq = args.baseline_sequence
        base_cond = baseline_scores.get(base_seq, np.zeros((0, 7)))[:, 0]
        base_cond = base_cond[np.isfinite(base_cond)]
        if base_cond.size == 0:
            raise RuntimeError(f"Baseline seq {base_seq!r} produced no cond scores")
        base_p = float(np.percentile(base_cond, args.threshold_percentile))
        cond_threshold = base_p * args.threshold_multiplier
        print(
            f"\nThreshold fit: Seq {base_seq} p{args.threshold_percentile:.0f}="
            f"{base_p:.4f} × {args.threshold_multiplier} = {cond_threshold:.4f}"
        )

    # ------------------------------------------------------------------
    # Pass B: downgrade
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print(f"PASS B: downgrade (threshold={cond_threshold:.4f} inflation={args.inflation_factor})")
    print("=" * 72)
    downgrade_ape: dict[str, float] = {}
    for seq in sequences:
        cache = _cache_for(seq)
        _, ape = _run_pass(
            cfg,
            seq,
            label="downgrade",
            analyzer=analyzer,
            inflation_factor=args.inflation_factor,
            cond_threshold=cond_threshold,
            cache=cache,
            max_frames=args.max_frames,
            out_dir=out_dir,
            closures_cache=closures_cache,
            sigma_mode=sigma_mode,
        )
        downgrade_ape[seq] = ape

    # ------------------------------------------------------------------
    # Aggregate + acceptance
    # ------------------------------------------------------------------
    summary_rows: list[dict[str, Any]] = []
    for seq in sequences:
        stats = _degeneracy_stats(baseline_scores[seq], cond_threshold)
        row = {
            "sequence": seq,
            "cond_count": stats.get("count", 0),
            "cond_p50": stats.get("p50", float("nan")),
            "cond_p95": stats.get("p95", float("nan")),
            "cond_p99": stats.get("p99", float("nan")),
            "cond_max": stats.get("max", float("nan")),
            "threshold": cond_threshold,
            "n_above_threshold": stats.get("n_above_threshold", 0),
            "ape_baseline": baseline_ape[seq],
            "ape_downgrade": downgrade_ape[seq],
        }
        row["ape_ratio"] = (
            row["ape_downgrade"] / row["ape_baseline"] if row["ape_baseline"] > 0 else float("nan")
        )
        summary_rows.append(row)

    summary_path = out_dir / "degeneracy_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    ape_path = out_dir / "ape_compare.csv"
    with ape_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "ape_baseline", "ape_downgrade", "ape_ratio", "n_downgraded"])
        for row in summary_rows:
            writer.writerow(
                [
                    row["sequence"],
                    f"{row['ape_baseline']:.4f}",
                    f"{row['ape_downgrade']:.4f}",
                    f"{row['ape_ratio']:.4f}",
                    row["n_above_threshold"],
                ]
            )

    print()
    print("=" * 72)
    print("Acceptance report")
    print("=" * 72)
    seq00 = next((r for r in summary_rows if r["sequence"] == "00"), None)
    seq01 = next((r for r in summary_rows if r["sequence"] == "01"), None)

    passed_1 = False
    if seq00 and seq01 and np.isfinite(seq00["cond_p95"]) and np.isfinite(seq01["cond_p50"]):
        bar = seq00["cond_p95"] * 2.0
        passed_1 = seq01["cond_p50"] >= bar
        print(
            f"[1] Seq 01 cond_p50={seq01['cond_p50']:.3f} vs 2×Seq 00 cond_p95={bar:.3f}"
            f" -> {'PASS' if passed_1 else 'FAIL'}"
        )

    passed_2 = False
    if seq01 and np.isfinite(seq01["ape_ratio"]):
        # Small tolerance to absorb LM convergence noise when the downgrade
        # has effectively no effect (e.g. no loop closures, graph uniquely
        # determined by initial values). "At least no harm" is the bar.
        passed_2 = seq01["ape_ratio"] <= 1.0 + 1e-4
        print(
            f"[2] Seq 01 APE baseline={seq01['ape_baseline']:.4f}m "
            f"downgrade={seq01['ape_downgrade']:.4f}m "
            f"ratio={seq01['ape_ratio']:.6f} "
            f"-> {'PASS' if passed_2 else 'FAIL'}"
        )

    print("[3] Visualization: run `python scripts/plot_sup07.py` to generate PNGs")
    overall = passed_1 and passed_2
    print(f"\nOverall: {'PASS' if overall else 'FAIL'}")

    manifest = BenchmarkManifest()
    manifest.append(
        task="SUP-07",
        config=cfg,
        sequences=sequences,
        artifacts=[
            str(summary_path),
            str(ape_path),
            *[str(out_dir / f"degeneracy_{s}_baseline.csv") for s in sequences],
        ],
        metrics={
            "threshold": cond_threshold,
            "threshold_percentile": args.threshold_percentile,
            "baseline_sequence": args.baseline_sequence,
            "inflation_factor": args.inflation_factor,
            "sigma_mode": sigma_mode,
            "summary": summary_rows,
            "acceptance": {
                "c1_cond_ratio": passed_1,
                "c2_ape_no_harm": passed_2,
                "overall": overall,
            },
        },
    )
    print(f"\nAppended SUP-07 record to {manifest.path}")


if __name__ == "__main__":
    main()
