"""Generate SUP-09 compose reproduction artifacts: results/ape.txt + trajectory.png.

Runs as the third step of the compose entrypoint (see
``scripts/sup09_entrypoint.sh``). Expects ``scripts/run_pipeline.py`` to have
already written ``poses_fused_<SEQ>.txt`` / ``poses_optimized_<SEQ>.txt`` /
``poses_<SEQ>.txt`` under ``--results-dir``. The full-sequence GT lives at
``<dataset_root>/poses/<SEQ>.txt`` (same tarball as the 200-frame velodyne
subset; the pipeline only consumes the leading ``max_frames`` rows).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.kitti_loader import load_poses  # noqa: E402
from src.odometry.kiss_icp_wrapper import evaluate_odometry  # noqa: E402


def _load_kitti_poses_txt(path: Path) -> np.ndarray:
    """Load a KITTI-format poses file as an (N, 4, 4) array."""
    return load_poses(path)


def _pick_est_poses(results_dir: Path, sequence: str) -> tuple[Path, np.ndarray]:
    """Choose the best available estimated-poses file (fused > optimized > raw)."""
    for stem in ("poses_fused", "poses_optimized", "poses"):
        candidate = results_dir / f"{stem}_{sequence}.txt"
        if candidate.exists():
            return candidate, _load_kitti_poses_txt(candidate)
    raise FileNotFoundError(
        f"no poses_{{fused,optimized,}}_{sequence}.txt found under {results_dir}"
    )


def _compute_ape(est: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    n = min(est.shape[0], gt.shape[0])
    est_list = [est[i] for i in range(n)]
    gt_list = [gt[i] for i in range(n)]
    metrics = evaluate_odometry(est_list, gt_list)
    ape = metrics["ape"]
    rpe = metrics.get("rpe", {})
    return {
        "n_frames": float(n),
        "ape_rmse": float(ape["rmse"]),
        "ape_mean": float(ape["mean"]),
        "ape_median": float(ape["median"]),
        "ape_std": float(ape["std"]),
        "ape_min": float(ape["min"]),
        "ape_max": float(ape["max"]),
        "rpe_rmse": float(rpe.get("rmse", float("nan"))),
    }


def _write_ape_txt(path: Path, stats: dict[str, float], est_source: Path) -> None:
    lines = [
        f"# SUP-09 APE report (source: {est_source.name})",
        f"n_frames={int(stats['n_frames'])}",
        f"ape_rmse={stats['ape_rmse']:.6f}",
        f"ape_mean={stats['ape_mean']:.6f}",
        f"ape_median={stats['ape_median']:.6f}",
        f"ape_std={stats['ape_std']:.6f}",
        f"ape_min={stats['ape_min']:.6f}",
        f"ape_max={stats['ape_max']:.6f}",
        f"rpe_rmse={stats['rpe_rmse']:.6f}",
    ]
    path.write_text("\n".join(lines) + "\n")


def _plot_trajectory(
    est: np.ndarray,
    gt: np.ndarray,
    output: Path,
    title: str,
) -> None:
    """Plot BEV trajectory comparison.

    KITTI camera frame: x right, y down, z forward. BEV uses (x, z).
    """
    n = min(est.shape[0], gt.shape[0])
    est_xz = est[:n, [0, 2], 3]
    gt_xz = gt[:n, [0, 2], 3]

    fig, ax = plt.subplots(figsize=(7, 7), dpi=120)
    ax.plot(gt_xz[:, 0], gt_xz[:, 1], "k-", linewidth=2.0, label="Ground truth")
    ax.plot(est_xz[:, 0], est_xz[:, 1], "r--", linewidth=1.5, label="Estimated")
    ax.scatter([gt_xz[0, 0]], [gt_xz[0, 1]], c="green", s=80, marker="o", zorder=5, label="Start")
    ax.scatter(
        [gt_xz[-1, 0]], [gt_xz[-1, 1]], c="blue", s=80, marker="s", zorder=5, label="End (GT)"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--sequence", default="00")
    ap.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="KITTI dataset root containing poses/<SEQ>.txt",
    )
    ap.add_argument(
        "--ape-out",
        type=Path,
        default=None,
        help="override output path for ape.txt (default: <results-dir>/ape.txt)",
    )
    ap.add_argument(
        "--plot-out",
        type=Path,
        default=None,
        help="override output path for trajectory.png (default: <results-dir>/trajectory.png)",
    )
    args = ap.parse_args(argv)

    results_dir: Path = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    est_path, est = _pick_est_poses(results_dir, args.sequence)
    gt_path = args.dataset_root / "poses" / f"{args.sequence}.txt"
    if not gt_path.exists():
        raise FileNotFoundError(f"GT poses file missing: {gt_path}")
    gt = _load_kitti_poses_txt(gt_path)

    stats = _compute_ape(est, gt)
    ape_out = args.ape_out or (results_dir / "ape.txt")
    _write_ape_txt(ape_out, stats, est_source=est_path)

    plot_out = args.plot_out or (results_dir / "trajectory.png")
    _plot_trajectory(
        est,
        gt,
        plot_out,
        title=(
            f"SUP-09 Seq {args.sequence} × {int(stats['n_frames'])} frames "
            f"(APE RMSE {stats['ape_rmse']:.2f} m)"
        ),
    )

    print(
        f"[sup09-report] wrote {ape_out} and {plot_out} "
        f"(APE RMSE={stats['ape_rmse']:.3f} m, n={int(stats['n_frames'])})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
