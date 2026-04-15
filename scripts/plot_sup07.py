#!/usr/bin/env python3
"""SUP-07: Visualizations from the eval_sup07.py artifacts.

Reads ``benchmarks/sup07/degeneracy_{seq}_baseline.csv`` and emits:
  - degeneracy_timeseries_seq{XX}.png   : cond_number vs frame (log-y)
  - degeneracy_bev_seq{XX}.png          : BEV trajectory colored by cond
  - cond_distribution_hist.png          : Seq 00 vs Seq 01 histograms

Usage::

    python scripts/plot_sup07.py --sequences 00,01

Threshold is read from benchmarks/sup07/degeneracy_summary.csv when present.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_pipeline import _apply_hysteresis  # noqa: E402

DEFAULT_DIR = Path("benchmarks/sup07")


def _load_csv(path: Path) -> dict[str, np.ndarray]:
    cols: dict[str, list[float]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                try:
                    cols.setdefault(k, []).append(float(v))
                except ValueError:
                    cols.setdefault(k, []).append(float("nan"))
    return {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}


def _load_poses_kitti(path: Path) -> np.ndarray:
    """Return ``(N, 3)`` xyz from a KITTI-format poses file."""
    rows = np.loadtxt(path)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    return rows[:, [3, 7, 11]]  # tx, ty, tz columns of the 3x4 pose


def _read_threshold(summary_path: Path, default: float = 0.0) -> float:
    if not summary_path.exists():
        return default
    with summary_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                return float(row["threshold"])
            except (KeyError, ValueError):
                continue
    return default


def plot_timeseries(seq: str, csv_path: Path, out_path: Path, threshold: float) -> None:
    data = _load_csv(csv_path)
    cond = data["cond_number"]
    frames = np.arange(cond.size)
    finite = np.isfinite(cond)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frames[finite], cond[finite], color="tab:blue", lw=0.8, label="cond_number")
    if threshold > 0:
        ax.axhline(
            threshold, color="tab:red", linestyle="--", lw=1.0, label=f"threshold={threshold:.2f}"
        )
    if finite.any():
        p50 = float(np.percentile(cond[finite], 50))
        p95 = float(np.percentile(cond[finite], 95))
        ax.axhline(p50, color="gray", linestyle=":", lw=0.8, label=f"p50={p50:.2f}")
        ax.axhline(p95, color="dimgray", linestyle=":", lw=0.8, label=f"p95={p95:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("frame")
    ax.set_ylabel("cond_number (log)")
    ax.set_title(f"SUP-07 degeneracy cond_number — KITTI Seq {seq}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return list of (start, end_exclusive) for each True run."""
    runs = []
    in_run = False
    start = 0
    for i, v in enumerate(mask):
        if v and not in_run:
            start = i
            in_run = True
        elif not v and in_run:
            runs.append((start, i))
            in_run = False
    if in_run:
        runs.append((start, mask.size))
    return runs


def plot_bev(
    seq: str,
    csv_path: Path,
    poses_path: Path,
    out_path: Path,
    threshold: float,
    ema_alpha: float = 0.3,
    min_consecutive: int = 5,
) -> None:
    if not poses_path.exists():
        print(f"  skip BEV: {poses_path} missing")
        return
    data = _load_csv(csv_path)
    cond = data["cond_number"]
    xyz = _load_poses_kitti(poses_path)
    n = min(cond.size, xyz.shape[0])
    cond = cond[:n]
    xyz = xyz[:n]

    # Camera frame: x right, z forward. BEV uses (x, z).
    fig, ax = plt.subplots(figsize=(8, 7))
    finite = np.isfinite(cond)

    # Background trajectory (lightgray connector).
    ax.plot(xyz[:, 0], xyz[:, 2], color="lightgray", lw=0.6, zorder=1)

    # Gradient-colored scatter of cond_number (log scale).
    vmin = max(1.0, float(np.nanmin(cond[finite])) if finite.any() else 1.0)
    vmax = float(np.nanmax(cond[finite])) if finite.any() else vmin * 10
    sc = ax.scatter(
        xyz[finite, 0],
        xyz[finite, 2],
        c=cond[finite],
        cmap="Reds",
        norm=LogNorm(vmin=vmin, vmax=vmax),
        s=8,
        alpha=0.8,
        zorder=2,
    )
    # Use make_axes_locatable so the colorbar matches the actual rendered
    # height of the axes. Without this, set_aspect('equal') on a long-thin
    # trajectory (Seq 01 highway) leaves the colorbar towering over a
    # flattened plot area.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(sc, cax=cax, label="cond_number (log)")

    n_sustained = 0
    sustained_frac = 0.0
    if threshold > 0:
        sustained = _apply_hysteresis(
            cond,
            threshold=threshold,
            ema_alpha=ema_alpha,
            min_consecutive=min_consecutive,
        )
        n_sustained = int(sustained.sum())
        sustained_frac = float(n_sustained) / max(int(finite.sum()), 1)

        # Adaptive rendering: two regimes.
        #
        # Sparse regime (sustained < 50% of finite frames, e.g. Seq 00): the
        # sustained mask itself is the story — overlay it as hollow circles.
        #
        # Dense regime (sustained >= 50%, e.g. Seq 01 highway, ground truth
        # per LOAM / KISS-ICP benchmarks): circles everywhere are
        # information-free. Instead, darken the trajectory line along
        # sustained runs, and highlight the **worst** sub-segments (top
        # quartile of cond within the sequence) with large red X markers.
        # This pairs with the time-series double-peak to show "even in a
        # fully-degenerate segment, these specific sub-segments are the
        # worst".
        if sustained.any() and sustained_frac <= 0.5:
            ax.scatter(
                xyz[sustained, 0],
                xyz[sustained, 2],
                facecolors="none",
                edgecolors="black",
                s=35,
                marker="o",
                linewidths=0.8,
                label=f"sustained (cond>{threshold:.2f}, run≥{min_consecutive})",
                zorder=3,
            )
        elif sustained_frac > 0.5 and finite.any():
            # Dense regime: trajectory line goes black on sustained runs.
            for s_start, s_end in _contiguous_runs(sustained):
                ax.plot(
                    xyz[s_start:s_end, 0],
                    xyz[s_start:s_end, 2],
                    color="black",
                    lw=1.8,
                    alpha=0.55,
                    zorder=2.5,
                )
            # Top-quartile overlay: within-sequence p75 of cond.
            finite_cond = cond[finite]
            if finite_cond.size:
                p75 = float(np.percentile(finite_cond, 75))
                top_mask = finite & (cond > p75)
                if top_mask.any():
                    ax.scatter(
                        xyz[top_mask, 0],
                        xyz[top_mask, 2],
                        marker="X",
                        s=55,
                        c="red",
                        edgecolors="black",
                        linewidths=0.6,
                        label=f"top-25% cond (>{p75:.1f}, n={int(top_mask.sum())})",
                        zorder=4,
                    )

        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize=8)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m] (camera frame)")
    ax.set_ylabel("z [m] (forward, camera frame)")
    subtitle = f"n_sustained={n_sustained}/{int(finite.sum())}"
    if sustained_frac > 0.5:
        subtitle += " · dense regime, top-25% highlighted"
    ax.set_title(f"SUP-07 BEV — KITTI Seq {seq}\n{subtitle}", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_hist(
    seq_to_csv: dict[str, Path],
    out_path: Path,
    threshold: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    bins = np.logspace(np.log10(1.0), np.log10(1000.0), 60)
    for i, (seq, path) in enumerate(seq_to_csv.items()):
        if not path.exists():
            continue
        data = _load_csv(path)
        cond = data["cond_number"]
        cond = cond[np.isfinite(cond)]
        if cond.size == 0:
            continue
        ax.hist(
            cond,
            bins=bins,
            alpha=0.55,
            label=f"Seq {seq} (n={cond.size}, p50={np.percentile(cond, 50):.2f})",
            color=colors[i % len(colors)],
        )
    if threshold > 0:
        ax.axvline(
            threshold, color="red", linestyle="--", lw=1.0, label=f"threshold={threshold:.2f}"
        )
    ax.set_xscale("log")
    ax.set_xlabel("cond_number (log)")
    ax.set_ylabel("count")
    ax.set_title("SUP-07 cond_number distribution")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DEFAULT_DIR))
    parser.add_argument("--sequences", default="00,01")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override cond_number threshold; defaults to value in degeneracy_summary.csv",
    )
    parser.add_argument("--ema-alpha", type=float, default=0.3)
    parser.add_argument("--min-consecutive", type=int, default=5)
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    sequences = [s.strip() for s in args.sequences.split(",") if s.strip()]

    threshold = (
        args.threshold
        if args.threshold is not None
        else _read_threshold(in_dir / "degeneracy_summary.csv", default=0.0)
    )
    print(f"Using threshold = {threshold:.4f}")

    seq_csvs: dict[str, Path] = {}
    for seq in sequences:
        csv_path = in_dir / f"degeneracy_{seq}_baseline.csv"
        if not csv_path.exists():
            print(f"  skip seq {seq}: {csv_path} missing")
            continue
        seq_csvs[seq] = csv_path

        plot_timeseries(
            seq,
            csv_path,
            in_dir / f"degeneracy_timeseries_seq{seq}.png",
            threshold,
        )

        poses_path = in_dir / "baseline" / seq / f"poses_optimized_{seq}.txt"
        plot_bev(
            seq,
            csv_path,
            poses_path,
            in_dir / f"degeneracy_bev_seq{seq}.png",
            threshold,
            ema_alpha=args.ema_alpha,
            min_consecutive=args.min_consecutive,
        )

    if seq_csvs:
        plot_hist(seq_csvs, in_dir / "cond_distribution_hist.png", threshold)


if __name__ == "__main__":
    main()
