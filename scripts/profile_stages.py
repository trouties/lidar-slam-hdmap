#!/usr/bin/env python3
"""SUP-03: Runtime profiling report.

Runs the pipeline on a KITTI sequence and collects per-stage timing,
then writes a profiling report identifying the Top-1 bottleneck.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scripts.run_pipeline import load_config, run_pipeline_cached
from src.benchmarks import BenchmarkManifest

BENCHMARKS_DIR = Path("benchmarks")


def profile_sequence(
    config: dict,
    sequence: str,
    label: str,
    max_frames: int | None = None,
) -> dict:
    """Run pipeline with all caches disabled to get timing for all stages."""
    out_dir = Path("results") / "profile" / label
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = run_pipeline_cached(
        config=config,
        sequence=sequence,
        cache=None,  # no cache — force all stages to compute
        max_frames=max_frames,
        output_dir=out_dir,
        verbose=True,
    )
    return summary


def write_report(
    summaries: list[dict],
    label: str,
) -> None:
    """Write profiling CSV and markdown report."""
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all timing data
    rows = []
    for s in summaries:
        seq = s["sequence"]
        for stage_name, timing in s.get("timing", {}).items():
            rows.append(
                {
                    "sequence": seq,
                    "stage": stage_name,
                    "n": timing["n"],
                    "p50_ms": f"{timing['p50']:.1f}",
                    "p95_ms": f"{timing['p95']:.1f}",
                    "max_ms": f"{timing['max']:.1f}",
                    "mean_ms": f"{timing['mean']:.1f}",
                    "total_ms": f"{timing['total_ms']:.1f}",
                }
            )

    # CSV
    csv_path = BENCHMARKS_DIR / f"runtime_profile_{label}.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # Find Top-1 bottleneck (sum across sequences)
    stage_totals: dict[str, float] = {}
    for r in rows:
        stage = r["stage"]
        stage_totals[stage] = stage_totals.get(stage, 0) + float(r["total_ms"])

    total_all = sum(stage_totals.values())
    top1_stage = max(stage_totals, key=stage_totals.get) if stage_totals else "N/A"
    top1_pct = stage_totals.get(top1_stage, 0) / total_all * 100 if total_all > 0 else 0

    # For each stage, aggregate p50/p95/max across sequences (median of medians).
    # When only one sequence was profiled the values are taken directly.
    stage_stats: dict[str, dict[str, str]] = {}
    for stage in stage_totals:
        matching = [r for r in rows if r["stage"] == stage]
        if len(matching) == 1:
            r = matching[0]
            stage_stats[stage] = {
                "n": r["n"],
                "p50_ms": r["p50_ms"],
                "p95_ms": r["p95_ms"],
                "max_ms": r["max_ms"],
            }
        else:
            # Multiple sequences: report median of per-sequence p50/p95/max.
            import statistics as _stats

            p50_vals = [float(r["p50_ms"]) for r in matching]
            p95_vals = [float(r["p95_ms"]) for r in matching]
            max_vals = [float(r["max_ms"]) for r in matching]
            total_n = sum(int(r["n"]) for r in matching)
            stage_stats[stage] = {
                "n": str(total_n),
                "p50_ms": f"{_stats.median(p50_vals):.1f}",
                "p95_ms": f"{_stats.median(p95_vals):.1f}",
                "max_ms": f"{max(max_vals):.1f}",
            }

    seqs_profiled = sorted({r["sequence"] for r in rows})

    # Markdown report
    md_path = BENCHMARKS_DIR / f"runtime_profile_{label}.md"
    lines = [
        f"# Runtime Profile: {label}",
        "",
        "## Summary",
        "",
        f"- **Sequences**: {', '.join(seqs_profiled)}",
        f"- **Top-1 bottleneck**: `{top1_stage}` ({top1_pct:.1f}% of total)",
        f"- **Total wall time**: {total_all / 1000:.1f} s",
        "",
        "## Per-Stage Breakdown",
        "",
        "| Stage | n | Total (s) | % | p50 (ms) | p95 (ms) | Max (ms) |",
        "|-------|---|----------|---|----------|----------|----------|",
    ]

    for stage in sorted(stage_totals, key=stage_totals.get, reverse=True):
        total_s = stage_totals[stage] / 1000
        pct = stage_totals[stage] / total_all * 100 if total_all > 0 else 0
        st = stage_stats.get(stage, {})
        n_val = st.get("n", "?")
        p50 = st.get("p50_ms", "?")
        p95 = st.get("p95_ms", "?")
        mx = st.get("max_ms", "?")
        lines.append(f"| {stage} | {n_val} | {total_s:.1f} | {pct:.1f}% | {p50} | {p95} | {mx} |")

    lines.extend(
        [
            "",
            "> **Note — batch stages**: `stage3`, `stage4`, `stage4b`, and `stage5`",
            "> are timed as single batch operations (n=1 per sequence), so",
            "> p50/p95/max all equal the total run time. Only `stage2` reports",
            "> a true per-frame latency distribution.",
            "",
            "> **SUP-02 interaction**: The `stage3_optimization` speedup (4.77×,",
            "> 819 s → 172 s) was measured with `sc_query_stride=5`. The current",
            "> production default (`query_stride=1`, locked by SUP-02 on 2026-04-11)",
            "> runs all frames through SC matching and will show higher Stage 3",
            "> wall time. The algorithmic gain in `_column_cosine_distance` still",
            "> applies; query coverage is the primary runtime driver.",
            "",
            f"*Label: {label}*",
            "",
        ]
    )

    md_path.write_text("\n".join(lines))

    print(f"\n{'=' * 60}")
    print(f"Profiling report written to {md_path}")
    print(f"CSV data: {csv_path}")
    print(f"Top-1 bottleneck: {top1_stage} ({top1_pct:.1f}%)")
    print(f"Total: {total_all / 1000:.1f} s")

    # Manifest
    manifest = BenchmarkManifest()
    manifest.append(
        task="SUP-03",
        config={},
        sequences=[s["sequence"] for s in summaries],
        artifacts=[str(csv_path), str(md_path)],
        metrics={"top1_stage": top1_stage, "top1_pct": top1_pct, "total_s": total_all / 1000},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SUP-03: Runtime Profiling")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--sequence", default="00")
    parser.add_argument("--label", default="baseline", help="Label for this profile run")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit frames per sequence (quick sanity run; full-sequence recommended for report)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    summary = profile_sequence(config, args.sequence, args.label, max_frames=args.max_frames)
    write_report([summary], args.label)


if __name__ == "__main__":
    main()
