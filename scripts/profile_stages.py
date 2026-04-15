#!/usr/bin/env python3
"""SUP-03: Runtime profiling report.

Runs the pipeline on a KITTI sequence and collects per-stage timing,
then writes a profiling report identifying the Top-1 bottleneck.

Stage 3 is reported with a sub-stage breakdown (sc_query / icp_verify /
graph_optimize) so that Stage 3 optimization iterations can target the
true sub-bottleneck under the production config.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from scripts.run_pipeline import load_config, run_pipeline_cached
from src.benchmarks import BenchmarkManifest
from src.cache import LayeredCache

BENCHMARKS_DIR = Path("benchmarks")

# Stage 3 sub-stage keys are accounted separately to avoid double-counting
# them in the Top-1 bottleneck sum. They show up in the breakdown table.
STAGE3_SUB_KEYS = ("stage3_sc_query", "stage3_icp_verify", "stage3_graph_optimize")


def profile_sequence(
    config: dict,
    sequence: str,
    label: str,
    max_frames: int | None = None,
    cache_upstream: bool = False,
) -> dict:
    """Run pipeline and collect per-stage timing.

    Args:
        config: Parsed YAML config dict.
        sequence: KITTI sequence id.
        label: Free-form label for this profile run.
        max_frames: Optional frame cap (sanity-test only; never use for
            published numbers — partial runs are not representative).
        cache_upstream: When True, reuses any existing layered cache for
            Stage 1–2 (odometry) and forces Stage 3+ to recompute. This is
            for fast Stage 3 optimization iterations and is NOT the
            published-profile mode (those should leave it False to keep
            measurements identical to the original SUP-03 methodology).
    """
    out_dir = Path("results") / "profile" / label
    out_dir.mkdir(parents=True, exist_ok=True)

    if cache_upstream:
        cache = LayeredCache(
            root=config.get("cache", {}).get("root", "cache/kitti"),
            sequence=sequence,
        )
        force_rebuild = "optimized"  # invalidate stage 3 + downstream
        print(
            "  [profile] cache-upstream enabled: reusing odometry cache, forcing Stage 3+ rebuild"
        )
    else:
        cache = None
        force_rebuild = "none"

    summary = run_pipeline_cached(
        config=config,
        sequence=sequence,
        cache=cache,
        force_rebuild=force_rebuild,
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

    # Find Top-1 bottleneck (sum across sequences). Stage 3 sub-stages are
    # excluded from the main totals — they would double-count against the
    # parent `stage3` rollup. They get their own breakdown table below.
    stage_totals: dict[str, float] = {}
    sub_totals: dict[str, float] = {}
    for r in rows:
        stage = r["stage"]
        if stage in STAGE3_SUB_KEYS:
            sub_totals[stage] = sub_totals.get(stage, 0) + float(r["total_ms"])
        else:
            stage_totals[stage] = stage_totals.get(stage, 0) + float(r["total_ms"])

    total_all = sum(stage_totals.values())
    top1_stage = max(stage_totals, key=stage_totals.__getitem__) if stage_totals else "N/A"
    top1_pct = stage_totals.get(top1_stage, 0) / total_all * 100 if total_all > 0 else 0

    # Stage 3 sub-stage rollup (independent of stage3 total) — used to
    # answer "which sub-phase dominates Stage 3 under production config".
    stage3_total_for_breakdown = stage_totals.get("stage3", 0.0)
    stage3_sub_pct: dict[str, float] = {}
    if stage3_total_for_breakdown > 0:
        for k, v in sub_totals.items():
            stage3_sub_pct[k] = v / stage3_total_for_breakdown * 100

    # For each stage, aggregate p50/p95/max across sequences (median of medians).
    # When only one sequence was profiled the values are taken directly.
    # Includes both top-level stages and Stage 3 sub-stages so the breakdown
    # table can render real distribution numbers (n>1 for sc_query/icp_verify).
    stage_stats: dict[str, dict[str, str]] = {}
    all_stage_keys = list(stage_totals.keys()) + list(sub_totals.keys())
    for stage in all_stage_keys:
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

    for stage in sorted(stage_totals, key=stage_totals.__getitem__, reverse=True):
        total_s = stage_totals[stage] / 1000
        pct = stage_totals[stage] / total_all * 100 if total_all > 0 else 0
        st = stage_stats.get(stage, {})
        n_val = st.get("n", "?")
        p50 = st.get("p50_ms", "?")
        p95 = st.get("p95_ms", "?")
        mx = st.get("max_ms", "?")
        lines.append(f"| {stage} | {n_val} | {total_s:.1f} | {pct:.1f}% | {p50} | {p95} | {mx} |")

    # Stage 3 sub-stage breakdown — only emitted when sub-stage timing was
    # actually collected (i.e., Stage 3 was recomputed, not loaded from cache).
    if sub_totals:
        lines.extend(
            [
                "",
                "## Stage 3 Sub-Stage Breakdown",
                "",
                "Per-sub-stage timing inside `stage3` (the rollup above). "
                "These rows are **excluded** from the main % column to avoid "
                "double-counting against the parent.",
                "",
                "| Sub-stage | n | Total (s) | % of stage3 | p50 (ms) | p95 (ms) | Max (ms) |",
                "|-----------|---|----------|------------|----------|----------|----------|",
            ]
        )
        for sub in sorted(sub_totals, key=sub_totals.__getitem__, reverse=True):
            total_s = sub_totals[sub] / 1000
            pct = stage3_sub_pct.get(sub, 0.0)
            st = stage_stats.get(sub, {})
            n_val = st.get("n", "?")
            p50 = st.get("p50_ms", "?")
            p95 = st.get("p95_ms", "?")
            mx = st.get("max_ms", "?")
            lines.append(f"| {sub} | {n_val} | {total_s:.1f} | {pct:.1f}% | {p50} | {p95} | {mx} |")
        # Sanity: sub-stages should account for ~all of stage3 (small overhead
        # from glue code is expected).
        sub_sum_s = sum(sub_totals.values()) / 1000
        stage3_s = stage3_total_for_breakdown / 1000
        if stage3_s > 0:
            coverage = sub_sum_s / stage3_s * 100
            lines.append(
                f"\n*Sub-stage coverage: {sub_sum_s:.1f}s / {stage3_s:.1f}s "
                f"= {coverage:.1f}% of stage3 rollup.*"
            )

    lines.extend(
        [
            "",
            "> **Note — batch stages**: `stage3`, `stage4`, `stage4b`, and `stage5`",
            "> are timed as single batch operations (n=1 per sequence), so",
            "> p50/p95/max all equal the total run time. Only `stage2` and the",
            "> Stage 3 sub-stages (`sc_query` per frame, `icp_verify` per",
            "> candidate) report true distributions.",
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

    # Manifest — include Stage 3 sub-stage totals so future runs can be
    # diffed against this entry to prove Step 3 optimizations actually moved
    # the needle.
    manifest_metrics = {
        "top1_stage": top1_stage,
        "top1_pct": top1_pct,
        "total_s": total_all / 1000,
        "label": label,
    }
    for sub in STAGE3_SUB_KEYS:
        if sub in sub_totals:
            manifest_metrics[f"{sub}_s"] = sub_totals[sub] / 1000
    if "stage3" in stage_totals:
        manifest_metrics["stage3_s"] = stage_totals["stage3"] / 1000

    manifest = BenchmarkManifest()
    manifest.append(
        task="SUP-03",
        config={},
        sequences=[s["sequence"] for s in summaries],
        artifacts=[str(csv_path), str(md_path)],
        metrics=manifest_metrics,
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
    parser.add_argument(
        "--cache-upstream",
        action="store_true",
        help=(
            "Reuse layered cache for Stage 1–2 (odometry); force Stage 3+ "
            "rebuild. Use this for Stage 3 optimization iterations to avoid "
            "re-running ~7 min of upstream work each iteration. NOT for "
            "published profile runs — those should leave this off."
        ),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    summary = profile_sequence(
        config,
        args.sequence,
        args.label,
        max_frames=args.max_frames,
        cache_upstream=args.cache_upstream,
    )
    write_report([summary], args.label)


if __name__ == "__main__":
    main()
