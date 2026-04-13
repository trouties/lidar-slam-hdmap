# Runtime Profile: stride1-cache

## Summary

- **Sequences**: 00
- **Top-1 bottleneck**: `stage3` (65.8% of total)
- **Total wall time**: 1992.2 s

## Per-Stage Breakdown

| Stage | n | Total (s) | % | p50 (ms) | p95 (ms) | Max (ms) |
|-------|---|----------|---|----------|----------|----------|
| stage3 | 1 | 1311.5 | 65.8% | 1311458.6 | 1311458.6 | 1311458.6 |
| stage4b | 1 | 635.1 | 31.9% | 635107.8 | 635107.8 | 635107.8 |
| stage5 | 1 | 44.5 | 2.2% | 44470.8 | 44470.8 | 44470.8 |
| stage4 | 1 | 1.1 | 0.1% | 1137.1 | 1137.1 | 1137.1 |

## Stage 3 Sub-Stage Breakdown

Per-sub-stage timing inside `stage3` (the rollup above). These rows are **excluded** from the main % column to avoid double-counting against the parent.

| Sub-stage | n | Total (s) | % of stage3 | p50 (ms) | p95 (ms) | Max (ms) |
|-----------|---|----------|------------|----------|----------|----------|
| stage3_icp_verify | 8285 | 650.5 | 49.6% | 77.2 | 176.1 | 442.1 |
| stage3_sc_query | 4541 | 575.5 | 43.9% | 132.7 | 177.3 | 273.2 |
| stage3_graph_optimize | 1 | 0.5 | 0.0% | 517.3 | 517.3 | 517.3 |

*Sub-stage coverage: 1226.5s / 1311.5s = 93.5% of stage3 rollup.*

> **Note — batch stages**: `stage3`, `stage4`, `stage4b`, and `stage5`
> are timed as single batch operations (n=1 per sequence), so
> p50/p95/max all equal the total run time. Only `stage2` and the
> Stage 3 sub-stages (`sc_query` per frame, `icp_verify` per
> candidate) report true distributions.

*Label: stride1-cache*
