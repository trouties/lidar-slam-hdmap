# Runtime Profile: stride1-baseline

## Summary

- **Sequences**: 00
- **Top-1 bottleneck**: `stage3` (63.8% of total)
- **Total wall time**: 4493.6 s

## Per-Stage Breakdown

| Stage | n | Total (s) | % | p50 (ms) | p95 (ms) | Max (ms) |
|-------|---|----------|---|----------|----------|----------|
| stage3 | 1 | 2866.2 | 63.8% | 2866200.0 | 2866200.0 | 2866200.0 |
| stage4b | 1 | 839.8 | 18.7% | 839808.6 | 839808.6 | 839808.6 |
| stage2 | 4541 | 742.9 | 16.5% | 156.1 | 214.7 | 1060.7 |
| stage5 | 1 | 43.5 | 1.0% | 43497.4 | 43497.4 | 43497.4 |
| stage4 | 1 | 1.1 | 0.0% | 1139.1 | 1139.1 | 1139.1 |

## Stage 3 Sub-Stage Breakdown

Per-sub-stage timing inside `stage3` (the rollup above). These rows are **excluded** from the main % column to avoid double-counting against the parent.

| Sub-stage | n | Total (s) | % of stage3 | p50 (ms) | p95 (ms) | Max (ms) |
|-----------|---|----------|------------|----------|----------|----------|
| stage3_icp_verify | 8285 | 2188.8 | 76.4% | 255.4 | 344.6 | 668.2 |
| stage3_sc_query | 4541 | 608.8 | 21.2% | 143.0 | 179.2 | 218.3 |
| stage3_graph_optimize | 1 | 1.0 | 0.0% | 991.0 | 991.0 | 991.0 |

*Sub-stage coverage: 2798.6s / 2866.2s = 97.6% of stage3 rollup.*

> **Note — batch stages**: `stage3`, `stage4`, `stage4b`, and `stage5`
> are timed as single batch operations (n=1 per sequence), so
> p50/p95/max all equal the total run time. Only `stage2` and the
> Stage 3 sub-stages (`sc_query` per frame, `icp_verify` per
> candidate) report true distributions.

*Label: stride1-baseline*
