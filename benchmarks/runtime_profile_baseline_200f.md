# Runtime Profile: baseline_200f

## Summary

- **Sequences**: 00
- **Top-1 bottleneck**: `stage2` (59.0% of total)
- **Total wall time**: 50.6 s

## Per-Stage Breakdown

| Stage | n | Total (s) | % | p50 (ms) | p95 (ms) | Max (ms) |
|-------|---|----------|---|----------|----------|----------|
| stage2 | 200 | 29.9 | 59.0% | 145.3 | 204.1 | 259.2 |
| stage4b | 1 | 15.0 | 29.7% | 15036.1 | 15036.1 | 15036.1 |
| stage3 | 1 | 4.6 | 9.1% | 4582.0 | 4582.0 | 4582.0 |
| stage5 | 1 | 1.0 | 2.1% | 1046.2 | 1046.2 | 1046.2 |
| stage4 | 1 | 0.0 | 0.1% | 49.4 | 49.4 | 49.4 |

> **Note — batch stages**: `stage3_optimization`, `stage4_fusion`,
> `stage4b_map_master`, and `stage5_features` are timed as single
> batch operations (n=1 per sequence), so p50/p95/max all equal the
> total run time. Only `stage2_odometry` reports a true per-frame
> latency distribution.

> **SUP-02 interaction**: The `stage3_optimization` speedup (4.77×,
> 819 s → 172 s) was measured with `sc_query_stride=5`. The current
> production default (`query_stride=1`, locked by SUP-02 on 2026-04-11)
> runs all frames through SC matching and will show higher Stage 3
> wall time. The algorithmic gain in `_column_cosine_distance` still
> applies; query coverage is the primary runtime driver.

*Label: baseline_200f*
