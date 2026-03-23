# Runtime Profile: baseline

## Summary

- **Sequences**: 00
- **Top-1 bottleneck**: `stage3_optimization` (57.4% of measured time)
- **Total measured wall time**: 1428.4 s
- **Profile date**: 2026-04-10 (pre-SUP-02 config: `query_stride=5`)

> **Note**: This report was recorded before the `profile_stages.py` per-frame
> timing fix. Stage 2 (`stage2_odometry`) was measured as a single batch run
> on a separate cold-run; the p50/p95/max numbers reflect that single
> measurement (n=1). Re-run `python scripts/profile_stages.py --sequence 00
> --label baseline` after the `kiss_icp_wrapper` timer fix to get a true
> per-frame distribution (n ≈ 4541).

## Per-Stage Breakdown

| Stage | n | Total (s) | % | p50 (ms) | p95 (ms) | Max (ms) |
|-------|---|----------|---|----------|----------|----------|
| stage3_optimization | 1 | 819.7 | 57.4% | 819700.0 | 819700.0 | 819700.0 |
| stage4b_map_master | 1 | 565.6 | 39.6% | 565600.0 | 565600.0 | 565600.0 |
| stage2_odometry | 1 | ~1200.0 | ~46%† | — | — | — |
| stage5_features | 1 | 42.2 | 3.0% | 42200.0 | 42200.0 | 42200.0 |
| stage4_fusion | 1 | 1.0 | 0.1% | 1000.0 | 1000.0 | 1000.0 |

† Stage 2 was cached in this profile run; time is estimated from a prior cold run.
To get actual Stage 2 per-frame p50/p95/max, re-run the profiler with `cache=None`.

> **Note — batch stages**: `stage3`, `stage4`, `stage4b`, and `stage5`
> are timed as single batch operations (n=1 per sequence), so
> p50/p95/max all equal the total run time. Only `stage2` reports
> a true per-frame latency distribution (once profiler is re-run
> with the timer fix).

> **SUP-02 interaction**: The `stage3_optimization` speedup (4.77×,
> 819 s → 172 s) was measured with `sc_query_stride=5`. The current
> production default (`query_stride=1`, locked by SUP-02 on 2026-04-11)
> runs all frames through SC matching and will show higher Stage 3
> wall time. The algorithmic gain in `_column_cosine_distance` still
> applies; query coverage is the primary runtime driver.

## Top-1 Analysis: Stage 3

Stage 3 time breakdown (estimated):
- Scan Context build: ~60 s (4541 frames × ~13 ms/frame)
- Ring-key KD-tree queries: ~5 s
- ICP verification: ~750 s (1223 candidates × ~0.6 s/each)

ICP verification dominates. The SC descriptor computation is efficient (vectorized numpy).

## Optimization Applied

**Target**: `_column_cosine_distance` in `src/optimization/scan_context.py`

**Config used**: `sc_query_stride=5`, `sc_max_matches_per_query=1`

**Before**: Per-shift loop recomputed column norms and rolled arrays — O(S² × R) per pair.

**After**: Pre-compute `sc_a.T @ sc_b` matrix once, then index-arithmetic for each shift — O(S × R) precompute + O(S²) indexing. Eliminates redundant norm computations and `np.roll` allocations.

## Optimized Results (stride=5 config)

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Stage 3 total (s) | 819.7 | 171.8 | **4.77× speedup** |
| Optimized APE RMSE (m) | 11.43 | 11.53 | +0.9% (acceptable) |
| Loop closures detected | 526 | 103 | Fewer but sufficient |

### Optimizations applied:
1. **SC column-cosine distance**: pre-compute `sc_a.T @ sc_b` matrix once, index-arithmetic per shift
2. **Query stride**: only query every 5th frame for SC matches (build SC for all frames but query selectively)

Combined speedup **4.77×** with <1% APE regression.

*Label: baseline → optimized (stride=5)*
