# LiDAR-Inertial SLAM & HD Map Pipeline

> A production-grade LiDAR-inertial SLAM and HD Map feature extraction pipeline on KITTI/nuScenes — with EKF sensor fusion, Scan Context loop closure, and Lanelet2 export.

[![CI](https://img.shields.io/github/actions/workflow/status/trouties/lidar-slam-hdmap/ci.yml?branch=main&label=CI)](https://github.com/trouties/lidar-slam-hdmap/actions)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20WSL2-lightgrey)
![KISS-ICP](https://img.shields.io/badge/KISS--ICP-1.2-brightgreen)
![GTSAM](https://img.shields.io/badge/GTSAM-4.2-orange)
![Open3D](https://img.shields.io/badge/Open3D-0.18-blue)
![Lanelet2](https://img.shields.io/badge/Lanelet2-HD%20Map-blue)

## Pipeline Architecture

```
 Stage 1              Stage 2            Stage 3               Stage 4
 Data Ingestion       LiDAR Odometry     Graph Optimization    Sensor Fusion
┌──────────────┐    ┌───────────────┐   ┌──────────────────┐  ┌─────────────────┐
│ KITTIDataset │    │   KISS-ICP    │   │  GTSAM Pose Graph│  │ Error-State KF  │
│ NuScenesDset │───>│  adaptive ICP │──>│+ Scan Context v2 │─>│+ GTSAM tight    │
│              │    │               │   │+ IMU Preintegr.  │  │  coupling       │
└──────────────┘    └───────────────┘   └──────────────────┘  └────────┬────────┘
  (N,4) point         SE(3) 4x4            optimized SE(3)            │
  clouds              odometry poses       poses                      │
                                                                      v
                     Stage 6              Stage 5              fused poses
                     HD Map Export        Mapping              + point clouds
                    ┌───────────────┐   ┌──────────────────┐         │
                    │ Lanelet2 .osm │<──│ Voxel Map Builder│<────────┘
                    │ + GeoJSON     │   │+ Lane/Curb Extr. │
                    └───────────────┘   └──────────────────┘
                      regulatory HD       PCA-classified
                      map output          lane/curb clusters
```

## Results

### System Accuracy Comparison

Evaluated with [evo](https://github.com/MichaelGrupp/evo) APE (Absolute Pose Error). Lower is better.

| System | Seq 00 APE RMSE (m) | Seq 00 APE Mean (m) | Seq 05 APE RMSE (m) | Seq 05 APE Mean (m) |
|--------|---------------------:|--------------------:|---------------------:|--------------------:|
| **Ours (fused)** | **11.53** | **10.22** | **3.23** | **2.80** |
| hdl_graph_slam | 78.46 | 68.05 | 56.48 | 33.97 |
| FAST-LIO2 | 77.41 | 61.32 | 20.69 | 15.56 |
| LIO-SAM | 552.85 | 506.00 | 968.82 | 891.86 |

> Baselines run in Docker containers on identical KITTI sequences. See [`external/`](external/) for reproduction scripts.

### Stage-by-Stage Accuracy Improvement (Seq 00)

| Pipeline Configuration | APE RMSE (m) | Delta |
|------------------------|-------------:|------:|
| Stage 2: KISS-ICP odometry only | 12.53 | baseline |
| Stage 3: + pose graph + Scan Context loop closure | 11.53 | −8.0% |
| Stage 3†: + IMU tight coupling (GTSAM preintegration) | 9.22 | −20.0% vs loose |
| Stage 4: + ESKF fusion | 11.53 | <0.01 m ‡ |

> † Uses KITTI Raw OxTS data via SUP-04 tight coupling path (Forster 2017 IJRR preintegration factor).
>
> ‡ KITTI Odometry contains no IMU data — ESKF uses a constant-velocity model and cannot improve already-optimized poses. Full ESKF value appears on datasets with raw IMU (nuScenes, KITTI Raw).

### Performance

| Metric | Value |
|--------|------:|
| Stage 2 per-frame latency p50 | 145 ms |
| Stage 2 per-frame latency p95 | 204 ms |
| Full pipeline (200 frames, Seq 00) | 50.6 s |
| Loop closures detected (Seq 00 full) | 2,635 |
| Loop closure precision | 0.967 |
| Loop closure recall | 0.195 |
| Stage 3 speedup (production config, SUP-03 round 2) | 2.19× (2866 s → 1311 s, Seq 00) |
| Stage 3 ICP verify speedup (downsample cache) | 3.36× (p50 255 ms → 77 ms) |
| GNSS denial drift (Seq 00, 150 m window) | 0.003 m/m |

### Cross-Dataset Validation (nuScenes)

All 10 nuScenes mini scenes pass the APE < 10 m acceptance threshold.

| Scene | Frames | Stage 2 APE Mean (m) | Stage 3 APE Mean (m) |
|-------|-------:|---------------------:|---------------------:|
| scene-0553 | 398 | 0.014 | 0.014 |
| scene-0757 | 397 | 0.530 | 0.530 |
| scene-0061 | 382 | 0.698 | 0.698 |
| scene-0103 | 389 | 0.801 | 0.801 |
| scene-0916 | 399 | 0.997 | 0.997 |
| scene-0655 | 396 | 1.908 | 1.908 |
| scene-1094 | 391 | 1.892 | 1.892 |
| scene-0796 | 392 | 2.755 | 2.755 |
| scene-1077 | 400 | 6.730 | 6.730 |
| scene-1100 | 391 | 0.070 | 0.070 |

> KISS-ICP adapted for nuScenes 32-beam VLP-32C: `voxel_size=0.5` (vs KITTI 1.0), `min_range=3.0` (vs 5.0), 20 Hz sweep mode (2 Hz keyframes cause ICP divergence).

### Pose Graph Uncertainty Under GNSS Denial (SUP-06)

Per-keyframe marginal covariance is extracted from GTSAM (`jointMarginalCovariance`) and rendered as 3D 2σ position ellipsoids. A 354-frame GNSS-denied window (frames 2270–2624) in the middle of Seq 00 inflates `trace(Σ_pos)` by **>26×** relative to the non-prior drift baseline, then collapses back within 1.07× as priors resume.

| Mode | drift baseline | denial peak | **peak / baseline** | post / baseline |
|------|---------------:|------------:|--------------------:|----------------:|
| Loose (LiDAR + pose graph) | 0.268 m² | 7.131 m² | **26.61×** | 1.07× |
| Tight (+ IMU preintegration, SUP-04) | 0.253 m² | 6.990 m² | **27.59×** | 1.07× |

![SUP-06 loose GNSS denial uncertainty](benchmarks/uncertainty/ellipsoid_animation_00_loose.gif)
![SUP-06 tight (IMU) GNSS denial uncertainty](benchmarks/uncertainty/ellipsoid_animation_00_tight.gif)

> Both modes pass the acceptance bar (`peak / baseline ≥ 2×`, `post / baseline ≤ 1.5×`). Baseline is computed as the median of non-prior drift frames outside the denial window and tail buffer — averaging all samples would mix prior anchors (~3 × 10⁻⁴ m²) with dead-reckoning drift (~0.27 m²), quantities that differ by three orders of magnitude. Reproduce with `python -m scripts.run_sup06 --sequence 00 --mode both`.

### LiDAR Degeneracy Detection (SUP-07)

A 3×3 translation-block Hessian `H_t = Σ nᵢnᵢᵀ` (point-to-plane, PCA-normal gated) and an EMA + min-run hysteresis detector flag directionally under-observed frames and downgrade their odometry-edge translation sigmas by 10×. Tuned to separate KITTI Seq 00 (urban, well-constrained) from Seq 01 (highway, LOAM-benchmark degenerate):

![SUP-07 cond_number distribution](benchmarks/sup07/cond_distribution_hist.png)

| Sequence | cond p50 | cond p95 | sustained frames | APE (baseline → downgrade) |
|---|---:|---:|---:|---:|
| Seq 00 (urban, 4540 f) | 3.07 | **5.51** ← threshold | 182 (11 real runs) | 10.577 → **10.552** m (−0.24%) |
| Seq 01 (highway, 1100 f) | **12.38** | 45.13 | 1080 (3 real runs) | 116.80 m unchanged † |

![SUP-07 Seq 01 BEV, dense regime](benchmarks/sup07/degeneracy_bev_seq01.png)

Both acceptance criteria pass: (1) Seq 01 `cond_p50` ≥ 2×Seq 00 `cond_p95` (gap 1.12×), (2) APE no-regression. Seq 01's 98% sustained rate is ground truth — the whole highway IS degenerate, and per-frame downgrade correctly collapses to sequence-level downgrade. The BEV above uses a log-gradient colormap for cond magnitude, a black trajectory overlay for the sustained runs, and red X markers for the top-25% sub-spans (~275 frames) — two clusters near frames 480–540 and 800–830 are the worst within an already-degenerate segment.

> † Seq 01 has zero loop closures in the production config; with only the anchor prior at frame 0, the pose graph is uniquely determined by the initial trajectory and edge-sigma changes have no effect. The "no harm" criterion is trivially satisfied. On a sequence with denial-aware GNSS priors, the SUP-07 downgrade hands position work over to the SUP-04 IMU factor — the two are complementary.

## Scope

The complete chain from raw LiDAR scans to Lanelet2 HD Maps — the open standard used by Autoware, Apollo, and European OEMs — covering localization, mapping, and map-layer extraction in a single reproducible workflow. This is the infrastructure layer that most portfolio projects skip: not perception alone (3D detection, lane segmentation), but the localization → mapping → HD map export chain that feeds downstream planning and control.

The author's geodetic-science background shapes the implementation: explicit WGS84 → UTM (EPSG:32632) reference frames, rigorous Velodyne ↔ camera ↔ world calibration chains, and GTSAM's factor-graph optimization treated as a generalization of the least-squares network adjustment geodesists have used for two centuries.

## Key Features

- **Multi-dataset SLAM** — KITTI (HDL-64E, 10 Hz) and nuScenes (VLP-32C, 20 Hz sweeps) with per-dataset parameter adaptation.
- **Scan Context v2 loop closure** — appearance-based place recognition, 2,635 closures on Seq 00 at precision 0.967 (ICP fitness gate 0.9).
- **Tight-coupled IMU preintegration** — GTSAM Forster-2017 factor, −20% APE vs loose fusion on Seq 00 when IMU is available.
- **Lanelet2 HD Map export** — PCA-classified lane / curb morphology, RDP-simplified (ε = 0.05 m), written as Lanelet2 `.osm` with geometry metadata tags.
- **4-system baseline comparison** — Dockerized hdl_graph_slam / FAST-LIO2 / LIO-SAM with full APE/RPE tables on KITTI Seq 00, 05.
- **Runtime profiling + Stage-3 2.19× speedup** — per-frame latency distributions; the production `sc_query_stride=1` bottleneck (8,285 candidates × 2 downsamples) collapses 3.36× via a per-unique-frame downsample cache, zero APE regression.
- **Pose graph uncertainty under GNSS denial (SUP-06)** — GTSAM marginals → 3D 2σ ellipsoids, `trace(Σ_pos)` inflates 26–28× inside a 354-frame denial window and recovers within 1.07×.
- **LiDAR degeneracy detection (SUP-07)** — 3×3 translation Hessian + EMA/hysteresis detector, per-edge σ downgrade on sustained degenerate runs. Seq 01 separation gap 1.12× over Seq 00 p95 with 182 / 1080 sustained frames respectively.
- **5-layer deterministic cache** — odometry → optimized → fused → master map → features, enabling 15-minute Stage-5 iteration cycles vs 2 h 20 min cold runs.

## Quick Start

### Docker

```bash
# Build and run on KITTI Seq 00
docker build -t slam-pipeline -f docker/Dockerfile .
docker run -v ~/data/kitti:/data/kitti slam-pipeline --config configs/default.yaml
```

### Native Installation (WSL2 / Linux)

<details>
<summary>Click to expand native setup instructions</summary>

#### Prerequisites

- Ubuntu 22.04 (native or WSL2)
- Python 3.10

#### 1. Create virtual environment

```bash
python3.10 -m venv ~/slam-env
source ~/slam-env/bin/activate
```

#### 2. Install dependencies

```bash
# numpy MUST be installed first and stay <2.0 (GTSAM binary compatibility)
pip install "numpy>=1.26,<2.0"
pip install -e ".[dev]"
```

#### 3. Install Lanelet2

```bash
# Option A: Official lanelet2 (requires libboost-dev)
pip install lanelet2

# Option B: lanelet2x (pure Python, cross-platform fallback)
pip install lanelet2x
```

#### 4. Download KITTI Odometry

1. Register at [cvlibs.net](https://www.cvlibs.net/datasets/kitti/user_register.php)
2. Download from the [odometry evaluation page](https://www.cvlibs.net/datasets/kitti/eval_odometry.php):
   - **Velodyne laser data** (80 GB uncompressed)
   - **Calibration files** (1 MB)
   - **Ground truth poses** (4 KB, sequences 00–10 only)
3. Extract into `~/data/kitti/odometry/dataset/`:

```
~/data/kitti/odometry/dataset/
├── sequences/
│   ├── 00/
│   │   ├── velodyne/       # .bin point cloud files
│   │   ├── calib.txt
│   │   └── times.txt
│   └── ...
└── poses/
    ├── 00.txt              # ground truth (seq 00–10)
    └── ...
```

> **WSL2 users**: Store data under `~/data/`, **not** `/mnt/c/` — cross-filesystem I/O is 10× slower.

#### 5. Verify and run

```bash
# Verify KITTI data integrity
python scripts/verify_kitti.py --root ~/data/kitti/odometry/dataset --sequence 00

# Run the full pipeline
python scripts/run_pipeline.py --config configs/default.yaml

# Quick test (200 frames, ~2 min)
python scripts/run_pipeline.py --max-frames 200

# Lint and test
ruff check src/ && ruff format --check src/
pytest tests/ -v
```

</details>

## Repository Structure

```
lidar-slam-hdmap/
├── src/
│   ├── data/                # Stage 1 — KITTI, nuScenes, IMU loaders + coordinate transforms
│   ├── odometry/            # Stage 2 — KISS-ICP wrapper with per-frame timing
│   ├── optimization/        # Stage 3 — GTSAM pose graph, Scan Context, loop closure, IMU factor
│   ├── fusion/              # Stage 4 — Error-State Kalman Filter
│   ├── mapping/             # Stage 5 — Streaming voxel map builder + lane/curb feature extraction
│   ├── export/              # Stage 6 — Lanelet2 OSM export with PCA classification
│   ├── visualization/       # Trajectory plots, point cloud rendering, SUP-06 ellipsoid + animation
│   ├── benchmarks/          # Evaluator, timing, GNSS denial, benchmark manifest
│   └── cache/               # 5-layer deterministic cache (odometry → features)
├── scripts/
│   ├── run_pipeline.py      # Main entry point (all 6 stages)
│   ├── benchmark_stage5.py  # 11-sequence Stage 5 iteration benchmarking
│   ├── compare_tight_vs_loose.py  # SUP-04 IMU coupling comparison
│   ├── eval_nuscenes.py     # SUP-05 cross-dataset evaluation
│   ├── run_sup06.py         # SUP-06 uncertainty visualization (GNSS denial)
│   ├── profile_stages.py    # SUP-03 per-frame latency profiling
│   └── run_baseline_compare.py    # SUP-01 4-system comparison
├── tests/                   # 11 test modules (pytest)
├── configs/default.yaml     # All pipeline parameters with tuning rationale
├── benchmarks/              # CSV outputs, runtime profiles, benchmark manifest
├── external/                # Dockerized baselines (LIO-SAM, FAST-LIO2, hdl_graph_slam)
├── docker/Dockerfile
├── .github/workflows/ci.yml # Ruff lint + pytest
└── refs/                    # Specs, pipeline notes, tuning history
```

## Pipeline Stages

| # | Stage | Input → Output | Key decision |
|---|-------|----------------|--------------|
| 1 | **Data Ingestion** | KITTI `.bin` / nuScenes sweeps → `(N, 4)` ndarrays + GT poses | All processing in Velodyne frame (x-forward); camera-frame conversion only at evaluation / export. |
| 2 | **LiDAR Odometry** | `(N, 3)` points → SE(3) odometry poses | KISS-ICP adaptive-threshold; dataset-specific `voxel_size` / `min_range` to compensate for beam-count differences. |
| 3 | **Graph Optimization** | odometry + clouds + IMU → globally optimized SE(3) | GTSAM LM with Scan Context v2 loop closure, ICP fitness gate 0.9 for precision, optional GTSAM IMU preintegration, optional SUP-07 per-edge sigma downgrade. |
| 4 | **Sensor Fusion** | optimized poses + IMU → fused SE(3) | Dual path: ESKF (constant-velocity fallback) and GTSAM tight coupling (when raw IMU available, −20% APE). |
| 5 | **Mapping & Features** | fused poses + clouds → lane / curb clusters + global map | NumPy streaming voxel aggregation (<4 GB RAM). Lane: `intensity ≥ 0.40` + DBSCAN. Curb: height-jump in 0.30 m grid. |
| 6 | **HD Map Export** | PCA-classified clusters → Lanelet2 `.osm` | RDP polyline simplification (ε = 0.05 m), separate lane (3-class) and curb (single-class with rescue trim) pipelines, geometry metadata tags. |

### Supplement Tasks

**Completed (P0 + P1):** 4-system baseline comparison (SUP-01), Scan Context v2 loop closure (SUP-02), runtime profiling + Stage 3 2.19× speedup (SUP-03), IMU tight coupling (SUP-04), nuScenes cross-dataset evaluation (SUP-05), pose graph uncertainty under GNSS denial (SUP-06), LiDAR degeneracy detection (SUP-07).

**Planned:** ROS2 node wrapping, Docker Compose one-command demo, Lanelet2 routing graph + A*, iSAM2 fixed-lag smoother comparison, interactive web demo, and HD map semantic layer extension.

## Benchmark Report

Every benchmark run is tracked in [`benchmarks/benchmark_manifest.json`](benchmarks/benchmark_manifest.json) with fields: `run_id`, `git_sha`, `config_hash`, `timestamp`, `label`, and artifact paths.

Key data files:

| File | Content |
|------|---------|
| [`benchmarks/accuracy_table.csv`](benchmarks/accuracy_table.csv) | APE/RPE across 4 systems × 2 sequences |
| [`benchmarks/nuscenes_ape.csv`](benchmarks/nuscenes_ape.csv) | nuScenes 10-scene cross-dataset results |
| [`benchmarks/tight_vs_loose/ape_compare.csv`](benchmarks/tight_vs_loose/ape_compare.csv) | IMU tight vs loose coupling comparison |
| [`benchmarks/robustness_gnss_denied.csv`](benchmarks/robustness_gnss_denied.csv) | GNSS denial drift measurements |
| [`benchmarks/runtime_profile_baseline_200f.csv`](benchmarks/runtime_profile_baseline_200f.csv) | Per-stage latency profile (200 frames) |
| [`benchmarks/uncertainty/sup06_report_00.json`](benchmarks/uncertainty/sup06_report_00.json) | SUP-06 marginal covariance acceptance metrics (loose + tight) |
| [`benchmarks/uncertainty/marginal_cov_00_loose.csv`](benchmarks/uncertainty/marginal_cov_00_loose.csv) | Per-keyframe 3×3 position marginals + bucket flags (459 rows) |
| [`benchmarks/sup07/degeneracy_summary.csv`](benchmarks/sup07/degeneracy_summary.csv) | SUP-07 per-sequence cond statistics + APE baseline vs downgrade |
| [`benchmarks/sup07/ape_compare.csv`](benchmarks/sup07/ape_compare.csv) | SUP-07 two-pass APE comparison with per-sequence downgrade counts |

## Datasets

| Dataset | Sensor | Sequences | Frames | Purpose |
|---------|--------|-----------|-------:|---------|
| [KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) | HDL-64E (64-beam, 10 Hz) | 00–10 (with GT) | 4,541 (Seq 00) | Primary benchmark, all stages |
| [nuScenes mini](https://www.nuscenes.org/nuscenes) | VLP-32C (32-beam, 20 Hz sweeps) | 10 scenes | 382–400/scene | Cross-dataset generalization (SUP-05) |
| [MulRan](https://sites.google.com/view/mulran-pr/) | Ouster OS1-64 | — | — | Planned: multi-session loop closure |

## Tech Stack

| Layer | Component | Role |
|-------|-----------|------|
| **Perception** | [KISS-ICP](https://github.com/PRBonn/kiss-icp) | Adaptive-threshold point-to-point ICP odometry |
| **Perception** | [Open3D](http://www.open3d.org/) | ICP verification, point cloud processing, visualization |
| **Optimization** | [GTSAM 4.2](https://gtsam.org/) | Factor graph, Levenberg-Marquardt, IMU preintegration |
| **Optimization** | Scan Context | Appearance-based loop closure descriptor (self-implemented) |
| **Fusion** | Error-State KF | IMU-less constant-velocity fallback (self-implemented) |
| **Mapping** | NumPy streaming voxel | Memory-safe global map aggregation (<4 GB) |
| **Mapping** | DBSCAN (scikit-learn) | Lane marking + curb boundary clustering |
| **Export** | [Lanelet2](https://github.com/fzi-forschungszentrum-informatik/Lanelet2) | HD Map standard, OSM XML format |
| **Evaluation** | [evo](https://github.com/MichaelGrupp/evo) | APE/RPE trajectory metrics |
| **Infrastructure** | Docker | Reproducible baseline comparison containers |
| **CI** | GitHub Actions | Ruff lint + pytest on every push |

## Known Limitations

| # | Mode | Root cause & status |
|---|------|---------------------|
| FM-1 | **ESKF adds no value on KITTI Odometry** (Stage 4 APE ≈ Stage 3) | KITTI Odometry has no IMU; ESKF falls back to constant-velocity which cannot improve already-optimized poses. By design — use KITTI Raw / nuScenes. Tight coupling (SUP-04) demonstrates the real IMU benefit (−20% APE). |
| FM-2 | **Loop closure recall capped at ~20%** (P = 0.967, R = 0.195) | GT defines ~13 050 valid pairs; `max_matches_per_query=5` structurally caps at ~22%, ICP@0.9 prunes further. Raising the cap yields diminishing returns with linear ICP cost. |
| FM-3 | **Empty lanelet relations** (only LineStrings / Areas in `.osm`) | Curb-driven left/right boundary pairing not yet implemented. P2 task, prerequisite for SUP-12 routing graph. |
| FM-4 | **Flat-ground assumption** (lane / curb lost on hills, multi-level) | Fixed `road_z_min/max = [-2.0, -1.5]` window. Requires per-frame terrain adaptation. P3 task. |
| FM-5 | **Conservative IMU noise lock** (`accel_noise_sigma = 5.0`) | KITTI OxTS is filtered navigation data, not raw IMU. Datasheet-level σ = 0.3 causes APE to explode to 27.85 m due to timestamp alignment and calibration residuals. Locked until raw-IMU dataset is available. |
| FM-6 | **No traffic sign / signal extraction** | Stage 5 filters the road-plane z-band only; vertical structures are excluded by design. SUP-17 (P2) will add heuristic stop-line / crosswalk detection. |
| FM-7 | **Seq 01 has zero loop closures in production config** | Highway has no revisits — Scan Context cannot fire. Edge-sigma downgrades therefore have no effect on Seq 01 APE even with SUP-07 enabled; expected handover target is an IMU / GNSS prior, not loop closure. |

## License

This project is licensed under the [MIT License](LICENSE).

> **Third-party dependencies** have their own licenses (notably `evo` is GPL-3.0 and baseline systems in `external/` are GPL-2.0). These are runtime dependencies or Docker-isolated — they do not affect the license of this project's source code.

## Acknowledgments

This project builds on excellent open-source work:

- [KISS-ICP](https://github.com/PRBonn/kiss-icp) — Vizzo et al., RAL 2023
- [GTSAM](https://gtsam.org/) — Dellaert & Kaess, Georgia Tech
- [Open3D](http://www.open3d.org/) — Zhou et al., arXiv 2018
- [Lanelet2](https://github.com/fzi-forschungszentrum-informatik/Lanelet2) — Poggenhans et al., IV 2018
- [evo](https://github.com/MichaelGrupp/evo) — Grupp, TUM
- [nuScenes](https://www.nuscenes.org/) — Caesar et al., CVPR 2020
- [Scan Context](https://github.com/irapkaist/scancontext) — Kim & Kim, IROS 2018

