# LiDAR SLAM HD Map Pipeline

> End-to-end LiDAR-inertial SLAM pipeline with HD Map feature extraction — from raw point clouds to Lanelet2 maps.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![KISS-ICP](https://img.shields.io/badge/KISS--ICP-1.2-green)
![GTSAM](https://img.shields.io/badge/GTSAM-4.2-orange)
![Status](https://img.shields.io/badge/Status-WIP-yellow)

> **⚠️ Work in Progress** — This project is under active development.

## Architecture

```
Raw LiDAR Scans
      │
      ▼
┌─────────────┐    ┌──────────────┐    ┌────────────────┐
│  KISS-ICP   │───▶│  Pose Graph  │───▶│  Global Map    │
│  Odometry   │    │  (GTSAM)     │    │  Construction  │
└─────────────┘    └──────────────┘    └────────────────┘
      │                   │                     │
      ▼                   ▼                     ▼
┌─────────────┐    ┌──────────────┐    ┌────────────────┐
│  IMU/GPS    │    │ Loop Closure │    │   Feature      │
│  Fusion     │    │  Detection   │    │   Extraction   │
│  (ESKF)     │    │              │    │                │
└─────────────┘    └──────────────┘    └────────────────┘
                                              │
                                              ▼
                                       ┌────────────────┐
                                       │  Lanelet2 HD   │
                                       │  Map Export    │
                                       └────────────────┘
```

## Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1 | `src/data/` | KITTI data loading, coordinate transforms |
| 2 | `src/odometry/` | LiDAR odometry via KISS-ICP |
| 3 | `src/optimization/` | Pose graph optimization with GTSAM + loop closure |
| 4 | `src/fusion/` | IMU/LiDAR fusion using Error-State Kalman Filter |
| 5 | `src/mapping/` | Point cloud map construction + feature extraction |
| 6 | `src/export/` | Lanelet2 HD Map export |

## Quick Start

### Docker
```bash
docker build -t slam-pipeline -f docker/Dockerfile .
docker run slam-pipeline --config configs/default.yaml
```

### Manual
```bash
# Activate virtual environment
source ~/slam-env/bin/activate

# Install the package
pip install -e ".[dev]"

# Run the pipeline
python scripts/run_pipeline.py --config configs/default.yaml
```

## Data

This pipeline uses the [KITTI Odometry Dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).

### Download

1. Register at [cvlibs.net](https://www.cvlibs.net/datasets/kitti/user_register.php) and go to the [odometry evaluation page](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).
2. Download these files:
   - **Velodyne laser data** (80 GB uncompressed; sequence 00 alone is ~2.3 GB)
   - **Calibration files** (1 MB)
   - **Ground truth poses** (4 KB, sequences 00-10 only)
3. Extract all archives into `~/data/kitti/odometry/dataset/`:

```
~/data/kitti/odometry/dataset/
├── sequences/
│   ├── 00/
│   │   ├── velodyne/       # .bin point cloud files
│   │   ├── calib.txt       # calibration matrices
│   │   └── times.txt       # timestamps
│   └── ...
├── poses/
│   ├── 00.txt              # ground truth (seq 00-10 only)
│   └── ...
```

> **WSL2 users**: Store data under `~/data/`, not `/mnt/c/` — cross-filesystem I/O is 10x slower.

4. Verify the dataset:
```bash
python scripts/verify_kitti.py --root ~/data/kitti/odometry/dataset --sequence 00
```

## Tech Stack

- **KISS-ICP** — Point-to-point ICP odometry
- **GTSAM** — Factor graph optimization
- **Open3D** — Point cloud processing and visualization
- **Lanelet2** — HD Map format
- **evo** — Trajectory evaluation (APE/RPE)
- **FilterPy** — Kalman filter implementation
