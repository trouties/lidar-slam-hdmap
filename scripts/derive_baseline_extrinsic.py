"""Derive correct LiDAR<->IMU extrinsic YAML for SUP-01 baselines from KITTI Raw.

SUP-01 P0-1 root cause: ``external/baselines/lio_sam/config/params.yaml`` hard-
codes a non-identity 180-ish rotation for ``extrinsicRot`` / ``extrinsicRPY``
that does not correspond to the real KITTI sensor rig. This produces ~550 m
drift on Seq 00 and is the dominant factor in our bogus baseline numbers.

This script parses the official KITTI Raw ``calib_imu_to_velo.txt`` for a given
date (Seq 00 -> 2011_10_03, Seq 05 -> 2011_09_30) and prints ready-to-paste
YAML snippets for:

  * LIO-SAM     (``extrinsicTrans`` / ``extrinsicRot`` / ``extrinsicRPY``)
  * FAST-LIO2   (``extrinsic_T`` / ``extrinsic_R``)

KITTI calib convention
----------------------
``calib_imu_to_velo.txt`` stores R, T such that::

    p_velo = R * p_imu + T

i.e. ``[R|T]`` is ``T_velo_from_imu``. ``T`` is the IMU origin expressed in the
Velodyne frame.

LIO-SAM convention (Shan 2020, params_kitti.yaml in the upstream repo)
---------------------------------------------------------------------
``extrinsicRot`` rotates IMU body-frame measurements (linear accel, angular
velocity) into the LiDAR frame, i.e. it is ``R_velo_from_imu`` -- exactly the
R in the calib file. ``extrinsicTrans`` is the IMU origin expressed in the
LiDAR frame, i.e. the T in the calib file. ``extrinsicRPY`` is identical to
``extrinsicRot`` for this convention.

FAST-LIO2 convention (Xu 2022, kitti.yaml)
------------------------------------------
``extrinsic_R`` is ``R_imu_from_velo`` (rotates LiDAR vectors into the IMU
frame), i.e. the *transpose* of the calib R. ``extrinsic_T`` is the LiDAR
origin expressed in the IMU frame, i.e. ``-R_imu_from_velo @ T``.

Usage
-----
    python scripts/derive_baseline_extrinsic.py \\
        --kitti-raw ~/data/kitti_raw --date 2011_10_03

Prints the two YAML snippets + a sanity report (determinant, identity-ish
check, transpose consistency between LIO-SAM and FAST-LIO2 blocks).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_kitti_calib(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (R, T) from a KITTI calib_imu_to_velo.txt file."""
    R = None
    T = None
    for line in path.read_text().splitlines():
        key, _, rest = line.partition(":")
        key = key.strip()
        if key == "R":
            R = np.array([float(x) for x in rest.split()]).reshape(3, 3)
        elif key == "T":
            T = np.array([float(x) for x in rest.split()])
    if R is None or T is None:
        raise ValueError(f"Could not parse R/T from {path}")
    return R, T


def format_yaml_matrix(M: np.ndarray, indent: int, per_row: int = 3) -> str:
    """Format a 3x3 matrix as a YAML flow-style list, 3 per row for readability."""
    flat = M.flatten()
    rows = []
    pad = " " * indent
    for i in range(0, len(flat), per_row):
        chunk = flat[i : i + per_row]
        rows.append(", ".join(f"{v: .6e}" for v in chunk))
    joined = (",\n" + pad).join(rows)
    return "[" + joined + "]"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--kitti-raw",
        type=Path,
        default=Path.home() / "data" / "kitti_raw",
        help="KITTI Raw root (default: ~/data/kitti_raw)",
    )
    ap.add_argument(
        "--date",
        required=True,
        help="KITTI Raw date folder, e.g. 2011_10_03 (Seq 00) or 2011_09_30 (Seq 05)",
    )
    args = ap.parse_args()

    calib_path = args.kitti_raw / args.date / "calib_imu_to_velo.txt"
    if not calib_path.exists():
        print(f"ERROR: calib file not found: {calib_path}", file=sys.stderr)
        print(
            "       (Seq 05 / 2011_09_30 calib is missing on this machine;"
            " download the KITTI Raw calib zip for that date.)",
            file=sys.stderr,
        )
        return 1

    R_velo_from_imu, T_velo_from_imu = parse_kitti_calib(calib_path)

    det = float(np.linalg.det(R_velo_from_imu))
    off_diag = float(np.max(np.abs(R_velo_from_imu - np.eye(3))))
    R_imu_from_velo = R_velo_from_imu.T
    T_imu_from_velo = -R_imu_from_velo @ T_velo_from_imu

    print("=" * 72)
    print(f"KITTI Raw calib: {calib_path}")
    print("=" * 72)
    print()
    print("Parsed R_velo_from_imu (row-major):")
    print(R_velo_from_imu)
    print(f"Parsed T_velo_from_imu (m): {T_velo_from_imu}")
    print()
    print("Sanity:")
    print(f"  det(R) = {det:.9f}  (must be ~1.0 for proper rotation)")
    print(f"  max |R - I| = {off_diag:.6f}  (KITTI IMU/Velo are nearly aligned)")
    print()
    print("-" * 72)
    print("LIO-SAM convention (extrinsicRot = R_velo_from_imu; extrinsicTrans = T)")
    print("-" * 72)
    print("Paste into external/baselines/lio_sam/config/params.yaml, under 'lio_sam:':")
    print()
    trans_str = ", ".join(f"{v: .6e}" for v in T_velo_from_imu)
    rot_yaml = format_yaml_matrix(R_velo_from_imu, indent=16)
    print(f"  extrinsicTrans: [{trans_str}]")
    print(f"  extrinsicRot: {rot_yaml}")
    print(f"  extrinsicRPY: {rot_yaml}")
    print()
    print("-" * 72)
    print("FAST-LIO2 convention (extrinsic_R = R_imu_from_velo; extrinsic_T = -R^T @ T)")
    print("-" * 72)
    print("Paste into external/baselines/fast_lio2/config/kitti.yaml, under 'mapping:':")
    print()
    trans_fl2 = ", ".join(f"{v: .6e}" for v in T_imu_from_velo)
    rot_fl2_yaml = format_yaml_matrix(R_imu_from_velo, indent=18)
    print(f"    extrinsic_T: [{trans_fl2}]")
    print(f"    extrinsic_R: {rot_fl2_yaml}")
    print()
    print("-" * 72)
    print("Cross-check:")
    print("-" * 72)
    # LIO-SAM's R and FAST-LIO2's R should be transposes of each other
    recon = R_velo_from_imu @ R_imu_from_velo
    id_err = float(np.max(np.abs(recon - np.eye(3))))
    print(f"  R_velo_imu @ R_imu_velo = I ?  max err = {id_err:.2e}  (should be ~0)")
    # T chain: p_imu = R_imu_velo @ p_velo + T_imu_velo
    # Check by mapping IMU origin (p_imu=0) through LIO-SAM convention and back
    origin_velo = T_velo_from_imu  # IMU origin in Velo frame
    origin_imu_recovered = R_imu_from_velo @ origin_velo + T_imu_from_velo
    t_err = float(np.max(np.abs(origin_imu_recovered)))
    print(f"  IMU origin -> Velo -> back to IMU. max err = {t_err:.2e}  (should be ~0)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
