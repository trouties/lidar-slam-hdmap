"""Validate KITTI odometry dataset directory structure.

Usage:
    python scripts/verify_kitti.py [--root PATH] [--sequence SEQ]

Checks that the expected files/directories exist and are consistent.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def check_sequence(root: Path, seq: str) -> list[str]:
    """Validate a single KITTI sequence. Returns a list of error messages."""
    errors: list[str] = []
    seq_dir = root / "sequences" / seq

    if not seq_dir.exists():
        errors.append(f"  sequences/{seq}/ does not exist")
        return errors

    # velodyne
    vel_dir = seq_dir / "velodyne"
    if not vel_dir.exists():
        errors.append(f"  sequences/{seq}/velodyne/ missing")
    else:
        bin_files = sorted(vel_dir.glob("*.bin"))
        if not bin_files:
            errors.append(f"  sequences/{seq}/velodyne/ contains no .bin files")
        else:
            print(f"  velodyne: {len(bin_files)} scans")
            # spot-check first .bin file
            try:
                pts = np.fromfile(bin_files[0], dtype=np.float32).reshape(-1, 4)
                print(f"  spot-check {bin_files[0].name}: {pts.shape[0]} points OK")
            except Exception as e:
                errors.append(f"  spot-check failed on {bin_files[0].name}: {e}")

    # calib.txt
    calib_path = seq_dir / "calib.txt"
    if not calib_path.exists():
        errors.append(f"  sequences/{seq}/calib.txt missing")
    else:
        try:
            with open(calib_path) as f:
                keys = [line.split(":")[0].strip() for line in f if ":" in line]
            print(f"  calib.txt: keys={keys}")
        except Exception as e:
            errors.append(f"  calib.txt parse error: {e}")

    # times.txt
    times_path = seq_dir / "times.txt"
    if not times_path.exists():
        errors.append(f"  sequences/{seq}/times.txt missing")
    else:
        timestamps = np.loadtxt(times_path)
        timestamps = np.atleast_1d(timestamps)
        n_times = len(timestamps)
        print(f"  times.txt: {n_times} entries")
        # consistency check
        if vel_dir.exists():
            n_bins = len(sorted(vel_dir.glob("*.bin")))
            if n_bins != n_times:
                errors.append(f"  mismatch: {n_bins} .bin files vs {n_times} timestamps")

    # poses (only sequences 00-10 have ground truth)
    poses_path = root / "poses" / f"{seq}.txt"
    seq_num = int(seq)
    if seq_num <= 10:
        if poses_path.exists():
            poses = np.loadtxt(poses_path)
            if poses.ndim == 1:
                poses = poses.reshape(1, -1)
            print(f"  poses: {poses.shape[0]} ground truth poses")
        else:
            errors.append(f"  poses/{seq}.txt missing (expected for seq 00-10)")
    else:
        if poses_path.exists():
            print("  poses: ground truth available (bonus)")
        else:
            print(f"  poses: no ground truth (expected for seq {seq})")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate KITTI odometry dataset")
    parser.add_argument(
        "--root",
        type=str,
        default="~/data/kitti/odometry/dataset",
        help="Root path to KITTI odometry dataset",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Specific sequence to check (e.g., '00'). Checks all found if omitted.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    print(f"KITTI root: {root}")

    if not root.exists():
        print(f"ERROR: root directory does not exist: {root}")
        sys.exit(1)

    seq_dir = root / "sequences"
    if not seq_dir.exists():
        print("ERROR: sequences/ directory does not exist")
        sys.exit(1)

    # Determine which sequences to check
    if args.sequence:
        sequences = [args.sequence]
    else:
        sequences = sorted(d.name for d in seq_dir.iterdir() if d.is_dir() and d.name.isdigit())
        if not sequences:
            print("No sequence directories found in sequences/")
            sys.exit(1)

    all_errors: list[str] = []
    for seq in sequences:
        print(f"\n--- Sequence {seq} ---")
        errors = check_sequence(root, seq)
        all_errors.extend(errors)
        for err in errors:
            print(f"ERROR: {err}")

    print(f"\n{'=' * 40}")
    if all_errors:
        print(f"FAILED: {len(all_errors)} error(s) found")
        sys.exit(1)
    else:
        print(f"OK: {len(sequences)} sequence(s) validated successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()
