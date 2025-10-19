"""KITTI dataset loader for LiDAR point clouds, GPS/IMU, and calibration data."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_velodyne_bin(path: Path) -> np.ndarray:
    """Load a Velodyne binary point cloud file.

    Args:
        path: Path to .bin file.

    Returns:
        Point cloud as (N, 4) array [x, y, z, reflectance].
    """
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    return points


def load_oxts(path: Path) -> dict:
    """Load GPS/IMU (OxTS) data from a KITTI Raw .txt file.

    Note: OxTS data is only available in KITTI Raw, not KITTI Odometry.

    Args:
        path: Path to OxTS .txt file (single line, 30 space-separated values).

    Returns:
        Dictionary with lat, lon, alt, roll, pitch, yaw, and velocity fields.
    """
    values = np.loadtxt(path)
    return {
        "lat": float(values[0]),
        "lon": float(values[1]),
        "alt": float(values[2]),
        "roll": float(values[3]),
        "pitch": float(values[4]),
        "yaw": float(values[5]),
        "vn": float(values[6]),
        "ve": float(values[7]),
        "vf": float(values[8]),
    }


def load_calibration(path: Path) -> dict[str, np.ndarray]:
    """Load calibration matrices from KITTI calib.txt.

    Each line has format "KEY: v1 v2 ... v12" representing a 3x4 matrix.
    Returns 4x4 matrices with [0, 0, 0, 1] appended as the last row.

    Args:
        path: Path to calib.txt.

    Returns:
        Dictionary mapping matrix names (P0..P3, Tr) to 4x4 numpy arrays.
    """
    calibration = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, _, values = line.partition(":")
            mat = np.eye(4)
            mat[:3, :] = np.fromstring(values, sep=" ").reshape(3, 4)
            calibration[key.strip()] = mat
    return calibration


def load_poses(path: Path) -> np.ndarray:
    """Load ground truth poses from a KITTI poses file.

    Each line contains 12 space-separated floats representing the 3x4
    row-major transformation matrix. Only sequences 00-10 have ground truth.

    Args:
        path: Path to poses .txt file (e.g., poses/00.txt).

    Returns:
        Array of shape (M, 4, 4) containing SE(3) transformation matrices.
    """
    raw = np.loadtxt(path)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    n = raw.shape[0]
    poses = np.zeros((n, 4, 4))
    poses[:, :3, :] = raw.reshape(n, 3, 4)
    poses[:, 3, 3] = 1.0
    return poses


def load_timestamps(path: Path) -> np.ndarray:
    """Load timestamps from KITTI times.txt.

    Args:
        path: Path to times.txt (one float per line, seconds from start).

    Returns:
        1D array of timestamps in seconds.
    """
    return np.atleast_1d(np.loadtxt(path))


class KITTIDataset:
    """KITTI odometry dataset wrapper.

    Provides indexed access to point clouds, ground truth poses,
    timestamps, and calibration for a given sequence.
    """

    def __init__(self, root_path: str | Path, sequence: str = "00") -> None:
        """Initialize dataset.

        Args:
            root_path: Root path to KITTI odometry dataset.
            sequence: Sequence number (e.g. "00", "01", ..., "21").
        """
        self.root = Path(root_path).expanduser()
        self.sequence = sequence

        seq_dir = self.root / "sequences" / sequence
        self.velodyne_dir = seq_dir / "velodyne"
        self.calib_path = seq_dir / "calib.txt"
        self.times_path = seq_dir / "times.txt"
        self.poses_path = self.root / "poses" / f"{sequence}.txt"

        if self.velodyne_dir.exists():
            self.scan_files = sorted(self.velodyne_dir.glob("*.bin"))
        else:
            self.scan_files = []

        # Eagerly load small metadata files
        self.calibration = load_calibration(self.calib_path) if self.calib_path.exists() else None
        self.timestamps = load_timestamps(self.times_path) if self.times_path.exists() else None
        self.poses = load_poses(self.poses_path) if self.poses_path.exists() else None

    def __len__(self) -> int:
        return len(self.scan_files)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray | None, float | None]:
        """Get a single frame.

        Args:
            idx: Frame index.

        Returns:
            Tuple of (pointcloud, pose, timestamp).
            - pointcloud: (N, 4) array [x, y, z, reflectance]
            - pose: (4, 4) ground truth SE(3) matrix, or None if unavailable
            - timestamp: seconds from start, or None if unavailable
        """
        pointcloud = load_velodyne_bin(self.scan_files[idx])

        pose = None
        if self.poses is not None and idx < len(self.poses):
            pose = self.poses[idx]

        timestamp = None
        if self.timestamps is not None and idx < len(self.timestamps):
            timestamp = float(self.timestamps[idx])

        return pointcloud, pose, timestamp
