"""PointCloud2 ↔ numpy helpers for SUP-08 nodes.

KISS-ICP expects raw ``(N, 3)`` ``float32`` xyz. ROS2 ``sensor_msgs/PointCloud2``
is a structured byte buffer whose fields vary by publisher (KITTI: xyz + intensity;
LIO-SAM: xyz + intensity + ring + time). This module centralizes the lossy
field extraction so both ``odom_node`` and ``pose_graph_node`` share one path.
"""

from __future__ import annotations

import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header


_XYZI_FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
]


def pc2_to_xyz(msg: PointCloud2) -> np.ndarray:
    """Extract ``(N, 3)`` float32 xyz from a PointCloud2 message.

    Uses ``sensor_msgs_py.point_cloud2.read_points_numpy`` which returns a
    contiguous 2-D array when passed explicit field names. NaN/inf filtering
    happens upstream of KISS-ICP so we do not repeat it here.
    """
    arr = pc2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
    return np.ascontiguousarray(arr, dtype=np.float32)


def pc2_to_xyzi(msg: PointCloud2) -> np.ndarray:
    """Extract ``(N, 4)`` float32 xyz+intensity.

    Falls back to zeros for intensity if the field is missing.
    """
    names = {f.name for f in msg.fields}
    if "intensity" in names:
        arr = pc2.read_points_numpy(
            msg, field_names=("x", "y", "z", "intensity"), skip_nans=True
        )
        return np.ascontiguousarray(arr, dtype=np.float32)
    xyz = pc2_to_xyz(msg)
    out = np.zeros((xyz.shape[0], 4), dtype=np.float32)
    out[:, :3] = xyz
    return out


def xyzi_to_pc2(points: np.ndarray, stamp, frame_id: str) -> PointCloud2:
    """Build a PointCloud2 message from an ``(N, 4)`` ``float32`` array.

    ``stamp`` must be a ``builtin_interfaces.msg.Time`` (e.g. from
    ``node.get_clock().now().to_msg()``). Fills the ``data`` buffer
    directly from the contiguous float32 array (no per-point Python
    iteration) — the Player runs this ≥10 Hz on 120 k-point KITTI scans
    and ``pc2.create_cloud(..., points.tolist())`` costs ~80 ms/msg.
    """
    if points.dtype != np.float32:
        points = points.astype(np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"expected (N, 3|4) array, got {points.shape}")
    if points.shape[1] == 3:
        out = np.zeros((points.shape[0], 4), dtype=np.float32)
        out[:, :3] = points
        points = out
    elif points.shape[1] > 4:
        points = np.ascontiguousarray(points[:, :4])

    n = points.shape[0]
    msg = PointCloud2()
    msg.header = Header(stamp=stamp, frame_id=frame_id)
    msg.height = 1
    msg.width = n
    msg.fields = _XYZI_FIELDS
    msg.is_bigendian = False
    msg.point_step = 16  # 4 × float32
    msg.row_step = 16 * n
    msg.is_dense = True
    msg.data = np.ascontiguousarray(points).tobytes()
    return msg
