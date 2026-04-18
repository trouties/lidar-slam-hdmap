"""SUP-08 KITTI player: replay KITTI Odometry velodyne scans as PointCloud2.

Reads ``sequences/<seq>/velodyne/*.bin`` and ``times.txt`` and publishes each
scan at the cadence recorded in ``times.txt`` (≈10 Hz on KITTI). Designed
for a single-shot run: once ``max_frames`` (or end of sequence) is reached
the node logs a summary and shuts itself down.

Usage:
    ros2 run lidar_slam_ros2 kitti_player_node --ros-args \
        -p kitti_root:=~/data/kitti/odometry/dataset -p sequence:=00
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import PointCloud2

from lidar_slam_ros2.pc2_utils import xyzi_to_pc2


def _load_velodyne_bin(path: Path) -> np.ndarray:
    """Load a KITTI velodyne .bin file as ``(N, 4) float32``."""
    return np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)


class KittiPlayerNode(Node):
    def __init__(self) -> None:
        super().__init__("kitti_player_node")

        self.declare_parameter("kitti_root", "~/data/kitti/odometry/dataset")
        self.declare_parameter("sequence", "00")
        self.declare_parameter("max_frames", 500)
        self.declare_parameter("rate_scale", 1.0)
        self.declare_parameter("frame_id", "velodyne")
        self.declare_parameter("topic", "/velodyne_points")

        root = Path(self.get_parameter("kitti_root").value).expanduser()
        # Accept either a quoted yaml string ("00") or an int (0); KITTI
        # sequences are zero-padded 2-digit strings on disk so we normalize
        # either form to ``"%02d"``. Launch's yaml serializer re-types
        # numeric-looking strings to int, so tolerate both.
        raw_seq = self.get_parameter("sequence").value
        seq = f"{int(raw_seq):02d}" if isinstance(raw_seq, int) else str(raw_seq)
        self._max_frames = int(self.get_parameter("max_frames").value)
        self._rate_scale = float(self.get_parameter("rate_scale").value)
        self._frame_id = str(self.get_parameter("frame_id").value)
        topic = str(self.get_parameter("topic").value)

        seq_dir = root / "sequences" / seq
        velo_dir = seq_dir / "velodyne"
        times_path = seq_dir / "times.txt"
        if not velo_dir.exists():
            raise RuntimeError(f"velodyne dir not found: {velo_dir}")
        if not times_path.exists():
            raise RuntimeError(f"times.txt not found: {times_path}")

        self._scan_files = sorted(velo_dir.glob("*.bin"))
        self._timestamps = np.atleast_1d(np.loadtxt(str(times_path), dtype=np.float64))
        n_avail = min(len(self._scan_files), len(self._timestamps))
        if self._max_frames > 0:
            n_avail = min(n_avail, self._max_frames)
        self._n = n_avail
        if self._n == 0:
            raise RuntimeError("no scans available after applying max_frames")

        # Reliable QoS with KEEP_LAST(10) — matches typical LiDAR driver defaults.
        qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._pub = self.create_publisher(PointCloud2, topic, qos)

        self._idx = 0
        self._t0_wall = self.get_clock().now().nanoseconds * 1e-9
        self._t0_seq = float(self._timestamps[0])
        # Drive the publishing loop with a short timer; the publish method
        # decides each call whether the next scan's target time has elapsed.
        self._timer = self.create_timer(0.005, self._tick)

        self.get_logger().info(
            f"KITTI player: seq={seq} root={root} frames={self._n} "
            f"rate_scale={self._rate_scale}"
        )

    def _tick(self) -> None:
        if self._idx >= self._n:
            self.get_logger().info(
                f"KITTI player: published {self._idx} frames — shutting down"
            )
            self._timer.cancel()
            # Let other nodes drain; issue a single shutdown request.
            rclpy.get_default_context().try_shutdown()
            return

        now_wall = self.get_clock().now().nanoseconds * 1e-9
        elapsed = (now_wall - self._t0_wall) * self._rate_scale
        target_dt = float(self._timestamps[self._idx]) - self._t0_seq
        if elapsed < target_dt:
            return

        scan = _load_velodyne_bin(self._scan_files[self._idx])
        stamp = self.get_clock().now().to_msg()
        msg = xyzi_to_pc2(scan, stamp=stamp, frame_id=self._frame_id)
        self._pub.publish(msg)

        if (self._idx + 1) % 100 == 0 or self._idx == self._n - 1:
            self.get_logger().info(f"  published {self._idx + 1}/{self._n}")
        self._idx += 1


def main() -> None:
    rclpy.init()
    node = KittiPlayerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
