"""SUP-08 odom_node: Stage 2 (KISS-ICP) as a streaming ROS2 node.

Subscribes to ``/velodyne_points`` (sensor_msgs/PointCloud2) and publishes:

* ``/odom`` (nav_msgs/Odometry) — frame-to-frame odometry estimate.
* ``/slam_path`` (nav_msgs/Path) — accumulated path for RViz visualization.
* TF: ``map → <lidar_frame>`` (dynamic broadcast per frame).

KISS-ICP's underlying ``kiss_icp.kiss_icp.KissICP`` is instantiated once and
fed one scan per ``/velodyne_points`` callback via ``register_frame``. Per-
frame processing time is logged in a per-frame CSV (optional) and
summarized every ``log_every_n`` frames so acceptance #3 (<500 ms) can be
verified from the node logs alone.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path as FsPath  # aliased to avoid clash with nav_msgs/Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from kiss_icp.config import KISSConfig
from kiss_icp.config.config import DataConfig, MappingConfig
from kiss_icp.kiss_icp import KissICP
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2
from tf2_ros import TransformBroadcaster

from lidar_slam_ros2.pc2_utils import pc2_to_xyz


def _pose_mat_to_ros(mat: np.ndarray) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Convert a 4×4 SE(3) matrix to (translation xyz, quaternion xyzw)."""
    t = mat[:3, 3]
    q = R.from_matrix(mat[:3, :3]).as_quat()  # scipy returns xyzw
    return (float(t[0]), float(t[1]), float(t[2])), (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


class OdomNode(Node):
    def __init__(self) -> None:
        super().__init__("odom_node")

        self.declare_parameter("kiss_max_range", 100.0)
        self.declare_parameter("kiss_min_range", 5.0)
        self.declare_parameter("kiss_voxel_size", 1.0)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("lidar_frame", "velodyne")
        self.declare_parameter("cloud_topic", "/velodyne_points")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("path_topic", "/slam_path")
        self.declare_parameter("log_every_n", 50)
        self.declare_parameter("csv_path", "")
        self.declare_parameter("traj_path", "")  # KITTI-format raw odometry trajectory

        self._map_frame = str(self.get_parameter("map_frame").value)
        self._lidar_frame = str(self.get_parameter("lidar_frame").value)
        self._log_every_n = int(self.get_parameter("log_every_n").value)

        cfg = KISSConfig(
            data=DataConfig(
                max_range=float(self.get_parameter("kiss_max_range").value),
                min_range=float(self.get_parameter("kiss_min_range").value),
                deskew=False,
            ),
            mapping=MappingConfig(
                voxel_size=float(self.get_parameter("kiss_voxel_size").value),
            ),
        )
        self._icp = KissICP(cfg)

        qos_sub = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        qos_pub = QoSProfile(
            depth=50,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self._odom_pub = self.create_publisher(
            Odometry, str(self.get_parameter("odom_topic").value), qos_pub
        )
        self._path_pub = self.create_publisher(
            Path, str(self.get_parameter("path_topic").value), qos_pub
        )
        self._tf = TransformBroadcaster(self)

        self._path = Path()
        self._path.header.frame_id = self._map_frame

        # Timing buffers — all in milliseconds.
        self._dts: list[float] = []
        self._csv_path = str(self.get_parameter("csv_path").value)
        self._csv_file = None
        self._csv_writer = None
        if self._csv_path:
            csv_parent = FsPath(self._csv_path).expanduser().parent
            csv_parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(FsPath(self._csv_path).expanduser(), "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(["frame_idx", "stamp_sec", "n_points", "icp_ms", "total_ms"])

        self._idx = 0
        self._traj_path = str(self.get_parameter("traj_path").value)
        self._poses: list[np.ndarray] = []
        self._sub = self.create_subscription(
            PointCloud2,
            str(self.get_parameter("cloud_topic").value),
            self._on_cloud,
            qos_sub,
        )
        self.get_logger().info(
            f"odom_node ready — kiss_voxel={cfg.mapping.voxel_size} "
            f"max_r={cfg.data.max_range} min_r={cfg.data.min_range}"
        )

    def _on_cloud(self, msg: PointCloud2) -> None:
        t0 = time.perf_counter()
        xyz = pc2_to_xyz(msg)
        n_pts = xyz.shape[0]
        if n_pts == 0:
            self.get_logger().warn("received empty cloud — skipping")
            return

        t_icp0 = time.perf_counter()
        self._icp.register_frame(xyz, np.zeros(n_pts, dtype=np.float64))
        pose = np.asarray(self._icp.last_pose, dtype=np.float64)
        t_icp1 = time.perf_counter()
        self._poses.append(pose)

        self._publish(pose, msg.header.stamp)

        total_ms = (time.perf_counter() - t0) * 1e3
        icp_ms = (t_icp1 - t_icp0) * 1e3
        self._dts.append(icp_ms)

        if self._csv_writer is not None:
            stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self._csv_writer.writerow([self._idx, f"{stamp_sec:.6f}", n_pts,
                                       f"{icp_ms:.3f}", f"{total_ms:.3f}"])
            self._csv_file.flush()

        if (self._idx + 1) % self._log_every_n == 0:
            arr = np.asarray(self._dts, dtype=np.float64)
            self.get_logger().info(
                f"[{self._idx + 1}] icp p50={np.percentile(arr, 50):.1f}ms "
                f"p95={np.percentile(arr, 95):.1f}ms max={arr.max():.1f}ms "
                f"(last icp={icp_ms:.1f}ms, total={total_ms:.1f}ms, n={n_pts})"
            )
        self._idx += 1

    def _publish(self, pose: np.ndarray, stamp) -> None:
        (tx, ty, tz), (qx, qy, qz, qw) = _pose_mat_to_ros(pose)

        # --- TF: map → lidar_frame ---
        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = self._map_frame
        tf_msg.child_frame_id = self._lidar_frame
        tf_msg.transform.translation.x = tx
        tf_msg.transform.translation.y = ty
        tf_msg.transform.translation.z = tz
        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw
        self._tf.sendTransform(tf_msg)

        # --- /odom ---
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self._map_frame
        odom.child_frame_id = self._lidar_frame
        odom.pose.pose.position.x = tx
        odom.pose.pose.position.y = ty
        odom.pose.pose.position.z = tz
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        self._odom_pub.publish(odom)

        # --- /slam_path ---
        ps = PoseStamped()
        ps.header = odom.header
        ps.pose = odom.pose.pose
        self._path.header.stamp = stamp
        self._path.poses.append(ps)
        self._path_pub.publish(self._path)

    def destroy_node(self) -> bool:  # type: ignore[override]
        if self._csv_file is not None:
            self._csv_file.close()
        if self._traj_path and self._poses:
            traj_out = FsPath(self._traj_path).expanduser()
            traj_out.parent.mkdir(parents=True, exist_ok=True)
            with traj_out.open("w") as f:
                for p in self._poses:
                    row = p[:3, :].reshape(-1)
                    f.write(" ".join(f"{v:.6e}" for v in row) + "\n")
            self.get_logger().info(f"saved {len(self._poses)} poses → {traj_out}")
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = OdomNode()
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
