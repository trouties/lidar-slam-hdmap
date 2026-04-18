"""SUP-08 pose_graph_node: Stage 3 as a periodic batch optimizer.

Subscribes to synchronized ``/velodyne_points`` + ``/odom`` (via
``message_filters.ApproximateTimeSynchronizer``) and accumulates keyframes
in memory. Every ``keyframe_stride`` new keyframes, triggers a batch
``PoseGraphOptimizer.build_graph + optimize`` plus ``LoopClosureDetector``
Scan Context v2 loop closure pass (both are re-runnable on the full
keyframe buffer — the design in ``scripts/run_pipeline.py``). The
optimized trajectory is published on ``/optimized_path``.

The node uses a dedicated worker thread to run GTSAM optimization so
that the ROS2 callback thread remains responsive — keyframe_stride=50
typically triggers a multi-second optimization on a 500-frame run, and
we do not want that to block the subscription queue.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Sequence

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from src.optimization.loop_closure import LoopClosureDetector
from src.optimization.pose_graph import PoseGraphOptimizer

from lidar_slam_ros2.pc2_utils import pc2_to_xyzi


def _pose_msg_to_mat(pose) -> np.ndarray:
    """Convert a geometry_msgs/Pose to a 4×4 SE(3) matrix."""
    mat = np.eye(4, dtype=np.float64)
    p = pose.position
    q = pose.orientation
    mat[:3, 3] = [p.x, p.y, p.z]
    mat[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    return mat


class _OnlineDataset:
    """Minimal duck-typed dataset compatible with ``LoopClosureDetector``.

    Exposes ``__getitem__`` returning ``(pointcloud, None, None)`` so the
    existing Stage 3 code does not need changes. Backed by a plain Python
    list that pose_graph_node appends to on each synchronized keyframe.
    """

    def __init__(self) -> None:
        self._clouds: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self._clouds)

    def __getitem__(self, idx: int):
        return self._clouds[idx], None, None

    def append(self, cloud: np.ndarray) -> None:
        self._clouds.append(cloud)


class PoseGraphNode(Node):
    def __init__(self) -> None:
        super().__init__("pose_graph_node")

        # Parameters (see config/slam_params.yaml for docstring)
        for name, default in [
            ("keyframe_stride", 50),
            ("max_keyframes", 2000),
            ("cloud_topic", "/velodyne_points"),
            ("odom_topic", "/odom"),
            ("optimized_path_topic", "/optimized_path"),
            ("loop_markers_topic", "/loop_markers"),
            ("map_frame", "map"),
            ("sync_slop", 0.05),
            ("loop_mode", "v2"),
            ("loop_distance_threshold", 15.0),
            ("loop_min_frame_gap", 100),
            ("loop_icp_fitness_threshold", 0.9),
            ("loop_icp_downsample_voxel", 1.0),
            ("sc_num_rings", 20),
            ("sc_num_sectors", 60),
            ("sc_max_range", 80.0),
            ("sc_distance_threshold", 0.4),
            ("sc_top_k", 25),
            ("sc_query_stride", 1),
            ("sc_max_matches_per_query", 5),
        ]:
            self.declare_parameter(name, default)

        self.declare_parameter("odom_sigmas", [0.1, 0.1, 0.1, 0.01, 0.01, 0.01])
        self.declare_parameter("prior_sigmas", [0.01, 0.01, 0.01, 0.001, 0.001, 0.001])

        self._keyframe_stride = int(self.get_parameter("keyframe_stride").value)
        self._max_keyframes = int(self.get_parameter("max_keyframes").value)
        self._map_frame = str(self.get_parameter("map_frame").value)

        self._dataset = _OnlineDataset()
        self._poses: list[np.ndarray] = []
        self._state_lock = threading.Lock()

        qos_pub = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._opt_path_pub = self.create_publisher(
            Path, str(self.get_parameter("optimized_path_topic").value), qos_pub
        )
        self._marker_pub = self.create_publisher(
            MarkerArray, str(self.get_parameter("loop_markers_topic").value), qos_pub
        )

        cloud_sub = Subscriber(self, PointCloud2, str(self.get_parameter("cloud_topic").value))
        odom_sub = Subscriber(self, Odometry, str(self.get_parameter("odom_topic").value))
        self._sync = ApproximateTimeSynchronizer(
            [cloud_sub, odom_sub],
            queue_size=50,
            slop=float(self.get_parameter("sync_slop").value),
        )
        self._sync.registerCallback(self._on_synced)

        self._worker_thread: threading.Thread | None = None
        self._worker_running = False
        self._last_closures: list[tuple[int, int, np.ndarray]] = []
        self._n_runs = 0

        self.get_logger().info(
            f"pose_graph_node ready — keyframe_stride={self._keyframe_stride} "
            f"loop_mode={self.get_parameter('loop_mode').value}"
        )

    # --- Keyframe ingestion ---

    def _on_synced(self, cloud_msg: PointCloud2, odom_msg: Odometry) -> None:
        if len(self._poses) >= self._max_keyframes:
            return
        xyz = pc2_to_xyzi(cloud_msg)  # LoopClosureDetector uses only xyz
        pose = _pose_msg_to_mat(odom_msg.pose.pose)
        with self._state_lock:
            self._dataset.append(xyz)
            self._poses.append(pose)
            n = len(self._poses)

        if n >= self._keyframe_stride and n % self._keyframe_stride == 0 and not self._worker_running:
            self._launch_optimization(n)

    # --- Batch optimization worker ---

    def _launch_optimization(self, n_at_trigger: int) -> None:
        with self._state_lock:
            poses_snapshot = list(self._poses)
        self._worker_running = True

        def _run():
            try:
                self._run_optimization(poses_snapshot, n_at_trigger)
            except Exception as exc:
                self.get_logger().error(f"optimization failed: {exc!r}")
            finally:
                self._worker_running = False

        self._worker_thread = threading.Thread(target=_run, name="pose_graph_worker", daemon=True)
        self._worker_thread.start()

    def _run_optimization(self, poses: Sequence[np.ndarray], n_at_trigger: int) -> None:
        odom_sigmas = list(self.get_parameter("odom_sigmas").value)
        prior_sigmas = list(self.get_parameter("prior_sigmas").value)
        optimizer = PoseGraphOptimizer(odom_sigmas=odom_sigmas, prior_sigmas=prior_sigmas)

        detector = LoopClosureDetector(
            distance_threshold=float(self.get_parameter("loop_distance_threshold").value),
            min_frame_gap=int(self.get_parameter("loop_min_frame_gap").value),
            icp_fitness_threshold=float(self.get_parameter("loop_icp_fitness_threshold").value),
            mode=str(self.get_parameter("loop_mode").value),
            sc_num_rings=int(self.get_parameter("sc_num_rings").value),
            sc_num_sectors=int(self.get_parameter("sc_num_sectors").value),
            sc_max_range=float(self.get_parameter("sc_max_range").value),
            sc_distance_threshold=float(self.get_parameter("sc_distance_threshold").value),
            sc_top_k=int(self.get_parameter("sc_top_k").value),
            sc_query_stride=int(self.get_parameter("sc_query_stride").value),
            sc_max_matches_per_query=int(self.get_parameter("sc_max_matches_per_query").value),
            icp_downsample_voxel=float(self.get_parameter("loop_icp_downsample_voxel").value),
        )

        # Loop closure detection needs (pointcloud, ...) dataset access — the
        # _OnlineDataset wrapper is safe to share read-only because pose_graph_node
        # only ever appends to it. Read a length snapshot to avoid races where a
        # new keyframe arrives mid-detect.
        with self._state_lock:
            n_frames = len(self._dataset)
        poses_for_opt = poses[:n_frames]

        closures = detector.detect(list(poses_for_opt), dataset=self._dataset)

        optimizer.build_graph(poses_for_opt)
        for i, j, rel_pose in closures:
            optimizer.add_loop_closure(i, j, rel_pose)
        optimized_poses = optimizer.optimize()

        self._n_runs += 1
        self._last_closures = closures
        self.get_logger().info(
            f"pose_graph run #{self._n_runs} @ {n_frames} keyframes → "
            f"{len(closures)} closures"
        )

        self._publish_optimized_path(optimized_poses)
        self._publish_loop_markers(optimized_poses, closures)

    # --- Publishers ---

    def _publish_optimized_path(self, poses: Sequence[np.ndarray]) -> None:
        path = Path()
        path.header.frame_id = self._map_frame
        path.header.stamp = self.get_clock().now().to_msg()
        for mat in poses:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(mat[0, 3])
            ps.pose.position.y = float(mat[1, 3])
            ps.pose.position.z = float(mat[2, 3])
            q = R.from_matrix(mat[:3, :3]).as_quat()
            ps.pose.orientation.x = float(q[0])
            ps.pose.orientation.y = float(q[1])
            ps.pose.orientation.z = float(q[2])
            ps.pose.orientation.w = float(q[3])
            path.poses.append(ps)
        self._opt_path_pub.publish(path)

    def _publish_loop_markers(
        self,
        poses: Sequence[np.ndarray],
        closures: Sequence[tuple[int, int, np.ndarray]],
    ) -> None:
        arr = MarkerArray()
        if not closures:
            # Publish an empty DELETEALL so RViz clears stale markers after a
            # run that finds no closures (e.g. Seq 00 first 200 frames).
            m = Marker()
            m.header.frame_id = self._map_frame
            m.header.stamp = self.get_clock().now().to_msg()
            m.action = Marker.DELETEALL
            arr.markers.append(m)
            self._marker_pub.publish(arr)
            return

        stamp = self.get_clock().now().to_msg()
        for k, (i, j, _rel) in enumerate(closures):
            m = Marker()
            m.header.frame_id = self._map_frame
            m.header.stamp = stamp
            m.ns = "loop_closures"
            m.id = k
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.5
            m.color = ColorRGBA(r=1.0, g=0.8, b=0.0, a=0.9)

            for idx in (i, j):
                p = Point()
                p.x = float(poses[idx][0, 3])
                p.y = float(poses[idx][1, 3])
                p.z = float(poses[idx][2, 3])
                m.points.append(p)
            arr.markers.append(m)
        self._marker_pub.publish(arr)


def main() -> None:
    rclpy.init()
    node = PoseGraphNode()
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
