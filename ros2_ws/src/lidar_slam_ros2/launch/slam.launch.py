"""SUP-08 launch: kitti_player + odom_node + pose_graph_node + RViz2.

    ros2 launch lidar_slam_ros2 slam.launch.py \
        sequence:=00 max_frames:=500

Optional args:
    kitti_root (str): path to KITTI Odometry dataset (sequences/<seq>/velodyne/*.bin)
    rate_scale (float): playback speed multiplier (1.0 = real-time)
    launch_rviz (bool): start rviz2 with slam.rviz (default True)
    csv_path (str): per-frame timing CSV for odom_node (default auto)
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    pkg_share = Path(get_package_share_directory("lidar_slam_ros2"))
    params_file = str(pkg_share / "config" / "slam_params.yaml")
    rviz_file = str(pkg_share / "rviz" / "slam.rviz")

    args = [
        DeclareLaunchArgument("kitti_root", default_value="~/data/kitti/odometry/dataset"),
        DeclareLaunchArgument("sequence", default_value="00"),
        DeclareLaunchArgument("max_frames", default_value="500"),
        DeclareLaunchArgument("rate_scale", default_value="1.0"),
        DeclareLaunchArgument("launch_rviz", default_value="true"),
        DeclareLaunchArgument(
            "csv_path",
            default_value="/home/troutie/projects/lidar-slam-hdmap/benchmarks/sup08/latency_per_frame.csv",
        ),
        DeclareLaunchArgument(
            "traj_path",
            default_value="/home/troutie/projects/lidar-slam-hdmap/benchmarks/sup08/slam_trajectory.txt",
        ),
    ]

    # ParameterValue(..., value_type=str) prevents launch's yaml serializer
    # from re-typing "00" back to an integer when it hits the temp param file.
    kitti_player = Node(
        package="lidar_slam_ros2",
        executable="kitti_player_node",
        name="kitti_player_node",
        output="screen",
        parameters=[
            params_file,
            {
                "kitti_root": ParameterValue(LaunchConfiguration("kitti_root"), value_type=str),
                "sequence": ParameterValue(LaunchConfiguration("sequence"), value_type=str),
                "max_frames": ParameterValue(LaunchConfiguration("max_frames"), value_type=int),
                "rate_scale": ParameterValue(LaunchConfiguration("rate_scale"), value_type=float),
            },
        ],
    )

    odom = Node(
        package="lidar_slam_ros2",
        executable="odom_node",
        name="odom_node",
        output="screen",
        parameters=[
            params_file,
            {
                "csv_path": ParameterValue(LaunchConfiguration("csv_path"), value_type=str),
                "traj_path": ParameterValue(LaunchConfiguration("traj_path"), value_type=str),
            },
        ],
    )

    pose_graph = Node(
        package="lidar_slam_ros2",
        executable="pose_graph_node",
        name="pose_graph_node",
        output="screen",
        parameters=[params_file],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_file],
        condition=IfCondition(LaunchConfiguration("launch_rviz")),
    )

    return LaunchDescription([*args, kitti_player, odom, pose_graph, rviz])
