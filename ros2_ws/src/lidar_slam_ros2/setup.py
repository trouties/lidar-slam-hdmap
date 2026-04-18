"""Ament-python setup for lidar_slam_ros2 (SUP-08)."""

from pathlib import Path

from setuptools import setup

package_name = "lidar_slam_ros2"


def _glob(subdir: str, pattern: str) -> list[str]:
    return [str(p) for p in Path(subdir).glob(pattern) if p.is_file()]


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", _glob("launch", "*.py")),
        (f"share/{package_name}/config", _glob("config", "*.yaml")),
        (f"share/{package_name}/rviz", _glob("rviz", "*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Haotian Zha",
    maintainer_email="haotian.zha@gmail.com",
    description="SUP-08 ROS2 Humble wrapping of Stage 2 + Stage 3",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            f"kitti_player_node = {package_name}.kitti_player_node:main",
            f"odom_node = {package_name}.odom_node:main",
            f"pose_graph_node = {package_name}.pose_graph_node:main",
        ],
    },
)
