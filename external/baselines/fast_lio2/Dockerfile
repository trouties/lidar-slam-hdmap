FROM slam-baselines/ros-base:noetic

# FAST-LIO2 dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-cv-bridge \
    ros-noetic-eigen-conversions \
    ros-noetic-tf-conversions \
    && rm -rf /var/lib/apt/lists/*

# livox_ros_driver (build dependency, even though we use Velodyne input)
RUN cd /catkin_ws/src && \
    git clone --depth 1 https://github.com/Livox-SDK/livox_ros_driver.git

# FAST-LIO2
RUN cd /catkin_ws/src && \
    git clone --depth 1 --recursive https://github.com/hku-mars/FAST_LIO.git

# SUP-01 Phase C: KITTI .bin is pre-deskewed (HDL-64E vendor motion compensation).
# FAST-LIO2's default UndistortPcl re-runs deskew and double-compensates, causing
# ~134 m APE_SE3 on Seq 00 (see results/diagnostics/phaseB_report.md). This patch
# adds `preprocess/skip_undistortion` (default false). When the kitti.yaml sets it
# to true, the per-point backward-propagation loop is short-circuited while IMU
# forward propagation / kf_state.predict remain untouched.
COPY fast_lio2/kitti_skip_deskew.patch /tmp/kitti_skip_deskew.patch
RUN cd /catkin_ws/src/FAST_LIO && \
    git apply --check /tmp/kitti_skip_deskew.patch && \
    git apply /tmp/kitti_skip_deskew.patch && \
    grep -q 'skip_undistortion' src/IMU_Processing.hpp && \
    grep -q 'skip_undistortion' src/laserMapping.cpp

# Build
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    cd /catkin_ws && \
    catkin_make -DCMAKE_BUILD_TYPE=Release -j$(nproc)"

# Copy scripts and config
COPY common/kitti_to_rosbag.py /scripts/kitti_to_rosbag.py
COPY common/extract_poses.py /scripts/extract_poses.py
COPY fast_lio2/config/ /config/
COPY fast_lio2/launch/ /launch/
COPY fast_lio2/run.sh /scripts/run.sh
RUN chmod +x /scripts/run.sh

ENTRYPOINT ["/scripts/run.sh"]
