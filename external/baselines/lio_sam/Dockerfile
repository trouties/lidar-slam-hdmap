FROM slam-baselines/ros-base:noetic

# SUP-01 Phase E (2026-04-18): switched from TixiaoShan/LIO-SAM upstream
# (P0-2 structural gravity/extrinsic bug, APE_SE3 = 582 m on Seq 00 even
# after Phase A/B fixes) to JokerJohn/LIO_SAM_6AXIS community fork.
# Rationale: 6-axis fork explicitly handles useImuHeadingInitialization +
# KITTI OxTS-style 6-axis IMU convention. Backup of upstream Dockerfile is
# in Dockerfile.bak_upstream for rollback.

# LIO-SAM-6AXIS dependencies (same as upstream LIO-SAM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-navigation \
    ros-noetic-robot-localization \
    ros-noetic-robot-state-publisher \
    ros-noetic-cv-bridge \
    ros-noetic-image-transport \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# GTSAM 4.x (required by LIO-SAM-6AXIS)
RUN add-apt-repository ppa:borglab/gtsam-release-4.1 && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    libgtsam-dev libgtsam-unstable-dev \
    && rm -rf /var/lib/apt/lists/*

# LIO-SAM-6AXIS (community KITTI-compatible fork; Phase E verified against
# commit d4318f70ef2e04dbe6e4b9d42a805140d41f09f7, 2025-12-15)
RUN cd /catkin_ws/src && \
    git clone --depth 1 --recursive https://github.com/JokerJohn/LIO_SAM_6AXIS.git

# Repo layout: LIO_SAM_6AXIS/ contains LIO-SAM-6AXIS/ (the ROS package,
# project = lio_sam_6axis) and rviz_satellite/ (GPS tile viz — depends on
# Qt tile libs we didn't apt install; not needed for headless KITTI runs).
# Mark rviz_satellite CATKIN_IGNORE so catkin_make skips it.
RUN touch /catkin_ws/src/LIO_SAM_6AXIS/rviz_satellite/CATKIN_IGNORE

# Patch deprecated opencv/cv.h → opencv2/opencv.hpp for OpenCV 4 (6AXIS
# carries this include from its upstream LIO-SAM ancestry).
RUN test -f /catkin_ws/src/LIO_SAM_6AXIS/LIO-SAM-6AXIS/include/utility.h && \
    sed -i 's|#include <opencv/cv.h>|#include <opencv2/opencv.hpp>|g' \
    /catkin_ws/src/LIO_SAM_6AXIS/LIO-SAM-6AXIS/include/utility.h || true

# Fix PCL 1.10 + FLANN serialization bug (Ubuntu 20.04 issue; same as upstream)
RUN sed -i '34s|.*|// &|' /usr/include/flann/util/serialization.h || true

# Ensure C++17 (6AXIS CMakeLists does not set it; needed for newer PCL/GTSAM)
RUN sed -i '/project(lio_sam_6axis)/a set(CMAKE_CXX_STANDARD 17)' \
    /catkin_ws/src/LIO_SAM_6AXIS/LIO-SAM-6AXIS/CMakeLists.txt

# Build
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    cd /catkin_ws && \
    catkin_make -DCMAKE_BUILD_TYPE=Release -j$(nproc)"

# Stage C runtime deps: gettext-base for envsubst (renders params.yaml.tpl
# at container start); python3-numpy for extrinsic_to_env.py.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gettext-base \
    python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# Copy scripts and config. Bag generation lives in the separate bag-builder
# image; this container just consumes /data/bags/kitti_${SEQ}_fixed.bag.
COPY common/extract_poses.py /scripts/extract_poses.py
COPY common/extrinsic_to_env.py /scripts/extrinsic_to_env.py
COPY common/inject_kitti_extrinsic.py /scripts/inject_kitti_extrinsic.py
COPY lio_sam/config/ /config/
COPY lio_sam/launch/ /launch/
COPY lio_sam/run.sh /scripts/run.sh
RUN chmod +x /scripts/run.sh

ENTRYPOINT ["/scripts/run.sh"]
