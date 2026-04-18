#!/usr/bin/env bash
# SUP-08 Step 0: 安装 ROS2 Humble + 建 ROS-aware venv
# 用法: sudo bash scripts/sup08_install_ros2.sh
#       然后普通用户再跑: bash scripts/sup08_install_ros2.sh --user-part

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${HOME}/slam-env-ros2"
# ROS2 Humble is built against python3.10 on Ubuntu 22.04. Do NOT let the venv
# inherit miniconda's python3 (often 3.13) from PATH — that breaks rclpy ABI
# and leaves open3d/kiss-icp without matching wheels.
PY310="/usr/bin/python3.10"

install_system() {
    echo "=== [SUP-08] Installing ROS2 Humble (root part) ==="

    apt update
    apt install -y curl gnupg lsb-release software-properties-common

    add-apt-repository -y universe

    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg

    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
        | tee /etc/apt/sources.list.d/ros2.list >/dev/null

    apt update
    apt install -y \
        ros-humble-desktop \
        ros-humble-tf-transformations \
        python3-colcon-common-extensions \
        python3-argcomplete \
        python3-rosdep

    if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
        rosdep init
    fi

    echo "=== [SUP-08] ROS2 Humble apt install DONE ==="
    echo "Next: run 'bash scripts/sup08_install_ros2.sh --user-part' as your normal user."
}

install_user() {
    echo "=== [SUP-08] Creating ROS-aware venv (user part) ==="

    if [ ! -f /opt/ros/humble/setup.bash ]; then
        echo "ERROR: /opt/ros/humble not found. Run the root part first." >&2
        exit 1
    fi

    # ROS2 setup.bash references unbound AMENT_* vars; temporarily relax -u.
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
    set -u

    rosdep update || true

    if [ ! -x "${PY310}" ]; then
        echo "ERROR: ${PY310} not found. Install it with 'sudo apt install python3.10 python3.10-venv'." >&2
        exit 1
    fi

    if [ -d "${VENV_DIR}" ]; then
        pyver="$("${VENV_DIR}/bin/python" -c 'import sys; print(sys.version_info[:2])')"
        if [[ "${pyver}" != "(3, 10)" ]]; then
            echo "existing venv has Python ${pyver} — deleting and rebuilding with ${PY310}"
            rm -rf "${VENV_DIR}"
        else
            echo "venv ${VENV_DIR} already exists (Python 3.10) — reusing"
        fi
    fi
    if [ ! -d "${VENV_DIR}" ]; then
        "${PY310}" -m venv --system-site-packages "${VENV_DIR}"
    fi

    set +u
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    set -u

    pip install --upgrade pip
    pip install -e "${REPO_ROOT}"
    pip install transforms3d  # for tf_transformations helper

    echo "=== [SUP-08] Smoke test ==="
    python3 -c "import rclpy, sensor_msgs_py, gtsam, kiss_icp, numpy; print('rclpy OK'); print('gtsam', gtsam.__file__); print('kiss_icp', kiss_icp.__file__)"

    echo "=== [SUP-08] DONE. Activate with: source ${VENV_DIR}/bin/activate && source /opt/ros/humble/setup.bash ==="
}

if [ "${1:-}" = "--user-part" ]; then
    install_user
elif [ "$(id -u)" = "0" ]; then
    install_system
else
    echo "Must run either as root (system part) or with --user-part flag."
    echo "  Root part: sudo bash scripts/sup08_install_ros2.sh"
    echo "  User part: bash scripts/sup08_install_ros2.sh --user-part"
    exit 2
fi
