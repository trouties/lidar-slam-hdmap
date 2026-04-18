#!/usr/bin/env bash
# SUP-08 acceptance harness. Runs after sup08_install_ros2.sh finishes.
# Usage:
#   bash scripts/sup08_acceptance.sh            # all 4 acceptance checks
#   bash scripts/sup08_acceptance.sh build      # only colcon build
#   bash scripts/sup08_acceptance.sh run        # only ros2 launch 500 frames

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="${REPO_ROOT}/benchmarks/sup08"
mkdir -p "${BENCH}"

activate_env() {
    # ROS2 + venv setup scripts reference unbound AMENT_*/VIRTUAL_ENV_*
    # shell vars; temporarily relax `set -u` while sourcing them.
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/humble/setup.bash
    # shellcheck disable=SC1091
    source "${HOME}/slam-env-ros2/bin/activate"
    set -u

    # colcon's generated entry-point shebangs point at /usr/bin/python3 (the
    # ament-detected python at build time), which does NOT see the venv's
    # site-packages where gtsam/kiss-icp live. Prepend both the venv and the
    # repo root to PYTHONPATH so the system python3 can resolve:
    #   * gtsam, kiss_icp, scipy, evo, open3d  (installed in venv)
    #   * src.optimization.*                   (editable install — resolved via repo root)
    local venv_sp="${HOME}/slam-env-ros2/lib/python3.10/site-packages"
    export PYTHONPATH="${venv_sp}:${REPO_ROOT}:${PYTHONPATH:-}"
}

run_build() {
    echo "=== SUP-08 Step 6a: colcon build ==="
    activate_env
    cd "${REPO_ROOT}/ros2_ws"
    # --symlink-install avoids having to rebuild on every edit
    colcon build --symlink-install 2>&1 | tee "${BENCH}/colcon_build.log"
    echo "=== build OK ==="
}

run_launch() {
    echo "=== SUP-08 Step 6b: ros2 launch Seq 00 × 500 frames ==="
    activate_env
    set +u
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/ros2_ws/install/setup.bash"
    set -u

    timeout 180 ros2 launch lidar_slam_ros2 slam.launch.py \
        sequence:=00 max_frames:=500 launch_rviz:=false \
        csv_path:="${BENCH}/latency_per_frame.csv" \
        2>&1 | tee "${BENCH}/launch_run.log" || echo "(launch exited / timed out)"

    echo "=== launch run DONE ==="
}

summarize() {
    echo "=== SUP-08 Step 6c: acceptance summary ==="
    if [ -f "${BENCH}/latency_per_frame.csv" ]; then
        python3 - <<PY
import csv, statistics
rows = list(csv.DictReader(open("${BENCH}/latency_per_frame.csv")))
icp = [float(r["icp_ms"]) for r in rows]
if not icp:
    print("NO ROWS")
else:
    print(f"frames logged: {len(icp)}")
    print(f"icp_ms  p50={statistics.median(icp):.1f}  "
          f"p95={statistics.quantiles(icp, n=20)[18]:.1f}  "
          f"max={max(icp):.1f}")
    print("PASS <500ms" if max(icp) < 500 else "FAIL >500ms")
PY
    else
        echo "latency CSV missing — launch may have failed"
    fi
}

case "${1:-all}" in
    build)  run_build ;;
    run)    run_launch ;;
    summary) summarize ;;
    all)    run_build; run_launch; summarize ;;
    *)      echo "usage: $0 [build|run|summary|all]"; exit 2 ;;
esac
