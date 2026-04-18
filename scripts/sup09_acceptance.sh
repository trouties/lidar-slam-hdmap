#!/usr/bin/env bash
# SUP-09 acceptance harness.
#   - default: "cached" mode (cache_sup09/ pre-populated → validates compose
#     + pipeline + report path end-to-end without re-downloading the tarball)
#   - "cold" mode: wipes cache_sup09/, serves the tarball via a loopback HTTP
#     server, points SUP09_SUBSET_URL at host.docker.internal → validates the
#     hydrate-from-URL path that a fresh-clone user experiences
#
# Usage:
#   bash scripts/sup09_acceptance.sh              # cached mode
#   bash scripts/sup09_acceptance.sh cached
#   bash scripts/sup09_acceptance.sh cold         # requires /tmp/kitti_seq00_200.tar.gz
#
# Writes benchmarks/sup09/acceptance_summary.json with timing + APE + sha256.

set -euo pipefail

MODE="${1:-cached}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="${REPO_ROOT}/benchmarks/sup09"
RESULTS_DIR="${REPO_ROOT}/results"
CACHE_DIR="${REPO_ROOT}/cache_sup09"
TARBALL="${SUP09_TARBALL:-/tmp/kitti_seq00_200.tar.gz}"

mkdir -p "${BENCH}"
cd "${REPO_ROOT}"

ape_rmse_threshold_m=50.0
up_seconds_budget=600
build_seconds_budget=900

cleanup_artifacts() {
    rm -f "${RESULTS_DIR}/ape.txt" "${RESULTS_DIR}/trajectory.png" \
          "${RESULTS_DIR}/poses_00.txt" "${RESULTS_DIR}/poses_optimized_00.txt" \
          "${RESULTS_DIR}/poses_fused_00.txt"
}

prepare_cached() {
    echo "=== SUP-09 acceptance: cached mode ==="
    if [ ! -f "${CACHE_DIR}/sequences/00/velodyne/000000.bin" ]; then
        echo "[prepare] seeding cache_sup09/ from ${TARBALL}"
        [ -f "${TARBALL}" ] || { echo "missing tarball at ${TARBALL}" >&2; exit 2; }
        mkdir -p "${CACHE_DIR}"
        tar -xzf "${TARBALL}" -C "${CACHE_DIR}"
    fi
    unset SUP09_SUBSET_URL SUP09_SUBSET_SHA256 || true
}

serve_tarball() {
    [ -f "${TARBALL}" ] || { echo "missing tarball at ${TARBALL}" >&2; exit 2; }
    local port="${SUP09_HTTP_PORT:-8098}"
    local serve_dir
    serve_dir="$(dirname "${TARBALL}")"
    python3 -m http.server "${port}" --bind 0.0.0.0 --directory "${serve_dir}" \
        >"${BENCH}/http_server.log" 2>&1 &
    HTTP_PID=$!
    sleep 1
    if ! kill -0 "${HTTP_PID}" 2>/dev/null; then
        echo "http.server failed to start" >&2
        cat "${BENCH}/http_server.log" >&2
        exit 3
    fi
    echo "[prepare] python http.server pid=${HTTP_PID} port=${port}"
    export SUP09_SUBSET_URL="http://host.docker.internal:${port}/$(basename "${TARBALL}")"
    if [ -f "${TARBALL}.sha256" ]; then
        export SUP09_SUBSET_SHA256="$(awk '{print $1}' "${TARBALL}.sha256")"
    fi
}

stop_http() {
    if [ -n "${HTTP_PID:-}" ] && kill -0 "${HTTP_PID}" 2>/dev/null; then
        kill "${HTTP_PID}" 2>/dev/null || true
        wait "${HTTP_PID}" 2>/dev/null || true
    fi
}
trap stop_http EXIT

prepare_cold() {
    echo "=== SUP-09 acceptance: cold mode ==="
    rm -rf "${CACHE_DIR}"
    mkdir -p "${CACHE_DIR}"
    serve_tarball
}

case "${MODE}" in
    cached) prepare_cached ;;
    cold)   prepare_cold ;;
    *)      echo "usage: $0 [cached|cold]" >&2; exit 2 ;;
esac

cleanup_artifacts

echo "=== docker compose build ==="
build_start=$(date +%s)
docker compose build 2>&1 | tee "${BENCH}/build.log" | tail -5
build_end=$(date +%s)
build_seconds=$((build_end - build_start))
echo "build_seconds=${build_seconds}"

echo "=== docker compose up ==="
up_start=$(date +%s)
docker compose up --abort-on-container-exit --exit-code-from sup09 \
    2>&1 | tee "${BENCH}/up.log" | tail -20
up_exit=${PIPESTATUS[0]}
up_end=$(date +%s)
up_seconds=$((up_end - up_start))
echo "up_seconds=${up_seconds}  exit_code=${up_exit}"

[ "${up_exit}" -eq 0 ] || { echo "compose up failed"; exit "${up_exit}"; }

# --- assertions ---------------------------------------------------------
[ -f "${RESULTS_DIR}/ape.txt" ]        || { echo "missing ape.txt" >&2; exit 10; }
[ -f "${RESULTS_DIR}/trajectory.png" ] || { echo "missing trajectory.png" >&2; exit 11; }

ape_rmse="$(awk -F= '/^ape_rmse=/{print $2; exit}' "${RESULTS_DIR}/ape.txt")"
n_frames="$(awk -F= '/^n_frames=/{print $2; exit}' "${RESULTS_DIR}/ape.txt")"
[ -n "${ape_rmse}" ] || { echo "could not parse ape_rmse" >&2; exit 12; }

python3 - "${ape_rmse}" "${ape_rmse_threshold_m}" <<'PY' || exit 13
import sys
rmse, thresh = float(sys.argv[1]), float(sys.argv[2])
if rmse > thresh:
    print(f"FAIL: ape_rmse={rmse:.3f} > {thresh}")
    raise SystemExit(1)
print(f"APE OK: ape_rmse={rmse:.3f} m <= {thresh}")
PY

if [ "${up_seconds}" -gt "${up_seconds_budget}" ]; then
    echo "FAIL: up_seconds=${up_seconds} > ${up_seconds_budget}" >&2
    exit 14
fi
if [ "${build_seconds}" -gt "${build_seconds_budget}" ]; then
    echo "WARN: build_seconds=${build_seconds} > ${build_seconds_budget}"
fi

# --- summary JSON -------------------------------------------------------
git_sha="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
subset_sha256="unknown"
if [ -f "${TARBALL}.sha256" ]; then
    subset_sha256="$(awk '{print $1}' "${TARBALL}.sha256")"
fi
image_digest="$(docker image inspect lidar-slam-hdmap-sup09:latest \
                 --format '{{index .Id}}' 2>/dev/null || echo unknown)"

python3 - <<PY
import json, pathlib, time
out = pathlib.Path("${BENCH}/acceptance_summary.json")
summary = {
    "mode": "${MODE}",
    "run_ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "git_sha": "${git_sha}",
    "image_digest": "${image_digest}",
    "subset_sha256": "${subset_sha256}",
    "build_seconds": ${build_seconds},
    "up_seconds": ${up_seconds},
    "ape_rmse": float("${ape_rmse}"),
    "n_frames": int("${n_frames}"),
    "up_budget_seconds": ${up_seconds_budget},
    "build_budget_seconds": ${build_seconds_budget},
    "ape_threshold_m": ${ape_rmse_threshold_m},
    "acceptance": "PASS",
}
out.write_text(json.dumps(summary, indent=2) + "\n")
print(f"wrote {out}")
PY

cp -f "${RESULTS_DIR}/trajectory.png" "${BENCH}/trajectory_seq00_200.png"
echo "=== SUP-09 acceptance PASS (mode=${MODE}, ape_rmse=${ape_rmse} m, up=${up_seconds}s, build=${build_seconds}s) ==="
