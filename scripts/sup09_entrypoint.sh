#!/usr/bin/env bash
# SUP-09 Docker Compose entrypoint.
#   1. If /data does not yet contain the KITTI Seq 00 subset, download and
#      extract the tarball from $SUP09_SUBSET_URL (overridable via compose env).
#   2. Run scripts/run_pipeline.py on the 200-frame subset with the compose
#      config (configs/sup09_docker.yaml).
#   3. Run scripts/sup09_report.py to emit results/ape.txt + results/trajectory.png.
#
# Assumes compose mounts:
#   ./cache_sup09  -> /data        (persistent subset cache; entrypoint hydrates on first run)
#   ./results      -> /app/results (host-visible artifacts)

set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
RESULTS_DIR="${RESULTS_DIR:-/app/results}"
SEQUENCE="${SEQUENCE:-00}"
MAX_FRAMES="${MAX_FRAMES:-200}"
CONFIG="${CONFIG:-configs/sup09_docker.yaml}"
SUP09_SUBSET_URL="${SUP09_SUBSET_URL:-}"
SUP09_SUBSET_SHA256="${SUP09_SUBSET_SHA256:-}"

mkdir -p "${DATA_DIR}" "${RESULTS_DIR}"

sentinel="${DATA_DIR}/sequences/${SEQUENCE}/velodyne/000000.bin"
if [ ! -f "${sentinel}" ]; then
    if [ -z "${SUP09_SUBSET_URL}" ]; then
        echo "[sup09-entrypoint] FATAL: subset missing at ${sentinel} and SUP09_SUBSET_URL is unset" >&2
        exit 2
    fi
    echo "[sup09-entrypoint] hydrating subset from ${SUP09_SUBSET_URL}"
    tmp_tar="$(mktemp /tmp/kitti_sup09.XXXXXX.tar.gz)"
    trap 'rm -f "${tmp_tar}"' EXIT
    curl -L --fail --silent --show-error -o "${tmp_tar}" "${SUP09_SUBSET_URL}"
    if [ -n "${SUP09_SUBSET_SHA256}" ]; then
        actual="$(sha256sum "${tmp_tar}" | awk '{print $1}')"
        if [ "${actual}" != "${SUP09_SUBSET_SHA256}" ]; then
            echo "[sup09-entrypoint] FATAL: sha256 mismatch (expected ${SUP09_SUBSET_SHA256}, got ${actual})" >&2
            exit 3
        fi
        echo "[sup09-entrypoint] sha256 OK"
    fi
    tar -xzf "${tmp_tar}" -C "${DATA_DIR}"
    rm -f "${tmp_tar}"
    trap - EXIT
    if [ ! -f "${sentinel}" ]; then
        echo "[sup09-entrypoint] FATAL: tarball extracted but ${sentinel} still missing" >&2
        exit 4
    fi
    echo "[sup09-entrypoint] subset ready under ${DATA_DIR}"
else
    echo "[sup09-entrypoint] subset already present under ${DATA_DIR}, skipping download"
fi

echo "[sup09-entrypoint] running pipeline (seq=${SEQUENCE}, max_frames=${MAX_FRAMES})"
python scripts/run_pipeline.py \
    --config "${CONFIG}" \
    --sequence "${SEQUENCE}" \
    --max-frames "${MAX_FRAMES}"

echo "[sup09-entrypoint] generating APE + trajectory artifacts"
python -m scripts.sup09_report \
    --results-dir "${RESULTS_DIR}" \
    --sequence "${SEQUENCE}" \
    --dataset-root "${DATA_DIR}"

echo "[sup09-entrypoint] done. Artifacts in ${RESULTS_DIR}:"
ls -la "${RESULTS_DIR}/ape.txt" "${RESULTS_DIR}/trajectory.png"
