#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

SUITE_PY="${OCEANGYM_SUITE_PY:-python}"
DATA_PY="${OCEANGYM_DATA_PY:-python}"
POST_PY="${OCEANGYM_POST_PY:-python}"

DATASET_NC="${OCEANGYM_DATASET_NC:-${REPO_ROOT}/Data_pipeline/Data/Combined/combined_environment.nc}"
CURRENT_NPZ="${OCEANGYM_CURRENT_NPZ:-${REPO_ROOT}/runs/_cache/data_grounding/currents/cmems_center_uovo.npz}"

DIFFICULTY="${OCEANGYM_DIFFICULTY:-easy}"
EPISODES="${OCEANGYM_EPISODES:-10}"
N_MULTIAGENT="${OCEANGYM_N_MULTIAGENT:-10}"
POLLUTION_MODEL="${OCEANGYM_POLLUTION_MODEL:-ocpnet_3d}"
DATASET_DAYS_PER_SIM_SECOND="${OCEANGYM_DATASET_DAYS_PER_SIM_SECOND:-0.1}"
TASKS="${OCEANGYM_TASKS:-go_to_goal_current station_keeping route_following_waypoints depth_profile_tracking formation_transit_multiagent surface_pollution_cleanup_multiagent__containment}"

TAG="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OCEANGYM_OUT_DIR:-${REPO_ROOT}/runs/oceangym_benchmark/hero_bundle_${TAG}}"

cd "${REPO_ROOT}"

if ! command -v "${SUITE_PY}" >/dev/null 2>&1 && [[ ! -x "${SUITE_PY}" ]]; then
  echo "[oceangym-benchmark] suite python not found: ${SUITE_PY}" >&2
  exit 2
fi
if ! command -v "${DATA_PY}" >/dev/null 2>&1 && [[ ! -x "${DATA_PY}" ]]; then
  echo "[oceangym-benchmark] data python not found: ${DATA_PY}" >&2
  exit 2
fi
if ! command -v "${POST_PY}" >/dev/null 2>&1 && [[ ! -x "${POST_PY}" ]]; then
  echo "[oceangym-benchmark] postprocess python not found: ${POST_PY}" >&2
  exit 2
fi
if [[ ! -f "${DATASET_NC}" ]]; then
  echo "[oceangym-benchmark] dataset not found: ${DATASET_NC}" >&2
  exit 2
fi

if [[ ! -f "${CURRENT_NPZ}" ]]; then
  echo "[oceangym-benchmark] exporting current series -> ${CURRENT_NPZ}"
  "${DATA_PY}" tracks/oceangym_benchmark/export_current_series_npz.py \
    --dataset "${DATASET_NC}" \
    --out_npz "${CURRENT_NPZ}"
fi

SSL_CERT_FILE="$("${SUITE_PY}" -c "import certifi; print(certifi.where())")"
export SSL_CERT_FILE

TASK_ARGS=()
if [[ -n "${TASKS//[[:space:]]/}" ]]; then
  # shellcheck disable=SC2206
  TASK_ARGS=(--tasks ${TASKS})
fi

echo "[oceangym-benchmark] running suite -> ${OUT_DIR}"
"${SUITE_PY}" tracks/oceangym_benchmark/run_task_suite.py \
  --preset ocean_worlds_camera \
  "${TASK_ARGS[@]}" \
  --episodes "${EPISODES}" \
  --difficulty "${DIFFICULTY}" \
  --n_multiagent "${N_MULTIAGENT}" \
  --pollution_model "${POLLUTION_MODEL}" \
  --current_npz "${CURRENT_NPZ}" \
  --dataset_days_per_sim_second "${DATASET_DAYS_PER_SIM_SECOND}" \
  --out_dir "${OUT_DIR}"

echo "[oceangym-benchmark] postprocess -> ${OUT_DIR}"
"${POST_PY}" tracks/oceangym_benchmark/postprocess_media.py --roots "${OUT_DIR}"

echo "[oceangym-benchmark] results_manifest: ${OUT_DIR}/results_manifest.json"
echo "[oceangym-benchmark] media_manifest: ${OUT_DIR}/media_manifest.json"
echo "[oceangym-benchmark] postprocess_manifest: ${OUT_DIR}/postprocess_media_manifest.json"
