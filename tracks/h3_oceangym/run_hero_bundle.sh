#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

# Python envs (override if needed).
SUITE_PY="${H3_SUITE_PY:-${REPO_ROOT}/.venv_h3_oceangym/bin/python}"
DATA_PY="${H3_DATA_PY:-${REPO_ROOT}/../.venv_ocean/bin/python}"
POST_PY="${H3_POST_PY:-/home/shuaijun/miniconda3/envs/demo2arm_sim_py310/bin/python}"

DATASET_NC="${H3_DATASET_NC:-/data/private/user2/workspace/ocean/OceanEnv/Data_pipeline/Data/Combined/combined_environment.nc}"
CURRENT_NPZ="${H3_CURRENT_NPZ:-${REPO_ROOT}/runs/_cache/data_grounding/currents/cmems_center_uovo.npz}"

DIFFICULTY="${H3_DIFFICULTY:-easy}"
EPISODES="${H3_EPISODES:-10}"
N_MULTIAGENT="${H3_N_MULTIAGENT:-10}"
POLLUTION_MODEL="${H3_POLLUTION_MODEL:-ocpnet_3d}"
DATASET_DAYS_PER_SIM_SECOND="${H3_DATASET_DAYS_PER_SIM_SECOND:-0.1}"
# Default: run the canonical H3 must-do subset + one pollution cleanup variant.
# Override with e.g. `H3_TASKS="surface_pollution_cleanup_multiagent__localization"`.
TASKS="${H3_TASKS:-go_to_goal_current station_keeping route_following_waypoints depth_profile_tracking formation_transit_multiagent surface_pollution_cleanup_multiagent__containment}"

TAG="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${H3_OUT_DIR:-${REPO_ROOT}/runs/oceangym_h3/hero_bundle_${TAG}}"

cd "${REPO_ROOT}"

if [[ ! -x "${SUITE_PY}" ]]; then
  echo "[h3] ERROR: suite python not found: ${SUITE_PY}" >&2
  exit 2
fi
if [[ ! -x "${DATA_PY}" ]]; then
  echo "[h3] ERROR: data python not found: ${DATA_PY}" >&2
  exit 2
fi
if [[ ! -x "${POST_PY}" ]]; then
  echo "[h3] ERROR: postprocess python not found: ${POST_PY}" >&2
  exit 2
fi

if [[ ! -f "${DATASET_NC}" ]]; then
  echo "[h3] ERROR: dataset not found: ${DATASET_NC}" >&2
  exit 2
fi

if [[ ! -f "${CURRENT_NPZ}" ]]; then
  echo "[h3] exporting currents npz -> ${CURRENT_NPZ}"
  "${DATA_PY}" tracks/h3_oceangym/export_current_series_npz.py \
    --dataset "${DATASET_NC}" \
    --out_npz "${CURRENT_NPZ}"
fi

SSL_CERT_FILE="$("${SUITE_PY}" -c "import certifi; print(certifi.where())")"
export SSL_CERT_FILE

echo "[h3] running suite -> ${OUT_DIR}"
TASK_ARGS=()
if [[ -n "${TASKS//[[:space:]]/}" ]]; then
  # shellcheck disable=SC2206
  TASK_ARGS=(--tasks ${TASKS})
fi
"${SUITE_PY}" tracks/h3_oceangym/run_task_suite.py \
  --preset ocean_worlds_camera \
  "${TASK_ARGS[@]}" \
  --episodes "${EPISODES}" \
  --difficulty "${DIFFICULTY}" \
  --n_multiagent "${N_MULTIAGENT}" \
  --pollution_model "${POLLUTION_MODEL}" \
  --current_npz "${CURRENT_NPZ}" \
  --dataset_days_per_sim_second "${DATASET_DAYS_PER_SIM_SECOND}" \
  --out_dir "${OUT_DIR}"

echo "[h3] postprocess (gif + keyframes) -> ${OUT_DIR}"
"${POST_PY}" tracks/h3_oceangym/postprocess_media.py --roots "${OUT_DIR}"

echo "[h3] DONE"
echo "[h3] results_manifest: ${OUT_DIR}/results_manifest.json"
echo "[h3] media_manifest:   ${OUT_DIR}/media_manifest.json"
echo "[h3] postprocess:      ${OUT_DIR}/postprocess_media_manifest.json"
