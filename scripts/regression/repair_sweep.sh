#!/usr/bin/env bash
# Regression sweep for the dropped-frame repair feature.
#
# Runs the existing verify-imx219-circlegrid and verify-thermal docker-compose
# services twice each — once with KALIBR_REPAIR_DISABLED=1 (baseline) and once
# without (repair enabled) — across all nav-operations circlegrid datasets.
# Each run uses a unique RID (results folder) so outputs don't clash.
#
# After both runs complete, the script extracts the key metrics
# (reprojection RMSE, identified markers, FCG count, repair_report.txt span count)
# into repair_sweep_summary.csv under the sweep output directory.
#
# Cache is intentionally bypassed by deleting any corners_cam*.csv under the
# dataset's corner cache directory before each run — the user requested
# "without cache" so every run reprocesses from raw images.
set -euo pipefail

COMPOSE=/home/kmro/praca/dev/kalibr-ws-dops/src/kalibr/docker-compose.verification.yml
if [[ ! -f "${COMPOSE}" ]]; then
  echo "docker-compose file not found at ${COMPOSE}" >&2
  exit 1
fi

: "${KALIBR_IMAGE:=dops-kalibr-dev}"
export KALIBR_IMAGE

DATASETS_ROOT=/home/kmro/praca/dev/datasets/nav-operations
DATASETS=(
  imx219_circlegrid_nord4_1
  imx219_circlegrid_nord4_2
  thermal_circlegrid_eposN_1
  thermal_circlegrid_eposN_2
  thermal_circlegrid_eposN_3
  thermal_circlegrid_eposN_4
)

SWEEP_OUT=/home/kmro/praca/dev/kalibr-ws-dops/repair_sweep_results
mkdir -p "${SWEEP_OUT}"
SUMMARY="${SWEEP_OUT}/repair_sweep_summary.csv"
echo "dataset,mode,rid,identified_total,fcg_count,spans,reproj_rmse" > "${SUMMARY}"

clear_caches() {
  local ds="$1"
  find "${DATASETS_ROOT}/${ds}" -maxdepth 3 -name 'corners_cam*.csv' -delete 2>/dev/null || true
  find "${DATASETS_ROOT}/${ds}" -maxdepth 3 -name 'cache_corners' -type d -exec rm -rf {} + 2>/dev/null || true
}

run_service() {
  local service="$1"
  local rid="$2"
  local env_extra="$3"
  echo ">>> ${service} RID=${rid} ${env_extra}"
  env RID="${rid}" ${env_extra} docker compose -f "${COMPOSE}" run --rm "${service}"
}

extract_metrics() {
  local ds="$1"
  local mode="$2"
  local rid="$3"
  local base="${DATASETS_ROOT}/${ds}/${rid}"
  local identified_total=""
  local fcg_count=""
  local spans=0
  local reproj=""
  if [[ -f "${base}/run.log" ]]; then
    reproj=$(grep -oE 'reprojection error.*[0-9.]+' "${base}/run.log" | tail -1 | grep -oE '[0-9.]+$' || true)
  fi
  if [[ -f "${base}/repair_report.txt" ]]; then
    spans=$(grep -c '^span ' "${base}/repair_report.txt" || echo 0)
  fi
  if [[ -d "${base}" ]]; then
    identified_total=$(find "${base}" -name 'corners_cam*.csv' -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}' || echo "")
  fi
  echo "${ds},${mode},${rid},${identified_total},${fcg_count},${spans},${reproj}" >> "${SUMMARY}"
}

run_sweep_pair() {
  local service="$1"
  local datasets_csv="$2"   # comma-separated for metrics extraction
  local mode="$3"
  local rid="$4"
  local env_extra="$5"

  IFS=',' read -ra ds_array <<< "${datasets_csv}"
  for ds in "${ds_array[@]}"; do
    clear_caches "${ds}"
  done
  run_service "${service}" "${rid}" "${env_extra}"
  for ds in "${ds_array[@]}"; do
    extract_metrics "${ds}" "${mode}" "${rid}"
  done
}

# imx219 pair: nord4_1 + nord4_2
run_sweep_pair verify-imx219-circlegrid "imx219_circlegrid_nord4_1,imx219_circlegrid_nord4_2" baseline results_sweep_baseline "KALIBR_REPAIR_DISABLED=1"
run_sweep_pair verify-imx219-circlegrid "imx219_circlegrid_nord4_1,imx219_circlegrid_nord4_2" repair   results_sweep_repair   "KALIBR_REPAIR_DISABLED=0"

# thermal quadruple: eposN_1..eposN_4
run_sweep_pair verify-thermal "thermal_circlegrid_eposN_1,thermal_circlegrid_eposN_2,thermal_circlegrid_eposN_3,thermal_circlegrid_eposN_4" baseline results_sweep_baseline "KALIBR_REPAIR_DISABLED=1"
run_sweep_pair verify-thermal "thermal_circlegrid_eposN_1,thermal_circlegrid_eposN_2,thermal_circlegrid_eposN_3,thermal_circlegrid_eposN_4" repair   results_sweep_repair   "KALIBR_REPAIR_DISABLED=0"

echo
echo "Sweep complete. Summary: ${SUMMARY}"
