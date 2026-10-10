#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/run_all.sh STAGE1_RUN_ROOT"
  exit 2
fi

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
proposal_run_root="$(cd "$1" && pwd)"
run_root="${RUN_ROOT:-$project_root/runs/jarvis_hpsafemoe}"
config="$project_root/configs/jarvis5_stage21.yaml"
environment="${STAGE2_ENV:-hpsafe-jarvis-stage2}"
gpu_lane="${GPU_LANE:-0}"
lock="$run_root/policy/deployment_validation_lock.json"
barrier="$run_root/policy/prediction_barrier.json"
child=""

clean_env=(
  env -u PYTHONPATH -u PYTHONHOME -u PYTHONUSERBASE -u PIP_TARGET -u PIP_PREFIX
  -u LD_LIBRARY_PATH -u CUDA_HOME -u CUDA_PATH PYTHONNOUSERSITE=1 PIP_USER=0
)

mkdir -p "$run_root/logs" "$run_root/launcher"
if ! mkdir "$run_root/launcher/active.lock" 2>/dev/null; then
  echo "Active/stale HP-SafeMoE lock exists: $run_root/launcher/active.lock"
  exit 1
fi
echo "$$" > "$run_root/launcher/active.lock/pid"
cleanup() {
  [[ -n "$child" ]] && kill -TERM "$child" 2>/dev/null || true
  rm -f "$run_root/launcher/active.lock/pid"
  rmdir "$run_root/launcher/active.lock" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

CUDA_VISIBLE_DEVICES="$gpu_lane" "${clean_env[@]}" \
  conda run --no-capture-output -n "$environment" \
  python -s "$project_root/scripts/preflight.py" \
  --proposal-run-root "$proposal_run_root" --config "$config" --output-root "$run_root" \
  > "$run_root/logs/preflight.log" 2>&1

CUDA_VISIBLE_DEVICES="$gpu_lane" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "${clean_env[@]}" conda run --no-capture-output -n "$environment" \
  python -s "$project_root/scripts/run_pipeline.py" calibrate-validation \
  --proposal-run-root "$proposal_run_root" --output-root "$run_root" --config "$config" \
  --device cuda --cpu-threads 4 > "$run_root/logs/calibrate_validation.log" 2>&1 &
child="$!"
wait "$child"
child=""

"${clean_env[@]}" conda run --no-capture-output -n "$environment" \
  python -s "$project_root/scripts/run_pipeline.py" predict-test \
  --proposal-run-root "$proposal_run_root" --output-root "$run_root" --lock "$lock" \
  > "$run_root/logs/predict_test.log" 2>&1
"${clean_env[@]}" conda run --no-capture-output -n "$environment" \
  python -s "$project_root/scripts/run_pipeline.py" commit-barrier \
  --proposal-run-root "$proposal_run_root" --output-root "$run_root" --lock "$lock" \
  > "$run_root/logs/commit_barrier.log" 2>&1
"${clean_env[@]}" conda run --no-capture-output -n "$environment" \
  python -s "$project_root/scripts/run_pipeline.py" score-test \
  --proposal-run-root "$proposal_run_root" --output-root "$run_root" --barrier "$barrier" \
  > "$run_root/logs/score_test.log" 2>&1
"${clean_env[@]}" conda run --no-capture-output -n "$environment" \
  python -s "$project_root/scripts/run_pipeline.py" summarize --output-root "$run_root" \
  > "$run_root/logs/summarize.log" 2>&1

trap - EXIT INT TERM
cleanup
echo "COMPLETE: $run_root/results/JARVIS_RESULTS.md"
