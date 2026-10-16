#!/usr/bin/env bash

set -eo pipefail

CONDA_ENV_NAME=chatgarment
CONDA_SH="${HOME}/anaconda3/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
  printf '[error] conda init script not found: %s\n' "$CONDA_SH" >&2
  exit 2
fi

source "$CONDA_SH"
conda activate "$CONDA_ENV_NAME"
set -u

# Configuration: edit these variables directly; this script takes no arguments.
GPU_IDS=(0)
PROCESSES_PER_GPU=8
BATCH_LIST=({148..171})

DATA_ROOT="/inspire/hdd/global_user/liming-253108120187/SartorAI/GCD-Verse/GCD-Verse"
BODY_MESH_DIR="/inspire/hdd/global_user/liming-253108120187/SartorAI/ReWeaver2/DataSet/5000_body_shapes_and_measures/meshes"
PYTHON_BIN="python"
RESOLUTION=(1024 1024)
BASE_SEED=42
SKIP_EXISTING=true

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RENDER_SCRIPT="$SCRIPT_DIR/render_random.py"
LOG_TERMINAL_DIR="$SCRIPT_DIR/log_terminal"

die() {
  printf '[error] %s\n' "$*" >&2
  exit 2
}

[[ -d "$DATA_ROOT" ]] || die "DATA_ROOT does not exist: $DATA_ROOT"
[[ -d "$BODY_MESH_DIR" ]] || die "BODY_MESH_DIR does not exist: $BODY_MESH_DIR"
[[ -f "$RENDER_SCRIPT" ]] || die "render_random.py not found: $RENDER_SCRIPT"
command -v setsid >/dev/null 2>&1 || die "setsid command not found"
[[ "$PROCESSES_PER_GPU" =~ ^[1-9][0-9]*$ ]] || \
  die "PROCESSES_PER_GPU must be a positive integer"
[[ ${#RESOLUTION[@]} -eq 2 ]] || die "RESOLUTION must contain width and height"
[[ "${RESOLUTION[0]}" =~ ^[1-9][0-9]*$ ]] || die "invalid width: ${RESOLUTION[0]}"
[[ "${RESOLUTION[1]}" =~ ^[1-9][0-9]*$ ]] || die "invalid height: ${RESOLUTION[1]}"
[[ "$BASE_SEED" =~ ^-?[0-9]+$ ]] || die "BASE_SEED must be an integer"
case "$SKIP_EXISTING" in
  true|false) ;;
  *) die "SKIP_EXISTING must be true or false" ;;
esac

declare -A SEEN_GPU_IDS=()
for gpu in "${GPU_IDS[@]}"; do
  [[ "$gpu" =~ ^[0-9]+$ ]] || die "invalid GPU ID: $gpu"
  [[ -z "${SEEN_GPU_IDS[$gpu]+x}" ]] || die "duplicate GPU ID: $gpu"
  SEEN_GPU_IDS["$gpu"]=1
done
((${#GPU_IDS[@]} > 0)) || die "GPU_IDS must not be empty"

declare -A SEEN_BATCH_IDS=()
BATCH_IDS=()
for raw_batch_id in "${BATCH_LIST[@]}"; do
  batch_id="${raw_batch_id#batch_}"
  [[ "$batch_id" =~ ^[0-9]+$ ]] || die "invalid batch ID: $raw_batch_id"
  if [[ -n "${SEEN_BATCH_IDS[$batch_id]+x}" ]]; then
    printf '[warning] duplicate batch ID ignored: %s\n' "$batch_id" >&2
    continue
  fi
  [[ -d "$DATA_ROOT/batch_$batch_id/random_body" ]] || \
    die "random_body directory not found: $DATA_ROOT/batch_$batch_id/random_body"
  SEEN_BATCH_IDS["$batch_id"]=1
  BATCH_IDS+=("$batch_id")
done
((${#BATCH_IDS[@]} > 0)) || die "BATCH_LIST must not be empty"

mkdir -p -- "$LOG_TERMINAL_DIR"
cd -- "$SCRIPT_DIR"

declare -A PID_TO_GPU=()
declare -A PID_TO_BATCH=()
declare -A PID_TO_LOG=()

next_batch=0
running=0
failed=0
total_batches=${#BATCH_IDS[@]}
gpu_count=${#GPU_IDS[@]}
total_slots=$((gpu_count * PROCESSES_PER_GPU))
max_concurrent=$total_slots
((total_batches < max_concurrent)) && max_concurrent=$total_batches
cpu_count=$(nproc)
cpu_threads_per_process=$((cpu_count / max_concurrent))
((cpu_threads_per_process < 1)) && cpu_threads_per_process=1

launch_next() {
  local gpu="$1"
  local batch_id="${BATCH_IDS[$next_batch]}"
  local batch_name="batch_$batch_id"
  local batch_root="$DATA_ROOT/$batch_name"
  local log_path="$LOG_TERMINAL_DIR/${batch_name}_render_random.log"
  local -a render_args=(
    --root "$batch_root"
    --body_mesh_dir "$BODY_MESH_DIR"
    --resolution "${RESOLUTION[0]}" "${RESOLUTION[1]}"
    --seed "$BASE_SEED"
  )
  if [[ "$SKIP_EXISTING" == true ]]; then
    render_args+=(--skip_existing)
  fi

  printf '[launch] %-10s GPU=%s log=%s\n' "$batch_name" "$gpu" "$log_path"
  printf '[config] batch=%s GPU=%s resolution=%sx%s seed=%s skip_existing=%s\n' \
    "$batch_name" "$gpu" "${RESOLUTION[0]}" "${RESOLUTION[1]}" \
    "$BASE_SEED" "$SKIP_EXISTING" >"$log_path"

  setsid env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS="$cpu_threads_per_process" \
    MKL_NUM_THREADS="$cpu_threads_per_process" \
    OPENBLAS_NUM_THREADS="$cpu_threads_per_process" \
    NUMEXPR_NUM_THREADS="$cpu_threads_per_process" \
    "$PYTHON_BIN" "$RENDER_SCRIPT" "${render_args[@]}" \
    >>"$log_path" 2>&1 &

  local pid=$!
  PID_TO_GPU["$pid"]="$gpu"
  PID_TO_BATCH["$pid"]="$batch_name"
  PID_TO_LOG["$pid"]="$log_path"
  next_batch=$((next_batch + 1))
  running=$((running + 1))
}

terminate_children() {
  local pid attempt alive
  for pid in "${!PID_TO_GPU[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  done

  for ((attempt = 0; attempt < 10; attempt++)); do
    alive=0
    for pid in "${!PID_TO_GPU[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        alive=1
        break
      fi
    done
    ((alive == 0)) && break
    sleep 1
  done

  for pid in "${!PID_TO_GPU[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
  done
  wait || true
}

stop_all() {
  trap - INT TERM
  printf '\n[stop] terminating %d running render process(es)\n' "$running" >&2
  terminate_children
  exit 130
}
trap stop_all INT TERM

printf '[config] conda_env=%s batches=%d GPUs=%d processes_per_GPU=%d total_slots=%d cpu_cores=%d cpu_threads_per_process=%d\n' \
  "$CONDA_DEFAULT_ENV" "$total_batches" "$gpu_count" "$PROCESSES_PER_GPU" \
  "$total_slots" "$cpu_count" "$cpu_threads_per_process"
printf '[config] data_root=%s log_terminal=%s\n' "$DATA_ROOT" "$LOG_TERMINAL_DIR"

for ((slot = 0; slot < PROCESSES_PER_GPU && next_batch < total_batches; slot++)); do
  for gpu in "${GPU_IDS[@]}"; do
    ((next_batch < total_batches)) || break
    launch_next "$gpu"
  done
done

while ((running > 0)); do
  finished_pid=""
  if wait -n -p finished_pid; then
    status=0
  else
    status=$?
  fi

  gpu="${PID_TO_GPU[$finished_pid]}"
  batch_name="${PID_TO_BATCH[$finished_pid]}"
  log_path="${PID_TO_LOG[$finished_pid]}"
  unset 'PID_TO_GPU[$finished_pid]' 'PID_TO_BATCH[$finished_pid]' 'PID_TO_LOG[$finished_pid]'
  running=$((running - 1))

  if ((status == 0)) && ! grep -Eq \
    '^done: success=[0-9]+, skipped=[0-9]+, failed=0, total=[0-9]+, seconds=' \
    "$log_path"; then
    printf '[error]  %-10s GPU=%s exited cleanly but its render summary is missing or reports failures\n' \
      "$batch_name" "$gpu" >&2
    status=1
  fi

  if ((status == 0)); then
    printf '[done]   %-10s GPU=%s\n' "$batch_name" "$gpu"
  else
    printf '[failed] %-10s GPU=%s exit=%d log=%s\n' \
      "$batch_name" "$gpu" "$status" "$log_path" >&2
    failed=$((failed + 1))
  fi

  if ((next_batch < total_batches)); then
    launch_next "$gpu"
  fi
done

if ((failed > 0)); then
  printf '[summary] %d/%d batches failed\n' "$failed" "$total_batches" >&2
  exit 1
fi

printf '[summary] all %d batches completed\n' "$total_batches"
