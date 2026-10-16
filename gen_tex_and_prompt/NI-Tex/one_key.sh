#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Defaults for the current dataset and the 8-GPU H200 deployment.
DATA_ROOT="$WORKSPACE_ROOT/GCD-Verse/GCD-Verse"
BASE_CONFIG="$SCRIPT_DIR/cfgs/inference_100k.yml"

BATCH_LIST="148-171"
GPU_LIST="0-7"
PROCESSES_PER_GPU=3
DETECT_JOBS=32

VIS_FLAG=false

PYTHON_BIN="${PYTHON_BIN:-python}"
CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
export HF_HOME="${HF_HOME:-/inspire/hdd/global_user/liming-253108120187/Datasets/HF_HOUSE}"

usage() {
  cat <<'EOF'
Usage: bash one_key.sh [options]

  --batch_list LIST       Batch IDs/ranges, e.g. 148-171 or 148,150-153 (default: 148-171)
  --data_root PATH       Dataset root containing batch_* directories
  --gpu_ids LIST         GPU IDs/ranges (default: 0-7)
  --processes_per_gpu N  Concurrent generation processes per GPU (default: 3)
  --detect_jobs N        Concurrent detection processes (default: 32)
  --base PATH            Inference YAML (default: cfgs/inference_100k.yml)
  --vis_flag             Save detector visualizations
  -h, --help             Show this help

Round 0 uses match rank 0. Round N uses rank N and only the case/part pairs
flagged by round N-1. Rounds are fixed at 0-4; complete texture outputs are skipped.
The workflow stops retrying early when detection finds no abnormal parts.
After the rounds finish, it gathers the selected batches into textured/.
Inference uses conda environment NI-Tex; detection and gathering use chatgarment.
EOF
}

die() { printf '[error] %s\n' "$*" >&2; exit 2; }
while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --vis_flag) VIS_FLAG=true; shift; continue ;;
  esac
  [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "Missing value for $1"
  case "$1" in
    --batch_list) BATCH_LIST=$2 ;;
    --data_root) DATA_ROOT=$2 ;;
    --gpu_ids) GPU_LIST=$2 ;;
    --processes_per_gpu) PROCESSES_PER_GPU=$2 ;;
    --detect_jobs) DETECT_JOBS=$2 ;;
    --base) BASE_CONFIG=$2 ;;
    *) die "Unknown argument: $1" ;;
  esac
  shift 2
done

[[ "$PROCESSES_PER_GPU" =~ ^[1-9][0-9]*$ ]] || die "--processes_per_gpu must be positive"
[[ "$DETECT_JOBS" =~ ^[1-9][0-9]*$ ]] || die "--detect_jobs must be positive"

# Parse without a subshell so invalid ranges cannot silently disappear.
parse_ids() {
  local value=$1 token start end id
  local -a tokens=()
  local -A seen=()
  PARSED_IDS=()
  read -r -a tokens <<< "${value//,/ }"
  for token in "${tokens[@]}"; do
    token=${token//batch_/}
    if [[ "$token" =~ ^([0-9]+)-([0-9]+)$ ]]; then
      start=$((10#${BASH_REMATCH[1]})); end=$((10#${BASH_REMATCH[2]}))
    elif [[ "$token" =~ ^[0-9]+$ ]]; then
      start=$((10#$token)); end=$start
    else
      die "Invalid ID/range: $token"
    fi
    ((start <= end)) || die "Reversed range: $token"
    for ((id=start; id<=end; id++)); do
      if [[ -z "${seen[$id]+x}" ]]; then
        PARSED_IDS+=("$id"); seen[$id]=1
      fi
    done
  done
  ((${#PARSED_IDS[@]})) || die "Empty ID list"
}
parse_ids "$BATCH_LIST"; BATCH_IDS=("${PARSED_IDS[@]}")
parse_ids "$GPU_LIST"; GPU_IDS=("${PARSED_IDS[@]}")

[[ -d "$DATA_ROOT" ]] || die "Dataset root not found: $DATA_ROOT"
DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
for id in "${BATCH_IDS[@]}"; do
  [[ -d "$DATA_ROOT/batch_$id/random_body" ]] || die "Missing random_body: batch_$id"
done
[[ -f "$CONDA_SH" ]] || die "Conda init script not found: $CONDA_SH"
cd "$SCRIPT_DIR"
[[ -f "$BASE_CONFIG" ]] || die "Inference config not found: $BASE_CONFIG"
[[ -f ckpt/RealESRGAN_x4plus.pth ]] || die "Missing ckpt/RealESRGAN_x4plus.pth"
export HF_HUB_CACHE="$HF_HOME/hub" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source "$CONDA_SH"
set -u

declare -A PID_TO_GPU=() PID_TO_BATCH=() PID_TO_READY=()
READY_DIR=""
cleanup() {
  local pid
  for pid in "${!PID_TO_BATCH[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait || true
  [[ -z "$READY_DIR" ]] || rm -rf -- "$READY_DIR"
  return 0
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

activate_env() {
  set +u
  conda activate "$1"
  set -u
}

report_dir() {
  if (($1 == 0)); then
    printf '%s/detect_weird_output' "$SCRIPT_DIR"
  else
    printf '%s/detect_weird_output_after_regen%s' "$SCRIPT_DIR" "$1"
  fi
}

launch_next() {
  local gpu=$1 id=${STAGE_BATCHES[$NEXT_BATCH]} pid log_path ready_file
  local batch_root="$DATA_ROOT/batch_$id"
  if [[ "$STAGE" == inference ]]; then
    log_path="$SCRIPT_DIR/log_terminal/batch_${id}_${OUT_SUBDIR}.log"
    ready_file="$READY_DIR/round_${CURRENT_ROUND}_batch_$id.ready"
    local -a filter_args=()
    if ((CURRENT_ROUND > 0)); then filter_args=(--weird_log_dir "$PREVIOUS_REPORT_DIR"); fi
    # Isolate concurrent copies of Diffusers' custom pipeline code.
    HF_MODULES_CACHE="$READY_DIR/modules_round_${CURRENT_ROUND}_batch_$id" \
    NI_TEX_READY_FILE="$ready_file" NI_TEX_CPU_THREADS="$CPU_THREADS" \
    OMP_NUM_THREADS="$CPU_THREADS" MKL_NUM_THREADS="$CPU_THREADS" \
    OPENBLAS_NUM_THREADS="$CPU_THREADS" NUMEXPR_NUM_THREADS="$CPU_THREADS" \
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -u -B "$SCRIPT_DIR/batch_gcd_textured.py" \
      --root "$batch_root" --rank "$CURRENT_ROUND" --base "$BASE_CONFIG" \
      --out_subdir "$OUT_SUBDIR" --skip_existing "${filter_args[@]}" >>"$log_path" 2>&1 &
    pid=$!
  else
    mkdir -p "$REPORT_DIR/$id"
    log_path="$REPORT_DIR/$id/detect_weird.log"
    ready_file=""
    local -a vis_args=()
    if [[ "$VIS_FLAG" == true ]]; then vis_args=(--vis_flag); fi
    "$PYTHON_BIN" -u -B "$SCRIPT_DIR/detect_weird.py" --root "$batch_root" \
      --out_dir "$REPORT_DIR/$id" --texture_subdir "$OUT_SUBDIR" \
      "${vis_args[@]}" >"$log_path" 2>&1 &
    pid=$!
  fi
  PID_TO_GPU[$pid]=$gpu; PID_TO_BATCH[$pid]=$id; PID_TO_READY[$pid]=$ready_file
  LAST_PID=$pid
  NEXT_BATCH=$((NEXT_BATCH + 1))
  printf '[launch] round=%s stage=%s batch_%s device=%s log=%s\n' \
    "$CURRENT_ROUND" "$STAGE" "$id" "$gpu" "$log_path"
}

wait_for_wave() {
  local pid
  for pid in "$@"; do
    while [[ ! -f "${PID_TO_READY[$pid]}" ]]; do
      kill -0 "$pid" 2>/dev/null || die "batch_${PID_TO_BATCH[$pid]} exited before model-ready; inspect its log"
      sleep 1
    done
  done
}

finish_one() {
  local pid finished_pid="" status gpu id
  # A no-op worker can already have exited while other models were loading.
  for pid in "${!PID_TO_BATCH[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then finished_pid=$pid; break; fi
  done
  if [[ -n "$finished_pid" ]]; then
    if wait "$finished_pid"; then status=0; else status=$?; fi
  else
    if wait -n -p finished_pid "${!PID_TO_BATCH[@]}"; then status=0; else status=$?; fi
    if [[ -z "${finished_pid:-}" ]]; then
      # Handle completion between the liveness check and wait -n.
      for pid in "${!PID_TO_BATCH[@]}"; do finished_pid=$pid; break; done
      if wait "$finished_pid"; then status=0; else status=$?; fi
    fi
  fi
  gpu=${PID_TO_GPU[$finished_pid]}; id=${PID_TO_BATCH[$finished_pid]}
  [[ -z "${PID_TO_READY[$finished_pid]}" ]] || rm -f -- "${PID_TO_READY[$finished_pid]}"
  unset 'PID_TO_GPU[$finished_pid]' 'PID_TO_BATCH[$finished_pid]' 'PID_TO_READY[$finished_pid]'
  if ((status)); then FAILED=$((FAILED + 1)); fi
  printf '[done] round=%s stage=%s batch_%s exit=%s\n' "$CURRENT_ROUND" "$STAGE" "$id" "$status"
  if ((NEXT_BATCH < ${#STAGE_BATCHES[@]})); then launch_next "$gpu"; fi
}

run_stage() {
  STAGE=$1
  NEXT_BATCH=0; FAILED=0
  PID_TO_GPU=(); PID_TO_BATCH=(); PID_TO_READY=()
  local max_parallel wave gpu
  local -a wave_pids=()
  if [[ "$STAGE" == inference ]]; then
    activate_env NI-Tex
    mkdir -p "$SCRIPT_DIR/log" "$SCRIPT_DIR/log_terminal"
    if [[ -z "$READY_DIR" ]]; then READY_DIR=$(mktemp -d "$SCRIPT_DIR/log/.ready.XXXXXX"); fi
    max_parallel=$((${#GPU_IDS[@]} * PROCESSES_PER_GPU))
    if ((max_parallel > ${#STAGE_BATCHES[@]})); then max_parallel=${#STAGE_BATCHES[@]}; fi
    CPU_THREADS=$(($(nproc) / max_parallel))
    if ((CPU_THREADS < 1)); then CPU_THREADS=1; fi
    for ((wave=0; wave<PROCESSES_PER_GPU && NEXT_BATCH<${#STAGE_BATCHES[@]}; wave++)); do
      wave_pids=()
      for gpu in "${GPU_IDS[@]}"; do
        ((NEXT_BATCH < ${#STAGE_BATCHES[@]})) || break
        launch_next "$gpu"; wave_pids+=("$LAST_PID")
      done
      wait_for_wave "${wave_pids[@]}"
    done
  else
    activate_env chatgarment
    for ((wave=0; wave<DETECT_JOBS && NEXT_BATCH<${#STAGE_BATCHES[@]}; wave++)); do launch_next cpu; done
  fi
  while ((${#PID_TO_BATCH[@]})); do finish_one; done
  ((FAILED == 0)) || die "$FAILED $STAGE batch process(es) failed; rerun the pipeline after fixing the error"
}

printf '[config] data_root=%s batches=%s rounds=0-4 GPUs=%s processes_per_GPU=%s\n' \
  "$DATA_ROOT" "${BATCH_IDS[*]}" "${GPU_IDS[*]}" "$PROCESSES_PER_GPU"

for ((CURRENT_ROUND=0; CURRENT_ROUND<=4; CURRENT_ROUND++)); do
  OUT_SUBDIR=NI-Tex_output
  if ((CURRENT_ROUND > 0)); then OUT_SUBDIR="NI-Tex_output_regen$CURRENT_ROUND"; fi
  REPORT_DIR=$(report_dir "$CURRENT_ROUND")
  PREVIOUS_REPORT_DIR=""
  STAGE_BATCHES=("${BATCH_IDS[@]}")
  if ((CURRENT_ROUND > 0)); then
    PREVIOUS_REPORT_DIR=$(report_dir "$((CURRENT_ROUND - 1))")
    STAGE_BATCHES=()
    for id in "${BATCH_IDS[@]}"; do
      report="$PREVIOUS_REPORT_DIR/$id/failed_cases.txt"
      [[ -f "$report" ]] || die "Missing previous-round report: $report"
      if [[ -s "$report" ]]; then STAGE_BATCHES+=("$id"); fi
    done
    if ((${#STAGE_BATCHES[@]} == 0)); then
      printf '[complete] No flagged parts remain after round %s.\n' "$((CURRENT_ROUND - 1))"
      break
    fi
  fi
  run_stage inference
  STAGE_BATCHES=("${BATCH_IDS[@]}")
  run_stage detect
  remaining=0
  for id in "${BATCH_IDS[@]}"; do
    count=$(wc -l < "$REPORT_DIR/$id/failed_cases.txt")
    remaining=$((remaining + count))
  done
  printf '[round %s] flagged_parts=%s reports=%s\n' "$CURRENT_ROUND" "$remaining" "$REPORT_DIR"
  if ((remaining == 0)); then break; fi
done
activate_env chatgarment
printf '[gather] Collecting selected batches: %s\n' "${BATCH_IDS[*]}"
"$PYTHON_BIN" -u -B "$SCRIPT_DIR/gather_texture.py" \
  --data_root "$DATA_ROOT" --batch_list "${BATCH_IDS[*]}"
printf '[complete] Selected stages finished.\n'
