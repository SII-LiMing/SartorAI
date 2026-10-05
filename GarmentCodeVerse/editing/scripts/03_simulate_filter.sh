#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/GarmentCodeRC/editing/scripts"
GCD_ROOT="/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/GarmentCodeRC"
INPUT_ROOT="$GCD_ROOT/garmentcode_144k"
TMP_ROOT="$GCD_ROOT/editing/tmp"
ROUND=${EDITING_ROUND:-1}
LOG_ROOT="$GCD_ROOT/editing/logs/03_simulate_filter_round_$ROUND"
MAX_PROCESSES=6
BATCH_LIST=all
SPLIT_PATH="$GCD_ROOT/editing/dataset_split/dataset_split_0_15_0_85.json"
SPLIT_NAME="dataset_0_15"

die() { echo "错误：$*" >&2; exit 2; }
[[ $ROUND =~ ^[1-9][0-9]*$ ]] || die "EDITING_ROUND 必须是正整数"
while (( $# )); do
    case "$1" in
        --batch_list) [[ -n ${2-} && ${2-} != --* ]] || die "--batch_list 需要参数"; BATCH_LIST=$2; shift 2 ;;
        -h|--help) echo "用法：$0 [--batch_list all|0,2,5|0-3]"; exit 0 ;;
        *) die "未知参数：$1" ;;
    esac
done

[[ -f $SPLIT_PATH ]] || die "split 文件不存在：$SPLIT_PATH"
SPLIT_LABEL=${SPLIT_NAME//[^a-zA-Z0-9_-]/_}

declare -a BATCHES=()
declare -A SEEN=()
add_batch() {
    local id=$1 batch
    [[ $id =~ ^[0-9]+$ ]] || die "非法 batch：$id"
    batch="batch_$id"
    [[ -d "$INPUT_ROOT/$batch" ]] || die "batch 不存在：$batch"
    [[ -n ${SEEN[$batch]+x} ]] || { BATCHES+=("$batch"); SEEN[$batch]=1; }
}
parse_batches() {
    local raw=${BATCH_LIST,,} token start end id
    if [[ $raw == all || $raw == \* ]]; then
        mapfile -t BATCHES < <(find "$INPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -regextype posix-extended -regex '.*/batch_[0-9]+' -printf '%f\n' | sort -V)
        return
    fi
    raw=${raw//,/ }
    for token in $raw; do
        token=${token#batch_}
        if [[ $token =~ ^([0-9]+)-([0-9]+)$ ]]; then
            start=${BASH_REMATCH[1]}; end=${BASH_REMATCH[2]}
            (( start <= end )) || die "范围不能倒序：$token"
            for ((id=start; id<=end; id++)); do add_batch "$id"; done
        else
            add_batch "$token"
        fi
    done
}

[[ ${CONDA_DEFAULT_ENV:-} == chatgarment ]] || die "请先激活 chatgarment"
parse_batches
(( ${#BATCHES[@]} > 0 )) || die "没有选中 batch"
mkdir -p "$LOG_ROOT"
PROCESSES=$(( ${#BATCHES[@]} < MAX_PROCESSES ? ${#BATCHES[@]} : MAX_PROCESSES ))

run_batch() {
    local batch=$1
    local workspace="$TMP_ROOT/${batch}_edit_output/.state/simulation_workspace"
    [[ -d "$workspace" && ! -L "$workspace" ]] || { echo "缺少 workspace：$workspace" >&2; return 2; }
    cd "$GCD_ROOT"
    python -u -B pattern_data_sim.py --data "$workspace" --config beautiful_sim_props.yaml
    python -u -B filter_fail_sim.py --gcd_dir "$workspace" --body_type random_body --no-delete
    python -u -B "$SCRIPT_DIR/filter_report_compat.py" "$workspace"
}

simulate_batch() {
    local batch=$1
    local log="$LOG_ROOT/terminal_log_${batch}_${SPLIT_LABEL}.log"
    echo "开始模拟和过滤：$batch"
    echo "数据划分：$SPLIT_NAME（workspace 已由 generate 筛选）"
    if run_batch "$batch" 2>&1 | tee "$log"; then
        echo "完成：$batch"
    else
        echo "失败：$batch（$log）" >&2
        return 1
    fi
}

export -f run_batch simulate_batch
export SCRIPT_DIR GCD_ROOT TMP_ROOT LOG_ROOT SPLIT_LABEL SPLIT_NAME

echo "共 ${#BATCHES[@]} 个 batch，并行进程数：$PROCESSES"
if printf '%s\0' "${BATCHES[@]}" |
    xargs -0 -n 1 -P "$PROCESSES" bash -c \
        'set -Eeuo pipefail; simulate_batch "$1"' _; then
    exit 0
fi
exit 1
