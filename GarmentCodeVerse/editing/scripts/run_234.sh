#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/GarmentCodeRC/editing/scripts"
GCD_ROOT="/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/GarmentCodeRC"
INPUT_ROOT="$GCD_ROOT/garmentcode_144k"
TMP_ROOT="$GCD_ROOT/editing/tmp"
BATCH_LIST=all
MAX_ROUNDS=3
ROUND_SPEC=
LOG_ROOT="$GCD_ROOT/editing/logs/run_234"
SPLIT_PATH="$GCD_ROOT/editing/dataset_split/dataset_split_0_15_0_85.json"
SPLIT_NAME="dataset_0_15"

die() { echo "错误：$*" >&2; exit 2; }
while (( $# )); do
    case "$1" in
        --batch_list)
            [[ -n ${2-} && ${2-} != --* ]] || die "--batch_list 需要参数"
            BATCH_LIST=$2
            shift 2
            ;;
        --round)
            [[ -n ${2-} && ${2-} != --* ]] || die "--round 需要参数"
            ROUND_SPEC=$2
            shift 2
            ;;
        -h|--help)
            echo "用法：$0 --round 1|2|3|2,3|1,2,3 [--batch_list all|0,2,5|0-3]"
            exit 0
            ;;
        *) die "未知参数：$1" ;;
    esac
done

[[ $ROUND_SPEC =~ ^[1-3](,[1-3])*$ ]] || \
    die "必须指定 --round，可传 1、2、3 或严格递增的逗号列表（如 2,3）"
IFS=, read -r -a ROUNDS <<< "$ROUND_SPEC"
previous_round=0
for round in "${ROUNDS[@]}"; do
    (( round > previous_round )) || die "--round 中的轮次必须严格递增且不能重复"
    previous_round=$round
done
[[ -f $SPLIT_PATH ]] || die "split 文件不存在：$SPLIT_PATH"

LOG_LABEL=${BATCH_LIST//[^a-zA-Z0-9_-]/_}
SPLIT_LOG_LABEL=${SPLIT_NAME//[^a-zA-Z0-9_-]/_}
mkdir -p "$LOG_ROOT"
[[ ${CONDA_DEFAULT_ENV:-} == chatgarment ]] || die "请先激活 chatgarment"

declare -a SELECTED=()
declare -A SEEN=()
add_batch() {
    local id=$1 batch
    [[ $id =~ ^[0-9]+$ ]] || die "非法 batch：$id"
    batch="batch_$id"
    [[ -d "$INPUT_ROOT/$batch" ]] || die "batch 不存在：$batch"
    [[ -n ${SEEN[$batch]+x} ]] || { SELECTED+=("$batch"); SEEN[$batch]=1; }
}
raw=${BATCH_LIST,,}
if [[ $raw == all || $raw == \* ]]; then
    mapfile -t SELECTED < <(find "$INPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -regextype posix-extended -regex '.*/batch_[0-9]+' -printf '%f\n' | sort -V)
else
    raw=${raw//,/ }
    for token in $raw; do
        token=${token#batch_}
        if [[ $token =~ ^([0-9]+)-([0-9]+)$ ]]; then
            start=${BASH_REMATCH[1]}
            end=${BASH_REMATCH[2]}
            (( start <= end )) || die "范围不能倒序：$token"
            for ((id=start; id<=end; id++)); do add_batch "$id"; done
        else
            add_batch "$token"
        fi
    done
fi
(( ${#SELECTED[@]} > 0 )) || die "没有选中 batch"

run_round() {
    local round=$1 batch workspace active_list
    local -a active=()

    echo "数据划分：$SPLIT_NAME（$SPLIT_PATH）"
    echo "========== 物理轮次 $round/$MAX_ROUNDS =========="
    EDITING_ROUND=$round \
        bash "$SCRIPT_DIR/02_generate.sh" --batch_list "$BATCH_LIST" || return

    for batch in "${SELECTED[@]}"; do
        workspace="$TMP_ROOT/${batch}_edit_output/.state/simulation_workspace"
        [[ -d "$workspace" && ! -L "$workspace" ]] && active+=("$batch")
    done
    if (( ${#active[@]} == 0 )); then
        echo "本轮没有新的待仿真候选。"
        echo "处理完成：$BATCH_LIST，round=$round"
        return 0
    fi

    active_list=$(IFS=,; echo "${active[*]}")
    EDITING_ROUND=$round \
        bash "$SCRIPT_DIR/03_simulate_filter.sh" --batch_list "$active_list" || return
    EDITING_ROUND=$round \
        bash "$SCRIPT_DIR/04_project.sh" --batch_list "$active_list" || return
    echo "处理完成：$BATCH_LIST，round=$round"
}

for round in "${ROUNDS[@]}"; do
    round_log="$LOG_ROOT/terminal_log_${LOG_LABEL}_${SPLIT_LOG_LABEL}_round_${round}.log"
    if ! run_round "$round" 2>&1 | tee "$round_log"; then
        die "round=$round 执行失败；不会继续后续轮次，请检查 $round_log"
    fi
done

echo "所有指定轮次处理完成：$BATCH_LIST，rounds=$ROUND_SPEC"
