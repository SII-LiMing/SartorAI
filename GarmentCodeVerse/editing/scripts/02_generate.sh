#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/GarmentCodeRC/editing/scripts"
GCD_ROOT="/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/GarmentCodeRC"
INPUT_ROOT="$GCD_ROOT/garmentcode_144k"
TMP_ROOT="$GCD_ROOT/editing/tmp"
ROUND=${EDITING_ROUND:-1}
LOG_ROOT="$GCD_ROOT/editing/logs/02_generate_round_$ROUND"
MAX_PROCESSES=32
BATCH_LIST=all
SPLIT_PATH="$GCD_ROOT/editing/dataset_split/dataset_split_0_15_0_85.json"
SPLIT_NAME="dataset_0_15"

die() { echo "错误：$*" >&2; exit 2; }
[[ $ROUND =~ ^[1-9][0-9]*$ ]] || die "EDITING_ROUND 必须是正整数"
while (( $# )); do
    case "$1" in
        --batch_list) [[ -n ${2-} && ${2-} != --* ]] || die "--batch_list 需要参数"; BATCH_LIST=$2; shift 2 ;;
        -h|--help)
            echo "用法：$0 [--batch_list all|0,2,5|0-3]"
            exit 0
            ;;
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
(( ${#BATCHES[@]} > 0 )) || { echo "错误：未发现 batch" >&2; exit 2; }
mkdir -p "$TMP_ROOT" "$LOG_ROOT"
PROCESSES=$(( ${#BATCHES[@]} < MAX_PROCESSES ? ${#BATCHES[@]} : MAX_PROCESSES ))

generate_batch() {
    local batch=$1
    local analysis="$TMP_ROOT/${batch}_edit_analysis"
    local output="$TMP_ROOT/${batch}_edit_output"
    local workspace="$output/.state/simulation_workspace"
    local log="$LOG_ROOT/terminal_log_${batch}_${SPLIT_LABEL}.log"

    if {
        echo "开始生成：$batch"
        echo "数据划分：$SPLIT_NAME（$SPLIT_PATH）"
        if [[ -e "$workspace" || -L "$workspace" ]]; then
            echo "已有 workspace，跳过：$workspace"
        elif [[ ! -f "$analysis/pilot_eligibility_report.json" ]]; then
            echo "错误：缺少 plan：$analysis" >&2
            false
        else
            cd "$GCD_ROOT"
            if ! python -u -B -m editing.semantic_edit generate \
                --input-root "$INPUT_ROOT" \
                --analysis-root "$analysis" \
                --output-root "$output" \
                --seed 42 \
                --split-path "$SPLIT_PATH" \
                --split-name "$SPLIT_NAME"; then
                echo "generate 失败：$batch" >&2
                false
            elif [[ -d "$workspace" && ! -L "$workspace" ]]; then
                echo "候选已生成：$workspace"
            else
                echo "本轮没有生成新候选：$batch"
            fi
        fi
    } >"$log" 2>&1; then
        echo "完成：$batch（$log）"
    else
        echo "失败：$batch（$log）" >&2
        return 1
    fi
}

export -f generate_batch
export INPUT_ROOT TMP_ROOT LOG_ROOT GCD_ROOT SPLIT_PATH SPLIT_NAME SPLIT_LABEL

echo "共 ${#BATCHES[@]} 个 batch，使用 $PROCESSES 个进程"
printf '%s\0' "${BATCHES[@]}" |
    xargs -0 -n 1 -P "$PROCESSES" bash -c 'generate_batch "$1"' _
