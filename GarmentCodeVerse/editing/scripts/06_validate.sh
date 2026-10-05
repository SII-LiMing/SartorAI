#!/usr/bin/env bash
set -Eeuo pipefail

GCD_ROOT="/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/GarmentCodeRC"
INPUT_ROOT="$GCD_ROOT/garmentcode_144k"
TMP_ROOT="$GCD_ROOT/editing/tmp"
LOG_ROOT="$GCD_ROOT/editing/logs/06_validate"
MAX_PROCESSES=32
SPLIT_PATH="$GCD_ROOT/editing/dataset_split/dataset_split_0_15_0_85.json"
SPLIT_NAME="dataset_0_15"

die() { echo "错误：$*" >&2; exit 2; }
(( $# == 0 )) || die "该脚本不接收参数"
[[ -f $SPLIT_PATH ]] || die "split 文件不存在：$SPLIT_PATH"
SPLIT_LABEL=${SPLIT_NAME//[^a-zA-Z0-9_-]/_}

[[ ${CONDA_DEFAULT_ENV:-} == chatgarment ]] || {
    echo "错误：请先激活 chatgarment" >&2
    exit 2
}

mapfile -t BATCHES < <(
    find "$INPUT_ROOT" -mindepth 1 -maxdepth 1 -type d \
        -regextype posix-extended -regex '.*/batch_[0-9]+' \
        -printf '%f\n' | sort -V
)
(( ${#BATCHES[@]} > 0 )) || { echo "错误：未发现 batch" >&2; exit 2; }
mkdir -p "$LOG_ROOT"
PROCESSES=$(( ${#BATCHES[@]} < MAX_PROCESSES ? ${#BATCHES[@]} : MAX_PROCESSES ))

validate_batch() {
    local batch=$1
    local output="$TMP_ROOT/${batch}_edit_output"
    local workspace="$output/.state/simulation_workspace"
    local final_batch="$output/$batch"
    local log="$LOG_ROOT/terminal_log_${batch}_${SPLIT_LABEL}.log"

    if {
        echo "开始最终验证：$batch"
        echo "数据划分状态：$SPLIT_NAME"
        cd "$GCD_ROOT"
        python -u -B -m editing.semantic_edit status \
            --output-root "$output" \
            --split-path "$SPLIT_PATH" \
            --split-name "$SPLIT_NAME"
        if [[ -e "$workspace" || -L "$workspace" ]]; then
            echo "错误：仍有 active workspace：$workspace" >&2
            false
        elif [[ ! -f "$final_batch/dataset_properties.yaml" ]]; then
            echo "错误：最终 batch 不完整：$final_batch" >&2
            false
        else
            cd "$GCD_ROOT"
            # 模拟候选已在步骤 3 完成物理过滤；这里只验证最终产物完整性。
            python -u -B check_integrity.py \
                --gcd_dir "$final_batch" \
                --body_type random_body
        fi
    } >"$log" 2>&1; then
        echo "完成：$batch（$log）"
    else
        echo "失败：$batch（$log）" >&2
        return 1
    fi
}

export -f validate_batch
export GCD_ROOT TMP_ROOT LOG_ROOT SPLIT_PATH SPLIT_NAME SPLIT_LABEL

echo "共 ${#BATCHES[@]} 个 batch，使用 $PROCESSES 个进程"
printf '%s\0' "${BATCHES[@]}" |
    xargs -0 -n 1 -P "$PROCESSES" bash -c 'validate_batch "$1"' _
