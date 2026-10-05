#!/usr/bin/env bash
set -Eeuo pipefail

EDITING_ROOT="/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/GarmentCodeRC/editing"
GCD_ROOT="/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/GarmentCodeRC"
PROJECT_ROOT="/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2"
INPUT_ROOT="$GCD_ROOT/garmentcode_144k"
CONFIG_PATH="$EDITING_ROOT/configs/semantic_edit.yaml"
PROMPT_BUILDER="$PROJECT_ROOT/gen_garment_texture/build_prompt_split_match/build_prompts.py"
LOG_ROOT="$EDITING_ROOT/logs/01_plan"
MAX_PROCESSES=32

mapfile -t BATCHES < <(
    find "$INPUT_ROOT" -mindepth 1 -maxdepth 1 -type d \
        -regextype posix-extended -regex '.*/batch_[0-9]+' \
        -printf '%f\n' | sort -V
)
(( ${#BATCHES[@]} > 0 )) || { echo "错误：未发现 batch" >&2; exit 2; }
mkdir -p "$EDITING_ROOT/tmp" "$LOG_ROOT"
PROCESSES=$(( ${#BATCHES[@]} < MAX_PROCESSES ? ${#BATCHES[@]} : MAX_PROCESSES ))

plan_batch() {
    local batch=$1
    local analysis="$EDITING_ROOT/tmp/${batch}_edit_analysis"
    local report="$analysis/pilot_eligibility_report.json"
    local log="$LOG_ROOT/terminal_log_${batch}.log"

    if {
        echo "开始规划：$batch"
        if [[ -f "$report" ]]; then
            echo "plan 已存在，跳过：$report"
        elif [[ -e "$analysis" ]]; then
            echo "错误：analysis 已存在但没有有效 plan：$analysis" >&2
            false
        else
            cd "$GCD_ROOT"
            python -B -m editing.semantic_edit plan \
                --input-root "$INPUT_ROOT" \
                --batch-id "$batch" \
                --analysis-root "$analysis" \
                --config "$CONFIG_PATH" \
                --prompt-builder "$PROMPT_BUILDER" \
                --seed 42
        fi
    } >"$log" 2>&1; then
        echo "完成：$batch（$log）"
    else
        echo "失败：$batch（$log）" >&2
        return 1
    fi
}

export -f plan_batch
export INPUT_ROOT EDITING_ROOT GCD_ROOT CONFIG_PATH PROMPT_BUILDER LOG_ROOT

echo "共 ${#BATCHES[@]} 个 batch，使用 $PROCESSES 个进程"
printf '%s\0' "${BATCHES[@]}" |
    xargs -0 -n 1 -P "$PROCESSES" bash -c 'plan_batch "$1"' _
