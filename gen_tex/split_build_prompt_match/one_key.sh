#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OP=""
BATCH_LIST=""
ROOT="/inspire/hdd/global_user/liming-253108120187/SartorAI/GCD-Verse/GCD-Verse"
OPENVTON_ROOT="/inspire/hdd/global_user/liming-253108120187/SartorAI/ReWeaver2/DataSet"
PARALLEL=12
DELETE_UNEXPECTED=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --op)
            OP="$2"
            shift 2
            ;;
        --batch_list)
            BATCH_LIST="$2"
            shift 2
            ;;
        --root)
            ROOT="$2"
            shift 2
            ;;
        --openvton_root)
            OPENVTON_ROOT="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --delete)
            DELETE_UNEXPECTED=true
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ "$OP" != "check_split" && "$OP" != "split" && "$OP" != "build_prompt" && "$OP" != "encode_prompt" && "$OP" != "match_garment_img" ]]; then
    echo "Only --op check_split, --op split, --op build_prompt, --op encode_prompt, and --op match_garment_img are supported"
    exit 1
fi

if [[ -z "$BATCH_LIST" ]]; then
    echo "--batch_list is required, e.g. --batch_list 0-12"
    exit 1
fi

if ! [[ "$PARALLEL" =~ ^[0-9]+$ ]]; then
    echo "--parallel must be a non-negative integer"
    exit 1
fi

if [[ "$DELETE_UNEXPECTED" == true && "$OP" != "check_split" ]]; then
    echo "--delete is only supported with --op check_split"
    exit 1
fi

if [[ "$OP" == "check_split" ]]; then
    if [[ "$PARALLEL" -eq 0 ]]; then
        echo "--parallel must be at least 1 for --op check_split"
        exit 1
    fi

    log_dir="log/check_split"
    mkdir -p "$log_dir"
    delete_args=()
    if [[ "$DELETE_UNEXPECTED" == true ]]; then
        delete_args+=(--delete)
    fi
    echo "Start check_split batches ${BATCH_LIST}, log: ${log_dir}"
    python check_split.py \
        --root "$ROOT" \
        --batch_list "$BATCH_LIST" \
        --log "$log_dir" \
        --workers "$PARALLEL" \
        "${delete_args[@]}"
    echo "All ${OP} jobs finished."
    exit 0
fi

if [[ "$OP" == "match_garment_img" ]]; then
    log_dir="log/match_garment_img"
    mkdir -p "$log_dir"
    log_path="${log_dir}/all_batches.log"
    echo "Start match_garment_img batches ${BATCH_LIST}, log: ${log_path}"
    python match_garment_img.py \
        --root "$ROOT" \
        --batch_list "$BATCH_LIST" \
        --openvton_root "$OPENVTON_ROOT" \
        --log "$log_path"
    echo "All ${OP} jobs finished."
    exit 0
fi

expand_batch_list() {
    local value="$1"
    local start end i item

    IFS=',' read -ra items <<< "$value"
    for item in "${items[@]}"; do
        if [[ "$item" == *-* ]]; then
            start="${item%-*}"
            end="${item#*-}"
            for ((i=start; i<=end; i++)); do
                echo "$i"
            done
        else
            echo "$item"
        fi
    done
}

run_one_batch() {
    local batch_id="$1"
    local batch_dir="${ROOT}/batch_${batch_id}"
    local log_dir log_path

    if [[ "$OP" == "split" ]]; then
        log_dir="log/split_garment_upper_bottom"
        mkdir -p "$log_dir"
        log_path="${log_dir}/batch_${batch_id}.log"
        echo "Start split batch_${batch_id}, log: ${log_path}"
        python split_garment_upper_bottom.py "$batch_dir" --log "$log_path" &
    elif [[ "$OP" == "build_prompt" ]]; then
        log_dir="log/build_prompts"
        mkdir -p "$log_dir"
        log_path="${log_dir}/batch_${batch_id}.log"
        echo "Start build_prompt batch_${batch_id}, log: ${log_path}"
        python build_prompts.py \
            --root "${batch_dir}/random_body" \
            --log "$log_path" &
    elif [[ "$OP" == "encode_prompt" ]]; then
        log_dir="log/encode_prompt"
        mkdir -p "$log_dir"
        log_path="${log_dir}/batch_${batch_id}.log"
        echo "Start encode_prompt batch_${batch_id}, log: ${log_path}"
        python encode_prompt.py \
            --root "$batch_dir" \
            --log "$log_path" &
    fi
}

for batch_id in $(expand_batch_list "$BATCH_LIST"); do
    while [[ "$PARALLEL" -gt 0 && "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]]; do
        wait -n
    done

    run_one_batch "$batch_id"
done

wait
echo "All ${OP} jobs finished."
