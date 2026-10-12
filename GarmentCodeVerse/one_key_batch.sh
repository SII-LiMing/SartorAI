#!/usr/bin/env bash
set -euo pipefail

name=garmentcode_144k
body_type=random_body
sample_body_type=both
size=140000
batch_size=1000
parallel_jobs=6
max_hours=0
seed=42
batch_list=all
op=""
script_start_ts=$(date +%s)

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"

usage() {
    cat <<'EOF'
Usage:
  bash one_key_batch.sh [--name NAME] [--body_type random_body|default_body|both] [--size SIZE] [--batch_size BATCH_SIZE] [--parallel_jobs JOBS] [--seed SEED] [--batch_list LIST] [--op sample|simulate|filter|check] [--max_hours HOURS]

Defaults are defined in the script. Command-line values override them.
Sampling defaults to both body types. --body_type selects which bodies to sample.
Later stages default to random_body; --body_type both also keeps that default.
If --op is omitted, the full pipeline runs: sample -> simulate -> filter -> check.
Only one --op is supported.
Batch stages run in parallel across batches, limited by --parallel_jobs.
--max_hours limits --op simulate only (0 = unlimited). At the shared deadline,
workers finish their current sample, save progress, and exit with status 75.
--batch_list defaults to all. Examples: --batch_list "0,2,5", --batch_list "batch_0,batch_2", --batch_list "0-3".
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 2
}

require_value() {
    local flag=$1
    local value=${2-}
    if [[ -z "$value" || "$value" == --* ]]; then
        die "${flag} requires a value"
    fi
}

op_seen=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --name)
            require_value "$1" "${2-}"
            name=$2
            shift 2
            ;;
        --body_type)
            require_value "$1" "${2-}"
            sample_body_type=$2
            if [[ "$sample_body_type" == "both" ]]; then
                body_type=random_body
            else
                body_type=$sample_body_type
            fi
            shift 2
            ;;
        --size)
            require_value "$1" "${2-}"
            size=$2
            shift 2
            ;;
        --batch_size)
            require_value "$1" "${2-}"
            batch_size=$2
            shift 2
            ;;
        --parallel_jobs)
            require_value "$1" "${2-}"
            parallel_jobs=$2
            shift 2
            ;;
        --seed)
            require_value "$1" "${2-}"
            seed=$2
            shift 2
            ;;
        --max_hours)
            require_value "$1" "${2-}"
            max_hours=$2
            shift 2
            ;;
        --batch_list)
            require_value "$1" "${2-}"
            batch_list=$2
            shift 2
            ;;
        --op)
            require_value "$1" "${2-}"
            if [[ $op_seen -ne 0 ]]; then
                die "multiple --op values are not supported"
            fi
            op=$2
            op_seen=1
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ -n "$name" ]] || die "--name must not be empty"
[[ "$name" != /* ]] || die "--name must be a relative dataset directory name"
[[ "$name" != *".."* ]] || die "--name must not contain '..'"

case "$sample_body_type" in
    random_body|default_body|both) ;;
    *) die "--body_type must be random_body, default_body or both" ;;
esac

[[ "$size" =~ ^[1-9][0-9]*$ ]] || die "--size must be a positive integer"
[[ "$batch_size" =~ ^[1-9][0-9]*$ ]] || die "--batch_size must be a positive integer"
[[ "$parallel_jobs" =~ ^[1-9][0-9]*$ ]] || die "--parallel_jobs must be a positive integer"
[[ "$max_hours" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "--max_hours must be a nonnegative number"
[[ -n "$seed" ]] || die "--seed must not be empty"
[[ -n "$batch_list" ]] || die "--batch_list must not be empty"

if [[ -n "$op" ]]; then
    case "$op" in
        sample|simulate|filter|check) ;;
        *) die "--op must be one of: sample, simulate, filter, check" ;;
    esac
    selected_ops=("$op")
else
    selected_ops=(sample simulate filter check)
fi

simulation_deadline=$(awk -v hours="$max_hours" -v start="$script_start_ts" 'BEGIN {printf "%.0f", (hours > 0 ? start + hours * 3600 : 0)}')
if [[ "$simulation_deadline" != 0 && "$op" != simulate ]]; then
    die "--max_hours requires --op simulate"
fi

operations_include_sample=0
downstream_ops=()
for selected_op in "${selected_ops[@]}"; do
    if [[ "$selected_op" == "sample" ]]; then
        operations_include_sample=1
    else
        downstream_ops+=("$selected_op")
    fi
done

join_by() {
    local delimiter=$1
    shift
    local IFS=$delimiter
    echo "$*"
}

operations_label=$(join_by " " "${selected_ops[@]}")
num_batches=$(((size + batch_size - 1) / batch_size))
root_dir="$script_dir/$name"

declare -a selected_batch_indices=()
declare -A selected_batch_seen=()

add_selected_batch_index() {
    local batch_index=$1
    [[ "$batch_index" =~ ^[0-9]+$ ]] || die "invalid batch id in --batch_list: ${batch_index}"
    if (( batch_index >= num_batches )); then
        die "batch id out of range in --batch_list: ${batch_index}; valid range is 0-$((num_batches - 1))"
    fi
    if [[ -z ${selected_batch_seen[$batch_index]+x} ]]; then
        selected_batch_indices+=("$batch_index")
        selected_batch_seen[$batch_index]=1
    fi
}

batch_list_selects_all=0
parse_batch_list() {
    local raw=$1
    local normalized token start_idx end_idx batch_index
    normalized=${raw,,}
    normalized=${normalized//,/ }

    selected_batch_indices=()
    selected_batch_seen=()

    if [[ "$normalized" == "all" || "$normalized" == "*" ]]; then
        batch_list_selects_all=1
        for ((batch_index = 0; batch_index < num_batches; batch_index++)); do
            add_selected_batch_index "$batch_index"
        done
        return 0
    fi

    batch_list_selects_all=0
    for token in $normalized; do
        token=${token//[[:space:]]/}
        token=${token//batch_/}
        [[ -n "$token" ]] || continue

        if [[ "$token" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            start_idx=${BASH_REMATCH[1]}
            end_idx=${BASH_REMATCH[2]}
            if (( start_idx > end_idx )); then
                die "invalid descending range in --batch_list: ${token}"
            fi
            for ((batch_index = start_idx; batch_index <= end_idx; batch_index++)); do
                add_selected_batch_index "$batch_index"
            done
        else
            add_selected_batch_index "$token"
        fi
    done

    if (( ${#selected_batch_indices[@]} == 0 )); then
        die "--batch_list did not select any batch"
    fi
}

parse_batch_list "$batch_list"
selected_batch_count=${#selected_batch_indices[@]}
selected_batches_label=$(join_by "," "${selected_batch_indices[@]}")
selected_batch_min=${selected_batch_indices[0]}
selected_batch_max=${selected_batch_indices[0]}
for batch_index in "${selected_batch_indices[@]}"; do
    if (( batch_index < selected_batch_min )); then
        selected_batch_min=$batch_index
    fi
    if (( batch_index > selected_batch_max )); then
        selected_batch_max=$batch_index
    fi
done

if [[ $operations_include_sample -eq 1 ]]; then
    if [[ -e "$root_dir" && $batch_list_selects_all -eq 1 ]]; then
        die "output directory already exists: $root_dir"
    fi
    mkdir -p "$root_dir/logs"
else
    if [[ ! -d "$root_dir" ]]; then
        die "dataset directory does not exist for --op ${op}: $root_dir"
    fi
    mkdir -p "$root_dir/logs"
fi

if (( parallel_jobs > selected_batch_count )); then
    effective_parallel_jobs=$selected_batch_count
else
    effective_parallel_jobs=$parallel_jobs
fi

run_id=$(date +"%Y%m%d_%H%M%S")
outer_logs_dir="$root_dir/logs"
outer_log_prefix="batch_${selected_batch_min}_${selected_batch_max}"
outer_terminal_log="$outer_logs_dir/${outer_log_prefix}_terminal.log"
outer_timing_log="$outer_logs_dir/${outer_log_prefix}_timing.log"
outer_summary_log="$outer_logs_dir/${outer_log_prefix}_summary.log"
safe_name=${name//\//_}
worker_tmp_dir=$(mktemp -d "/tmp/${safe_name}_${run_id}_workers.XXXXXX")

cleanup() {
    rm -rf "$worker_tmp_dir"
}
trap cleanup EXIT

init_outer_log_file() {
    local log_path=$1
    touch "$log_path"
}

init_outer_log_file "$outer_terminal_log"
init_outer_log_file "$outer_timing_log"
init_outer_log_file "$outer_summary_log"

timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

format_seconds() {
    local total_seconds=$1
    printf "%02d:%02d:%02d" \
        $((total_seconds / 3600)) \
        $(((total_seconds % 3600) / 60)) \
        $((total_seconds % 60))
}

log_outer() {
    local message=$1
    local line="[$(timestamp)] $message"
    echo "$line" | tee -a "$outer_terminal_log"
    echo "$line" >> "$outer_timing_log"
    echo "$line" >> "$outer_summary_log"
}

current_batch_size() {
    local batch_index=$1
    local remaining=$((size - batch_index * batch_size))
    if (( remaining < batch_size )); then
        echo "$remaining"
    else
        echo "$batch_size"
    fi
}

batch_seed_for_index() {
    local batch_index=$1
    printf '%s::batch_%s\n' "$seed" "$batch_index"
}

check_sample_preconditions() {
    local sample_prefix=$1
    local sample_target=$2

    if [[ -e "$sample_target" ]]; then
        echo "Target batch directory already exists: $sample_target" >&2
        return 1
    fi

    shopt -s nullglob
    local matches=( "${sample_prefix}"_* )
    shopt -u nullglob

    if (( ${#matches[@]} > 0 )); then
        echo "Sample output candidates already exist for prefix: ${sample_prefix}_" >&2
        printf '  %s\n' "${matches[@]}" >&2
        return 1
    fi

    return 0
}

rename_sample_output() {
    local sample_prefix=$1
    local sample_target=$2

    if [[ -e "$sample_target" ]]; then
        echo "Target batch directory already exists: $sample_target" >&2
        return 1
    fi

    shopt -s nullglob
    local matches=( "${sample_prefix}"_* )
    shopt -u nullglob

    if (( ${#matches[@]} != 1 )); then
        echo "Expected exactly one generated directory for prefix ${sample_prefix}_, found ${#matches[@]}" >&2
        if (( ${#matches[@]} > 0 )); then
            printf '  %s\n' "${matches[@]}" >&2
        fi
        return 1
    fi

    if [[ ! -d "${matches[0]}" ]]; then
        echo "Generated path is not a directory: ${matches[0]}" >&2
        return 1
    fi

    mv -- "${matches[0]}" "$sample_target"
}

check_existing_batch_preconditions() {
    local step=$1
    local batch_abs_dir=$2
    local body_root="$batch_abs_dir/$body_type"

    if [[ ! -d "$batch_abs_dir" ]]; then
        echo "Batch directory does not exist: $batch_abs_dir" >&2
        return 1
    fi

    case "$step" in
        simulate)
            if [[ ! -f "$batch_abs_dir/dataset_properties.yaml" ]]; then
                echo "Missing dataset_properties.yaml: $batch_abs_dir/dataset_properties.yaml" >&2
                return 1
            fi
            if [[ ! -d "$body_root/data" ]]; then
                echo "Missing sampled data directory: $body_root/data" >&2
                return 1
            fi
            ;;
        filter|check)
            if [[ ! -d "$body_root" ]]; then
                echo "Missing body directory: $body_root" >&2
                return 1
            fi
            if [[ ! -f "$body_root/dataset_properties_${body_type}.yaml" ]]; then
                echo "Missing simulation properties file: $body_root/dataset_properties_${body_type}.yaml" >&2
                return 1
            fi
            if [[ "$step" == "check" && ! -f "$body_root/filter_fail_report.yaml" ]]; then
                echo "Missing filter report for check: $body_root/filter_fail_report.yaml" >&2
                return 1
            fi
            ;;
        *)
            echo "Unknown precondition step: $step" >&2
            return 1
            ;;
    esac

    return 0
}

run_pattern_data_sim() {
    local batch_rel=$1
    if [[ "$body_type" == "default_body" ]]; then
        python pattern_data_sim.py --data "$batch_rel" --default_body --deadline "$simulation_deadline"
    else
        python pattern_data_sim.py --data "$batch_rel" --deadline "$simulation_deadline"
    fi
}

run_filter_fail_sim() {
    local batch_abs_dir=$1
    python filter_fail_sim.py --gcd_dir "$batch_abs_dir" --body_type "$body_type"
}

run_check_integrity() {
    local batch_abs_dir=$1
    local expected_count=$2
    python check_integrity.py --gcd_dir "$batch_abs_dir" --body_type "$body_type" --expected_count "$expected_count" --require_filter_report
}

sample_worker_log() {
    local terminal_log=$1
    local timing_log=$2
    local batch_name=$3
    local message=$4
    local line="[$(timestamp)] [${batch_name}] $message"
    echo "$line" >> "$terminal_log"
    echo "$line" >> "$timing_log"
}

sample_worker_run_step() {
    local terminal_log=$1
    local timing_log=$2
    local batch_name=$3
    local step_name=$4
    shift 4
    local start_ts end_ts elapsed status

    sample_worker_log "$terminal_log" "$timing_log" "$batch_name" "START ${step_name}"
    start_ts=$(date +%s)
    set +e
    "$@" >> "$terminal_log" 2>&1
    status=$?
    set -e
    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))

    if [[ $status -eq 0 ]]; then
        sample_worker_log "$terminal_log" "$timing_log" "$batch_name" "END ${step_name}: elapsed=$(format_seconds "$elapsed") (${elapsed}s)"
    else
        sample_worker_log "$terminal_log" "$timing_log" "$batch_name" "FAILED ${step_name}: exit_code=${status} elapsed=$(format_seconds "$elapsed") (${elapsed}s)"
    fi

    return "$status"
}

write_sample_status() {
    local status_file=$1
    local batch_name=$2
    local status=$3
    local failed_step=$4
    local global_seed=$5
    local batch_seed=$6
    local terminal_log=$7
    local timing_log=$8

    {
        echo "batch_name=${batch_name}"
        echo "status=${status}"
        echo "failed_step=${failed_step}"
        echo "global_seed=${global_seed}"
        echo "batch_seed=${batch_seed}"
        echo "terminal_log=${terminal_log}"
        echo "timing_log=${timing_log}"
    } > "$status_file"
}

run_sample_worker() {
    local batch_index=$1
    local batch_data_size=$2
    local batch_name="batch_${batch_index}"
    local batch_rel="${name}/${batch_name}"
    local batch_abs_dir="$root_dir/${batch_name}"
    local sample_prefix="$batch_rel"
    local sample_target="$batch_rel"
    local terminal_log="$worker_tmp_dir/${batch_name}_sample_terminal.log"
    local timing_log="$worker_tmp_dir/${batch_name}_sample_timing.log"
    local status_file="$worker_tmp_dir/${batch_name}_sample.status"
    local final_terminal_log="$batch_abs_dir/logs/${run_id}_terminal.log"
    local final_timing_log="$batch_abs_dir/logs/${run_id}_timing.log"
    local batch_start_ts batch_end_ts batch_elapsed
    local batch_status=0
    local failed_step=none
    local batch_seed

    batch_seed=$(batch_seed_for_index "$batch_index")

    : > "$terminal_log"
    : > "$timing_log"

    batch_start_ts=$(date +%s)
    sample_worker_log "$terminal_log" "$timing_log" "$batch_name" "Sample batch started: size=${batch_data_size} body_type=${sample_body_type} global_seed=${seed} batch_seed=${batch_seed}"

    if ! sample_worker_run_step "$terminal_log" "$timing_log" "$batch_name" "precheck sample" check_sample_preconditions "$sample_prefix" "$sample_target"; then
        batch_status=1
        failed_step=precheck_sample
    elif ! sample_worker_run_step "$terminal_log" "$timing_log" "$batch_name" "pattern_sampler" python pattern_sampler.py --name "$sample_prefix" --size "$batch_data_size" --seed "$batch_seed" --body_type "$sample_body_type"; then
        batch_status=1
        failed_step=pattern_sampler
    elif ! sample_worker_run_step "$terminal_log" "$timing_log" "$batch_name" "rename dataset directory" rename_sample_output "$sample_prefix" "$sample_target"; then
        batch_status=1
        failed_step=rename_dataset_directory
    else
        mkdir -p "$batch_abs_dir/logs"
        sample_worker_log "$terminal_log" "$timing_log" "$batch_name" "Logs initialized: terminal_log=${final_terminal_log} timing_log=${final_timing_log}"
    fi

    batch_end_ts=$(date +%s)
    batch_elapsed=$((batch_end_ts - batch_start_ts))

    if [[ $batch_status -eq 0 ]]; then
        sample_worker_log "$terminal_log" "$timing_log" "$batch_name" "Sample batch finished successfully: elapsed=$(format_seconds "$batch_elapsed") (${batch_elapsed}s)"
        cp "$terminal_log" "$final_terminal_log"
        cp "$timing_log" "$final_timing_log"
    else
        sample_worker_log "$terminal_log" "$timing_log" "$batch_name" "Sample batch failed at ${failed_step}: elapsed=$(format_seconds "$batch_elapsed") (${batch_elapsed}s)"
    fi

    write_sample_status "$status_file" "$batch_name" "$batch_status" "$failed_step" "$seed" "$batch_seed" "$terminal_log" "$timing_log"
    return "$batch_status"
}

append_sample_logs_to_outer() {
    local batch_name=$1
    local terminal_log timing_log

    terminal_log=$(read_sample_status_field "$batch_name" terminal_log)
    timing_log=$(read_sample_status_field "$batch_name" timing_log)

    if [[ -f "$terminal_log" ]]; then
        {
            echo ""
            echo "===== BEGIN ${batch_name} sample terminal log ====="
            cat "$terminal_log"
            echo "===== END ${batch_name} sample terminal log ====="
        } >> "$outer_terminal_log"
    fi

    if [[ -f "$timing_log" ]]; then
        {
            echo ""
            echo "===== BEGIN ${batch_name} sample timing log ====="
            cat "$timing_log"
            echo "===== END ${batch_name} sample timing log ====="
        } >> "$outer_timing_log"
        {
            echo ""
            echo "===== BEGIN ${batch_name} sample summary log ====="
            cat "$timing_log"
            echo "===== END ${batch_name} sample summary log ====="
        } >> "$outer_summary_log"
    fi
}

read_sample_failed_step() {
    local batch_name=$1
    local failed_step
    failed_step=$(read_sample_status_field "$batch_name" failed_step)
    if [[ -n "$failed_step" ]]; then
        echo "$failed_step"
    else
        echo "missing_status_file"
    fi
}

read_sample_status_field() {
    local batch_name=$1
    local field_name=$2
    local status_file="$worker_tmp_dir/${batch_name}_sample.status"
    if [[ -f "$status_file" ]]; then
        awk -F= -v field="$field_name" '$1 == field {print substr($0, length(field) + 2); exit}' "$status_file"
    fi
}

downstream_worker_log() {
    local terminal_log=$1
    local timing_log=$2
    local batch_name=$3
    local message=$4
    local line="[$(timestamp)] [${batch_name}] $message"
    echo "$line" >> "$terminal_log"
    echo "$line" >> "$timing_log"
}

downstream_worker_run_step() {
    local terminal_log=$1
    local timing_log=$2
    local batch_name=$3
    local step_name=$4
    shift 4
    local start_ts end_ts elapsed status

    downstream_worker_log "$terminal_log" "$timing_log" "$batch_name" "START ${step_name}"
    start_ts=$(date +%s)
    set +e
    "$@" >> "$terminal_log" 2>&1
    status=$?
    set -e
    end_ts=$(date +%s)
    elapsed=$((end_ts - start_ts))

    if [[ $status -eq 0 ]]; then
        downstream_worker_log "$terminal_log" "$timing_log" "$batch_name" "END ${step_name}: elapsed=$(format_seconds "$elapsed") (${elapsed}s)"
    elif [[ $status -eq 75 ]]; then
        downstream_worker_log "$terminal_log" "$timing_log" "$batch_name" "PAUSED ${step_name}: elapsed=$(format_seconds "$elapsed") (${elapsed}s)"
    else
        downstream_worker_log "$terminal_log" "$timing_log" "$batch_name" "FAILED ${step_name}: exit_code=${status} elapsed=$(format_seconds "$elapsed") (${elapsed}s)"
    fi

    return "$status"
}

write_downstream_status() {
    local status_file=$1
    local batch_name=$2
    local status=$3
    local failed_step=$4
    local terminal_log=$5
    local timing_log=$6

    {
        echo "batch_name=${batch_name}"
        echo "status=${status}"
        echo "failed_step=${failed_step}"
        echo "terminal_log=${terminal_log}"
        echo "timing_log=${timing_log}"
    } > "$status_file"
}

run_downstream_batch() {
    local batch_index=$1
    local batch_data_size=$2
    local batch_name="batch_${batch_index}"
    local batch_rel="${name}/${batch_name}"
    local batch_abs_dir="$root_dir/${batch_name}"
    local status_file="$worker_tmp_dir/${batch_name}_downstream.status"
    local terminal_log="$worker_tmp_dir/${batch_name}_downstream_terminal.log"
    local timing_log="$worker_tmp_dir/${batch_name}_downstream_timing.log"
    local final_terminal_log="$batch_abs_dir/logs/${run_id}_terminal.log"
    local final_timing_log="$batch_abs_dir/logs/${run_id}_timing.log"
    local batch_start_ts batch_end_ts batch_elapsed
    local batch_status=0
    local failed_step=none

    : > "$terminal_log"
    : > "$timing_log"

    batch_start_ts=$(date +%s)
    downstream_worker_log "$terminal_log" "$timing_log" "$batch_name" "Downstream batch started: size=${batch_data_size} operations=$(join_by " " "${downstream_ops[@]}")"

    for selected_op in "${downstream_ops[@]}"; do
        case "$selected_op" in
            simulate)
                if ! downstream_worker_run_step "$terminal_log" "$timing_log" "$batch_name" "precheck simulate" check_existing_batch_preconditions simulate "$batch_abs_dir"; then
                    batch_status=1
                    failed_step=precheck_simulate
                    break
                fi
                if downstream_worker_run_step "$terminal_log" "$timing_log" "$batch_name" "pattern_data_sim" run_pattern_data_sim "$batch_rel"; then
                    :
                else
                    batch_status=$?
                    if [[ $batch_status -eq 75 ]]; then
                        failed_step=paused
                    else
                        batch_status=1
                        failed_step=simulate
                    fi
                    break
                fi
                ;;
            filter)
                if ! downstream_worker_run_step "$terminal_log" "$timing_log" "$batch_name" "precheck filter" check_existing_batch_preconditions filter "$batch_abs_dir"; then
                    batch_status=1
                    failed_step=precheck_filter
                    break
                fi
                if ! downstream_worker_run_step "$terminal_log" "$timing_log" "$batch_name" "filter_fail_sim" run_filter_fail_sim "$batch_abs_dir"; then
                    batch_status=1
                    failed_step=filter
                    break
                fi
                ;;
            check)
                if ! downstream_worker_run_step "$terminal_log" "$timing_log" "$batch_name" "precheck check" check_existing_batch_preconditions check "$batch_abs_dir"; then
                    batch_status=1
                    failed_step=precheck_check
                    break
                fi
                if ! downstream_worker_run_step "$terminal_log" "$timing_log" "$batch_name" "check_integrity" run_check_integrity "$batch_abs_dir" "$batch_data_size"; then
                    batch_status=1
                    failed_step=check
                    break
                fi
                ;;
            *)
                batch_status=1
                failed_step="unknown_operation_${selected_op}"
                break
                ;;
        esac
    done

    batch_end_ts=$(date +%s)
    batch_elapsed=$((batch_end_ts - batch_start_ts))

    if [[ $batch_status -eq 0 ]]; then
        downstream_worker_log "$terminal_log" "$timing_log" "$batch_name" "Downstream batch finished successfully: elapsed=$(format_seconds "$batch_elapsed") (${batch_elapsed}s)"
    elif [[ $batch_status -eq 75 ]]; then
        downstream_worker_log "$terminal_log" "$timing_log" "$batch_name" "Downstream batch paused safely: elapsed=$(format_seconds "$batch_elapsed") (${batch_elapsed}s)"
    else
        downstream_worker_log "$terminal_log" "$timing_log" "$batch_name" "Downstream batch failed at ${failed_step}: elapsed=$(format_seconds "$batch_elapsed") (${batch_elapsed}s)"
    fi

    if [[ -d "$batch_abs_dir" ]]; then
        mkdir -p "$batch_abs_dir/logs"
        cat "$terminal_log" >> "$final_terminal_log"
        cat "$timing_log" >> "$final_timing_log"
    fi

    write_downstream_status "$status_file" "$batch_name" "$batch_status" "$failed_step" "$terminal_log" "$timing_log"
    return "$batch_status"
}

declare -a failed_batches=()
declare -a paused_batches=()
declare -a failed_reasons=()
declare -A batch_failed=()
declare -A pid_to_batch=()

mark_batch_failed() {
    local batch_name=$1
    local reason=$2
    if [[ -z ${batch_failed[$batch_name]+x} ]]; then
        failed_batches+=("$batch_name")
        failed_reasons+=("${batch_name}: ${reason}")
        batch_failed[$batch_name]=1
    fi
}

is_batch_failed() {
    local batch_name=$1
    [[ -n ${batch_failed[$batch_name]+x} ]]
}

active_sample_jobs=0

collect_sample_job() {
    local finished_pid=$1
    local status=$2
    local batch_name=${pid_to_batch[$finished_pid]:-}
    local failed_step

    if [[ -z "$batch_name" ]]; then
        log_outer "Sample worker finished with unknown pid=${finished_pid} status=${status}"
        return
    fi

    append_sample_logs_to_outer "$batch_name"
    failed_step=$(read_sample_failed_step "$batch_name")

    if [[ $status -eq 0 ]]; then
        log_outer "END ${batch_name}: sample status=success"
    else
        mark_batch_failed "$batch_name" "sample:${failed_step}:exit_code=${status}"
        log_outer "END ${batch_name}: sample status=failed failed_step=${failed_step} exit_code=${status}"
    fi

    unset "pid_to_batch[$finished_pid]"
}

wait_for_one_sample_job() {
    local finished_pid=""
    local status

    set +e
    wait -n -p finished_pid
    status=$?
    set -e

    if [[ -n "${finished_pid:-}" ]]; then
        collect_sample_job "$finished_pid" "$status"
        active_sample_jobs=$((active_sample_jobs - 1))
    fi
}

launch_sample_job() {
    local batch_index=$1
    local batch_data_size=$2
    local batch_name="batch_${batch_index}"
    local pid

    log_outer "START ${batch_name}: sample size=${batch_data_size}"
    run_sample_worker "$batch_index" "$batch_data_size" &
    pid=$!
    pid_to_batch[$pid]=$batch_name
    active_sample_jobs=$((active_sample_jobs + 1))
}

run_parallel_sample_stage() {
    local batch_index batch_data_size
    local sample_start_ts sample_end_ts sample_elapsed

    sample_start_ts=$(date +%s)
    log_outer "Parallel sample started: parallel_jobs=${effective_parallel_jobs} requested_parallel_jobs=${parallel_jobs} selected_batches=${selected_batches_label}"

    for batch_index in "${selected_batch_indices[@]}"; do
        while (( active_sample_jobs >= effective_parallel_jobs )); do
            wait_for_one_sample_job
        done
        batch_data_size=$(current_batch_size "$batch_index")
        launch_sample_job "$batch_index" "$batch_data_size"
    done

    while (( active_sample_jobs > 0 )); do
        wait_for_one_sample_job
    done

    sample_end_ts=$(date +%s)
    sample_elapsed=$((sample_end_ts - sample_start_ts))
    log_outer "Parallel sample finished: elapsed=$(format_seconds "$sample_elapsed") (${sample_elapsed}s)"
}

read_downstream_status_field() {
    local batch_name=$1
    local field_name=$2
    local status_file="$worker_tmp_dir/${batch_name}_downstream.status"
    if [[ -f "$status_file" ]]; then
        awk -F= -v field="$field_name" '$1 == field {print substr($0, length(field) + 2); exit}' "$status_file"
    fi
}

read_downstream_failed_step() {
    local batch_name=$1
    local failed_step
    failed_step=$(read_downstream_status_field "$batch_name" failed_step)
    if [[ -n "$failed_step" ]]; then
        echo "$failed_step"
    else
        echo "missing_status_file"
    fi
}

append_downstream_logs_to_outer() {
    local batch_name=$1
    local terminal_log timing_log

    terminal_log=$(read_downstream_status_field "$batch_name" terminal_log)
    timing_log=$(read_downstream_status_field "$batch_name" timing_log)

    if [[ -n "$terminal_log" && -f "$terminal_log" ]]; then
        {
            echo ""
            echo "===== BEGIN ${batch_name} downstream terminal log ====="
            cat "$terminal_log"
            echo "===== END ${batch_name} downstream terminal log ====="
        } >> "$outer_terminal_log"
    fi

    if [[ -n "$timing_log" && -f "$timing_log" ]]; then
        {
            echo ""
            echo "===== BEGIN ${batch_name} downstream timing log ====="
            cat "$timing_log"
            echo "===== END ${batch_name} downstream timing log ====="
        } >> "$outer_timing_log"
        {
            echo ""
            echo "===== BEGIN ${batch_name} downstream summary log ====="
            cat "$timing_log"
            echo "===== END ${batch_name} downstream summary log ====="
        } >> "$outer_summary_log"
    fi
}

active_downstream_jobs=0

collect_downstream_job() {
    local finished_pid=$1
    local status=$2
    local batch_name=${pid_to_batch[$finished_pid]:-}
    local failed_step

    if [[ -z "$batch_name" ]]; then
        log_outer "Downstream worker finished with unknown pid=${finished_pid} status=${status}"
        return
    fi

    append_downstream_logs_to_outer "$batch_name"
    failed_step=$(read_downstream_failed_step "$batch_name")

    if [[ $status -eq 0 && "$failed_step" != "missing_status_file" ]]; then
        log_outer "END ${batch_name}: downstream status=success"
    elif [[ $status -eq 75 && "$failed_step" == paused ]]; then
        paused_batches+=("$batch_name")
        log_outer "END ${batch_name}: downstream status=paused checkpoint_saved=true"
    else
        mark_batch_failed "$batch_name" "downstream:${failed_step}:exit_code=${status}"
        log_outer "END ${batch_name}: downstream status=failed failed_step=${failed_step} exit_code=${status}"
    fi

    unset "pid_to_batch[$finished_pid]"
}

wait_for_one_downstream_job() {
    local finished_pid=""
    local status

    set +e
    wait -n -p finished_pid
    status=$?
    set -e

    if [[ -n "${finished_pid:-}" ]]; then
        collect_downstream_job "$finished_pid" "$status"
        active_downstream_jobs=$((active_downstream_jobs - 1))
    fi
}

launch_downstream_job() {
    local batch_index=$1
    local batch_data_size=$2
    local batch_name="batch_${batch_index}"
    local pid

    log_outer "START ${batch_name}: downstream size=${batch_data_size}"
    run_downstream_batch "$batch_index" "$batch_data_size" &
    pid=$!
    pid_to_batch[$pid]=$batch_name
    active_downstream_jobs=$((active_downstream_jobs + 1))
}

run_parallel_downstream_stage() {
    local batch_index batch_data_size batch_name
    local downstream_start_ts downstream_end_ts downstream_elapsed

    if (( ${#downstream_ops[@]} == 0 )); then
        return 0
    fi

    downstream_start_ts=$(date +%s)
    log_outer "Parallel downstream stage started: operations=$(join_by " " "${downstream_ops[@]}") parallel_jobs=${effective_parallel_jobs} requested_parallel_jobs=${parallel_jobs} selected_batches=${selected_batches_label}"

    for batch_index in "${selected_batch_indices[@]}"; do
        batch_name="batch_${batch_index}"
        batch_data_size=$(current_batch_size "$batch_index")

        if is_batch_failed "$batch_name"; then
            log_outer "SKIP ${batch_name}: previous step failed"
            continue
        fi

        while (( active_downstream_jobs >= effective_parallel_jobs )); do
            wait_for_one_downstream_job
        done
        if (( simulation_deadline > 0 && $(date +%s) >= simulation_deadline )); then
            paused_batches+=("$batch_name")
            log_outer "PAUSE ${batch_name}: deadline reached; batch not started"
            continue
        fi
        launch_downstream_job "$batch_index" "$batch_data_size"
    done

    while (( active_downstream_jobs > 0 )); do
        wait_for_one_downstream_job
    done

    downstream_end_ts=$(date +%s)
    downstream_elapsed=$((downstream_end_ts - downstream_start_ts))
    log_outer "Parallel downstream stage finished: elapsed=$(format_seconds "$downstream_elapsed") (${downstream_elapsed}s)"
}

log_outer "Pipeline started: name=${name} body_type=${body_type} sample_body_type=${sample_body_type} size=${size} batch_size=${batch_size} num_batches=${num_batches} selected_batch_count=${selected_batch_count} batch_list=${batch_list} selected_batches=${selected_batches_label} operations=${operations_label} seed=${seed} parallel_jobs=${parallel_jobs} effective_parallel_jobs=${effective_parallel_jobs} run_id=${run_id}"
if (( simulation_deadline > 0 )); then
    log_outer "Automatic pause: max_hours=${max_hours} deadline_unix=${simulation_deadline}; workers finish their current sample before exiting"
fi
log_outer "Outer logs: terminal_log=${outer_terminal_log} timing_log=${outer_timing_log} summary_log=${outer_summary_log}"

if [[ $operations_include_sample -eq 1 ]]; then
    run_parallel_sample_stage
fi

run_parallel_downstream_stage

failed_count=${#failed_batches[@]}
paused_count=${#paused_batches[@]}
success_count=$((selected_batch_count - failed_count - paused_count))
script_end_ts=$(date +%s)
script_elapsed=$((script_end_ts - script_start_ts))

log_outer "Pipeline finished: elapsed=$(format_seconds "$script_elapsed") (${script_elapsed}s)"
log_outer "Summary: success_batches=${success_count} failed_batches=${failed_count} paused_batches=${paused_count} selected_batches=${selected_batch_count} total_batches=${num_batches}"

if (( paused_count > 0 )); then
    log_outer "Paused batch list: $(join_by ", " "${paused_batches[@]}")"
fi

if (( failed_count > 0 )); then
    failed_list=$(join_by ", " "${failed_batches[@]}")
    log_outer "Failed batch list: ${failed_list}"
    log_outer "Failed batch details:"
    for failed_reason in "${failed_reasons[@]}"; do
        log_outer "  ${failed_reason}"
    done
    exit 1
fi

log_outer "Failed batch list: none"
if (( paused_count > 0 )); then
    log_outer "Simulation paused safely. All workers have exited; rerun the same command to resume."
    exit 75
fi
exit 0
