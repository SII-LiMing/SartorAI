#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="/inspire/hdd/global_user/liming-253108120187/ReWeaver2/ReWeaver2/GarmentCodeRC/garmentcode_144k"
LOG_PATH="$SCRIPT_DIR/log/batch_gen_caption.log"

cd "$SCRIPT_DIR"
mkdir -p "$SCRIPT_DIR/log"

python "$SCRIPT_DIR/gen_caption.py" "$DATASET_ROOT" --log_path "$LOG_PATH" "$@"
