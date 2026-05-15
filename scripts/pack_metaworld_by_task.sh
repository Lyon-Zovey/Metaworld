#!/bin/bash
# Pack each MetaWorld task into its own webdataset subdirectory.
# Usage: bash scripts/pack_metaworld_by_task.sh [DATA_ROOT] [OUTPUT_ROOT]

set -e

DATA_ROOT="${1:-/mnt2/liangzhuowei/Metaworld/datasets_500_fixed}"
OUTPUT_ROOT="${2:-/mnt2/liangzhuowei/Metaworld/webdataset_metaworld}"
PYTHON=/mnt2/liangzhuowei/miniconda3/envs/metaworld/bin/python
PACK_SCRIPT=/mnt2/liangzhuowei/rbs-data-utils/src/wbs_utils/pack_shards.py
LOG="$OUTPUT_ROOT/_pack.log"

mkdir -p "$OUTPUT_ROOT"
echo "=== pack_metaworld_by_task start $(date) ===" | tee -a "$LOG"

for task_dir in "$DATA_ROOT"/*/; do
    task=$(basename "$task_dir")
    out="$OUTPUT_ROOT/$task"
    echo "--- $task ---" | tee -a "$LOG"
    $PYTHON "$PACK_SCRIPT" \
        --dataset metaworld \
        --data-root "$task_dir" \
        --output "$out" \
        2>&1 | tee -a "$LOG"
done

echo "=== done $(date) ===" | tee -a "$LOG"
