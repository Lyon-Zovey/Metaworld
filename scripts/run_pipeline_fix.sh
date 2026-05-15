#!/bin/bash
# 批量重新数采到 datasets_500_fix
# 用法: bash scripts/run_pipeline_fix.sh
# 每个 task 独占一个进程，Step2→3→4 串行完成后再处理下一条 traj

set -euo pipefail

PYTHON=/mnt2/liangzhuowei/miniconda3/envs/metaworld/bin/python
ROOT=/mnt2/liangzhuowei/Metaworld
OUT_ROOT=/mnt2/liangzhuowei/Metaworld/datasets_500_fix
ROLLOUT_ROOT=/mnt2/liangzhuowei/Metaworld/rollout_data
LOG_DIR=/tmp/pipeline_fix_logs
PROGRESS_FILE=/tmp/pipeline_fix_progress.txt

mkdir -p "$LOG_DIR" "$OUT_ROOT"
> "$PROGRESS_FILE"

TASKS=(
  assembly-v3 basketball-v3 bin-picking-v3 box-close-v3
  button-press-topdown-v3 button-press-topdown-wall-v3
  button-press-v3 button-press-wall-v3
  coffee-button-v3 coffee-pull-v3 coffee-push-v3
  dial-turn-v3 disassemble-v3
  door-close-v3 door-lock-v3 door-open-v3 door-unlock-v3
  drawer-close-v3 drawer-open-v3
  faucet-close-v3 faucet-open-v3
  hammer-v3 hand-insert-v3
  handle-press-side-v3 handle-press-v3
  handle-pull-side-v3 handle-pull-v3
  lever-pull-v3
  peg-insert-side-v3 peg-unplug-side-v3
  pick-out-of-hole-v3 pick-place-v3 pick-place-wall-v3
  plate-slide-back-side-v3 plate-slide-back-v3
  plate-slide-side-v3 plate-slide-v3
  push-back-v3 push-v3 push-wall-v3
  reach-v3 reach-wall-v3
  shelf-place-v3 soccer-v3
  stick-pull-v3 stick-push-v3
  sweep-into-v3 sweep-v3
  window-close-v3 window-open-v3
)

run_task() {
  local TASK=$1
  local H5="$ROLLOUT_ROOT/$TASK/trajectory.state.mocap_xyz.mujoco_cpu.h5"
  local JSON="$ROLLOUT_ROOT/$TASK/trajectory.state.mocap_xyz.mujoco_cpu.json"
  local OUT_DIR="$OUT_ROOT/$TASK"
  local LOG="$LOG_DIR/$TASK.log"

  # 读取总 traj 数
  local TOTAL
  TOTAL=$($PYTHON -c "import json; d=json.load(open('$JSON')); print(len(d['episodes']))" 2>/dev/null || echo 500)

  echo "[$(date +%H:%M:%S)] START $TASK  ($TOTAL trajs)" >> "$LOG"

  for ((i=0; i<TOTAL; i++)); do
    # Step 2: replay
    MUJOCO_GL=egl $PYTHON "$ROOT/rbs_sceneflow_scripts/replay_record_trajectories.py" \
      --h5 "$H5" --json "$JSON" \
      --output-dir "$OUT_DIR" \
      --traj-id "$i" \
      --camera corner --width 832 --height 480 \
      >> "$LOG" 2>&1

    local CAM_DIR="$OUT_DIR/camera_data/traj_$i"

    # Step 3: sceneflow
    $PYTHON "$ROOT/rbs_sceneflow_scripts/traj2sceneflow/convert_camera_depths.py" \
      "$CAM_DIR" >> "$LOG" 2>&1

    # Step 4a: flow compress
    $PYTHON "$ROOT/rbs_sceneflow_scripts/traj2sceneflow/flow_compress.py" \
      compress --out_dir "$CAM_DIR" >> "$LOG" 2>&1

    # Step 4b: depth compress
    $PYTHON "$ROOT/rbs_sceneflow_scripts/traj2sceneflow/point_compress.py" \
      --seg_dir "$CAM_DIR" >> "$LOG" 2>&1

    # Step 4c: seg compress
    $PYTHON "$ROOT/rbs_sceneflow_scripts/traj2sceneflow/seg_compress.py" \
      compress --seg-dir "$CAM_DIR" >> "$LOG" 2>&1

    # 进度打点
    echo "$TASK traj_$i" >> "$PROGRESS_FILE"
  done

  echo "[$(date +%H:%M:%S)] DONE $TASK" >> "$LOG"
}

export -f run_task
export PYTHON ROOT OUT_ROOT ROLLOUT_ROOT LOG_DIR PROGRESS_FILE

# 50个 task 全部并行，每个 task 自己串行跑500条
printf '%s\n' "${TASKS[@]}" | \
  xargs -P 50 -I{} bash -c 'run_task "$@"' _ {}

echo "ALL DONE" >> "$PROGRESS_FILE"
