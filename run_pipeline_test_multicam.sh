#!/usr/bin/env bash
# Test pipeline: 1 task, 3 cameras simultaneously, NO compression.
# Produces raw sceneflow .npy files viewable with:
#   python scripts/viser_raw_sceneflow_viewer.py --traj-dir <DATASET_DIR>/traj_0
#
# Usage:
#   bash run_pipeline_test_multicam.sh
#   ENV_NAME=drawer-close-v3 NUM_EPISODES=3 bash run_pipeline_test_multicam.sh
set -euo pipefail

# ── 修改这里 ─────────────────────────────────────────────────────────────────
ENV_NAME="${ENV_NAME:-drawer-close-v3}"
NUM_EPISODES="${NUM_EPISODES:-3}"
CAMERAS="${CAMERAS:-corner corner2 corner3}"
SAVE_DIR="${SAVE_DIR:-/mnt2/liangzhuowei/Metaworld/test_multicam_out}"
REPLAY_FPS="${REPLAY_FPS:-16}"
REPLAY_WIDTH="${REPLAY_WIDTH:-832}"
REPLAY_HEIGHT="${REPLAY_HEIGHT:-480}"
# ─────────────────────────────────────────────────────────────────────────────

H5_FILE="rollout_data/${ENV_NAME}/trajectory.state.mocap_xyz.mujoco_cpu.h5"
JSON_FILE="rollout_data/${ENV_NAME}/trajectory.state.mocap_xyz.mujoco_cpu.json"
DATASET_DIR="${SAVE_DIR}/${ENV_NAME}/camera_data"

echo ""
echo "========================================================="
echo "  ENV     : ${ENV_NAME}   EPISODES : ${NUM_EPISODES}"
echo "  CAMERAS : ${CAMERAS}"
echo "  OUTPUT  : ${SAVE_DIR}/${ENV_NAME}"
echo "========================================================="

echo ""
echo "=== [1/3] Rollout scripted policy ==="
python rbs_sceneflow_scripts/rollout_scripted_policy.py \
  --env-name "${ENV_NAME}" \
  --num-episodes "${NUM_EPISODES}" \
  --output-dir rollout_data

echo ""
echo "=== [2/3] Replay & record (3 cameras simultaneously, no random) ==="
# shellcheck disable=SC2086
python rbs_sceneflow_scripts/replay_record_trajectories.py \
  --h5   "${H5_FILE}" \
  --json "${JSON_FILE}" \
  --output-dir "${SAVE_DIR}/${ENV_NAME}" \
  --all-trajs --success-only \
  --multi-cameras ${CAMERAS} \
  --fps "${REPLAY_FPS}" \
  --width "${REPLAY_WIDTH}" \
  --height "${REPLAY_HEIGHT}"

echo ""
echo "=== [3/3] Convert depths → raw sceneflow .npy (NO compression) ==="
python rbs_sceneflow_scripts/traj2sceneflow/convert_camera_depths.py \
  "${DATASET_DIR}"

echo ""
echo "========================================================="
echo "  Done!  Raw sceneflow in: ${DATASET_DIR}"
echo ""
echo "  用 Viser 看数据（按 traj 编号选）："
echo "    python scripts/viser_raw_sceneflow_viewer.py \\"
echo "      --traj-dir ${DATASET_DIR}/traj_0"
echo ""
echo "  指定 ref 帧 / 相机："
echo "    python scripts/viser_raw_sceneflow_viewer.py \\"
echo "      --traj-dir ${DATASET_DIR}/traj_0 \\"
echo "      --cam corner2 --ref 0"
echo "========================================================="
