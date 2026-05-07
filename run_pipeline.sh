#!/usr/bin/env bash
# Full data collection pipeline for a single Metaworld env.
# Usage:
#   bash run_pipeline.sh
#   或直接改下面两行后运行
set -euo pipefail

# ── 修改这两行（或通过环境变量覆盖）──────────────────────────
ENV_NAME="${ENV_NAME:-stick-push-v3}"
NUM_EPISODES="${NUM_EPISODES:-5}"
# ────────────────────────────────────────────────────────────

H5_FILE="rollout_data/${ENV_NAME}/trajectory.state.mocap_xyz.mujoco_cpu.h5"
JSON_FILE="rollout_data/${ENV_NAME}/trajectory.state.mocap_xyz.mujoco_cpu.json"
DATASET_DIR="dataset/${ENV_NAME}/camera_data"

echo ""
echo "========================================================="
echo "  ENV : ${ENV_NAME}   EPISODES : ${NUM_EPISODES}"
echo "========================================================="

echo ""
echo "=== [1/7] Rollout scripted policy ==="
python rbs_sceneflow_scripts/rollout_scripted_policy.py \
  --env-name "${ENV_NAME}" \
  --num-episodes "${NUM_EPISODES}" \
  --output-dir rollout_data

echo ""
echo "=== [2/7] Replay & record trajectories ==="
python rbs_sceneflow_scripts/replay_record_trajectories.py \
  --h5   "${H5_FILE}" \
  --json "${JSON_FILE}" \
  --output-dir "dataset/${ENV_NAME}" \
  --all-trajs --success-only

echo ""
echo "=== [3/7] Convert camera depths ==="
python rbs_sceneflow_scripts/traj2sceneflow/convert_camera_depths.py \
  "${DATASET_DIR}"

echo ""
echo "=== [4/7] Flow compress ==="
python rbs_sceneflow_scripts/traj2sceneflow/flow_compress.py \
  compress --out_root "${DATASET_DIR}"

echo ""
echo "=== [5/7] Point compress ==="
python rbs_sceneflow_scripts/traj2sceneflow/point_compress.py \
  --mode compress --root "${DATASET_DIR}"

echo ""
echo "=== [6/7] Seg compress (all traj dirs) ==="
for traj_dir in "${DATASET_DIR}"/traj_*/; do
  echo "  -> ${traj_dir}"
  python rbs_sceneflow_scripts/traj2sceneflow/seg_compress.py \
    compress --seg-dir "${traj_dir}"
done

echo ""
echo "=== [7/7] Inspect sceneflow first frame (traj_0) ==="
conda run -n metaworld python \
  rbs_sceneflow_scripts/traj2sceneflow/inspect_sceneflow_first_frame.py \
  --traj-dir "${DATASET_DIR}/traj_0"

echo ""
echo "========================================================="
echo "  Done: ${ENV_NAME}"
echo "========================================================="
