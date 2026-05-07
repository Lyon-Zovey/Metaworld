#!/usr/bin/env bash
# Full data collection pipeline for a single Metaworld env.
# Usage:
#   bash run_pipeline.sh
#   或直接改下面两行后运行
set -euo pipefail

# ── 修改这两行（或通过环境变量覆盖）──────────────────────────
ENV_NAME="${ENV_NAME:-drawer-close-v3}"
NUM_EPISODES="${NUM_EPISODES:-5}"
# ────────────────────────────────────────────────────────────

SAVE_DIR="${SAVE_DIR:-dataset}"
H5_FILE="rollout_data/${ENV_NAME}/trajectory.state.mocap_xyz.mujoco_cpu.h5"
JSON_FILE="rollout_data/${ENV_NAME}/trajectory.state.mocap_xyz.mujoco_cpu.json"
DATASET_DIR="${SAVE_DIR}/${ENV_NAME}/camera_data"

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
# METAWORLD_CAMERAS 可由外部脚本（如 run_all_tasks_parallel.sh）传入覆盖
_CAMERAS="${METAWORLD_CAMERAS:-corner corner2 corner3}"
# shellcheck disable=SC2086
python rbs_sceneflow_scripts/replay_record_trajectories.py \
  --h5   "${H5_FILE}" \
  --json "${JSON_FILE}" \
  --output-dir "${SAVE_DIR}/${ENV_NAME}" \
  --all-trajs --success-only \
  --random-camera --cameras ${_CAMERAS}

echo ""
echo "=== [3/7] Convert camera depths ==="
python rbs_sceneflow_scripts/traj2sceneflow/convert_camera_depths.py \
  "${DATASET_DIR}"

echo ""
echo "=== [4/7] Flow compress (+ delete scene_point_flow_ref*.npy) ==="
python rbs_sceneflow_scripts/traj2sceneflow/flow_compress.py \
  compress --out_root "${DATASET_DIR}" --delete_npy

echo ""
echo "=== [5/7] Point compress (+ delete depth_video.npy) ==="
python rbs_sceneflow_scripts/traj2sceneflow/point_compress.py \
  --mode compress --root "${DATASET_DIR}" --delete-existing

echo ""
echo "=== [6/7] Seg compress (+ delete seg.npy, all traj dirs) ==="
for traj_dir in "${DATASET_DIR}"/traj_*/; do
  echo "  -> ${traj_dir}"
  python rbs_sceneflow_scripts/traj2sceneflow/seg_compress.py \
    compress --seg-dir "${traj_dir}" --delete-source
done

echo ""
INSPECT_TRAJ_DIR="$(ls -d "${DATASET_DIR}"/traj_* 2>/dev/null | sort -V | head -n 1)"
if [[ -z "${INSPECT_TRAJ_DIR}" ]]; then
  echo "No trajectory directory found under ${DATASET_DIR}; skip inspect."
  exit 0
fi
echo "=== [7/7] Inspect sceneflow first frame (${INSPECT_TRAJ_DIR##*/}) ==="
conda run -n metaworld python \
  rbs_sceneflow_scripts/traj2sceneflow/inspect_sceneflow_first_frame.py \
  --traj-dir "${INSPECT_TRAJ_DIR}"

echo ""
echo "========================================================="
echo "  Done: ${ENV_NAME}"
echo "========================================================="
