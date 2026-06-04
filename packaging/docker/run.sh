#!/usr/bin/env bash
# Run the Metaworld pipeline-runner image.
#
# Usage:
#   bash packaging/docker/run.sh shell                                       # interactive bash
#   bash packaging/docker/run.sh pipeline ENV_NAME=hammer-v3 NUM_EPISODES=5  # one task end-to-end
#   bash packaging/docker/run.sh parallel NUM_EPISODES=1000 N_WORKERS=8      # all 50 tasks parallel
#
# Env knobs (host):
#   IMAGE        image:tag to run                 (default: metaworld-pipeline:0.1)
#   DATA_ROOT    host path mounted as datasets    (default: $REPO/rbs_datasets)
#   USE_GPU=1    pass --gpus all and use EGL      (default: off; uses mesa-EGL or osmesa)
#   NO_GL=1      force MUJOCO_GL=osmesa           (CPU software renderer, slowest but always works)
#   SHM_SIZE     /dev/shm size                    (default: 8g)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-metaworld-pipeline:0.1}"
DATA_ROOT="${DATA_ROOT:-${REPO}/rbs_datasets}"
SHM_SIZE="${SHM_SIZE:-8g}"

mkdir -p "${DATA_ROOT}"

# GPU vs CPU rendering
if [[ "${NO_GL:-0}" == "1" ]]; then
    GPU_FLAGS=()
    GL_ENV=(-e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa)
elif [[ "${USE_GPU:-0}" == "1" ]]; then
    GPU_FLAGS=(--gpus all)
    GL_ENV=(-e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl)
else
    GPU_FLAGS=()
    GL_ENV=(-e MUJOCO_GL=egl -e PYOPENGL_PLATFORM=egl)
fi

MODE="${1:-shell}"; shift || true

# Forward remaining KEY=VAL args as -e flags (for ENV_NAME=, NUM_EPISODES=, ...)
ENV_FORWARD=()
for kv in "$@"; do
    if [[ "${kv}" == *=* ]]; then
        ENV_FORWARD+=(-e "${kv}")
    fi
done

# Code is baked into /opt/metaworld at build time; we only mount the dataset
# directory so trained outputs survive container removal.
COMMON=(
    --rm
    "${GPU_FLAGS[@]}"
    --shm-size="${SHM_SIZE}"
    -v "${DATA_ROOT}:/opt/metaworld/rbs_datasets"
    -e SAVE_DIR="/opt/metaworld/rbs_datasets/datasets_test"
    -e ROLLOUT_DIR="/opt/metaworld/rbs_datasets/datasets_test/_rollout"
    "${GL_ENV[@]}"
    "${ENV_FORWARD[@]}"
    -w /opt/metaworld
)

case "${MODE}" in
    shell)
        exec docker run -it "${COMMON[@]}" "${IMAGE}" bash
        ;;
    pipeline)
        exec docker run "${COMMON[@]}" "${IMAGE}" \
            bash rbs_sceneflow_scripts/run_pipeline.sh
        ;;
    parallel)
        exec docker run "${COMMON[@]}" "${IMAGE}" \
            bash rbs_sceneflow_scripts/run_all_tasks_parallel.sh
        ;;
    *)
        echo "unknown mode: ${MODE} (expected: shell | pipeline | parallel)" >&2
        sed -n '2,16p' "$0"
        exit 1
        ;;
esac
