#!/usr/bin/env bash
# Batch pipeline: run run_pipeline.sh for every Metaworld task with 5 episodes each.
set -euo pipefail

NUM_EPISODES=5

TASKS=(
    "assembly-v3"
    "basketball-v3"
    "bin-picking-v3"
    "box-close-v3"
    "button-press-topdown-v3"
    "button-press-topdown-wall-v3"
    "button-press-v3"
    "button-press-wall-v3"
    "coffee-button-v3"
    "coffee-pull-v3"
    "coffee-push-v3"
    "dial-turn-v3"
    "disassemble-v3"
    "door-close-v3"
    "door-lock-v3"
    "door-open-v3"
    "door-unlock-v3"
    "hand-insert-v3"
    "drawer-close-v3"
    "drawer-open-v3"
    "faucet-open-v3"
    "faucet-close-v3"
    "hammer-v3"
    "handle-press-side-v3"
    "handle-press-v3"
    "handle-pull-side-v3"
    "handle-pull-v3"
    "lever-pull-v3"
    "pick-place-wall-v3"
    "pick-out-of-hole-v3"
    "pick-place-v3"
    "plate-slide-v3"
    "plate-slide-side-v3"
    "plate-slide-back-v3"
    "plate-slide-back-side-v3"
    "peg-insert-side-v3"
    "peg-unplug-side-v3"
    "soccer-v3"
    "stick-push-v3"
    "stick-pull-v3"
    "push-v3"
    "push-wall-v3"
    "push-back-v3"
    "reach-v3"
    "reach-wall-v3"
    "shelf-place-v3"
    "sweep-into-v3"
    "sweep-v3"
    "window-open-v3"
    "window-close-v3"
)

TOTAL=${#TASKS[@]}
FAILED=()

for i in "${!TASKS[@]}"; do
    ENV_NAME="${TASKS[$i]}"
    IDX=$((i + 1))
    echo ""
    echo "#########################################################"
    echo "  TASK ${IDX}/${TOTAL}: ${ENV_NAME}"
    echo "#########################################################"

    if ENV_NAME="${ENV_NAME}" NUM_EPISODES="${NUM_EPISODES}" conda run -n metaworld --no-capture-output bash run_pipeline.sh; then
        echo "  [OK] ${ENV_NAME}"
    else
        echo "  [FAILED] ${ENV_NAME}"
        FAILED+=("${ENV_NAME}")
    fi
done

echo ""
echo "#########################################################"
echo "  All tasks finished.  Total: ${TOTAL}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  Failed tasks (${#FAILED[@]}):"
    for t in "${FAILED[@]}"; do
        echo "    - ${t}"
    done
    exit 1
else
    echo "  All tasks succeeded!"
fi
echo "#########################################################"
