#!/usr/bin/env bash
# ============================================================================
# Metaworld 并行多任务数采脚本
# 参考 MIKASA-Robo/run_scripts/collect_gpu_batched.sh 结构
#
# 并行模式：每个任务（task）作为一个独立进程跑完完整的 7 步 pipeline，
#           最多同时跑 N_WORKERS 个任务。任务互相独立，落盘路径不重叠。
#
# 用法:
#   bash run_all_tasks_parallel.sh
#   N_WORKERS=8 NUM_EPISODES=20 bash run_all_tasks_parallel.sh
# ============================================================================

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                          可 调 参 数  ← 直接改这里                         ║
# ╠══════════════════════════════════════════════════════════════════════════╣

NUM_EPISODES="${NUM_EPISODES:-500}"
# 每个任务最多尝试采集的 episodes 数（success-only 过滤后可能更少）

N_WORKERS="${N_WORKERS:-24}"
# 同时并行的任务数。
# Metaworld 是纯 CPU（mujoco_cpu），每个任务约占 2-4 核；
# 建议 N_WORKERS ≤ CPU核心数 / 4，避免内存/调度竞争。

REPLAY_PROCS="${REPLAY_PROCS:-2}"
# replay_record_trajectories.py 内部的并行进程数（每个任务内拆分轨迹并行）。
# 总占核 ≈ N_WORKERS × REPLAY_PROCS；建议乘积 ≤ CPU 核心数。

REPLAY_FPS="${REPLAY_FPS:-16}"
REPLAY_WIDTH="${REPLAY_WIDTH:-832}"
REPLAY_HEIGHT="${REPLAY_HEIGHT:-480}"
# 回放录像参数，需与下游训练数据规范一致。

CAMERAS="${CAMERAS:-corner corner2 corner3}"
# 随机摄像头池（空格分隔），传给 replay_record_trajectories.py --cameras

SAVE_DIR="${SAVE_DIR:-/mnt2/liangzhuowei/Metaworld/datasets_500}"
# 落盘根目录（相对脚本所在位置），每个任务写入 $SAVE_DIR/<task_name>/

CONDA_ENV="${CONDA_ENV:-metaworld}"
# conda 环境名

RUN_INSPECT="${RUN_INSPECT:-1}"
# 1 = 数采完成后对所有成功任务生成点云可视化图；0 = 跳过

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

# ╚══════════════════════════════════════════════════════════════════════════╝

# ─── 路径 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${SAVE_DIR}" = /* ]]; then
    DATA_ROOT="${SAVE_DIR}"
    LOG_DIR="${SAVE_DIR}/_logs"
else
    DATA_ROOT="${SCRIPT_DIR}/${SAVE_DIR}"
    LOG_DIR="${SCRIPT_DIR}/${SAVE_DIR}/_logs"
fi
mkdir -p "${LOG_DIR}"

TOTAL=${#TASKS[@]}
START_TS=$(date +%s)

# ─── Banner ───────────────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║            Metaworld 并行多任务数采                           ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  tasks    : %-48s║\n" "${TOTAL}"
printf "║  episodes : %-48s║\n" "${NUM_EPISODES}"
printf "║  workers  : %-48s║\n" "${N_WORKERS}"
printf "║  replay-procs: %-45s║\n" "${REPLAY_PROCS}"
printf "║  replay-fps: %-47s║\n" "${REPLAY_FPS}"
printf "║  replay-res: %-47s║\n" "${REPLAY_WIDTH}x${REPLAY_HEIGHT}"
printf "║  cameras  : %-48s║\n" "${CAMERAS}"
printf "║  save_dir : %-48s║\n" "${SAVE_DIR}"
printf "║  logs     : %-48s║\n" "${LOG_DIR}"
echo "╚══════════════════════════════════════════════════════════════╝"
echo

# ─── 单任务采集函数 ───────────────────────────────────────────────────────
_run_task() {
    local env_name="$1"
    cd "${SCRIPT_DIR}"
    # 将摄像头参数注入给 run_pipeline.sh（见该脚本 step [2/7]）
    ENV_NAME="${env_name}" \
    NUM_EPISODES="${NUM_EPISODES}" \
    METAWORLD_CAMERAS="${CAMERAS}" \
    SAVE_DIR="${SAVE_DIR}" \
    REPLAY_PROCS="${REPLAY_PROCS}" \
    REPLAY_FPS="${REPLAY_FPS}" \
    REPLAY_WIDTH="${REPLAY_WIDTH}" \
    REPLAY_HEIGHT="${REPLAY_HEIGHT}" \
    MUJOCO_GL=egl \
    /mnt2/liangzhuowei/miniconda3/bin/conda run -n "${CONDA_ENV}" --no-capture-output bash run_pipeline.sh
}

# ─── 并行调度：N_WORKERS 槽位 ─────────────────────────────────────────────
declare -a PIDS=()
declare -a ENVNAMES=()
FAILED=()
SUCCEEDED=()
DONE_COUNT=0

# 轮询一次已运行的进程，回收已结束的槽位
_reap_finished() {
    local i exit_code
    for i in "${!PIDS[@]}"; do
        if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
            exit_code=0
            wait "${PIDS[$i]}" || exit_code=$?
            DONE_COUNT=$(( DONE_COUNT + 1 ))
            if [[ ${exit_code} -ne 0 ]]; then
                echo "  [FAILED] ${ENVNAMES[$i]}  (exit=${exit_code}, ${DONE_COUNT}/${TOTAL})"
                FAILED+=("${ENVNAMES[$i]}")
            else
                echo "  [OK]     ${ENVNAMES[$i]}  (${DONE_COUNT}/${TOTAL} done)"
                SUCCEEDED+=("${ENVNAMES[$i]}")
            fi
            unset "PIDS[$i]" "ENVNAMES[$i]"
            PIDS=("${PIDS[@]}")
            ENVNAMES=("${ENVNAMES[@]}")
            return 0
        fi
    done
    return 1   # 没有进程结束
}

# 阻塞直到有一个槽位空出
_wait_one_slot() {
    while [[ ${#PIDS[@]} -ge ${N_WORKERS} ]]; do
        _reap_finished && return
        sleep 2
    done
}

# 派发所有任务
for env_name in "${TASKS[@]}"; do
    _wait_one_slot
    log="${LOG_DIR}/${env_name}.log"
    printf "[launch] %-35s → %s\n" "${env_name}" "${log}"
    ( _run_task "${env_name}" ) >"${log}" 2>&1 &
    PIDS+=($!)
    ENVNAMES+=("${env_name}")
done

# 等待剩余任务完成
echo
echo "[wait] 等待剩余 ${#PIDS[@]} 个任务完成..."
while [[ ${#PIDS[@]} -gt 0 ]]; do
    _reap_finished || sleep 2
done

# ─── 采集汇总 ─────────────────────────────────────────────────────────────
END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))
echo
echo "════════════════════════════════════════════════════════════════"
printf "  采集完成  耗时 %dm%ds\n" $(( ELAPSED / 60 )) $(( ELAPSED % 60 ))
printf "  成功 %d / %d 个任务\n" "${#SUCCEEDED[@]}" "${TOTAL}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  失败任务 (${#FAILED[@]}):"
    for t in "${FAILED[@]}"; do
        printf "    ✗ %s  (log: %s/%s.log)\n" "${t}" "${LOG_DIR}" "${t}"
    done
fi
echo "════════════════════════════════════════════════════════════════"

# ─── 点云可视化（对全部成功任务并行 inspect）─────────────────────────────
if [[ "${RUN_INSPECT}" == "1" && ${#SUCCEEDED[@]} -gt 0 ]]; then
    echo
    echo "════════════════════════════════════════════════════════════════"
    echo "[inspect] 对 ${#SUCCEEDED[@]} 个成功任务生成点云可视化图..."
    echo "════════════════════════════════════════════════════════════════"

    INSPECT_PIDS=()
    for env_name in "${SUCCEEDED[@]}"; do
        cam_root="${DATA_ROOT}/${env_name}/camera_data"
        if [[ ! -d "${cam_root}" ]]; then
            echo "  [skip] ${env_name}: camera_data/ 不存在"
            continue
        fi
        ilog="${LOG_DIR}/${env_name}_inspect.log"
        printf "  [inspect] %-30s → %s\n" "${env_name}" "${ilog}"
        (
            cd "${SCRIPT_DIR}"
            conda run -n "${CONDA_ENV}" --no-capture-output python \
                rbs_sceneflow_scripts/traj2sceneflow/inspect_sceneflow_first_frame.py \
                --root "${cam_root}"
        ) >"${ilog}" 2>&1 &
        INSPECT_PIDS+=($!)
    done

    for pid in "${INSPECT_PIDS[@]}"; do
        wait "${pid}" || echo "  [warn] inspect 进程 pid=${pid} 返回非零"
    done
    echo "  [inspect] 完成"
fi

# ─── 最终状态 ─────────────────────────────────────────────────────────────
echo
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  全部完成！                                                    ║"
printf "║  数据路径 : %-49s║\n" "${DATA_ROOT}/"
printf "║  日志路径 : %-49s║\n" "${LOG_DIR}/"
echo "║  点云图   : 各 traj 目录下的 _sceneflow_check_ref*.png         ║"
echo "╚══════════════════════════════════════════════════════════════╝"

[[ ${#FAILED[@]} -gt 0 ]] && exit 1
exit 0
