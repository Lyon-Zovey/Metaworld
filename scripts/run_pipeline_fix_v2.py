#!/usr/bin/env python3
"""
批量重新数采 datasets_500_fix
策略: 50 task 并行，每 task 内按 --num-procs 拆成多子进程，
      每条 traj 完成 Step2 后立即串行跑 Step3/4，不等整个 task 结束。
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PYTHON     = "/mnt2/liangzhuowei/miniconda3/envs/metaworld/bin/python"
ROOT       = Path("/mnt2/liangzhuowei/Metaworld")
ROLLOUT    = ROOT / "rollout_data"
SCRIPTS    = ROOT / "rbs_sceneflow_scripts"
PROGRESS   = Path("/tmp/pipeline_fix_progress.txt")
LOG_DIR    = Path("/tmp/pipeline_fix_logs")

TASKS = [
    "assembly-v3","basketball-v3","bin-picking-v3","box-close-v3",
    "button-press-topdown-v3","button-press-topdown-wall-v3",
    "button-press-v3","button-press-wall-v3",
    "coffee-button-v3","coffee-pull-v3","coffee-push-v3",
    "dial-turn-v3","disassemble-v3",
    "door-close-v3","door-lock-v3","door-open-v3","door-unlock-v3",
    "drawer-close-v3","drawer-open-v3",
    "faucet-close-v3","faucet-open-v3",
    "hammer-v3","hand-insert-v3",
    "handle-press-side-v3","handle-press-v3",
    "handle-pull-side-v3","handle-pull-v3",
    "lever-pull-v3",
    "peg-insert-side-v3","peg-unplug-side-v3",
    "pick-out-of-hole-v3","pick-place-v3","pick-place-wall-v3",
    "plate-slide-back-side-v3","plate-slide-back-v3",
    "plate-slide-side-v3","plate-slide-v3",
    "push-back-v3","push-v3","push-wall-v3",
    "reach-v3","reach-wall-v3",
    "shelf-place-v3","soccer-v3",
    "stick-pull-v3","stick-push-v3",
    "sweep-into-v3","sweep-v3",
    "window-close-v3","window-open-v3",
]


def run(cmd, log_fh, env=None):
    e = os.environ.copy()
    if env:
        e.update(env)
    r = subprocess.run(cmd, stdout=log_fh, stderr=log_fh, env=e)
    return r.returncode == 0


def steps34(cam_dir: Path, log_fh):
    """Step3 + Step4 for a single traj dir."""
    ok = True
    ok &= run([PYTHON, str(SCRIPTS/"traj2sceneflow/convert_camera_depths.py"), str(cam_dir)], log_fh)
    ok &= run([PYTHON, str(SCRIPTS/"traj2sceneflow/flow_compress.py"),
               "compress", "--out_dir", str(cam_dir)], log_fh)
    ok &= run([PYTHON, str(SCRIPTS/"traj2sceneflow/point_compress.py"),
               "--seg_dir", str(cam_dir)], log_fh)
    ok &= run([PYTHON, str(SCRIPTS/"traj2sceneflow/seg_compress.py"),
               "compress", "--seg-dir", str(cam_dir)], log_fh)
    return ok


def run_task(task: str, out_root: Path, workers_per_task: int,
             already_done: set, width: int, height: int):
    h5   = ROLLOUT / task / "trajectory.state.mocap_xyz.mujoco_cpu.h5"
    js   = ROLLOUT / task / "trajectory.state.mocap_xyz.mujoco_cpu.json"
    out  = out_root / task
    log  = LOG_DIR / f"{task}.log"

    episodes = json.loads(js.read_text())["episodes"]
    total    = len(episodes)

    # 找出还没完成的 traj
    pending = [i for i in range(total) if f"{task} traj_{i}" not in already_done]
    if not pending:
        return task, total, 0, []

    log_fh = open(log, "a")
    errors = []

    # 把 pending 按 workers_per_task 分片，每片一个 replay 子进程
    chunk = max(1, len(pending) // workers_per_task)
    slices = []
    for k in range(workers_per_task):
        s = k * chunk
        e_ = s + chunk if k < workers_per_task - 1 else len(pending)
        if s < len(pending):
            slices.append(pending[s:e_])

    # 每个分片：先 Step2（replay 整片），再逐条 Step3/4
    def run_slice(traj_ids: list):
        slice_errors = []
        # Step 2: replay 整片
        id_args = []
        for tid in traj_ids:
            id_args += ["--traj-id", str(tid)]

        # replay 不支持多 --traj-id，改用 --all-trajs 不行；
        # 用临时 json 指定 traj 列表也太重，直接逐条跑 Step2 然后 Step3/4
        for tid in traj_ids:
            if f"{task} traj_{tid}" in already_done:
                continue
            cam_dir = out / "camera_data" / f"traj_{tid}"
            # Step 2
            ok2 = run(
                [PYTHON, str(SCRIPTS/"replay_record_trajectories.py"),
                 "--h5", str(h5), "--json", str(js),
                 "--output-dir", str(out),
                 "--traj-id", str(tid),
                 "--camera", "corner",
                 "--width", str(width), "--height", str(height)],
                log_fh,
                env={"MUJOCO_GL": "egl"},
            )
            if not ok2:
                slice_errors.append((task, tid, "step2_failed"))
                continue
            # Step 3/4
            ok34 = steps34(cam_dir, log_fh)
            if not ok34:
                slice_errors.append((task, tid, "step34_failed"))
                continue
            # 打点
            with open(PROGRESS, "a") as pf:
                pf.write(f"{task} traj_{tid}\n")
        return slice_errors

    # 各分片并行
    with ThreadPoolExecutor(max_workers=len(slices)) as ex:
        futs = [ex.submit(run_slice, sl) for sl in slices]
        for fut in as_completed(futs):
            errors.extend(fut.result())

    log_fh.close()
    done_count = len(pending) - len(errors)
    return task, total, done_count, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="/mnt2/liangzhuowei/Metaworld/datasets_500_fix")
    parser.add_argument("--workers-per-task", type=int, default=3,
                        help="每个 task 内部并行 replay 的进程数 (default=3, 50×3=150核)")
    parser.add_argument("--width",  type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--task",   default=None, help="只跑某一个 task（测试用）")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS.touch(exist_ok=True)

    # 读取已完成的
    already_done = set(PROGRESS.read_text().strip().splitlines())
    n_already = len(already_done)

    tasks = [args.task] if args.task else TASKS
    total_trajs = len(tasks) * 500

    print(f"目标路径   : {out_root}")
    print(f"Task 数    : {len(tasks)}")
    print(f"总 traj    : {total_trajs}")
    print(f"已完成     : {n_already}")
    print(f"每task并行 : {args.workers_per_task}  →  总并行进程: {len(tasks) * args.workers_per_task}")
    print(f"分辨率     : {args.width}×{args.height}")
    print()

    t0 = time.time()

    # 50 个 task 全部并行
    with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
        futs = {
            ex.submit(run_task, t, out_root, args.workers_per_task,
                      already_done, args.width, args.height): t
            for t in tasks
        }
        done_tasks = 0
        all_errors = []
        for fut in as_completed(futs):
            task_name, total, done_cnt, errs = fut.result()
            done_tasks += 1
            all_errors.extend(errs)
            elapsed = time.time() - t0
            cur_done = len(set(PROGRESS.read_text().strip().splitlines()))
            speed = (cur_done - n_already) / elapsed if elapsed > 0 else 0
            remain = total_trajs - cur_done
            eta = remain / speed if speed > 0 else 0
            print(f"[{done_tasks:2d}/50 tasks]  {task_name}  done={done_cnt}  "
                  f"总进度={cur_done}/{total_trajs}  "
                  f"速度={speed:.2f}/s  ETA={eta/3600:.1f}h",
                  flush=True)

    elapsed = time.time() - t0
    final_done = len(set(PROGRESS.read_text().strip().splitlines()))
    print(f"\n=== 完成 ===")
    print(f"总完成: {final_done}/{total_trajs}")
    print(f"错误数: {len(all_errors)}")
    print(f"耗时  : {elapsed/3600:.2f} 小时")
    if all_errors:
        print("前10个错误:")
        for t, i, e in all_errors[:10]:
            print(f"  {t} traj_{i}: {e}")


if __name__ == "__main__":
    main()
