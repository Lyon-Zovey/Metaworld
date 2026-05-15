#!/usr/bin/env python3
"""实时显示 run_pipeline_fix.sh 的进度"""
import time, os, sys
from pathlib import Path
from datetime import datetime, timedelta

PROGRESS_FILE = "/tmp/pipeline_fix_progress.txt"
TOTAL_TRAJS   = 25000
LOG_DIR       = Path("/tmp/pipeline_fix_logs")

def read_done():
    try:
        lines = Path(PROGRESS_FILE).read_text().strip().splitlines()
        return [l for l in lines if l and l != "ALL DONE"]
    except FileNotFoundError:
        return []

def per_task_done():
    counts = {}
    for line in read_done():
        task = line.split()[0]
        counts[task] = counts.get(task, 0) + 1
    return counts

t_start = None

print(f"监控 datasets_500_fix 数采进度 (共 {TOTAL_TRAJS} 条 traj)")
print(f"{'':=<70}")

while True:
    done_lines = read_done()
    n_done = len(done_lines)

    if n_done > 0 and t_start is None:
        t_start = time.time()

    elapsed = time.time() - t_start if t_start else 0
    speed   = n_done / elapsed if elapsed > 0 else 0
    remain  = TOTAL_TRAJS - n_done
    eta_sec = remain / speed if speed > 0 else 0

    per_task = per_task_done()
    # 已完成的 task（500条都跑完的）
    finished_tasks = [t for t, c in per_task.items() if c >= 500]
    running_tasks  = {t: c for t, c in per_task.items() if c < 500}

    pct = 100 * n_done / TOTAL_TRAJS
    bar_len = 40
    filled  = int(bar_len * n_done / TOTAL_TRAJS)
    bar     = "█" * filled + "░" * (bar_len - filled)

    now = datetime.now().strftime("%H:%M:%S")
    eta_str = str(timedelta(seconds=int(eta_sec))) if eta_sec > 0 else "--:--:--"

    os.system("clear")
    print(f"[{now}]  datasets_500_fix 重新数采进度")
    print(f"{'':=<70}")
    print(f"  [{bar}] {pct:.1f}%")
    print(f"  已完成: {n_done:5d} / {TOTAL_TRAJS}  |  速度: {speed:.2f} traj/s  |  ETA: {eta_str}")
    print(f"  已完成 task: {len(finished_tasks)}/50  |  运行中 task: {len(running_tasks)}")
    print(f"{'':=<70}")

    # 显示各 task 进度（只显示运行中的）
    if running_tasks:
        print(f"  {'Task':<35}  {'完成':>5}  {'进度':>6}")
        print(f"  {'-'*50}")
        for task, cnt in sorted(running_tasks.items(), key=lambda x: -x[1])[:20]:
            bar2 = "█" * int(20 * cnt / 500) + "░" * (20 - int(20 * cnt / 500))
            print(f"  {task:<35}  {cnt:>5}  [{bar2}]")

    if Path(PROGRESS_FILE).exists() and "ALL DONE" in Path(PROGRESS_FILE).read_text():
        print(f"\n{'':=<70}")
        print(f"  全部完成！总耗时: {str(timedelta(seconds=int(elapsed)))}")
        break

    time.sleep(5)
