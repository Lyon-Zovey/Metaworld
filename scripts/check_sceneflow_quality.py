#!/usr/bin/env python3
"""
Check sceneflow anchor quality across all datasets_500 trajs.

Method: for each traj, pick the body with most anchor pixels,
compare anchor mean (OpenGL cam coords) vs h5 camera_position[0] (OpenCV cam coords).
If anchor is correct OpenGL: anchor_mean * [1,-1,-1] should closely match camera_position[0].
Cosine similarity < 0.9 flags the traj as bad.

Output: CSV + per-camera summary.
"""
import numpy as np
import h5py
import blosc2
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict
import csv
import sys

FLIP3 = np.array([1., -1., -1.], dtype=np.float32)
ROOT = Path("/mnt2/liangzhuowei/Metaworld/datasets_500")
BAD_THRESHOLD = 0.9


def check_traj(traj_dir: Path):
    traj_dir = Path(traj_dir)
    try:
        cam = open(str(traj_dir / "camera_name.txt")).read().strip()
        anchor = np.load(str(sorted(traj_dir.glob("scene_point_flow_ref*.anchor.npy"))[0]))
        seg = blosc2.open(str(traj_dir / "seg.b2nd"))[0]  # only frame 0
        h5_path = sorted(traj_dir.glob("*.h5"))[0]

        anchor_flat = anchor.reshape(-1, 3)
        seg_flat = seg.reshape(-1)

        best_cos = -2.0
        best_sid = None

        with h5py.File(str(h5_path), "r") as f:
            grp = f[list(f.keys())[0]]
            id_poses = grp["id_poses"]
            for sid_str in id_poses.keys():
                cp0 = id_poses[sid_str]["camera_position"][0].astype(np.float32)
                norm_cp = np.linalg.norm(cp0)
                if norm_cp < 0.1 or norm_cp > 10:
                    continue
                mask = seg_flat == int(sid_str)
                if mask.sum() < 20:
                    continue
                pts = anchor_flat[mask]
                valid = np.isfinite(pts).all(1) & (pts != 0).any(1)
                if valid.sum() < 10:
                    continue
                anchor_cv = pts[valid].mean(0) * FLIP3
                na = np.linalg.norm(anchor_cv)
                if na < 1e-4:
                    continue
                cos = float(np.dot(anchor_cv, cp0) / (na * norm_cp))
                if cos > best_cos:
                    best_cos = cos
                    best_sid = sid_str

        task = traj_dir.parent.parent.name
        traj = traj_dir.name
        return task, traj, cam, round(best_cos, 4), best_sid
    except Exception as e:
        return None


def main():
    traj_dirs = []
    for task_dir in sorted(ROOT.iterdir()):
        cam_data = task_dir / "camera_data"
        if not cam_data.exists():
            continue
        for t in sorted(cam_data.iterdir()):
            if (t / "cam2world.npy").exists() and (t / "seg.b2nd").exists():
                traj_dirs.append(t)

    print(f"Scanning {len(traj_dirs)} trajs with {mp.cpu_count()} workers...", flush=True)

    with mp.Pool(processes=min(16, mp.cpu_count())) as pool:
        results = pool.map(check_traj, traj_dirs)

    results = [r for r in results if r is not None and r[3] > -2]

    # Write CSV
    out_csv = Path("/mnt2/liangzhuowei/Metaworld/sceneflow_quality.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "traj", "camera", "cos_sim", "best_sid", "status"])
        for task, traj, cam, cos, sid in sorted(results):
            status = "OK" if cos >= BAD_THRESHOLD else "BAD"
            w.writerow([task, traj, cam, cos, sid, status])
    print(f"Written: {out_csv}")

    # Per-camera summary
    cam_vals = defaultdict(list)
    bad_by_cam = defaultdict(list)
    for task, traj, cam, cos, sid in results:
        cam_vals[cam].append(cos)
        if cos < BAD_THRESHOLD:
            bad_by_cam[cam].append(f"{task}/{traj} (cos={cos})")

    print("\n=== Per-camera summary ===")
    for cam in sorted(cam_vals):
        vals = cam_vals[cam]
        n_bad = len(bad_by_cam[cam])
        print(f"{cam:10s}: n={len(vals):5d}  bad={n_bad:5d} ({100*n_bad/len(vals):.0f}%)  "
              f"cos mean={np.mean(vals):.3f}  min={np.min(vals):.3f}")

    total_bad = sum(len(v) for v in bad_by_cam.values())
    print(f"\n=== BAD trajs total: {total_bad} / {len(results)} "
          f"({100*total_bad/len(results):.1f}%) ===")
    for cam in sorted(bad_by_cam):
        print(f"\n  {cam} ({len(bad_by_cam[cam])} bad):")
        for x in bad_by_cam[cam][:10]:
            print(f"    {x}")
        if len(bad_by_cam[cam]) > 10:
            print(f"    ... +{len(bad_by_cam[cam])-10} more")


if __name__ == "__main__":
    main()
