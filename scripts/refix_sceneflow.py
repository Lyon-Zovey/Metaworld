#!/usr/bin/env python3
"""
Batch re-fix sceneflow for datasets_500.

For each traj dir:
  1. Decompress seg.b2nd -> seg (in memory)
  2. Decompress depth_video_int16mm_dt.b2nd -> depth (in memory)
  3. Re-run depth_to_camera_points with the FIXED y = (vv-cy)*z/fy formula
     to rebuild anchor .npy files in-place
  4. Re-run track_anchor_file_exact to produce corrected scene_point_flow_*.npy
  5. Re-compress via flow_compress.py compress --out_dir (overwrite mp4/json)
  6. Clean up intermediate .npy

Progress is printed to stdout as a tqdm bar.
"""

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import blosc2
import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Inline the fixed depth_to_camera_points (y-axis corrected)
# ---------------------------------------------------------------------------

def depth_to_camera_points_fixed(depth: np.ndarray, K: np.ndarray):
    H, W = depth.shape
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v, indexing="xy")

    z_mm = depth.reshape(-1).astype(np.float32)
    valid = (z_mm > 0) & np.isfinite(z_mm)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32), valid

    uu = uu.reshape(-1)[valid].astype(np.float32)
    vv = vv.reshape(-1)[valid].astype(np.float32)
    z_m = z_mm[valid]

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    x = (uu - cx) * z_m / fx
    y = (vv - cy) * z_m / fy   # FIXED: was (cy - vv)
    z_cam = -z_m
    return np.stack([x, y, z_cam], axis=1), valid


# ---------------------------------------------------------------------------
# Tracking logic (copied from convert_camera_depths.py)
# ---------------------------------------------------------------------------

def quat_to_rot_matrix(quat: np.ndarray) -> np.ndarray:
    q = quat
    if q.ndim == 1:
        q = q[None, ...]
    w, x, y, z = q[...,0], q[...,1], q[...,2], q[...,3]
    rot = np.empty((q.shape[0], 3, 3), dtype=np.float32)
    rot[:,0,0] = 1-2*(y*y+z*z); rot[:,0,1] = 2*(x*y-w*z); rot[:,0,2] = 2*(x*z+w*y)
    rot[:,1,0] = 2*(x*y+w*z);   rot[:,1,1] = 1-2*(x*x+z*z); rot[:,1,2] = 2*(y*z-w*x)
    rot[:,2,0] = 2*(x*z-w*y);   rot[:,2,1] = 2*(y*z+w*x); rot[:,2,2] = 1-2*(x*x+y*y)
    return rot[0] if rot.shape[0] == 1 else rot


def find_h5_with_id_poses(folder: Path):
    for p in folder.glob("*.h5"):
        try:
            with h5py.File(p, "r") as f:
                for k in f.keys():
                    g = f[k]
                    if isinstance(g, h5py.Group) and "id_poses" in g.keys():
                        return p, k
        except Exception:
            continue
    return None, None


def load_tracking_context_from_arrays(seg_all: np.ndarray, folder: Path):
    if seg_all.ndim == 2:
        seg_all = seg_all[None, ...]
    pixel_shape = (int(seg_all.shape[1]), int(seg_all.shape[2]))

    h5_path, traj_group = find_h5_with_id_poses(folder)
    if h5_path is None:
        raise FileNotFoundError(f"No .h5 with id_poses in {folder}")

    sid_data = {}
    T = None
    with h5py.File(h5_path, "r") as f:
        traj = f[traj_group]
        id_poses = traj["id_poses"]
        for key in id_poses.keys():
            g = id_poses[key]
            if not isinstance(g, h5py.Group):
                continue
            if "camera_position" in g and "camera_quaternion" in g:
                pos_arr = np.asarray(g["camera_position"])
                quat_arr = np.asarray(g["camera_quaternion"])
            elif "position" in g and "quaternion" in g:
                pos_arr = np.asarray(g["position"])
                quat_arr = np.asarray(g["quaternion"])
            else:
                continue
            if pos_arr.ndim == 3 and pos_arr.shape[1] > 1:
                pos_arr = pos_arr[:, 0, :]
            if quat_arr.ndim == 3 and quat_arr.shape[1] > 1:
                quat_arr = quat_arr[:, 0, :]
            pos_arr = pos_arr.astype(np.float32)
            quat_arr = quat_arr.astype(np.float32)
            if T is None:
                T = int(pos_arr.shape[0])
            sid_data[str(int(key))] = {
                "pos": pos_arr,
                "rot": quat_to_rot_matrix(quat_arr),
            }
    if T is None:
        raise RuntimeError("Cannot determine T from id_poses")
    return {"seg_all": seg_all, "pixel_shape": pixel_shape, "sid_data": sid_data, "T": T}


def track_anchor(anchor_hw: np.ndarray, anchor_idx: int, context: dict) -> np.ndarray:
    seg_all   = context["seg_all"]
    pixel_shape = context["pixel_shape"]
    sid_data  = context["sid_data"]
    T         = context["T"]
    H, W      = pixel_shape

    a = max(0, min(anchor_idx, int(seg_all.shape[0]) - 1))
    seg_flat    = seg_all[a].reshape(-1)
    unique_sids = np.unique(seg_flat)

    pts = anchor_hw.reshape(-1, 3)
    N   = pts.shape[0]

    p_local = np.full((N, 3), np.nan, dtype=np.float32)
    for sid in unique_sids:
        if sid == 0:
            continue
        sid_str = str(int(sid))
        if sid_str not in sid_data:
            continue
        pose = sid_data[sid_str]
        ref_i = min(anchor_idx, int(pose["pos"].shape[0]) - 1)
        pos_a = pose["pos"][ref_i]
        R_a   = pose["rot"][ref_i] if pose["rot"].ndim == 3 else pose["rot"]
        T_a   = np.eye(4, dtype=np.float32)
        T_a[:3,:3] = R_a; T_a[:3,3] = pos_a
        T_a_inv = np.linalg.inv(T_a)
        mask = seg_flat == sid
        if not np.any(mask):
            continue
        pts_sel = pts[mask]
        homo = np.concatenate([pts_sel, np.ones((pts_sel.shape[0],1), dtype=np.float32)], axis=1)
        p_local[mask] = (T_a_inv @ homo.T).T[:, :3]

    frames = np.zeros((T, N, 3), dtype=np.float32)
    bg_mask = seg_flat == 0
    if np.any(bg_mask):
        frames[:, bg_mask, :] = pts[bg_mask][None, :, :]

    for sid in unique_sids:
        if sid == 0:
            continue
        sid_str = str(int(sid))
        mask = seg_flat == sid
        if sid_str not in sid_data or not np.any(mask):
            continue
        pose = sid_data[sid_str]
        R_ts = pose["rot"]; t_ts = pose["pos"]
        local_sel = p_local[mask]
        if np.isnan(local_sel).all():
            continue
        local_h = np.concatenate([local_sel, np.ones((local_sel.shape[0],1), dtype=np.float32)], axis=1)
        for t in range(T):
            R    = R_ts[t] if R_ts.ndim == 3 else R_ts
            tvec = t_ts[t]
            Tt   = np.eye(4, dtype=np.float32)
            Tt[:3,:3] = R; Tt[:3,3] = tvec
            frames[t, mask, :] = (Tt @ local_h.T).T[:, :3]

    return frames.reshape((T, H, W, 3))


# ---------------------------------------------------------------------------
# Decode depth b2nd
# ---------------------------------------------------------------------------

def decode_depth_b2nd(folder: Path) -> np.ndarray:
    meta_path = folder / "depth_video_int16mm_dt.meta.json"
    b2nd_path = folder / "depth_video_int16mm_dt.b2nd"
    meta  = json.loads(meta_path.read_text())
    scale = meta.get("scale", 200.0)
    d_int16 = blosc2.open(str(b2nd_path))[:]
    # XOR delta decode
    for t in range(1, d_int16.shape[0]):
        d_int16[t] ^= d_int16[t-1]
    return (d_int16.astype(np.float32) / scale)  # metres


# ---------------------------------------------------------------------------
# Per-folder worker
# ---------------------------------------------------------------------------

def process_folder(folder_str: str) -> tuple:
    folder = Path(folder_str)
    try:
        # Check required files
        b2nd_seg   = folder / "seg.b2nd"
        b2nd_depth = folder / "depth_video_int16mm_dt.b2nd"
        intr_path  = folder / "cam_intrinsics.npy"

        if not b2nd_seg.exists() or not b2nd_depth.exists() or not intr_path.exists():
            return folder_str, "skip_missing", 0.0

        # Find anchor files
        anchor_files = sorted(folder.glob("scene_point_flow_ref*.anchor.npy"))
        if not anchor_files:
            return folder_str, "skip_no_anchors", 0.0

        t0 = time.time()

        # 1. Decode seg and depth
        seg = blosc2.open(str(b2nd_seg))[:]
        depth = decode_depth_b2nd(folder)
        K     = np.load(intr_path)

        T_frames, H, W = depth.shape

        # 2. Build tracking context
        ctx = load_tracking_context_from_arrays(seg, folder)

        # 3. For each anchor: rebuild anchor.npy + track + save .npy
        npy_paths = []
        for anchor_path in anchor_files:
            # Parse ref index from filename
            stem = anchor_path.stem  # scene_point_flow_ref00000.anchor
            ref_str = stem.replace("scene_point_flow_ref", "").replace(".anchor", "")
            ref_idx = int(ref_str)

            frame_depth = depth[ref_idx]
            pts_cam, valid = depth_to_camera_points_fixed(frame_depth, K)

            anchor_hw = np.zeros((H, W, 3), dtype=np.float32)
            anchor_hw.reshape(-1, 3)[valid] = pts_cam

            # Overwrite anchor.npy with corrected values
            np.save(str(anchor_path), anchor_hw)

            # Track and save .npy
            flow = track_anchor(anchor_hw, ref_idx, ctx)
            npy_path = anchor_path.with_name(anchor_path.name.replace(".anchor.npy", ".npy"))
            np.save(str(npy_path), flow)
            npy_paths.append(npy_path)

        # 4. Re-compress via flow_compress.compress_out_dir (suppress its stdout)
        sys.path.insert(0, str(Path(__file__).parent.parent / "rbs_sceneflow_scripts" / "traj2sceneflow"))
        from flow_compress import compress_out_dir
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            compress_out_dir(folder)

        # 5. Clean up intermediate .npy
        for p in npy_paths:
            if p.exists():
                p.unlink()

        elapsed = time.time() - t0
        return folder_str, "ok", elapsed

    except Exception as e:
        return folder_str, f"error: {e}", 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch re-fix sceneflow Y-axis bug")
    parser.add_argument("root", help="Root dir, e.g. datasets_500")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--task", default=None, help="Only process this task, e.g. assembly-v3")
    args = parser.parse_args()

    root = Path(args.root)

    # Collect all traj dirs (flat: root/task/camera_data/traj_N)
    folders = []
    task_dirs = sorted(root.iterdir()) if args.task is None else [root / args.task]
    for task_dir in task_dirs:
        cam_data = task_dir / "camera_data"
        if not cam_data.is_dir():
            continue
        for traj_dir in sorted(cam_data.iterdir()):
            if traj_dir.is_dir() and (traj_dir / "seg.b2nd").exists():
                folders.append(str(traj_dir))

    total = len(folders)
    print(f"Found {total} traj dirs to process  (workers={args.workers})")
    if total == 0:
        return

    done = 0
    errors = []
    skipped = 0
    t_start = time.time()

    try:
        from tqdm import tqdm
        pbar = tqdm(total=total, unit="traj", dynamic_ncols=True)
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        print(f"{'Progress':>10}  {'Done':>6}  {'Skip':>5}  {'Err':>4}  {'Elapsed':>8}  {'ETA':>8}  Speed")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(process_folder, f): f for f in folders}
        for fut in as_completed(futures):
            folder_str, status, elapsed = fut.result()
            done += 1
            if status == "ok":
                pass
            elif status.startswith("skip"):
                skipped += 1
            else:
                errors.append((folder_str, status))

            elapsed_total = time.time() - t_start
            speed = done / elapsed_total if elapsed_total > 0 else 0
            remaining = total - done
            eta = remaining / speed if speed > 0 else 0

            if use_tqdm:
                pbar.update(1)
                pbar.set_postfix(
                    skip=skipped, err=len(errors),
                    speed=f"{speed:.1f}/s", eta=f"{eta/60:.1f}min"
                )
            else:
                pct = 100 * done / total
                print(f"\r{pct:9.1f}%  {done:6d}  {skipped:5d}  {len(errors):4d}  "
                      f"{elapsed_total:7.0f}s  {eta/60:7.1f}m  {speed:.2f}/s",
                      end="", flush=True)

    if use_tqdm:
        pbar.close()
    else:
        print()

    elapsed_total = time.time() - t_start
    print(f"\n=== Done ===")
    print(f"  Total    : {total}")
    print(f"  OK       : {total - skipped - len(errors)}")
    print(f"  Skipped  : {skipped}")
    print(f"  Errors   : {len(errors)}")
    print(f"  Time     : {elapsed_total/60:.1f} min")
    if errors:
        print(f"\nFirst 10 errors:")
        for f, e in errors[:10]:
            print(f"  {f}: {e}")


if __name__ == "__main__":
    main()
