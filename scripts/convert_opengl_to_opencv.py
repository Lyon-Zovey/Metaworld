#!/usr/bin/env python3
"""In-place OpenGL -> OpenCV conversion for a trajectory directory.

Converts all camera-frame quantities (cam_poses, anchors, sceneflow videos,
H5 camera_position/camera_quaternion) from OpenGL convention (X right, Y up,
Z back, forward = -Z) to OpenCV convention (X right, Y down, Z forward).

Math:
    flip3 = diag(1, -1, -1)         # 3x3
    flip4 = diag(1, -1, -1, 1)      # 4x4 homogeneous, det = +1

    cam->world (column-vector form):
        T_wc_cv = T_wc_gl @ flip4
        -> translation unchanged; rotation columns 1,2 negated

    camera-frame points:
        p_cv = p_gl * [1, -1, -1]

    body->cam (column-vector form):
        T_bc_cv = flip4 @ T_bc_gl
        -> R rows 1,2 negated; t = t * [1, -1, -1]
        For quaternion (wxyz Hamilton):  q_cv = q_flip * q_gl,  q_flip=(0,1,0,0)

Run on the rebuild_test copy only -- do NOT point this at the originals.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "rbs_sceneflow_scripts" / "traj2sceneflow"))
from flow_compress import compress_one_flow, decompress_one_flow  # noqa: E402

FLIP3 = np.array([1.0, -1.0, -1.0], dtype=np.float32)


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product (q1 * q2) with quaternions stored as (w, x, y, z)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1).astype(q1.dtype)


def convert_cam_poses(traj_dir: Path) -> float:
    t0 = time.time()
    p = traj_dir / "cam_poses.npy"
    cp = np.load(p)
    cp[..., :3, 1:3] *= -1.0
    np.save(p, cp.astype(np.float32))
    return time.time() - t0


def convert_anchors(traj_dir: Path) -> float:
    t0 = time.time()
    for f in sorted(traj_dir.glob("scene_point_flow_ref*.anchor.npy")):
        a = np.load(f)
        a = a * FLIP3
        np.save(f, a.astype(np.float32))
    return time.time() - t0


def convert_h5_camera_fields(traj_dir: Path) -> float:
    t0 = time.time()
    h5_files = list(traj_dir.glob("*.h5"))
    assert len(h5_files) == 1, f"expected exactly one .h5 in {traj_dir}, got {h5_files}"
    q_flip = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    with h5py.File(h5_files[0], "r+") as f:
        traj_grp = f[list(f.keys())[0]]
        id_poses = traj_grp["id_poses"]
        for sid in id_poses:
            g = id_poses[sid]
            cp = g["camera_position"][...]
            cq = g["camera_quaternion"][...]
            cp_new = cp * FLIP3
            cq_new = quat_mul_wxyz(np.broadcast_to(q_flip, cq.shape), cq)
            g["camera_position"][...] = cp_new
            g["camera_quaternion"][...] = cq_new
    return time.time() - t0


def convert_sceneflow_videos(traj_dir: Path) -> float:
    """Decompress -> flip flow -> recompress with same codec/crf/bits."""
    t0 = time.time()
    anchors = sorted(traj_dir.glob("scene_point_flow_ref*.anchor.npy"))
    for anchor_path in anchors:
        ref_stem = anchor_path.name.replace(".anchor.npy", "")
        videos = sorted(traj_dir.glob(f"{ref_stem}_*.mp4"))
        if not videos:
            videos = sorted(traj_dir.glob(f"{ref_stem}_*.mkv")) + \
                     sorted(traj_dir.glob(f"{ref_stem}_*.webm"))
        if not videos:
            raise FileNotFoundError(f"no compressed flow video for {anchor_path}")
        if len(videos) > 1:
            raise RuntimeError(f"ambiguous flow videos for {anchor_path}: {videos}")
        vid = videos[0]
        sidecar = json.loads(vid.with_suffix(".json").read_text())

        anchor_cv = np.load(anchor_path)
        anchor_gl = anchor_cv * FLIP3
        flow_gl = decompress_one_flow(vid, anchor_gl, sidecar)
        flow_cv = (flow_gl.astype(np.float32) * FLIP3).astype(np.float32)

        codec = sidecar.get("codec", "libx265")
        crf = int(sidecar.get("crf", 0))
        bits = int(sidecar.get("bits", 10))
        suffix = sidecar.get("suffix")
        if suffix is None:
            stem = vid.stem
            assert stem.startswith(ref_stem + "_"), stem
            suffix = stem[len(ref_stem) + 1:]

        # Use the canonical ref_stem so compress_one_flow produces the
        # expected filename "<ref_stem>_<suffix>.<ext>" directly.
        tmp_npy = traj_dir / f"{ref_stem}.npy"
        np.save(tmp_npy, flow_cv)

        vid.unlink()
        vid.with_suffix(".json").unlink()

        compress_one_flow(
            tmp_npy,
            anchor_cv,
            codec=codec,
            crf=crf,
            bits=bits,
            suffix=suffix,
            delete_npy=True,
        )

        ext_map = {"libx265": ".mp4", "ffv1": ".mkv", "libvpx-vp9": ".webm"}
        produced = traj_dir / f"{ref_stem}_{suffix}{ext_map.get(codec, '.mp4')}"
        if not produced.exists():
            raise RuntimeError(f"recompressed flow video not found: {produced}")
    return time.time() - t0


def remove_sceneflow_checks(traj_dir: Path) -> int:
    removed = 0
    for f in traj_dir.glob("_sceneflow_check_*.png"):
        f.unlink()
        removed += 1
    return removed


def write_convention_marker(traj_dir: Path, skip_flow: bool = False) -> None:
    meta_path = traj_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {}
    meta["camera_convention"] = "opencv"
    meta["camera_convention_note"] = (
        "X right, Y down, Z forward. Converted from OpenGL by "
        "scripts/convert_opengl_to_opencv.py."
    )
    # Explicitly mark flow/anchor convention when doing pose-only migration.
    if skip_flow:
        meta["flow_convention"] = "opengl"
        meta["flow_convention_note"] = (
            "scene_point_flow_ref*.mp4 and scene_point_flow_ref*.anchor.npy remain OpenGL "
            "for dataloader-time conversion."
        )
    else:
        meta["flow_convention"] = "opencv"
    meta_path.write_text(json.dumps(meta, indent=2))



def convert_one_traj(traj_dir: Path, drop_checks: bool, skip_flow: bool) -> dict:
    timings = {}
    timings["cam_poses"] = convert_cam_poses(traj_dir)
    timings["h5"] = convert_h5_camera_fields(traj_dir)
    if skip_flow:
        timings["anchors"] = 0.0
        timings["sceneflow"] = 0.0
    else:
        timings["anchors"] = convert_anchors(traj_dir)
        timings["sceneflow"] = convert_sceneflow_videos(traj_dir)
    if drop_checks:
        timings["checks_removed"] = remove_sceneflow_checks(traj_dir)
    write_convention_marker(traj_dir, skip_flow=skip_flow)
    return timings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="directory containing traj_* subdirs")
    ap.add_argument("--drop-checks", action="store_true",
                    help="delete _sceneflow_check_*.png (regenerate later)")
    ap.add_argument("--skip-flow", action="store_true",
                    help="only convert cam_poses + H5 camera_* fields; keep anchors/flow videos unchanged")
    args = ap.parse_args()

    root = Path(args.root)
    traj_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("traj_")])
    if not traj_dirs:
        print(f"no traj_* under {root}")
        return 1

    print(f"Converting {len(traj_dirs)} trajectories under {root}")
    grand_total = 0.0
    per_traj = []
    for d in traj_dirs:
        t0 = time.time()
        timings = convert_one_traj(d, drop_checks=args.drop_checks, skip_flow=args.skip_flow)
        dt = time.time() - t0
        grand_total += dt
        per_traj.append((d.name, dt, timings))
        print(f"  {d.name}: total={dt:.2f}s  detail={timings}")

    print("\n=== summary ===")
    print(f"total: {grand_total:.2f}s for {len(traj_dirs)} trajs "
          f"(mean {grand_total/len(traj_dirs):.2f}s/traj)")
    if per_traj:
        keys = ["cam_poses", "anchors", "h5", "sceneflow"]
        for k in keys:
            avg = sum(t[2].get(k, 0.0) for t in per_traj) / len(per_traj)
            print(f"  avg {k}: {avg:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
