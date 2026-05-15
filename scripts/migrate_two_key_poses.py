#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_npy(path: Path) -> np.ndarray:
    return np.load(str(path))


def save_npy(path: Path, arr: np.ndarray) -> None:
    np.save(str(path), arr.astype(np.float32))


def invert_homogeneous(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float32)
    Rt = R.T
    out[:3, :3] = Rt
    out[:3, 3] = -(Rt @ t)
    return out


def convert_pose_body_world_to_body_cam(body_world: np.ndarray, cam_world: np.ndarray) -> np.ndarray:
    cam_world_inv = np.linalg.inv(cam_world).astype(np.float32)
    return (cam_world_inv[None, ...] @ body_world).astype(np.float32)


def process_traj(traj_dir: Path, overwrite: bool = False) -> dict:
    old_cam = traj_dir / "cam_poses.npy"
    cam2world = traj_dir / "cam2world.npy"
    meta_path = traj_dir / "meta.json"

    if not old_cam.exists() and not cam2world.exists():
        return {"traj": traj_dir.name, "status": "skip:no_cam_poses"}

    if old_cam.exists() and not cam2world.exists():
        old_cam.rename(cam2world)
    elif old_cam.exists() and cam2world.exists() and overwrite:
        old_cam.unlink()
    elif old_cam.exists() and cam2world.exists() and not overwrite:
        old_cam.unlink()

    if not cam2world.exists():
        return {"traj": traj_dir.name, "status": "skip:no_cam2world"}

    abs_pose = load_npy(cam2world)
    if abs_pose.ndim != 3 or abs_pose.shape[1:] != (4, 4):
        return {"traj": traj_dir.name, "status": f"skip:bad_cam2world_shape:{abs_pose.shape}"}

    rel = np.empty_like(abs_pose, dtype=np.float32)
    T0_inv = invert_homogeneous(abs_pose[0])
    for i in range(abs_pose.shape[0]):
        rel[i] = T0_inv @ abs_pose[i]
    rel[0] = np.eye(4, dtype=np.float32)

    save_npy(traj_dir / "cam_poses.npy", rel)

    cam2world0 = abs_pose[0].astype(np.float32)
    pose_files = sorted(traj_dir.glob("pose_*.npy"))
    converted = 0
    for p in pose_files:
        body_world = load_npy(p)
        if body_world.ndim != 3 or body_world.shape[1:] != (4, 4):
            continue
        body_cam = convert_pose_body_world_to_body_cam(body_world, cam2world0)
        save_npy(p, body_cam)
        converted += 1

    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
    meta["camera_pose_layout"] = "cam2world absolute; cam_poses relative to cam0"
    meta["cam2world_file"] = "cam2world.npy"
    meta["cam_poses_file"] = "cam_poses.npy"
    meta["pose_layout"] = "body->cam"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    return {
        "traj": traj_dir.name,
        "status": "ok",
        "frames": int(abs_pose.shape[0]),
        "poses_converted": int(converted),
    }


def collect_trajs(root: Path):
    if root.is_dir() and root.name.startswith("traj_"):
        yield root
        return
    for p in sorted(root.rglob("traj_*")):
        if p.is_dir():
            yield p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    results = []
    for traj_dir in collect_trajs(root):
        results.append(process_traj(traj_dir, overwrite=args.overwrite))

    for r in results:
        print(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
