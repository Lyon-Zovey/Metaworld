#!/usr/bin/env python3
"""Fix pose_*.npy files corrupted by double application of migrate_two_key_poses.py.

Root cause:
  migrate_two_key_poses.py was run twice on data where pose_*.npy was already
  body->cam (OpenCV). The second run treated body->cam as body->world and applied
  inv(c2w[0]) again, resulting in:
      pose_wrong[t] = inv(c2w[0]) @ inv(c2w[0]) @ body_world[t]

Fix:
  pose_correct[t] = c2w[0] @ pose_wrong[t]
                  = inv(c2w[0]) @ body_world[t]   (correct body->cam)

Usage (dry run first):
    python scripts/fix_double_migrate_poses.py --root /media/home/liangzhuowei/mikasa/InterceptFast-v0-256/camera_data --dry-run

Usage (apply):
    python scripts/fix_double_migrate_poses.py --root /media/home/liangzhuowei/mikasa/InterceptFast-v0-256/camera_data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def find_traj_dirs(root: Path) -> list[Path]:
    if root.is_dir() and root.name.startswith("traj_"):
        return [root]
    return sorted(d for d in root.rglob("traj_*") if d.is_dir())


def fix_one_traj(traj_dir: Path, dry_run: bool) -> dict:
    c2w_path = traj_dir / "cam2world.npy"
    if not c2w_path.exists():
        return {"traj": str(traj_dir), "status": "skip:no_cam2world"}

    c2w = np.load(str(c2w_path))
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        return {"traj": str(traj_dir), "status": f"skip:bad_cam2world_shape:{c2w.shape}"}

    c2w0 = c2w[0].astype(np.float32)

    pose_files = sorted(p for p in traj_dir.glob("pose_*.npy") if "_cv" not in p.stem)
    if not pose_files:
        return {"traj": str(traj_dir), "status": "skip:no_pose_files"}

    results = []
    for pf in pose_files:
        pose = np.load(str(pf))
        if pose.ndim != 3 or pose.shape[1:] != (4, 4):
            results.append(f"{pf.name}:bad_shape")
            continue

        fixed = (c2w0 @ pose).astype(np.float32)

        # Sanity check: translation z of first frame should be plausible in cam space
        # (positive Z = in front of camera in OpenCV convention)
        z0_before = pose[0, 2, 3]
        z0_after  = fixed[0, 2, 3]

        if not dry_run:
            np.save(str(pf), fixed)

        results.append(
            f"{pf.name}: z[0] {z0_before:.4f} -> {z0_after:.4f}"
            + (" [DRY RUN]" if dry_run else " [SAVED]")
        )

    return {"traj": str(traj_dir), "status": "ok", "files": results}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Root directory containing traj_* subdirs (or a single traj_* dir)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be done without writing anything")
    args = ap.parse_args()

    root = Path(args.root)
    traj_dirs = find_traj_dirs(root)
    if not traj_dirs:
        print(f"No traj_* dirs found under {root}")
        return 1

    print(f"{'[DRY RUN] ' if args.dry_run else ''}Processing {len(traj_dirs)} traj dir(s) under {root}")
    print()

    n_ok = n_skip = 0
    for traj_dir in traj_dirs:
        r = fix_one_traj(traj_dir, dry_run=args.dry_run)
        status = r["status"]
        if status == "ok":
            n_ok += 1
            print(f"  {traj_dir.name}:")
            for line in r["files"]:
                print(f"    {line}")
        else:
            n_skip += 1
            print(f"  {traj_dir.name}: {status}")

    print()
    print(f"done: ok={n_ok}  skipped={n_skip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
