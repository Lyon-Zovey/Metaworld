#!/usr/bin/env python3
"""Keep only target-object pose/mesh files, delete everything else.

For each traj_N/ reads meta.json["target_object"]["seg_ids"], then:
  - Deletes pose_<obj>.npy  for non-target bodies
  - Deletes mesh_<obj>.ply  for non-target bodies
  - Trims meta.json["body_poses"] and meta.json["meshes"] to target bodies only
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import traceback
from pathlib import Path


def process_one_traj(traj_dir: Path, dry_run: bool = False) -> tuple[Path, str]:
    meta_path = traj_dir / "meta.json"
    if not meta_path.exists():
        return traj_dir, "skip (no meta.json)"

    meta = json.loads(meta_path.read_text())

    target = meta.get("target_object", {})
    target_seg_ids = set(int(s) for s in target.get("seg_ids", []))
    if not target_seg_ids:
        return traj_dir, "skip (no target_object.seg_ids)"

    n_del_pose = n_del_mesh = 0

    # ---------- trim body_poses ----------
    bp = meta.get("body_poses", {})
    bp_seg_ids: dict[str, int] = {k: int(v) for k, v in bp.get("seg_ids", {}).items()}
    bp_files: dict[str, str]   = bp.get("files", {})

    keep_pose: set[str] = {obj for obj, sid in bp_seg_ids.items() if sid in target_seg_ids}
    drop_pose: set[str] = set(bp_files.keys()) - keep_pose

    for obj in drop_pose:
        f = traj_dir / bp_files[obj]
        if f.exists():
            if not dry_run:
                f.unlink()
            n_del_pose += 1

    if "body_poses" in meta:
        meta["body_poses"]["files"]   = {k: v for k, v in bp_files.items()   if k in keep_pose}
        meta["body_poses"]["seg_ids"] = {k: v for k, v in bp_seg_ids.items() if k in keep_pose}

    # ---------- trim meshes ----------
    ms = meta.get("meshes", {})
    ms_seg_ids: dict[str, int] = {k: int(v) for k, v in ms.get("seg_ids", {}).items()}
    ms_files: dict[str, str]   = ms.get("files", {})
    ms_nverts: dict[str, int]  = ms.get("num_vertices", {})
    ms_nfaces: dict[str, int]  = ms.get("num_faces", {})

    keep_mesh: set[str] = {obj for obj, sid in ms_seg_ids.items() if sid in target_seg_ids}
    drop_mesh: set[str] = set(ms_files.keys()) - keep_mesh

    for obj in drop_mesh:
        f = traj_dir / ms_files[obj]
        if f.exists():
            if not dry_run:
                f.unlink()
            n_del_mesh += 1

    if "meshes" in meta:
        meta["meshes"]["files"]        = {k: v for k, v in ms_files.items()   if k in keep_mesh}
        meta["meshes"]["seg_ids"]      = {k: v for k, v in ms_seg_ids.items() if k in keep_mesh}
        meta["meshes"]["num_vertices"] = {k: v for k, v in ms_nverts.items()  if k in keep_mesh}
        meta["meshes"]["num_faces"]    = {k: v for k, v in ms_nfaces.items()  if k in keep_mesh}

    if not dry_run:
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    return traj_dir, (
        f"del_pose={n_del_pose}  del_mesh={n_del_mesh}  "
        f"kept_pose={len(keep_pose)}  kept_mesh={len(keep_mesh)}"
    )


def _worker(args):
    traj_dir, dry_run = args
    try:
        td, status = process_one_traj(Path(traj_dir), dry_run=dry_run)
        return (str(td), status, None)
    except Exception as e:
        return (str(traj_dir), "ERROR", f"{e}\n{traceback.format_exc()}")


def collect_traj_dirs(root: Path) -> list[Path]:
    root = root.resolve()
    if (root / "meta.json").exists() and any(root.glob("traj_*.h5")):
        return [root]
    if root.name == "camera_data":
        return [p for p in sorted(root.glob("traj_*")) if p.is_dir()]
    if (root / "camera_data").is_dir():
        return [p for p in sorted((root / "camera_data").glob("traj_*")) if p.is_dir()]
    direct = [p for p in sorted(root.glob("traj_*")) if p.is_dir()]
    if direct and any((p / "meta.json").exists() for p in direct):
        return direct
    out: list[Path] = []
    for env_dir in sorted(root.iterdir()):
        cd = env_dir / "camera_data"
        if cd.is_dir():
            out.extend(sorted(p for p in cd.glob("traj_*") if p.is_dir()))
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--traj-dir", type=Path)
    g.add_argument("--env-dir",  type=Path)
    g.add_argument("--root",     type=Path)
    ap.add_argument("--dry-run",   action="store_true",
                    help="report what would be deleted without touching files")
    ap.add_argument("--num-procs", type=int, default=1)
    args = ap.parse_args()

    if args.traj_dir is not None:
        traj_dirs = [args.traj_dir]
    elif args.env_dir is not None:
        traj_dirs = collect_traj_dirs(args.env_dir)
    else:
        traj_dirs = collect_traj_dirs(args.root)

    if not traj_dirs:
        print("[prune_to_target] no traj dirs found", file=sys.stderr)
        sys.exit(1)

    mode = "DRY-RUN" if args.dry_run else "LIVE"
    print(f"[prune_to_target] {len(traj_dirs)} traj dir(s)  procs={args.num_procs}  mode={mode}")
    items = [(str(p), args.dry_run) for p in traj_dirs]

    if args.num_procs <= 1:
        results = [_worker(it) for it in items]
    else:
        with mp.Pool(processes=args.num_procs) as pool:
            results = pool.map(_worker, items)

    n_ok = n_err = 0
    for path, status, err in results:
        if status == "ERROR":
            n_err += 1
            print(f"  [ERR] {path}: {err.splitlines()[0]}")
        else:
            n_ok += 1
    print(f"[prune_to_target] ok={n_ok}  err={n_err}")
    if n_err:
        sys.exit(2)


if __name__ == "__main__":
    main()
