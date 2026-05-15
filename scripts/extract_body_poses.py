#!/usr/bin/env python3
"""Extract per-body 4x4 world-frame poses from existing traj_N.h5.

For each traj_N/ writes:
  pose_<obj>.npy        (T, 4, 4) float32     body-to-world homogeneous transform
                                              one file per body
and updates traj_N/meta.json with a "body_poses" section indexed by name.

<obj> is the body name with the leading "body:" prefix stripped.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
import traceback
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_name(raw: str) -> str:
    """Strip 'body:' / 'link:' prefix and replace unsafe filename chars."""
    if raw.startswith("body:"):
        raw = raw[len("body:"):]
    elif raw.startswith("link:"):
        raw = raw[len("link:"):]
    return _SAFE_NAME.sub("_", raw)


def quat_wxyz_to_R(quat_wxyz: np.ndarray) -> np.ndarray:
    """(T,4) wxyz  ->  (T,3,3) rotation matrices."""
    q = np.asarray(quat_wxyz, dtype=np.float64)
    xyzw = np.stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]], axis=1)
    return Rotation.from_quat(xyzw).as_matrix().astype(np.float32)


def build_poses_world(positions: np.ndarray, quats_wxyz: np.ndarray) -> np.ndarray:
    T = positions.shape[0]
    R = quat_wxyz_to_R(quats_wxyz)
    poses = np.zeros((T, 4, 4), dtype=np.float32)
    poses[:, :3, :3] = R
    poses[:, :3,  3] = positions.astype(np.float32)
    poses[:,  3,  3] = 1.0
    return poses


_DEFAULT_MAPPING = Path(__file__).parent / "target_objects.json"


def _resolve_target_bodies(traj_dir: Path, mapping: dict) -> set[str]:
    """Return target body names from target_objects.json, including child_of expansions."""
    task_id = traj_dir.parent.parent.name
    if task_id not in mapping:
        return set()
    entry = mapping[task_id]
    target = set(entry.get("target_bodies", []))
    child_of = set(entry.get("child_of", []))
    if not child_of:
        return target

    # expand child_of: include the anonymous body immediately following the parent in actors
    for src in [traj_dir / "meta.json", traj_dir / "traj_task.json"]:
        if src.exists():
            actors = json.loads(src.read_text()).get("actors", [])
            for i, a in enumerate(actors):
                if not a["name"].startswith("body:"):
                    continue
                short = a["name"][len("body:"):]
                if short in child_of and i + 1 < len(actors):
                    nxt = actors[i + 1]
                    if nxt["name"].startswith("body:"):
                        target.add(nxt["name"][len("body:"):])
            break
    return target


def find_traj_h5(traj_dir: Path) -> Path:
    cands = sorted(traj_dir.glob("traj_*.h5"))
    if not cands:
        raise FileNotFoundError(f"no traj_*.h5 under {traj_dir}")
    if len(cands) > 1:
        raise RuntimeError(f"ambiguous traj_*.h5 under {traj_dir}: {cands}")
    return cands[0]


def process_one_traj(traj_dir: Path, overwrite: bool = False, target_bodies=None) -> tuple[Path, str]:
    h5_path = find_traj_h5(traj_dir)
    meta_path = traj_dir / "meta.json"

    with h5py.File(str(h5_path), "r") as f:
        traj_keys = [k for k in f.keys() if k.startswith("traj_")]
        if len(traj_keys) != 1:
            raise RuntimeError(f"{h5_path}: expected 1 traj_* group, got {traj_keys}")
        grp = f[traj_keys[0]]
        if "id_poses" not in grp:
            raise RuntimeError(f"{h5_path}: missing id_poses/")
        id_grp = grp["id_poses"]

        bids_sorted = sorted((int(k) for k in id_grp.keys()))
        files_map: dict[str, str] = {}
        seg_id_map: dict[str, int] = {}
        T_ref: int | None = None
        used_names: dict[str, int] = {}

        for bid in bids_sorted:
            sub = id_grp[str(bid)]
            raw_name = sub.attrs.get("name", f"body:{bid}")
            if isinstance(raw_name, bytes):
                raw_name = raw_name.decode()
            obj_name = safe_name(str(raw_name)) or f"body_{bid}"

            if obj_name in used_names:
                used_names[obj_name] += 1
                obj_name = f"{obj_name}_{used_names[obj_name]}"
            else:
                used_names[obj_name] = 0

            pos  = sub["position"][:]
            quat = sub["quaternion"][:]
            if T_ref is None:
                T_ref = pos.shape[0]
            elif pos.shape[0] != T_ref:
                raise RuntimeError(
                    f"{h5_path} body {bid}: T mismatch {pos.shape[0]} vs {T_ref}"
                )

            out_npy = traj_dir / f"pose_{obj_name}.npy"
            if target_bodies and obj_name not in target_bodies:
                continue
            if out_npy.exists() and not overwrite:
                continue
            poses = build_poses_world(pos, quat)
            np.save(str(out_npy), poses)

            files_map[obj_name] = out_npy.name
            seg_id_map[obj_name] = int(bid)

    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {}
    meta["body_poses"] = {
        "num_frames": int(T_ref) if T_ref is not None else 0,
        "format": "(T, 4, 4) float32  body-to-world homogeneous transform "
                  "(column-vec: p_world = R @ p_body + t); quaternion order (w,x,y,z).",
        "files":   files_map,
        "seg_ids": seg_id_map,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    return traj_dir, f"ok  bodies={len(files_map)}  T={T_ref}"


def _worker(args):
    traj_dir, overwrite, mapping = args
    try:
        target_bodies = _resolve_target_bodies(Path(traj_dir), mapping)
        td, status = process_one_traj(Path(traj_dir), overwrite=overwrite, target_bodies=target_bodies or None)
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
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--num-procs", type=int, default=1)
    ap.add_argument("--mapping", type=Path, default=_DEFAULT_MAPPING,
                    help="target_objects.json; omit to extract all bodies")
    args = ap.parse_args()

    if args.traj_dir is not None:
        traj_dirs = [args.traj_dir]
    elif args.env_dir is not None:
        traj_dirs = collect_traj_dirs(args.env_dir)
    else:
        traj_dirs = collect_traj_dirs(args.root)

    if not traj_dirs:
        print("[extract_body_poses] no traj_N dirs found", file=sys.stderr)
        sys.exit(1)

    print(f"[extract_body_poses] {len(traj_dirs)} trajectory dir(s)  procs={args.num_procs}")
    mapping = json.loads(args.mapping.read_text()) if args.mapping and args.mapping.exists() else {}
    items = [(str(p), args.overwrite, mapping) for p in traj_dirs]

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
    print(f"[extract_body_poses] ok={n_ok}  err={n_err}")
    if n_err:
        sys.exit(2)


if __name__ == "__main__":
    main()
