#!/usr/bin/env python3
"""
Produce all camera-pose and object-pose variants for a trajectory directory.

Input (must exist beforehand):
  cam_poses.npy        (T, 4, 4) float32  cam-to-world, OpenGL/RUB, absolute
  pose_<obj>.npy       (T, 4, 4) float32  body-to-world, MuJoCo world frame
                                          (written by extract_body_poses.py)

Output:
  cam2world_gl.npy     (T, 4, 4)  cam-to-world, OpenGL/RUB   <- rename of cam_poses.npy
  cam2world_cv.npy     (T, 4, 4)  cam-to-world, OpenCV/RDF
  cam_poses_gl.npy     (T, 4, 4)  relative to frame-0, OpenGL/RUB
  cam_poses_cv.npy     (T, 4, 4)  relative to frame-0, OpenCV/RDF
  pose_<obj>_world.npy (T, 4, 4)  body-to-world              <- rename of pose_<obj>.npy
  pose_<obj>_cv.npy    (T, 4, 4)  body-to-cam0, OpenCV/RDF

Convention:
  OpenGL/RUB : X right, Y up,   Z backward  (cam looks -Z)
  OpenCV/RDF : X right, Y down, Z forward   (cam looks +Z)

Math (column-vector convention):
  flip4          = diag(1, -1, -1, 1)
  cam2world_cv   = cam2world_gl  @ flip4
  cam_poses_cv   = flip4 @ cam_poses_gl @ flip4    (= inv(T0_cv) @ Ti_cv)
  pose_obj_cv[t] = inv(cam2world_cv[0]) @ pose_obj_world[t]

Usage:
  # single traj dir
  python rbs_sceneflow_scripts/build_pose_variants.py --root datasets_500_fixed_test/assembly-v3/camera_data/traj_0

  # whole task (all traj_* under camera_data/)
  python rbs_sceneflow_scripts/build_pose_variants.py --root datasets_500_fixed_test/assembly-v3/camera_data

  # whole dataset (all tasks)
  python rbs_sceneflow_scripts/build_pose_variants.py --root datasets_500_fixed_test

  # overwrite already-produced files
  python rbs_sceneflow_scripts/build_pose_variants.py --root ... --overwrite
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path

import numpy as np

_DEFAULT_MAPPING = Path(__file__).parent / "target_objects.json"

FLIP4 = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
_POSE_RE = re.compile(r"^pose_(.+)\.npy$")


def _inv44(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = R.T
    out[:3, 3] = -(R.T @ t)
    return out


def _make_relative(T_abs: np.ndarray) -> np.ndarray:
    """(T,4,4) absolute → relative to frame 0."""
    rel = (_inv44(T_abs[0])[None] @ T_abs).astype(np.float32)
    rel[0] = np.eye(4, dtype=np.float32)
    return rel


def _resolve_target_bodies(traj_dir: Path, mapping: dict) -> list[str]:
    """Return target body names for this traj from target_objects.json or meta.json.

    Priority:
      1. meta.json target_object.body_names  (already computed by generate_target_obj_mask)
      2. target_objects.json keyed by task_id inferred from path
    """
    # 1. meta.json
    meta_path = traj_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        body_names = meta.get("target_object", {}).get("body_names")
        if body_names:
            return list(body_names)

    # 2. infer task_id from path: .../camera_data/traj_N -> parent.parent.name
    task_id = traj_dir.parent.parent.name
    if task_id not in mapping:
        return []

    entry = mapping[task_id]
    target = list(entry.get("target_bodies", []))
    child_of = set(entry.get("child_of", []))

    # include the anonymous child body that follows a child_of parent in actors
    # (same logic as generate_target_obj_mask.py)
    meta_src = traj_dir / "meta.json" if (traj_dir / "meta.json").exists() else traj_dir / "traj_task.json"
    if meta_src.exists():
        actors = json.loads(meta_src.read_text()).get("actors", [])
        for i, a in enumerate(actors):
            if not a["name"].startswith("body:"):
                continue
            short = a["name"][len("body:"):]
            if short in child_of and i + 1 < len(actors):
                nxt = actors[i + 1]
                if nxt["name"].startswith("body:"):
                    child_name = nxt["name"][len("body:"):]
                    if child_name not in target:
                        target.append(child_name)

    return target


def process_traj(traj_dir: Path, overwrite: bool, mapping: dict) -> dict:
    # ── 1. locate cam2world_gl source ────────────────────────────────────────
    raw_cam = traj_dir / "cam_poses.npy"
    cam2world_gl_path = traj_dir / "cam2world_gl.npy"

    if cam2world_gl_path.exists():
        cam2world_gl = np.load(cam2world_gl_path).astype(np.float32)
    elif raw_cam.exists():
        cam2world_gl = np.load(raw_cam).astype(np.float32)
        raw_cam.rename(cam2world_gl_path)
    else:
        return {"traj": traj_dir.name, "status": "skip:no_cam_poses"}

    if cam2world_gl.ndim != 3 or cam2world_gl.shape[1:] != (4, 4):
        return {"traj": traj_dir.name, "status": f"skip:bad_shape:{cam2world_gl.shape}"}

    # ── 2. cam2world_cv ───────────────────────────────────────────────────────
    cam2world_cv_path = traj_dir / "cam2world_cv.npy"
    if overwrite or not cam2world_cv_path.exists():
        cam2world_cv = (cam2world_gl @ FLIP4).astype(np.float32)
        np.save(cam2world_cv_path, cam2world_cv)
    else:
        cam2world_cv = np.load(cam2world_cv_path).astype(np.float32)

    # ── 3. relative cam poses (gl + cv) ──────────────────────────────────────
    cam_poses_gl_path = traj_dir / "cam_poses_gl.npy"
    cam_poses_cv_path = traj_dir / "cam_poses_cv.npy"

    if overwrite or not cam_poses_gl_path.exists():
        cam_poses_gl = _make_relative(cam2world_gl)
        np.save(cam_poses_gl_path, cam_poses_gl)
    else:
        cam_poses_gl = np.load(cam_poses_gl_path).astype(np.float32)

    if overwrite or not cam_poses_cv_path.exists():
        # flip4 @ T_rel_gl @ flip4  ==  inv(T0_cv) @ Ti_cv
        cam_poses_cv = (FLIP4 @ cam_poses_gl @ FLIP4).astype(np.float32)
        np.save(cam_poses_cv_path, cam_poses_cv)

    # ── 4. object poses (target bodies only) ─────────────────────────────────
    target_bodies = set(_resolve_target_bodies(traj_dir, mapping))
    T0_cv_inv = _inv44(cam2world_cv[0])
    pose_files = [
        f for f in sorted(traj_dir.iterdir())
        if _POSE_RE.match(f.name)
        and not f.name.endswith("_world.npy")
        and not f.name.endswith("_cv.npy")
        and (not target_bodies or _POSE_RE.match(f.name).group(1) in target_bodies)
    ]

    obj_names: list[str] = []
    for pf in pose_files:
        m = _POSE_RE.match(pf.name)
        obj = m.group(1)
        obj_names.append(obj)

        world_path = traj_dir / f"pose_{obj}_world.npy"
        cv_path    = traj_dir / f"pose_{obj}_cv.npy"

        # rename pose_<obj>.npy → pose_<obj>_world.npy
        if not world_path.exists():
            pf.rename(world_path)
            pose_world = np.load(world_path).astype(np.float32)
        elif overwrite:
            pf.unlink()
            pose_world = np.load(world_path).astype(np.float32)
        else:
            pose_world = np.load(world_path).astype(np.float32)
            if pf.exists():
                pf.unlink()

        if overwrite or not cv_path.exists():
            pose_cv = (T0_cv_inv[None] @ pose_world).astype(np.float32)
            np.save(cv_path, pose_cv)

    # ── 5. meta.json ─────────────────────────────────────────────────────────
    meta_path = traj_dir / "meta.json"
    legacy_path = traj_dir / "traj_task.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    elif legacy_path.exists():
        meta = json.loads(legacy_path.read_text())
    else:
        meta = {}
    # merge traj_task.json fields (actors/links) into meta if not already present
    if legacy_path.exists() and meta_path != legacy_path:
        legacy = json.loads(legacy_path.read_text())
        for k, v in legacy.items():
            if k not in meta:
                meta[k] = v

    meta["camera_convention"] = "opencv"
    meta["camera_convention_note"] = "X right, Y down, Z forward (RDF). OpenGL source kept as cam2world_gl.npy."
    meta["flow_convention"] = "opengl"
    meta["flow_convention_note"] = (
        "scene_point_flow_ref*.mp4 and *.anchor.npy remain OpenGL/RUB; "
        "apply flip3=diag(1,-1,-1) in the dataloader."
    )
    meta["cam_pose_files"] = {
        "cam2world_gl":  "cam2world_gl.npy",
        "cam2world_cv":  "cam2world_cv.npy",
        "cam_poses_gl":  "cam_poses_gl.npy",
        "cam_poses_cv":  "cam_poses_cv.npy",
    }
    meta["cam_pose_layout"] = (
        "cam2world_*: absolute cam-to-world. "
        "cam_poses_*: relative to frame-0 (T0_inv @ Ti). "
        "_gl = OpenGL/RUB, _cv = OpenCV/RDF."
    )
    if obj_names:
        meta["object_pose_files"] = {
            obj: {
                "world": f"pose_{obj}_world.npy",
                "cv":    f"pose_{obj}_cv.npy",
            }
            for obj in obj_names
        }
        meta["object_pose_layout"] = (
            "pose_<obj>_world: body-to-world (MuJoCo world frame). "
            "pose_<obj>_cv: body-to-cam0, OpenCV/RDF."
        )

    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    if legacy_path.exists():
        legacy_path.unlink()

    return {
        "traj": traj_dir.name,
        "status": "ok",
        "T": int(cam2world_gl.shape[0]),
        "objects": obj_names,
    }


def collect_traj_dirs(root: Path) -> list[Path]:
    if root.name.startswith("traj_") and root.is_dir():
        return [root]
    # camera_data/ or task dir containing traj_* directly
    direct = sorted(p for p in root.glob("traj_*") if p.is_dir())
    if direct:
        return direct
    # dataset root containing <task>/camera_data/traj_*/
    out: list[Path] = []
    for task_dir in sorted(root.iterdir()):
        cd = task_dir / "camera_data"
        if cd.is_dir():
            out.extend(sorted(p for p in cd.glob("traj_*") if p.is_dir()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1].strip())
    ap.add_argument("--root", required=True, help="traj dir, camera_data dir, task dir, or dataset root")
    ap.add_argument("--mapping", type=Path, default=_DEFAULT_MAPPING,
                    help="target_objects.json (default: rbs_sceneflow_scripts/target_objects.json)")
    ap.add_argument("--overwrite", action="store_true", help="overwrite already-produced files")
    args = ap.parse_args()

    mapping = json.loads(args.mapping.read_text()) if args.mapping.exists() else {}

    traj_dirs = collect_traj_dirs(Path(args.root))
    if not traj_dirs:
        print(f"no traj_* directories found under {args.root}", file=sys.stderr)
        return 1

    ok = err = skip = 0
    for td in traj_dirs:
        try:
            r = process_traj(td, overwrite=args.overwrite, mapping=mapping)
        except Exception as e:
            print(f"  ERROR {td}: {e}\n{traceback.format_exc()}", file=sys.stderr)
            err += 1
            continue
        status = r["status"]
        if status == "ok":
            ok += 1
            print(f"  ok   {r['traj']}  T={r['T']}  objects={r['objects']}")
        else:
            skip += 1
            print(f"  skip {r['traj']}  reason={status}")

    print(f"\ndone: {ok} ok, {skip} skipped, {err} errors  (total {len(traj_dirs)})")
    return 0 if err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
