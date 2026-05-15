#!/usr/bin/env python3
"""Align MIKASA per-traj outputs with Metaworld's pose/mesh/target-mask layout.

Produces, per traj dir:
  pose_<obj>.npy        (T, 4, 4) float32   body-to-world (column-vec)
  mesh_<obj>.ply        binary little-endian PLY in body-local frame
  mask_<obj>.npz        (T, H, W) uint8 strict binary {0,255}

and writes meta.json with the same section ordering as Metaworld:
  task_id, traj_name, actors, links, target_object, target_obj_mask,
  body_poses, meshes

Mesh sources (per actor):
  geometry_params_json["collision_shapes"] preferred (parametric, small);
  falls back to render_shapes if no usable collision primitive.
  Triangle/ConvexMesh shapes that need an external .glb/.stl are skipped
  with a warning since we're only exporting target objects.

Naming:
  actor:red_ball[env0]                       -> red_ball
  link:panda_wristcam/panda_link0[env0]      -> panda_wristcam_panda_link0
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import subprocess
import sys
import traceback
from pathlib import Path

import blosc2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation


_TRAJ_RE = re.compile(r"^traj_(\d+)(?:_.+)?$")
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


# Default per-task target spec. Add new entries here as we expand to other tasks.
DEFAULT_TARGET_SPEC: dict[str, list[str]] = {
    "InterceptFast-v0": ["red_ball"],
    "InterceptMedium-v0": ["red_ball"],
    "InterceptGrabFast-v0": ["red_ball"],
    "InterceptGrabMedium-v0": ["red_ball"],
    "InterceptGrabSlow-v0": ["red_ball"],
    "ShellGamePush-v0": ["red_ball", "025_mug-left-0", "025_mug-center-0", "025_mug-right-0"],
    "ShellGameTouch-v0": ["red_ball", "025_mug-left-0", "025_mug-center-0", "025_mug-right-0"],
    "RotateStrictPosNeg-v0": ["Blue rectangular"],
}


# ---------------------------------------------------------------------------
# Name handling
# ---------------------------------------------------------------------------

def strip_actor_name(raw: str) -> str:
    """Normalize actor names to short names without prefixes/suffixes."""
    s = raw
    if s.startswith("actor:"):
        s = s[len("actor:"):]
    elif s.startswith("link:"):
        s = s[len("link:"):]
    if s.startswith("body:"):
        s = s[len("body:"):]
    s = re.sub(r"\[env\d+\]$", "", s)
    return s


def safe_name(short: str) -> str:
    return _SAFE_NAME.sub("_", short)


# ---------------------------------------------------------------------------
# Pose
# ---------------------------------------------------------------------------

def build_poses_world(positions: np.ndarray, quats_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quats_wxyz, dtype=np.float64)
    xyzw = np.stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]], axis=1)
    R = Rotation.from_quat(xyzw).as_matrix().astype(np.float32)
    T = positions.shape[0]
    poses = np.zeros((T, 4, 4), dtype=np.float32)
    poses[:, :3, :3] = R
    poses[:, :3, 3] = positions.astype(np.float32)
    poses[:, 3, 3] = 1.0
    return poses


# ---------------------------------------------------------------------------
# Primitive mesh generators (vertices in actor-local frame)
# ---------------------------------------------------------------------------

def box_mesh(half: list[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hx, hy, hz = (float(x) for x in half[:3])
    V = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ], dtype=np.float64)
    F = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ], dtype=np.int32)
    return V, F


def icosphere(subdiv: int = 1) -> tuple[np.ndarray, np.ndarray]:
    t = (1.0 + np.sqrt(5.0)) / 2.0
    V = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=np.float64)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    F = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int32)
    for _ in range(subdiv):
        edge_cache: dict[tuple[int, int], int] = {}
        V_list = V.tolist()
        new_F: list[list[int]] = []

        def midpoint(a: int, b: int) -> int:
            k = (a, b) if a < b else (b, a)
            if k in edge_cache:
                return edge_cache[k]
            mid = (V[a] + V[b]) / 2.0
            mid /= np.linalg.norm(mid)
            V_list.append(mid.tolist())
            idx = len(V_list) - 1
            edge_cache[k] = idx
            return idx

        for tri in F:
            a, b, c = (int(x) for x in tri)
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_F.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]])
        V = np.asarray(V_list, dtype=np.float64)
        F = np.asarray(new_F, dtype=np.int32)
    return V, F


def sphere_mesh(radius: float) -> tuple[np.ndarray, np.ndarray]:
    V, F = icosphere(subdiv=2)
    return V * float(radius), F


def cylinder_mesh(radius: float, half_length: float, segs: int = 24) -> tuple[np.ndarray, np.ndarray]:
    r = float(radius)
    h = float(half_length)
    angles = np.linspace(0.0, 2.0 * np.pi, segs, endpoint=False)
    cx, cy = np.cos(angles) * r, np.sin(angles) * r
    bottom = np.stack([cx, cy, -np.full(segs, h)], axis=1)
    top = np.stack([cx, cy, np.full(segs, h)], axis=1)
    bot_c = np.array([[0.0, 0.0, -h]])
    top_c = np.array([[0.0, 0.0, h]])
    V = np.concatenate([bottom, top, bot_c, top_c], axis=0)
    bc, tc = 2 * segs, 2 * segs + 1
    F: list[list[int]] = []
    for i in range(segs):
        j = (i + 1) % segs
        F.append([i, j, segs + j])
        F.append([i, segs + j, segs + i])
        F.append([bc, j, i])
        F.append([tc, segs + i, segs + j])
    return V, np.asarray(F, dtype=np.int32)


def capsule_mesh(radius: float, half_length: float, segs: int = 16,
                 hemi_rings: int = 4) -> tuple[np.ndarray, np.ndarray]:
    r = float(radius)
    h = float(half_length)
    cyl_V, cyl_F_full = cylinder_mesh(r, h, segs=segs)
    cyl_V = cyl_V[: 2 * segs]
    cyl_F = np.asarray(
        [tri.tolist() for tri in cyl_F_full
         if 2 * segs not in tri and 2 * segs + 1 not in tri],
        dtype=np.int32,
    )

    def hemisphere(top: bool) -> tuple[np.ndarray, np.ndarray]:
        thetas = np.linspace(0.0, np.pi / 2.0, hemi_rings + 1)[1:]
        phis = np.linspace(0.0, 2.0 * np.pi, segs, endpoint=False)
        verts: list[list[float]] = []
        for th in thetas:
            zr = np.cos(th) * r
            rr = np.sin(th) * r
            for ph in phis:
                verts.append([rr * np.cos(ph), rr * np.sin(ph),
                              (h + zr) if top else (-h - zr)])
        verts.append([0.0, 0.0, (h + r) if top else (-h - r)])
        V = np.asarray(verts, dtype=np.float64)
        faces: list[list[int]] = []
        for k in range(hemi_rings - 1):
            for i in range(segs):
                a = k * segs + i
                b = k * segs + (i + 1) % segs
                c = (k + 1) * segs + i
                d = (k + 1) * segs + (i + 1) % segs
                if top:
                    faces.append([a, c, d]); faces.append([a, d, b])
                else:
                    faces.append([a, d, c]); faces.append([a, b, d])
        pole_idx = len(V) - 1
        last_ring = (hemi_rings - 1) * segs
        for i in range(segs):
            a = last_ring + i
            b = last_ring + (i + 1) % segs
            faces.append([a, pole_idx, b] if top else [a, b, pole_idx])
        return V, np.asarray(faces, dtype=np.int32)

    V_top, F_top = hemisphere(top=True)
    V_bot, F_bot = hemisphere(top=False)

    n_cyl = cyl_V.shape[0]
    n_top = V_top.shape[0]

    bottom_ring_cyl = np.arange(0, segs)
    top_ring_cyl = np.arange(segs, 2 * segs)

    F_top_off = F_top + n_cyl
    F_bot_off = F_bot + n_cyl + n_top

    stitch: list[list[int]] = []
    for i in range(segs):
        j = (i + 1) % segs
        a = int(top_ring_cyl[i]); b = int(top_ring_cyl[j])
        c = i + n_cyl; d = j + n_cyl
        stitch.append([a, b, d]); stitch.append([a, d, c])
    for i in range(segs):
        j = (i + 1) % segs
        a = int(bottom_ring_cyl[i]); b = int(bottom_ring_cyl[j])
        c = i + n_cyl + n_top; d = j + n_cyl + n_top
        stitch.append([a, d, b]); stitch.append([a, c, d])

    V = np.concatenate([cyl_V, V_top, V_bot], axis=0)
    F = np.concatenate([cyl_F, F_top_off, F_bot_off,
                        np.asarray(stitch, dtype=np.int32)], axis=0)
    return V, F


def _shape_to_mesh(shape: dict) -> tuple[np.ndarray, np.ndarray] | None:
    t = shape.get("shape_type", "")
    if t.endswith("Sphere"):
        return sphere_mesh(shape["radius"])
    if t.endswith("Box"):
        half = shape.get("half_size") or shape.get("halfExtent")
        if half is None:
            return None
        return box_mesh(half)
    if t.endswith("Cylinder"):
        return cylinder_mesh(shape["radius"], shape["half_length"])
    if t.endswith("Capsule"):
        return capsule_mesh(shape["radius"], shape["half_length"])
    return None


def build_actor_mesh(geometry_params_json: str, obj_name: str
                     ) -> tuple[np.ndarray, np.ndarray] | None:
    """Concatenate all primitive parts into one (V, F) in actor-local frame."""
    try:
        gp = json.loads(geometry_params_json) if geometry_params_json else {}
    except json.JSONDecodeError:
        return None

    parts: list[tuple[np.ndarray, np.ndarray]] = []
    skipped: list[str] = []
    for src in ("collision_shapes", "render_shapes"):
        for shape in gp.get(src, []):
            r = _shape_to_mesh(shape)
            if r is None:
                skipped.append(shape.get("shape_type", "?"))
                continue
            parts.append(r)
        if parts:
            break  # prefer collision; only fall through to render if no collision parts produced

    if not parts:
        if skipped:
            print(f"  [warn] {obj_name}: no parametric primitives "
                  f"(only {sorted(set(skipped))}); skip mesh.", file=sys.stderr)
        return None

    V_all: list[np.ndarray] = []
    F_all: list[np.ndarray] = []
    offset = 0
    for V, F in parts:
        V_all.append(V.astype(np.float32))
        F_all.append((F + offset).astype(np.int32))
        offset += V.shape[0]
    return np.concatenate(V_all, axis=0), np.concatenate(F_all, axis=0)


def write_ply_binary(path: Path, V: np.ndarray, F: np.ndarray) -> None:
    V = np.ascontiguousarray(V, dtype=np.float32)
    F = np.ascontiguousarray(F, dtype=np.int32)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {V.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {F.shape[0]}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    face_dtype = np.dtype([("n", "u1"), ("idx", "<i4", (3,))])
    face_arr = np.empty(F.shape[0], dtype=face_dtype)
    face_arr["n"] = 3
    face_arr["idx"] = F
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(V.tobytes(order="C"))
        f.write(face_arr.tobytes(order="C"))


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Per-traj processor
# ---------------------------------------------------------------------------

def find_traj_h5(traj_dir: Path) -> Path | None:
    cands = sorted(traj_dir.glob("traj_*.h5"))
    if not cands:
        return None
    if len(cands) > 1:
        raise RuntimeError(f"ambiguous traj_*.h5 under {traj_dir}: {cands}")
    return cands[0]


def process_one_traj(traj_dir: Path,
                     target_short_names: list[str],
                     overwrite: bool,
                     finalize: bool) -> tuple[Path, str]:
    h5_path = find_traj_h5(traj_dir)
    if h5_path is None:
        return traj_dir, "skip:no_h5"
    seg_b2nd = traj_dir / "seg.b2nd"
    rgb_mp4 = traj_dir / "rgb.mp4"
    meta_path = traj_dir / "meta.json"
    legacy_json = traj_dir / "traj_task.json"

    if legacy_json.exists():
        existing = json.loads(legacy_json.read_text())
    elif meta_path.exists():
        existing = json.loads(meta_path.read_text())
    else:
        existing = {}

    actors_list = existing.get("actors", [])
    links_list = existing.get("links", [])
    task_id = existing.get("task_id", "")
    traj_name = existing.get("traj_name", traj_dir.name)

    matched: list[tuple[int, str, str]] = []  # (seg_id, short_name, raw_name)
    with h5py.File(str(h5_path), "r") as f:
        traj_keys = [k for k in f.keys() if k.startswith("traj_")]
        if len(traj_keys) != 1:
            raise RuntimeError(f"{h5_path}: expected 1 traj_* group, got {traj_keys}")
        grp = f[traj_keys[0]]
        if "id_poses" not in grp:
            raise RuntimeError(f"{h5_path}: missing id_poses/")
        id_grp = grp["id_poses"]

        wanted = set(target_short_names)
        seen_short: set[str] = set()
        T_ref: int | None = None

        bp_files: dict[str, str] = {}
        bp_seg_ids: dict[str, int] = {}
        ms_files: dict[str, str] = {}
        ms_seg_ids: dict[str, int] = {}
        ms_nverts: dict[str, int] = {}
        ms_nfaces: dict[str, int] = {}

        bids_sorted = sorted((int(k) for k in id_grp.keys()))
        for bid in bids_sorted:
            sub = id_grp[str(bid)]
            raw = sub.attrs.get("name", f"id:{bid}")
            if isinstance(raw, bytes):
                raw = raw.decode()
            short = strip_actor_name(str(raw))
            if short not in wanted or short in seen_short:
                continue
            seen_short.add(short)
            obj_name = safe_name(short) or f"actor_{bid}"

            pos = sub["position"][:]
            quat = sub["quaternion"][:]
            if T_ref is None:
                T_ref = int(pos.shape[0])
            elif pos.shape[0] != T_ref:
                raise RuntimeError(
                    f"{h5_path} actor {bid}: T mismatch {pos.shape[0]} vs {T_ref}"
                )

            out_npy = traj_dir / f"pose_{obj_name}.npy"
            if overwrite or not out_npy.exists():
                np.save(str(out_npy), build_poses_world(pos, quat))
            bp_files[obj_name] = out_npy.name
            bp_seg_ids[obj_name] = int(bid)
            matched.append((int(bid), short, str(raw)))

            gp_attr = sub.attrs.get("geometry_params_json", "")
            if isinstance(gp_attr, bytes):
                gp_attr = gp_attr.decode()
            mesh_res = build_actor_mesh(str(gp_attr), obj_name)
            if mesh_res is not None:
                V, F = mesh_res
                out_ply = traj_dir / f"mesh_{obj_name}.ply"
                if overwrite or not out_ply.exists():
                    write_ply_binary(out_ply, V, F)
                ms_files[obj_name] = out_ply.name
                ms_seg_ids[obj_name] = int(bid)
                ms_nverts[obj_name] = int(V.shape[0])
                ms_nfaces[obj_name] = int(F.shape[0])

    if not matched:
        return traj_dir, f"skip:no_match({target_short_names})"

    target_seg_ids = sorted({bid for bid, _, _ in matched})
    target_body_names = [short for _, short, _ in matched]

    if not seg_b2nd.is_file():
        return traj_dir, "skip:no_seg_b2nd"
    if not rgb_mp4.is_file():
        return traj_dir, "skip:no_rgb_mp4"

    out_npz = traj_dir / f"mask_{safe_name(target_body_names[0]) if target_body_names else 'target'}.npz"
    if overwrite or not out_npz.exists():
        seg = blosc2.open(str(seg_b2nd))[:]
        mask = np.isin(seg, target_seg_ids)
        frames = (mask.astype(np.uint8) * 255)
        T_mask, H, W = frames.shape
        np.savez_compressed(out_npz, mask=frames)
    else:
        existing_mask = np.load(str(out_npz))["mask"]
        T_mask, H, W = existing_mask.shape

    meta = {
        "task_id": task_id,
        "traj_name": traj_name,
        "actors": actors_list,
        "links": links_list,
        "target_object": {
            "body_names": target_body_names,
            "seg_ids": target_seg_ids,
        },
        "target_obj_mask": {
            "file": out_npz.name,
            "format": "npz_uint8_binary",
            "binary_values": [0, 255],
            "num_frames": int(T_mask),
            "height": int(H),
            "width": int(W),
        },
        "body_poses": {
            "num_frames": int(T_ref) if T_ref is not None else 0,
            "format": "(T, 4, 4) float32  body-to-world homogeneous transform "
                      "(column-vec: p_world = R @ p_body + t); quaternion order (w,x,y,z).",
            "files": bp_files,
            "seg_ids": bp_seg_ids,
        },
        "meshes": {
            "format": "binary_little_endian PLY; vertices in body-local frame "
                      "(p_world = pose_<obj>[t] @ [v; 1]).",
            "files": ms_files,
            "seg_ids": ms_seg_ids,
            "num_vertices": ms_nverts,
            "num_faces": ms_nfaces,
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    if finalize and legacy_json.exists() and legacy_json != meta_path:
        legacy_json.unlink()

    return traj_dir, (
        f"ok  pose={len(bp_files)} mesh={len(ms_files)} "
        f"target={target_body_names}->{target_seg_ids}  T={T_ref}"
    )


def _worker(args):
    traj_dir, target_names, overwrite, finalize = args
    try:
        td, status = process_one_traj(Path(traj_dir), target_names, overwrite, finalize)
        return (str(td), status, None)
    except Exception as e:
        return (str(traj_dir), "ERROR", f"{e}\n{traceback.format_exc()}")


def collect_traj_dirs(env_dir: Path) -> list[Path]:
    cd = env_dir / "camera_data" if (env_dir / "camera_data").is_dir() else env_dir
    return [p for p in sorted(cd.iterdir()) if p.is_dir() and _TRAJ_RE.match(p.name)]


def resolve_target_names(env_dir: Path, override: list[str] | None) -> list[str]:
    if override:
        return [strip_actor_name(x) for x in override]
    name = env_dir.name
    name_no_suffix = re.sub(r"-\d+$", "", name)
    if name_no_suffix in DEFAULT_TARGET_SPEC:
        return DEFAULT_TARGET_SPEC[name_no_suffix]
    raise SystemExit(
        f"no target spec for env '{name_no_suffix}'; "
        f"pass --target NAME [NAME ...] explicitly"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-dir", type=Path, required=True,
                    help="MIKASA env dir (e.g. /media/.../InterceptFast-v0-256)")
    ap.add_argument("--target", nargs="*", default=None,
                    help="actor short names to export (default: lookup by env dir name)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--finalize", action="store_true",
                    help="delete legacy traj_task.json after writing meta.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-procs", type=int, default=1)
    args = ap.parse_args()

    target_names = resolve_target_names(args.env_dir, args.target)
    traj_dirs = collect_traj_dirs(args.env_dir)
    if args.limit is not None:
        traj_dirs = traj_dirs[: args.limit]
    if not traj_dirs:
        print(f"no traj dirs under {args.env_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[mikasa_align] env={args.env_dir.name}  target={target_names}  "
          f"trajs={len(traj_dirs)}  procs={args.num_procs}  finalize={args.finalize}")
    items = [(str(p), target_names, args.overwrite, args.finalize) for p in traj_dirs]

    if args.num_procs <= 1:
        results = [_worker(it) for it in items]
    else:
        with mp.Pool(processes=args.num_procs) as pool:
            results = pool.map(_worker, items)

    n_ok = n_err = n_skip = 0
    for path, status, err in results:
        if status == "ERROR":
            n_err += 1
            print(f"  [ERR ] {path}: {err.splitlines()[0]}")
        elif status.startswith("skip"):
            n_skip += 1
            print(f"  [SKIP] {path}: {status}")
        else:
            n_ok += 1
    print(f"[mikasa_align] ok={n_ok}  skip={n_skip}  err={n_err}")
    if n_err:
        sys.exit(2)


if __name__ == "__main__":
    main()
