#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def read_ply_vertices_faces(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path}: bad ply header")
            s = line.decode("ascii").strip()
            header.append(s)
            if s == "end_header":
                break

        n_verts = None
        n_faces = None
        for s in header:
            if s.startswith("element vertex "):
                n_verts = int(s.split()[-1])
            elif s.startswith("element face "):
                n_faces = int(s.split()[-1])
        if n_verts is None or n_faces is None:
            raise ValueError(f"{path}: missing vertex/face counts")

        verts = np.fromfile(f, dtype=np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")]), count=n_verts)
        v = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float32)

        faces_raw = np.fromfile(f, dtype=np.dtype([("n", "u1"), ("idx", "<i4", (3,))]), count=n_faces)
        tri = faces_raw["idx"].astype(np.int32)
    return v, tri


def project_points_opencv(pts_cam: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = pts_cam[:, 2]
    valid = z > 1e-6
    uv = np.full((pts_cam.shape[0], 2), -1.0, dtype=np.float32)
    if np.any(valid):
        xyz = pts_cam[valid]
        u = K[0, 0] * (xyz[:, 0] / xyz[:, 2]) + K[0, 2]
        v = K[1, 1] * (xyz[:, 1] / xyz[:, 2]) + K[1, 2]
        uv[valid] = np.stack([u, v], axis=1)
    return uv, valid


def draw_wireframe_np(frame: np.ndarray, uv: np.ndarray, valid: np.ndarray, faces: np.ndarray, color=(0, 255, 0)) -> np.ndarray:
    h, w = frame.shape[:2]
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    for tri in faces:
        i0, i1, i2 = int(tri[0]), int(tri[1]), int(tri[2])
        if not (valid[i0] and valid[i1] and valid[i2]):
            continue
        p0 = uv[i0]
        p1 = uv[i1]
        p2 = uv[i2]
        if not (0 <= p0[0] < w and 0 <= p0[1] < h and 0 <= p1[0] < w and 0 <= p1[1] < h and 0 <= p2[0] < w and 0 <= p2[1] < h):
            continue
        a = (float(p0[0]), float(p0[1]))
        b = (float(p1[0]), float(p1[1]))
        c = (float(p2[0]), float(p2[1]))
        draw.line([a, b], fill=color, width=1)
        draw.line([b, c], fill=color, width=1)
        draw.line([c, a], fill=color, width=1)
    return np.asarray(img)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    traj = args.traj_dir
    meta = json.loads((traj / "meta.json").read_text())
    K = np.load(traj / "cam_intrinsics.npy").astype(np.float32)
    if K.ndim == 3:
        K = K[0]

    pose_files = meta.get("body_poses", {}).get("files", {})
    mesh_files = meta.get("meshes", {}).get("files", {})
    obj_names = [k for k in pose_files.keys() if k in mesh_files]
    if not obj_names:
        raise RuntimeError(f"{traj}: no overlapping pose/mesh object names")

    obj_data = []
    for name in obj_names:
        T_bc = np.load(traj / pose_files[name]).astype(np.float32)
        V_body, F = read_ply_vertices_faces(traj / mesh_files[name])
        obj_data.append((name, T_bc, V_body, F))

    rgb_path = traj / "rgb.mp4"
    reader = imageio.get_reader(str(rgb_path))
    meta_vid = reader.get_meta_data()
    fps = float(meta_vid.get("fps", 30.0))

    out_path = args.out or (traj / "mesh_overlay.mp4")
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8)

    count = 0
    for idx, frame in enumerate(reader):
        if args.max_frames > 0 and idx >= args.max_frames:
            break
        canvas = frame
        for j, (_name, T_seq, V_body, F) in enumerate(obj_data):
            if idx >= T_seq.shape[0]:
                continue
            T = T_seq[idx]
            V_h = np.concatenate([V_body, np.ones((V_body.shape[0], 1), dtype=np.float32)], axis=1)
            V_cam = (T @ V_h.T).T[:, :3]
            uv, valid = project_points_opencv(V_cam, K)
            color = ((37 * (j + 3)) % 255, (97 * (j + 5)) % 255, (53 * (j + 7)) % 255)
            canvas = draw_wireframe_np(canvas, uv, valid, F, color=color)
        writer.append_data(canvas)
        count += 1

    reader.close()
    writer.close()
    print(f"wrote {out_path} frames={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
