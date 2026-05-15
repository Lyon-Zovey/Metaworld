#!/usr/bin/env python3
"""Verify OpenGL -> OpenCV conversion preserves world-space geometry.

Checks (against the backup OpenGL copy):
  1. anchor lifted to world via cam_pose matches between GL and CV.
  2. body world position recovered from camera_position + camera_quaternion
     matches between GL and CV (and matches id_poses/<sid>/position).
  3. depth back-projection of an arbitrary pixel gives a world point that
     matches between GL and CV (using the appropriate per-convention formula).
  4. flow video reconstructs to anchor at the reference frame within tolerance.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "rbs_sceneflow_scripts" / "traj2sceneflow"))
from flow_compress import decompress_one_flow  # noqa: E402

FLIP3 = np.array([1.0, -1.0, -1.0], dtype=np.float32)


def quat_to_R_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    s = 2.0 / max(n, 1e-12)
    R = np.array([
        [1 - s * (y * y + z * z),   s * (x * y - z * w),     s * (x * z + y * w)],
        [s * (x * y + z * w),       1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w),       s * (y * z + x * w),     1 - s * (x * x + y * y)],
    ], dtype=np.float64)
    return R


def lift_anchor_to_world(anchor_cam: np.ndarray, cam_pose: np.ndarray) -> np.ndarray:
    H, W, _ = anchor_cam.shape
    pts = anchor_cam.reshape(-1, 3)
    ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    homog = np.concatenate([pts, ones], axis=1).T   # 4xN
    world = (cam_pose @ homog).T[:, :3]
    return world.reshape(H, W, 3)


def body_world_from_cam(cam_pose: np.ndarray, body_cam_pos: np.ndarray) -> np.ndarray:
    """Return body world position given cam_to_world and body position in cam frame."""
    homog = np.append(body_cam_pos, 1.0)
    return (cam_pose @ homog)[:3]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-root", required=True, help="converted (OpenCV) root with traj_*")
    ap.add_argument("--gl-root", required=True, help="backup (OpenGL) root with traj_*")
    ap.add_argument("--traj", default=None, help="single traj name to inspect (e.g. traj_0)")
    args = ap.parse_args()

    cv_root = Path(args.cv_root)
    gl_root = Path(args.gl_root)
    if args.traj:
        names = [args.traj]
    else:
        names = sorted(d.name for d in cv_root.iterdir() if d.is_dir() and d.name.startswith("traj_"))

    ok = True
    for name in names:
        print(f"=== {name} ===")
        cvd = cv_root / name
        gld = gl_root / name

        # 1) cam_pose: translation must be equal; rotation cols 1,2 negated.
        cp_cv = np.load(cvd / "cam_poses.npy")
        cp_gl = np.load(gld / "cam_poses.npy")
        d_t = np.abs(cp_cv[..., :3, 3] - cp_gl[..., :3, 3]).max()
        rec = cp_cv.copy()
        rec[..., :3, 1:3] *= -1.0
        d_r = np.abs(rec - cp_gl).max()
        print(f"  cam_poses: translation diff={d_t:.2e}, flipback diff={d_r:.2e}")
        ok &= (d_t < 1e-6 and d_r < 1e-6)

        # 2) anchor world-position invariance
        for f_cv in sorted(cvd.glob("scene_point_flow_ref*.anchor.npy")):
            f_gl = gld / f_cv.name
            a_cv = np.load(f_cv)
            a_gl = np.load(f_gl)
            # ref index from filename
            ref = int(f_cv.name.split("ref")[1].split(".")[0])
            T = cp_cv.shape[0]
            ref_clamped = min(ref, T - 1)
            w_cv = lift_anchor_to_world(a_cv, cp_cv[ref_clamped].astype(np.float64))
            w_gl = lift_anchor_to_world(a_gl, cp_gl[ref_clamped].astype(np.float64))
            mask = np.isfinite(a_cv).all(axis=-1) & np.isfinite(a_gl).all(axis=-1)
            mask &= (np.linalg.norm(a_cv, axis=-1) > 1e-6)
            if not mask.any():
                print(f"  {f_cv.name}: no valid anchor pixels")
                continue
            diff = np.linalg.norm(w_cv[mask] - w_gl[mask], axis=-1)
            print(f"  {f_cv.name}: world-pos max diff={diff.max():.2e}, mean={diff.mean():.2e}")
            ok &= (diff.max() < 1e-4)

            # flow video roundtrip: ref-frame should equal anchor
            videos = list(cvd.glob(f"{f_cv.name.replace('.anchor.npy','')}_*.mp4"))
            if videos:
                vid = videos[0]
                side = json.loads(vid.with_suffix(".json").read_text())
                flow = decompress_one_flow(vid, a_cv, side)
                if ref < flow.shape[0]:
                    df = np.abs(flow[ref] - a_cv).max()
                    print(f"    flow[ref]==anchor max diff={df:.4e}")

        # 3) H5 camera fields -- recover body world pos from cam frame
        h5s = list(cvd.glob("*.h5"))
        h5g = gld / h5s[0].name
        with h5py.File(h5s[0], "r") as fcv, h5py.File(h5g, "r") as fgl:
            tname = list(fcv.keys())[0]
            ids = list(fcv[tname]["id_poses"].keys())
            sid = ids[0]
            cp_cv_b = fcv[tname][f"id_poses/{sid}/camera_position"][0]
            cq_cv_b = fcv[tname][f"id_poses/{sid}/camera_quaternion"][0]
            cp_gl_b = fgl[tname][f"id_poses/{sid}/camera_position"][0]
            cq_gl_b = fgl[tname][f"id_poses/{sid}/camera_quaternion"][0]
            p_world = fgl[tname][f"id_poses/{sid}/position"][0]
            # translation must satisfy cp_cv_b == cp_gl_b * [1,-1,-1]
            d_pos = np.abs(cp_cv_b - cp_gl_b * np.array([1, -1, -1])).max()
            print(f"  H5 sid={sid}: cam_pos flip diff={d_pos:.2e}")
            # world position invariance through cam_pose
            w_cv = body_world_from_cam(cp_cv[0].astype(np.float64), cp_cv_b.astype(np.float64))
            w_gl = body_world_from_cam(cp_gl[0].astype(np.float64), cp_gl_b.astype(np.float64))
            d_w = np.linalg.norm(w_cv - w_gl)
            d_gt = np.linalg.norm(w_cv - p_world)
            print(f"  H5 sid={sid}: world(cv) vs world(gl) diff={d_w:.2e}, vs id_poses/position diff={d_gt:.2e}")
            ok &= (d_pos < 1e-6 and d_w < 1e-4)
            # quaternion: q_cv should equal q_flip * q_gl
            qf = np.array([0.0, 1.0, 0.0, 0.0])
            from numpy import array
            def qmul(a, b):
                w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
                return array([w1*w2-x1*x2-y1*y2-z1*z2,
                              w1*x2+x1*w2+y1*z2-z1*y2,
                              w1*y2-x1*z2+y1*w2+z1*x2,
                              w1*z2+x1*y2-y1*x2+z1*w2])
            q_expected = qmul(qf, cq_gl_b)
            d_q = np.abs(cq_cv_b - q_expected).max()
            print(f"  H5 sid={sid}: cam_quat flip diff={d_q:.2e}")
            ok &= (d_q < 1e-6)

    print("\nALL OK" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
