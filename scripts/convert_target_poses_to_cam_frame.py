"""
Convert target object poses from body-to-WORLD to body-to-CAMERA, in both
OpenGL/RUB and OpenCV/RDF conventions, then replace the original files.

Input (per traj dir):
    pose_<name>.npy            (T, 4, 4)  body-to-world  (MuJoCo world frame)
    cam2world_gl.npy           (T, 4, 4)  cam-to-world   (OpenGL/RUB)
    cam2world_cv.npy           (T, 4, 4)  cam-to-world   (OpenCV/RDF)

Output (per traj dir):
    pose_<name>_gl.npy         (T, 4, 4)  body-to-cam    (OpenGL/RUB)
    pose_<name>_cv.npy         (T, 4, 4)  body-to-cam    (OpenCV/RDF)
    (original pose_<name>.npy is removed)

Math:
    T_body_to_cam = inv(cam2world) @ T_body_to_world
"""
import argparse
import json
from pathlib import Path

import numpy as np


def invert_se3(T: np.ndarray) -> np.ndarray:
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = np.swapaxes(R, -1, -2)
    out = np.zeros_like(T)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, t)
    out[..., 3, 3] = 1.0
    return out


def convert_traj(traj_dir: Path, dry_run: bool = False) -> str:
    pose_files = sorted(p for p in traj_dir.glob("pose_*.npy")
                        if "_cv" not in p.stem and "_gl" not in p.stem)
    if not pose_files:
        return f"SKIP (no pose_*.npy): {traj_dir}"

    c2w_gl_path = traj_dir / "cam2world_gl.npy"
    c2w_cv_path = traj_dir / "cam2world_cv.npy"
    if not c2w_gl_path.exists() or not c2w_cv_path.exists():
        return f"SKIP (missing cam2world_{{gl,cv}}.npy): {traj_dir}"

    c2w_gl = np.load(c2w_gl_path).astype(np.float32)
    c2w_cv = np.load(c2w_cv_path).astype(np.float32)
    w2c_gl = invert_se3(c2w_gl)
    w2c_cv = invert_se3(c2w_cv)

    written = []
    for pf in pose_files:
        body2world = np.load(pf).astype(np.float32)
        if body2world.shape[0] != c2w_gl.shape[0]:
            raise ValueError(f"{pf}: T={body2world.shape[0]} != cam T={c2w_gl.shape[0]}")
        body2cam_gl = (w2c_gl @ body2world).astype(np.float32)
        body2cam_cv = (w2c_cv @ body2world).astype(np.float32)
        out_gl = pf.with_name(pf.stem + "_gl.npy")
        out_cv = pf.with_name(pf.stem + "_cv.npy")
        if not dry_run:
            np.save(out_gl, body2cam_gl)
            np.save(out_cv, body2cam_cv)
            pf.unlink()
        written.append((pf.name, out_gl.name, out_cv.name))

    meta_path = traj_dir / "meta.json"
    if meta_path.exists() and not dry_run:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        bp = meta.get("body_poses", {})
        bp["format"] = (
            "(T, 4, 4) float32  body-to-CAMERA homogeneous transform "
            "(p_cam = R @ p_body + t). Two conventions: _gl = OpenGL/RUB, "
            "_cv = OpenCV/RDF."
        )
        files = bp.get("files", {})
        new_files = {}
        for name in list(files.keys()):
            new_files[name] = {
                "gl": f"pose_{name}_gl.npy",
                "cv": f"pose_{name}_cv.npy",
            }
        bp["files"] = new_files
        meta["body_poses"] = bp
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    pairs = ", ".join(f"{old}->{gl}+{cv}" for old, gl, cv in written)
    return f"OK {traj_dir}: {pairs}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path,
                    help="Either a single traj dir or a parent dir containing */camera_data/traj_*/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if (args.root / "meta.json").exists() and any(args.root.glob("pose_*.npy")):
        targets = [args.root]
    else:
        targets = sorted(args.root.glob("*/camera_data/traj_*"))

    for d in targets:
        print(convert_traj(d, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
