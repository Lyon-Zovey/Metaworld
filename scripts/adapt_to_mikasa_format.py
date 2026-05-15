"""
Adapt a Metaworld camera_data trajectory to the mikasa file/meta layout
so `vis_sceneflow.py` (and downstream dataloaders) can read it without flags.

For each traj dir:
    cam2world_cv.npy        -> cam2world.npy
    cam_poses_cv.npy        -> cam_poses.npy
    pose_<name>_cv.npy      -> pose_<name>.npy
    (cam2world_gl.npy, cam_poses_gl.npy, pose_<name>_gl.npy are deleted)

meta.json is rewritten to mikasa-style fields:
    camera_pose_layout, cam2world_file, cam_poses_file, pose_layout
    body_poses.files.<name> = "pose_<name>.npy"   (was {gl, cv})
    body_poses.format       = "(T,4,4) ... body-to-cam in OpenCV camera frame."
    meshes.format           = simplified to match mikasa wording
    cam_pose_files          -> removed (was nested {gl, cv})
    cam_pose_layout         -> removed (typo of camera_pose_layout)

Use --dry-run to preview without changing files.
"""
import argparse
import json
from pathlib import Path


def adapt_traj(traj_dir: Path, dry_run: bool) -> str:
    actions: list[str] = []

    def rename(src: Path, dst: Path) -> None:
        if not src.exists():
            return
        if dst.exists():
            actions.append(f"SKIP rename (dst exists): {src.name} -> {dst.name}")
            return
        actions.append(f"rename {src.name} -> {dst.name}")
        if not dry_run:
            src.rename(dst)

    def delete(p: Path) -> None:
        if not p.exists():
            return
        actions.append(f"delete {p.name}")
        if not dry_run:
            p.unlink()

    rename(traj_dir / "cam2world_cv.npy", traj_dir / "cam2world.npy")
    rename(traj_dir / "cam_poses_cv.npy", traj_dir / "cam_poses.npy")
    delete(traj_dir / "cam2world_gl.npy")
    delete(traj_dir / "cam_poses_gl.npy")

    for pf in sorted(traj_dir.glob("pose_*_cv.npy")):
        name = pf.stem[len("pose_"):-len("_cv")]
        rename(pf, pf.with_name(f"pose_{name}.npy"))
    for pf in sorted(traj_dir.glob("pose_*_gl.npy")):
        delete(pf)

    meta_path = traj_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta.pop("cam_pose_layout", None)
        meta.pop("cam_pose_files", None)
        meta["camera_pose_layout"] = "cam2world absolute; cam_poses relative to cam0"
        meta["cam2world_file"] = "cam2world.npy"
        meta["cam_poses_file"] = "cam_poses.npy"
        meta["pose_layout"] = "body->cam"

        bp = meta.get("body_poses", {})
        bp["format"] = (
            "(T, 4, 4) float32  body-to-cam homogeneous transform "
            "(column-vec: p_cam = R @ p_body + t); quaternion order (w,x,y,z)."
        )
        new_files: dict[str, str] = {}
        for name, val in bp.get("files", {}).items():
            new_files[name] = f"pose_{name}.npy"
        bp["files"] = new_files
        meta["body_poses"] = bp

        if "meshes" in meta:
            meta["meshes"]["format"] = (
                "binary_little_endian PLY; vertices in body-local frame "
                "(p_cam = pose_<obj>[t] @ [v; 1])."
            )

        actions.append("rewrite meta.json (camera_pose_layout/cam2world_file/cam_poses_file/pose_layout, "
                       "body_poses.files->str, meshes.format)")
        if not dry_run:
            meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                 encoding="utf-8")

    if not actions:
        return f"NOOP {traj_dir}"
    return f"OK {traj_dir}:\n  - " + "\n  - ".join(actions)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path,
                    help="A traj dir, or a parent containing */camera_data/traj_*/")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if (args.root / "meta.json").exists():
        targets = [args.root]
    else:
        targets = sorted(args.root.glob("*/camera_data/traj_*"))

    for d in targets:
        print(adapt_traj(d, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
