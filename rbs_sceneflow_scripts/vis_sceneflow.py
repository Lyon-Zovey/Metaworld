"""
Visualize sceneflow point cloud, camera trajectory, and object pose trajectory in viser.

Usage:
    conda run -n maniskill python vis_sceneflow.py \
        demos/LiftPegUpright-v1/motionplanning/camera_data/traj_0

Opens a viser web UI at http://localhost:8080
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
import viser.transforms as tf

sys.path.insert(0, str(Path(__file__).parent / "traj2sceneflow"))
from flow_compress import decompress_one_flow


def load_meshes(traj_dir: Path) -> dict[str, trimesh.Trimesh]:
    meshes = {}
    for mp in sorted(traj_dir.glob("mesh_*.ply")):
        name = mp.stem.replace("mesh_", "")
        try:
            m = trimesh.load(str(mp), force="mesh")
        except Exception as e:
            print(f"  failed to load {mp.name}: {e}")
            continue
        if not isinstance(m, trimesh.Trimesh) or len(m.vertices) == 0:
            print(f"  skipped {mp.name} (not a triangle mesh)")
            continue
        meshes[name] = m
        print(f"  {mp.name}: V={len(m.vertices)} F={len(m.faces)}")
    return meshes


FLIP3 = np.array([1.0, -1.0, -1.0], dtype=np.float32)
FLIP4 = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
ROT_Z_180 = np.diag([-1.0, -1.0, 1.0, 1.0]).astype(np.float32)


def load_first_flow(traj_dir: Path):
    anchors = sorted(traj_dir.glob("scene_point_flow_ref*.anchor.npy"))
    if not anchors:
        raise FileNotFoundError(f"No anchor files in {traj_dir}")
    anchor_path = anchors[0]
    anchor = np.load(str(anchor_path))

    stem = anchor_path.stem.replace(".anchor", "")
    video_candidates = list(traj_dir.glob(f"{stem}*.mp4")) + list(traj_dir.glob(f"{stem}*.mkv"))
    if not video_candidates:
        raise FileNotFoundError(f"No flow video for {stem}")
    video_path = video_candidates[0]

    flow = decompress_one_flow(video_path, anchor)
    return flow


def read_meta(traj_dir: Path) -> dict:
    p = traj_dir / "meta.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("traj_dir", type=Path)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--downsample", type=int, default=4,
                        help="Spatial downsample factor for point cloud")
    parser.add_argument("--opencv", action="store_true",
                        help="Use OpenCV-convention files (*_cv.npy)")
    parser.add_argument("--rdf", action="store_true",
                        help="Convert RUB(OpenGL) camera-frame data to RDF(OpenCV) at runtime")
    parser.add_argument("--flip-flow-yz", action="store_true",
                        help="Flip only sceneflow point coordinates: y,z -> -y,-z (for debugging)")
    parser.add_argument("--flip-x", action="store_true",
                        help="Flip sceneflow x axis: x -> -x")
    parser.add_argument("--flip-y", action="store_true",
                        help="Flip sceneflow y axis: y -> -y")
    parser.add_argument("--flip-z", action="store_true",
                        help="Flip sceneflow z axis: z -> -z")
    parser.add_argument("--flip-xyz", action="store_true",
                        help="Flip all three sceneflow axes: x,y,z -> -x,-y,-z "
                             "(shortcut for --flip-x --flip-y --flip-z)")
    parser.add_argument("--rot-z-180", action="store_true",
                        help="Rotate sceneflow 180 degrees around the z-axis: "
                             "(x,y,z) -> (-x,-y,z). Equivalent to --flip-x --flip-y.")
    parser.add_argument("--auto", action="store_true",
                        help="Read meta.json and auto-apply OpenGL->OpenCV flow conversion when "
                             "flow_convention==opengl and cam2world.npy is in OpenCV convention")
    parser.add_argument("--fix-pose", action="store_true",
                        help="Fix double-migrate bug: pose was body->cam but inv(c2w[0]) was applied "
                             "a second time. Multiply c2w[0] back to recover correct body->cam.")
    parser.add_argument("--cam0", action="store_true",
                        help="Use cam0 as world frame (use cam_poses relative to cam0 instead of cam2world)")
    parser.add_argument("--world", choices=["cam_poses", "cam2world_cv", "cam2world_gl"],
                        default=None,
                        help="Explicit camera-pose file to use as world frame. Overrides --cam0 / "
                             "--opencv defaults. cam_poses=relative to cam0 (CV); "
                             "cam2world_cv=absolute world (CV/RDF); cam2world_gl=absolute world (GL/RUB). "
                             "Note: with cam2world_gl the flow is already in GL — DO NOT pass "
                             "--flip-flow-yz; with the other two the flow is in CV after --flip-flow-yz.")
    parser.add_argument("--no-mesh", action="store_true",
                        help="Skip mesh visualization even if mesh_*.ply files exist")
    parser.add_argument("--flip-cam-yz", action="store_true",
                        help="Right-multiply cam2world by FLIP4 (negate cam y/z axes). "
                             "Converts a GL cam2world to CV in-place. flow/pose untouched.")
    args = parser.parse_args()

    traj_dir = args.traj_dir

    meta = read_meta(traj_dir)
    if meta:
        print(f"  meta.json: flow_convention={meta.get('flow_convention','?')}, "
              f"camera_convention={meta.get('camera_convention','?')}, "
              f"cam2world_file={meta.get('cam2world_file','?')}")

    print("Loading sceneflow...")
    if args.opencv:
        flow_npy = sorted(traj_dir.glob("scene_point_flow_ref*_cv.npy"))
        if not flow_npy:
            raise FileNotFoundError("No *_cv.npy flow files found. Generate them first.")
        flow = np.load(str(flow_npy[0]))
        print(f"  flow (opencv): {flow.shape}")
    else:
        flow = load_first_flow(traj_dir)
        print(f"  flow (opengl): {flow.shape}")
    T, H, W, _ = flow.shape

    if args.rdf:
        flow = (flow.astype(np.float32) * FLIP3).astype(np.float32)
        print("  applied RUB->RDF conversion to flow")
    elif args.flip_flow_yz:
        flow = (flow.astype(np.float32) * FLIP3).astype(np.float32)
        print("  applied debug flip to flow only: y,z negated")

    flip_x = args.flip_x or args.flip_xyz or args.rot_z_180
    flip_y = args.flip_y or args.flip_xyz or args.rot_z_180
    flip_z = args.flip_z or args.flip_xyz
    if flip_x or flip_y or flip_z:
        sign = np.array([
            -1.0 if flip_x else 1.0,
            -1.0 if flip_y else 1.0,
            -1.0 if flip_z else 1.0,
        ], dtype=np.float32)
        flow = flow * sign
        axes = "".join(a for a, f in zip("xyz", [flip_x, flip_y, flip_z]) if f)
        print(f"  flipped flow axes: {axes}")

    # Prefer cam2world.npy (absolute, may be OpenCV) over cam_poses.npy (may be
    # relative-to-cam0 identity matrix in newer datasets).
    if args.world is not None:
        c2w_candidates = [f"{args.world}.npy"]
    elif args.cam0:
        c2w_candidates = ["cam_poses_cv.npy", "cam_poses.npy"] if args.opencv else ["cam_poses_gl.npy", "cam_poses.npy"]
    elif args.opencv:
        c2w_candidates = ["cam2world_cv.npy", "cam_poses_cv.npy"]
    else:
        c2w_candidates = ["cam2world.npy", "cam2world_cv.npy", "cam_poses.npy"]
    c2w_path = None
    for name in c2w_candidates:
        p = traj_dir / name
        if p.exists():
            c2w_path = p
            break
    if c2w_path is None:
        raise FileNotFoundError(f"No camera pose file found in {traj_dir}; tried {c2w_candidates}")

    cam2world = np.load(str(c2w_path)).astype(np.float32)
    print(f"  {c2w_path.name}: {cam2world.shape}")

    # For --fix-pose we always need the absolute cam2world[0], regardless of --cam0.
    # Load it separately so fix-pose stays correct even in cam0 mode.
    abs_c2w0 = cam2world[0]
    if args.cam0 and args.fix_pose:
        abs_candidates = ["cam2world_cv.npy", "cam2world.npy"] if args.opencv else ["cam2world.npy"]
        for name in abs_candidates:
            p = traj_dir / name
            if p.exists():
                abs_c2w0 = np.load(str(p)).astype(np.float32)[0]
                print(f"  [fix-pose] using {name}[0] as absolute cam2world[0]")
                break

    # Detect whether cam2world is OpenCV and flow is OpenGL.
    # Strategy: convert everything to a single consistent world frame (OpenCV).
    #   - sceneflow points (OpenGL cam) -> flip y,z -> OpenCV cam -> c2w_cv -> world
    #   - pose_*.npy (body->cam OpenCV) -> c2w_cv -> world
    # This keeps point cloud and pose axes in the same world frame for viser.
    auto_flip_flow = False
    cam2world_for_flow = cam2world
    cam2world_for_pose = cam2world

    if args.flip_cam_yz:
        cam2world = (cam2world @ FLIP4[None]).astype(np.float32)
        cam2world_for_flow = cam2world
        cam2world_for_pose = cam2world
        print("  [flip-cam-yz] applied cam2world @ FLIP4 (GL->CV on cam axes)")

    if args.rdf:
        cam2world_for_flow = (cam2world @ FLIP4).astype(np.float32)
        cam2world_for_pose = (cam2world @ FLIP4).astype(np.float32)
        print("  applied RUB->RDF conversion to cam poses")
    elif args.auto:
        cam_conv = meta.get("camera_convention", "")
        flow_conv = meta.get("flow_convention", "opengl")
        if cam_conv == "opencv" and flow_conv == "opengl":
            # flip flow points to OpenCV cam space, then use c2w_cv for both
            auto_flip_flow = True
            cam2world_for_flow = cam2world   # c2w_cv
            cam2world_for_pose = cam2world   # c2w_cv
            print("  [auto] converting OpenGL flow to OpenCV, using c2w_cv for both flow and poses")

    pose_files = sorted(traj_dir.glob("pose_*.npy"))
    poses = {}
    for pf in pose_files:
        if args.opencv:
            if not pf.stem.endswith("_cv"):
                continue
            name = pf.stem.replace("pose_", "").replace("_cv", "")
        else:
            if "_cv" in pf.stem:
                continue
            name = pf.stem.replace("pose_", "")
        poses[name] = np.load(str(pf))
        if args.rdf:
            poses[name] = (FLIP4 @ poses[name].astype(np.float32)).astype(np.float32)
        if args.fix_pose:
            poses[name] = (abs_c2w0 @ poses[name].astype(np.float32)).astype(np.float32)
            print(f"  [fix-pose] applied c2w[0] @ pose to undo double body->cam migration")
        print(f"  {pf.name}: {poses[name].shape}")

    meshes: dict[str, trimesh.Trimesh] = {}
    if not args.no_mesh:
        print("Loading meshes...")
        meshes = load_meshes(traj_dir)
        if meshes:
            missing = [n for n in meshes if n not in poses]
            if missing:
                print(f"  warning: meshes without matching pose_*.npy: {missing}")

    ds = args.downsample
    flow_ds = flow[:, ::ds, ::ds, :]

    # When --rot-z-180 is set, sceneflow points get (x,y,z)->(-x,-y,z) in the
    # CAMERA frame (before cam2world). Apply the same cam-frame rotation to
    # obj_pose by left-multiplying ROT_Z_180 into the body->cam pose.
    pose_cam_xform = ROT_Z_180 if args.rot_z_180 else np.eye(4, dtype=np.float32)

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    print(f"Viser running at http://0.0.0.0:{args.port}")

    frame_slider = server.gui.add_slider(
        "Frame", min=0, max=T - 1, step=1, initial_value=0
    )

    point_size_slider = server.gui.add_slider(
        "Point size", min=0.001, max=0.05, step=0.001, initial_value=0.005
    )

    cam_axis_len = 0.05
    obj_axis_len = 0.08

    pc_handle = None
    cam_frame_handle = None
    obj_frame_handles = {}
    mesh_handles: dict[str, object] = {}

    def render_frame(t: int) -> None:
        nonlocal pc_handle, cam_frame_handle, obj_frame_handles

        pts_cam = flow_ds[t].reshape(-1, 3)

        valid = np.isfinite(pts_cam).all(axis=1) & (np.abs(pts_cam) < 100).all(axis=1)
        pts_cam = pts_cam[valid]

        # Convert OpenGL cam coords to OpenCV before projecting with c2w_cv
        if auto_flip_flow:
            pts_cam = pts_cam * FLIP3

        c2w_flow = cam2world_for_flow[t]
        pts = (c2w_flow[:3, :3] @ pts_cam.T).T + c2w_flow[:3, 3]

        colors = np.full((pts.shape[0], 3), 180, dtype=np.uint8)

        if pc_handle is not None:
            pc_handle.remove()
        pc_handle = server.scene.add_point_cloud(
            "/points",
            points=pts,
            colors=colors,
            point_size=point_size_slider.value,
        )

        if cam_frame_handle is not None:
            cam_frame_handle.remove()
        cam_frame_handle = server.scene.add_frame(
            "/camera",
            wxyz=tf.SO3.from_matrix(c2w_flow[:3, :3]).wxyz,
            position=c2w_flow[:3, 3],
            axes_length=cam_axis_len,
            axes_radius=0.002,
        )

        for name, pose_arr in poses.items():
            pose = pose_arr[t]
            pose_world = cam2world_for_pose[t] @ pose_cam_xform @ pose

            if name in obj_frame_handles:
                obj_frame_handles[name].remove()
            obj_frame_handles[name] = server.scene.add_frame(
                f"/obj_{name}",
                wxyz=tf.SO3.from_matrix(pose_world[:3, :3]).wxyz,
                position=pose_world[:3, 3],
                axes_length=obj_axis_len,
                axes_radius=0.003,
            )

            mesh = meshes.get(name)
            if mesh is not None:
                if name in mesh_handles:
                    mesh_handles[name].remove()
                mesh_handles[name] = server.scene.add_mesh_trimesh(
                    f"/mesh_{name}",
                    mesh=mesh,
                    wxyz=tf.SO3.from_matrix(pose_world[:3, :3]).wxyz,
                    position=pose_world[:3, 3],
                )

    @frame_slider.on_update
    def _(_) -> None:
        render_frame(int(frame_slider.value))

    render_frame(0)

    cam_positions = cam2world_for_flow[:, :3, 3]
    cam_segments = np.stack(
        [cam_positions[:-1], cam_positions[1:]], axis=1
    )  # (T-1, 2, 3)
    server.scene.add_line_segments(
        "/cam_traj",
        points=cam_segments,
        colors=np.array([50, 130, 255], dtype=np.uint8),
    )

    for name, pose_arr in poses.items():
        obj_world_positions = np.array([
            (cam2world_for_pose[t] @ pose_cam_xform @ pose_arr[t])[:3, 3] for t in range(T)
        ])
        obj_segments = np.stack(
            [obj_world_positions[:-1], obj_world_positions[1:]], axis=1
        )  # (T-1, 2, 3)
        server.scene.add_line_segments(
            f"/obj_traj_{name}",
            points=obj_segments,
            colors=np.array([255, 80, 50], dtype=np.uint8),
        )

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
