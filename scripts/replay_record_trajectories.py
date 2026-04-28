#!/usr/bin/env python3
"""
Replay Metaworld trajectories with simultaneous camera-data recording.

Reads:
  trajectory.state.mocap_xyz.mujoco_cpu.h5
  trajectory.state.mocap_xyz.mujoco_cpu.json

Writes (inside <output_dir>/):
  trajectory.rgb+depth+segmentation.mocap_xyz.mujoco_cpu.h5
  trajectory.rgb+depth+segmentation.mocap_xyz.mujoco_cpu.json
  camera_data/
    traj_N/
      rgb.mp4                  H×W×3  uint8
      depth_video.npy          (T+1, H, W)  float16  metres
      seg.npy                  (T+1, H, W)  int32    MuJoCo body-IDs (-1 = bg)
      cam_poses.npy            (T+1, 4, 4)  float32  cam-to-world (OpenGL/SAPIEN conv.)
      cam_intrinsics.npy       (3, 3)       float32  pinhole K matrix
      traj_N.h5                copy of main traj_N group + id_poses/

id_poses/ layout (within each traj_N group):
  .attrs  {str(body_id): "body:<name>", ...}
  <body_id>/
    .attrs  {name: "body:<name>", seg_id: body_id}
    position          (T+1, 3)  float32  world-frame position
    quaternion        (T+1, 4)  float32  (w,x,y,z)
    camera_position   (T+1, 3)  float32  camera centre in world frame
    camera_quaternion (T+1, 4)  float32  (w,x,y,z)

Replay uses env_states (qpos + qvel + target_pos) for exact frame-by-frame
reproduction – no policy is needed.

Usage:
  python scripts/replay_record_trajectories.py \\
      --h5   rollout_data/reach-v3/trajectory.state.mocap_xyz.mujoco_cpu.h5 \\
      --json rollout_data/reach-v3/trajectory.state.mocap_xyz.mujoco_cpu.json \\
      --output-dir replay_data/reach-v3

  # only record successful trajectories
  python scripts/replay_record_trajectories.py \\
      --h5   rollout_data/reach-v3/trajectory.state.mocap_xyz.mujoco_cpu.h5 \\
      --json rollout_data/reach-v3/trajectory.state.mocap_xyz.mujoco_cpu.json \\
      --output-dir replay_data/reach-v3 --success-only

  # specify camera / resolution
  python scripts/replay_record_trajectories.py \\
      --h5  ... --json ... --output-dir ... \\
      --camera corner --width 640 --height 480 --fps 30
"""

import argparse
import copy
import json
import os
from pathlib import Path

import h5py
import numpy as np

import mujoco
import metaworld

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OUT_OBS_MODE    = "rgb+depth+segmentation"
OUT_CTRL_MODE   = "mocap_xyz"
OUT_BACKEND     = "mujoco_cpu"
OUT_BASENAME    = f"trajectory.{OUT_OBS_MODE}.{OUT_CTRL_MODE}.{OUT_BACKEND}"


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def mat3_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix → quaternion (w, x, y, z) via scipy."""
    try:
        from scipy.spatial.transform import Rotation
        xyzw = Rotation.from_matrix(R.astype(np.float64)).as_quat()
        return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)
    except ImportError:
        # Shepperd's method (no scipy)
        trace = R[0, 0] + R[1, 1] + R[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            return np.array([0.25 / s,
                             (R[2, 1] - R[1, 2]) * s,
                             (R[0, 2] - R[2, 0]) * s,
                             (R[1, 0] - R[0, 1]) * s], dtype=np.float32)
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                             (R[0, 1] + R[1, 0]) / s,
                             (R[0, 2] + R[2, 0]) / s], dtype=np.float32)
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                             0.25 * s, (R[1, 2] + R[2, 1]) / s], dtype=np.float32)
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                             (R[1, 2] + R[2, 1]) / s, 0.25 * s], dtype=np.float32)


def build_env(env_name: str, seed: int, width: int, height: int):
    """Build a Metaworld env.

    We pass render_mode="rgb_array" together with the target resolution so that
    gymnasium's MujocoRenderer expands the model's offscreen framebuffer to at
    least (width, height) before we create our own mujoco.Renderer instances.
    Without this, mujoco.Renderer raises:
      ValueError: Image width N > framebuffer width M
    We do NOT call env.render() afterwards – all rendering goes through the
    three standalone mujoco.Renderer objects in main().
    """
    mt1 = metaworld.MT1(env_name, seed=seed)
    env_cls = mt1.train_classes[env_name]
    env = env_cls(render_mode="rgb_array", width=width, height=height)
    env.seed(seed)
    env.set_task(mt1.train_tasks[0])
    env.reset()
    return env, mt1


def get_camera_id(model, cam_name: str) -> int:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        raise ValueError(
            f"Camera '{cam_name}' not found in model. "
            "Available: corner, topview, behindGripper, gripperPOV"
        )
    return cam_id


def compute_intrinsics(model, cam_id: int, width: int, height: int) -> np.ndarray:
    """Pinhole K matrix from MuJoCo camera FOV (square pixels assumed)."""
    fovy_deg = float(model.cam_fovy[cam_id])
    fy = (height / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)
    fx = fy  # MuJoCo renders with square pixels
    cx = width  / 2.0
    cy = height / 2.0
    return np.array([[fx, 0., cx],
                     [0., fy, cy],
                     [0., 0., 1.]], dtype=np.float32)


def get_cam_pose(data, cam_id: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (position (3,), rotation_matrix (3,3)) of camera in world frame.

    MuJoCo's cam_xmat uses OpenGL camera convention:
      +x right, +y up, -z forward  (z points *backward* out of lens)
    which matches the cam2world_gl convention used by MIKASA / SAPIEN.
    """
    cam_pos  = np.copy(data.cam_xpos[cam_id]).astype(np.float32)         # (3,)
    cam_xmat = np.copy(data.cam_xmat[cam_id]).reshape(3, 3).astype(np.float32)  # (3,3)
    return cam_pos, cam_xmat


def render_rgb(renderer: mujoco.Renderer, data, cam_id: int) -> np.ndarray:
    """Render RGB (H, W, 3) uint8.

    mujoco.Renderer internally applies np.flipud() (EGL/OSMesa/GLFW backends),
    so row-0 = top of image = camera +y direction.
    For Metaworld's corner camera, the y-axis is (-0.2, -0.2, -1) – mostly
    *downward* in world coordinates – so after that internal flip, world-down
    ends up at the image top, making the scene appear upside-down.
    We apply one more [::-1] to put world-up back at the image top.
    """
    renderer.update_scene(data, camera=cam_id)
    return renderer.render()[::-1].copy()   # (H, W, 3) uint8


def render_depth(renderer: mujoco.Renderer, data, cam_id: int) -> np.ndarray:
    """Render depth (H, W) float32 metres.  Same extra flip as render_rgb."""
    renderer.update_scene(data, camera=cam_id)
    return renderer.render()[::-1].copy()   # (H, W) float32


def render_seg_body_ids(renderer: mujoco.Renderer, data, cam_id: int,
                        model) -> np.ndarray:
    """Render per-pixel MuJoCo body IDs (-1 = background).  Same extra flip.

    mujoco.Renderer with enable_segmentation_rendering() returns (H, W, 2) int32:
      [..., 0] = objid    – object index within its type (geom_id when type==GEOM)
      [..., 1] = objtype  – mjtObj enum value (mjOBJ_GEOM=5, mjOBJ_SITE=6, …)
    Background pixels have both channels == -1.
    """
    renderer.update_scene(data, camera=cam_id)
    seg_raw = renderer.render()[::-1]   # (H, W, 2) int32, flip corrects orientation

    # Select pixels where a visible geom was rendered
    geom_mask = (
        (seg_raw[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) &   # channel 1 = objtype
        (seg_raw[..., 0] >= 0)                              # channel 0 = objid (valid)
    )
    # Safely map geom_id → body_id via model.geom_bodyid
    safe_geom_ids = np.clip(seg_raw[..., 0], 0, model.ngeom - 1)
    body_ids      = model.geom_bodyid[safe_geom_ids]
    return np.where(geom_mask, body_ids, -1).astype(np.int32)  # (H, W)


def save_rgb_video(frames: list[np.ndarray], path: str, fps: float) -> None:
    """Save list of (H,W,3) uint8 frames as MP4."""
    try:
        import imageio.v3 as iio
        iio.imwrite(path, np.stack(frames, axis=0), fps=int(fps))
    except ImportError:
        import imageio
        with imageio.get_writer(path, fps=int(fps)) as w:
            for f in frames:
                w.append_data(f)


# ---------------------------------------------------------------------------
# Core replay + record
# ---------------------------------------------------------------------------

def replay_and_record_traj(
    env,
    mt1,
    in_h5f:     h5py.File,
    out_h5f:    h5py.File,
    ep_idx:     int,
    episode:    dict,
    cam_id:     int,
    cam_K:      np.ndarray,
    rgb_ren:    mujoco.Renderer,
    depth_ren:  mujoco.Renderer,
    seg_ren:    mujoco.Renderer,
    cam_data_root: Path,
    fps:        float,
) -> None:
    """
    Replay one trajectory from env_states, recording camera data and id_poses.
    Writes into out_h5f[f"traj_{ep_idx}"] and camera_data/traj_{ep_idx}/.
    """
    traj_key = f"traj_{ep_idx}"
    if traj_key not in in_h5f:
        print(f"  [SKIP] {traj_key} not found in input H5")
        return

    in_grp = in_h5f[traj_key]
    qpos_seq   = in_grp["env_states/qpos"][:]        # (T+1, Dq)
    qvel_seq   = in_grp["env_states/qvel"][:]        # (T+1, Dv)
    has_target = "env_states/target_pos" in in_grp
    target_seq = in_grp["env_states/target_pos"][:] if has_target else None
    T_plus_1   = qpos_seq.shape[0]

    # ------------------------------------------------------------------ env setup
    task_idx = episode.get("task_idx", ep_idx % len(mt1.train_tasks))
    env.set_task(mt1.train_tasks[task_idx])
    env.reset()

    # ------------------------------------------------------------------ buffers
    rgb_frames   : list[np.ndarray] = []
    depth_frames : list[np.ndarray] = []
    seg_frames   : list[np.ndarray] = []
    cam_poses    : list[np.ndarray] = []  # (4,4)

    n_bodies = env.model.nbody
    # body 0 = world; track bodies 1..n_bodies-1
    body_ids_tracked = list(range(1, n_bodies))
    id_buf: dict[int, dict[str, list]] = {
        bid: {"position": [], "quaternion": [],
              "camera_position": [], "camera_quaternion": []}
        for bid in body_ids_tracked
    }

    # ------------------------------------------------------------------ loop over frames
    for t in range(T_plus_1):
        env.set_env_state((qpos_seq[t], qvel_seq[t]))
        if target_seq is not None:
            env._target_pos = target_seq[t].astype(np.float64)

        # RGB – use mujoco.Renderer directly (no gymnasium flip, see render_rgb docstring)
        rgb = render_rgb(rgb_ren, env.data, cam_id)                # (H, W, 3) uint8
        rgb_frames.append(rgb)

        # Depth (metres, float32)
        depth = render_depth(depth_ren, env.data, cam_id)
        depth_frames.append(depth)

        # Segmentation body-IDs
        seg = render_seg_body_ids(seg_ren, env.data, cam_id, env.model)
        seg_frames.append(seg)

        # Camera pose
        cam_pos, cam_xmat = get_cam_pose(env.data, cam_id)
        T_c2w = np.eye(4, dtype=np.float32)
        T_c2w[:3, :3] = cam_xmat
        T_c2w[:3, 3]  = cam_pos
        cam_poses.append(T_c2w)

        cam_quat = mat3_to_quat_wxyz(cam_xmat)

        # id_poses: per-body world-frame pose
        for bid in body_ids_tracked:
            pos  = np.copy(env.data.xpos [bid]).astype(np.float32)   # (3,)
            quat = np.copy(env.data.xquat[bid]).astype(np.float32)   # (4,) w,x,y,z
            id_buf[bid]["position"].append(pos)
            id_buf[bid]["quaternion"].append(quat)
            id_buf[bid]["camera_position"].append(cam_pos.copy())
            id_buf[bid]["camera_quaternion"].append(cam_quat.copy())

    # ------------------------------------------------------------------ save camera_data/traj_N/
    cam_traj_dir = cam_data_root / traj_key
    cam_traj_dir.mkdir(parents=True, exist_ok=True)

    # rgb.mp4
    save_rgb_video(rgb_frames, str(cam_traj_dir / "rgb.mp4"), fps)

    # depth_video.npy  (T+1, H, W) float16 metres
    np.save(str(cam_traj_dir / "depth_video.npy"),
            np.stack(depth_frames, axis=0).astype(np.float16))

    # seg.npy  (T+1, H, W) int32
    np.save(str(cam_traj_dir / "seg.npy"),
            np.stack(seg_frames, axis=0))

    # cam_poses.npy  (T+1, 4, 4) float32
    np.save(str(cam_traj_dir / "cam_poses.npy"),
            np.stack(cam_poses, axis=0))

    # cam_intrinsics.npy  (3, 3) float32  (constant for fixed camera)
    np.save(str(cam_traj_dir / "cam_intrinsics.npy"), cam_K)

    # ------------------------------------------------------------------ write main H5 traj group
    # Copy all datasets from input (obs, actions, rewards, env_states, …)
    in_h5f.copy(traj_key, out_h5f)
    out_grp = out_h5f[traj_key]

    # id_poses/ group (MIKASA-compatible)
    id_grp = out_grp.create_group("id_poses", track_order=True)
    for bid in body_ids_tracked:
        body_name = env.model.body(bid).name
        id_grp.attrs[str(bid)] = f"body:{body_name}"

    for bid in body_ids_tracked:
        body_name = env.model.body(bid).name
        seg_grp = id_grp.create_group(str(bid), track_order=True)
        seg_grp.attrs["name"]   = f"body:{body_name}"
        seg_grp.attrs["seg_id"] = bid
        buf = id_buf[bid]
        seg_grp.create_dataset("position",
            data=np.array(buf["position"],          dtype=np.float32))
        seg_grp.create_dataset("quaternion",
            data=np.array(buf["quaternion"],         dtype=np.float32))
        seg_grp.create_dataset("camera_position",
            data=np.array(buf["camera_position"],    dtype=np.float32))
        seg_grp.create_dataset("camera_quaternion",
            data=np.array(buf["camera_quaternion"],  dtype=np.float32))

    # ------------------------------------------------------------------ per-episode traj_N.h5
    per_h5_path = cam_traj_dir / f"{traj_key}.h5"
    with h5py.File(str(per_h5_path), "w") as per_h5:
        try:
            out_h5f.copy(traj_key, per_h5)
        except Exception as e:
            print(f"  [WARN] per-episode H5 copy failed ({e}); writing children individually")
            for k in out_grp.keys():
                out_h5f.copy(f"{traj_key}/{k}", per_h5, name=k)

    print(
        f"  traj_{ep_idx:04d}  steps={episode['elapsed_steps']:3d}"
        f"  success={episode['success']}"
        f"  → {cam_traj_dir}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    json_data  = load_json(args.json)
    env_info   = json_data["env_info"]
    env_name   = env_info["env_id"]
    seed       = env_info["env_kwargs"].get("seed", 42)
    episodes   = json_data["episodes"]

    # Filter trajectories
    if args.all_trajs:
        selected = list(range(len(episodes)))
    else:
        selected = [args.traj_id]

    if args.success_only:
        selected = [i for i in selected if episodes[i]["success"]]
        print(f"  (--success-only: {len(selected)} successful trajectories)")

    if not selected:
        print("No trajectories to process. Exiting.")
        return

    # Output paths
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_data_root = out_dir / "camera_data"
    cam_data_root.mkdir(exist_ok=True)
    out_h5_path   = out_dir / f"{OUT_BASENAME}.h5"
    out_json_path = out_dir / f"{OUT_BASENAME}.json"

    # Build env + aux renderers
    print(f"\nBuilding env  : {env_name}  (seed={seed})")
    env, mt1 = build_env(env_name, seed, args.width, args.height)

    cam_id = get_camera_id(env.model, args.camera)
    cam_K  = compute_intrinsics(env.model, cam_id, args.width, args.height)

    # Three independent mujoco.Renderer instances – all use the same no-flip
    # convention so RGB / depth / seg are spatially consistent with each other.
    rgb_ren   = mujoco.Renderer(env.model, height=args.height, width=args.width)

    depth_ren = mujoco.Renderer(env.model, height=args.height, width=args.width)
    depth_ren.enable_depth_rendering()

    seg_ren = mujoco.Renderer(env.model, height=args.height, width=args.width)
    seg_ren.enable_segmentation_rendering()

    print(f"Camera        : {args.camera}  (id={cam_id})")
    print(f"Resolution    : {args.width}×{args.height}")
    print(f"Trajectories  : {selected}\n")

    # Output JSON (copy + update obs_mode)
    out_json = copy.deepcopy(json_data)
    out_json["env_info"]["env_kwargs"]["obs_mode"] = OUT_OBS_MODE
    out_json["source_desc"] = (
        f"Metaworld scripted policy replay+record "
        f"(env={env_name}, seed={seed}, camera={args.camera})"
    )

    # Process trajectories
    with (
        h5py.File(args.h5,           "r") as in_h5f,
        h5py.File(str(out_h5_path),  "w") as out_h5f,
    ):
        for ep_idx in selected:
            replay_and_record_traj(
                env=env,
                mt1=mt1,
                in_h5f=in_h5f,
                out_h5f=out_h5f,
                ep_idx=ep_idx,
                episode=episodes[ep_idx],
                cam_id=cam_id,
                cam_K=cam_K,
                rgb_ren=rgb_ren,
                depth_ren=depth_ren,
                seg_ren=seg_ren,
                cam_data_root=cam_data_root,
                fps=args.fps,
            )

    # Write output JSON
    # Keep only processed episodes in the JSON
    out_json["episodes"] = [episodes[i] for i in selected]
    with open(str(out_json_path), "w") as f:
        json.dump(out_json, f, indent=2)

    rgb_ren.close()
    depth_ren.close()
    seg_ren.close()
    env.close()

    print(f"\nOutput H5   : {out_h5_path}")
    print(f"Output JSON : {out_json_path}")
    print(f"Camera data : {cam_data_root}")
    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Replay Metaworld trajectories and record RGB/depth/seg/id_poses."
    )
    p.add_argument("--h5",   required=True,
                   help="Path to trajectory.state.mocap_xyz.mujoco_cpu.h5")
    p.add_argument("--json", required=True,
                   help="Path to trajectory.state.mocap_xyz.mujoco_cpu.json")
    p.add_argument("--output-dir", required=True,
                   help="Directory where output H5, JSON and camera_data/ are written.")
    p.add_argument(
        "--traj-id", type=int, default=0,
        help="Index of the single trajectory to process (ignored with --all-trajs).",
    )
    p.add_argument(
        "--all-trajs", action="store_true",
        help="Process all trajectories in the file.",
    )
    p.add_argument(
        "--success-only", action="store_true",
        help="Only process trajectories where success=True.",
    )
    p.add_argument("--fps",    type=float, default=30.0, help="FPS for rgb.mp4.")
    p.add_argument("--camera", type=str,   default="corner",
                   help="MuJoCo camera name: corner | topview | behindGripper | gripperPOV")
    p.add_argument("--width",  type=int,   default=640)
    p.add_argument("--height", type=int,   default=480)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
