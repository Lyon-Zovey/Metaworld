#!/usr/bin/env python3
"""
Viser viewer for RAW (uncompressed) Metaworld sceneflow.

Reads the files written by convert_camera_depths.py directly:
  <traj_dir>/<cam_name>/scene_point_flow_ref<NNNNN>.npy   (T, H, W, 3)
  <traj_dir>/<cam_name>/scene_point_flow_ref<NNNNN>.anchor.npy  (H, W, 3)
  <traj_dir>/<cam_name>/cam_poses.npy                     (T, 4, 4)
  <traj_dir>/<cam_name>/rgb.mp4                           (optional)

Multi-camera layout (produced by --multi-cameras):
  camera_data/traj_0/corner/
  camera_data/traj_0/corner2/
  camera_data/traj_0/corner3/

Single-camera layout (produced by --camera):
  camera_data/traj_0/

Usage:
  python rbs_sceneflow_scripts/viser_raw_sceneflow_viewer.py \\
      --traj-dir /path/to/camera_data/traj_0

  # pick a specific camera (multi-camera layout)
  python rbs_sceneflow_scripts/viser_raw_sceneflow_viewer.py \\
      --traj-dir /path/to/camera_data/traj_0 --cam corner2

  # pick a specific ref frame
  python rbs_sceneflow_scripts/viser_raw_sceneflow_viewer.py \\
      --traj-dir /path/to/camera_data/traj_0 --ref 0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import viser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_cam_dir(traj_dir: Path, cam: str | None) -> Path:
    """Return the directory containing cam_poses.npy + sceneflow files."""
    # Multi-camera layout: traj_dir/<cam_name>/cam_poses.npy
    subdirs = [d for d in sorted(traj_dir.iterdir()) if d.is_dir()]
    cam_subdirs = [d for d in subdirs if (d / "cam_poses.npy").exists()]

    if cam_subdirs:
        if cam is not None:
            matches = [d for d in cam_subdirs if d.name == cam]
            if not matches:
                available = [d.name for d in cam_subdirs]
                raise FileNotFoundError(
                    f"Camera '{cam}' not found. Available: {available}"
                )
            return matches[0]
        # default: first camera alphabetically
        chosen = cam_subdirs[0]
        print(f"[info] Multi-camera layout detected. Using camera: {chosen.name}")
        print(f"[info] All available cameras: {[d.name for d in cam_subdirs]}")
        return chosen

    # Single-camera layout: traj_dir/cam_poses.npy
    if (traj_dir / "cam_poses.npy").exists():
        if cam is not None:
            print(f"[warn] --cam={cam} ignored: single-camera layout detected")
        return traj_dir

    raise FileNotFoundError(
        f"No cam_poses.npy found in {traj_dir} or its subdirectories.\n"
        "Make sure convert_camera_depths.py has been run."
    )


def _find_flow_file(cam_dir: Path, ref: int) -> tuple[Path, Path]:
    """Return (flow_npy_path, anchor_npy_path) for the given ref frame."""
    stem = f"scene_point_flow_ref{ref:05d}"
    flow_path   = cam_dir / f"{stem}.npy"
    anchor_path = cam_dir / f"{stem}.anchor.npy"

    if not flow_path.exists():
        # list what is available
        available = sorted(cam_dir.glob("scene_point_flow_ref*.npy"))
        avail_refs = []
        for p in available:
            if ".anchor" not in p.name:
                try:
                    avail_refs.append(int(p.stem.split("ref")[1]))
                except (ValueError, IndexError):
                    pass
        raise FileNotFoundError(
            f"Flow file not found: {flow_path}\n"
            f"Available refs: {avail_refs}"
        )
    if not anchor_path.exists():
        raise FileNotFoundError(f"Anchor file not found: {anchor_path}")

    return flow_path, anchor_path


def _list_available_refs(cam_dir: Path) -> list[int]:
    refs = []
    for p in sorted(cam_dir.glob("scene_point_flow_ref*.npy")):
        if ".anchor" not in p.name:
            try:
                refs.append(int(p.stem.split("ref")[1]))
            except (ValueError, IndexError):
                pass
    return refs


def opengl_to_opencv(p: np.ndarray) -> np.ndarray:
    q = p.copy()
    q[..., 1] *= -1.0
    q[..., 2] *= -1.0
    return q


def add_pose_axes(server: viser.ViserServer, name: str, T: np.ndarray, axis_len: float):
    o  = T[:3, 3]
    R  = T[:3, :3]
    x  = o + R[:, 0] * axis_len
    y  = o + R[:, 1] * axis_len
    z  = o + R[:, 2] * axis_len
    hx = server.scene.add_line_segments(
        f"/axes/{name}/x",
        points=np.array([[o, x]], dtype=np.float32),
        colors=np.array([[[255, 70, 70], [255, 70, 70]]], dtype=np.uint8),
        line_width=2.0,
    )
    hy = server.scene.add_line_segments(
        f"/axes/{name}/y",
        points=np.array([[o, y]], dtype=np.float32),
        colors=np.array([[[70, 255, 70], [70, 255, 70]]], dtype=np.uint8),
        line_width=2.0,
    )
    hz = server.scene.add_line_segments(
        f"/axes/{name}/z",
        points=np.array([[o, z]], dtype=np.float32),
        colors=np.array([[[70, 70, 255], [70, 70, 255]]], dtype=np.uint8),
        line_width=2.0,
    )
    return [hx, hy, hz]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Viser viewer for raw Metaworld sceneflow (.npy)")
    ap.add_argument("--traj-dir", required=True, help="Path to camera_data/traj_N/")
    ap.add_argument("--cam",  default=None, help="Camera name (multi-cam layout), e.g. corner2")
    ap.add_argument("--ref",  type=int, default=None, help="Reference frame index (default: first available)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--step", type=int, default=8,
                    help="Pixel subsampling stride (lower = more points, slower)")
    ap.add_argument("--axis-len", type=float, default=0.05)
    args = ap.parse_args()

    traj_dir = Path(args.traj_dir).resolve()
    if not traj_dir.exists():
        raise FileNotFoundError(f"traj_dir not found: {traj_dir}")

    # Locate camera subdirectory
    cam_dir = _find_cam_dir(traj_dir, args.cam)
    cam_name = cam_dir.name if cam_dir != traj_dir else "(single-cam)"
    print(f"[info] Camera dir : {cam_dir}")

    # Pick ref frame
    available_refs = _list_available_refs(cam_dir)
    if not available_refs:
        raise FileNotFoundError(
            f"No scene_point_flow_ref*.npy found in {cam_dir}.\n"
            "Run: python rbs_sceneflow_scripts/traj2sceneflow/convert_camera_depths.py <camera_data_dir>"
        )
    ref = args.ref if args.ref is not None else available_refs[0]
    print(f"[info] Available refs: {available_refs}  →  using ref={ref}")

    # Load data
    flow_path, anchor_path = _find_flow_file(cam_dir, ref)
    print(f"[info] Loading anchor : {anchor_path}")
    anchor_gl = np.load(anchor_path).astype(np.float32)   # (H, W, 3)
    print(f"[info] Loading flow   : {flow_path}")
    flow_gl   = np.load(flow_path).astype(np.float32)     # (T, H, W, 3)
    print(f"[info] Loading cam_poses")
    cam_poses = np.load(cam_dir / "cam_poses.npy").astype(np.float32)  # (T, 4, 4)

    T_flow = flow_gl.shape[0]
    T_cam  = cam_poses.shape[0]
    T      = min(T_flow, T_cam)
    print(f"[info] T={T}  flow shape={flow_gl.shape}  anchor shape={anchor_gl.shape}")

    # Convert OpenGL → OpenCV for Viser (right-hand, Y-up)
    anchor_cv = opengl_to_opencv(anchor_gl)          # (H, W, 3)
    flow_cv   = opengl_to_opencv(flow_gl[:T])         # (T, H, W, 3)

    # Subsampled indices
    H, W = anchor_cv.shape[:2]
    ys = np.arange(0, H, args.step)
    xs = np.arange(0, W, args.step)
    YY, XX = np.meshgrid(ys, xs, indexing="ij")

    anchor_sub = anchor_cv[YY, XX]                    # (Hs, Ws, 3)
    valid = (
        np.isfinite(anchor_sub).all(axis=-1) &
        (np.linalg.norm(anchor_sub, axis=-1) > 1e-6)
    )
    a   = anchor_sub.reshape(-1, 3)[valid.reshape(-1)]   # (N, 3) anchor points
    vm  = valid.reshape(-1)

    # ── Viser server ─────────────────────────────────────────────────────────
    server = viser.ViserServer(host=args.host, port=args.port)

    server.gui.add_markdown(
        f"### Raw Sceneflow Viewer\n"
        f"- **traj**: `{traj_dir.name}`\n"
        f"- **camera**: `{cam_name}`\n"
        f"- **ref frame**: `{ref}`\n"
        f"- **T**: `{T}` frames  |  **points (subsampled)**: `{a.shape[0]}`\n"
        f"- **all refs**: `{available_refs}`"
    )

    frame_gui      = server.gui.add_slider("frame",      min=0, max=max(T - 1, 0), step=1, initial_value=0)
    autoplay_gui   = server.gui.add_checkbox("autoplay",  initial_value=False)
    fps_gui        = server.gui.add_slider("fps",         min=1, max=60, step=1, initial_value=12)
    pt_size_gui    = server.gui.add_slider("point_size",  min=0.001, max=0.03, step=0.001, initial_value=0.004)
    show_anch_gui  = server.gui.add_checkbox("show_anchor",    initial_value=True)
    show_flow_gui  = server.gui.add_checkbox("show_flow",      initial_value=True)
    show_cam_gui   = server.gui.add_checkbox("show_cam_poses", initial_value=True)
    color_mode_gui = server.gui.add_dropdown("color_by",
                                             options=["displacement", "height_z", "uniform"],
                                             initial_value="displacement")

    # Static camera path
    cam_centers = cam_poses[:T, :3, 3]
    server.scene.add_point_cloud(
        "/cam/path",
        points=cam_centers,
        colors=np.tile(np.array([[255, 210, 60]], dtype=np.uint8), (T, 1)),
        point_size=0.01,
    )

    handles: dict = {"anchor": None, "flow": None, "axes": {}}

    def _make_colors(p: np.ndarray, ref_pts: np.ndarray, mode: str) -> np.ndarray:
        """Return (N, 3) uint8 colours for the flow point cloud."""
        if mode == "displacement":
            d = p - ref_pts
            mag = np.linalg.norm(d, axis=1)
            if len(mag) > 0 and float(mag.max()) > float(mag.min()):
                n = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
            else:
                n = np.zeros_like(mag)
            return np.stack([
                (255 * n).astype(np.uint8),
                (30 + 210 * (1.0 - n)).astype(np.uint8),
                (255 * (1.0 - n)).astype(np.uint8),
            ], axis=1)
        elif mode == "height_z":
            z = p[:, 2]
            mn, mx = float(z.min()), float(z.max())
            n = (z - mn) / (mx - mn + 1e-8) if mx > mn else np.zeros_like(z)
            return np.stack([
                (255 * n).astype(np.uint8),
                np.full(n.shape, 120, dtype=np.uint8),
                (255 * (1.0 - n)).astype(np.uint8),
            ], axis=1)
        else:  # uniform teal
            return np.tile(np.array([[60, 200, 200]], dtype=np.uint8), (p.shape[0], 1))

    def render(t: int):
        if handles["flow"]   is not None:
            handles["flow"].remove();   handles["flow"]   = None
        if handles["anchor"] is not None:
            handles["anchor"].remove(); handles["anchor"] = None
        for k, hs in list(handles["axes"].items()):
            for h in hs:
                h.remove()
            del handles["axes"][k]

        flow_t = flow_cv[t][YY, XX].reshape(-1, 3)[vm]   # (N, 3)

        if show_flow_gui.value:
            colors = _make_colors(flow_t, a, color_mode_gui.value)
            handles["flow"] = server.scene.add_point_cloud(
                "/flow/current",
                points=flow_t,
                colors=colors,
                point_size=float(pt_size_gui.value),
            )

        if show_anch_gui.value:
            gray = np.tile(np.array([[130, 130, 130]], dtype=np.uint8), (a.shape[0], 1))
            handles["anchor"] = server.scene.add_point_cloud(
                "/flow/anchor",
                points=a,
                colors=gray,
                point_size=float(pt_size_gui.value),
            )

        if show_cam_gui.value:
            handles["axes"]["cam_t"] = add_pose_axes(server, "cam_t", cam_poses[t], axis_len=args.axis_len)

    for ctrl in (frame_gui, pt_size_gui, show_anch_gui, show_flow_gui,
                 show_cam_gui, color_mode_gui):
        @ctrl.on_update
        def _(_evt):
            render(int(frame_gui.value))

    render(0)

    print(f"\nViser running at: http://127.0.0.1:{args.port}")
    print("If remote: ssh -L 8080:127.0.0.1:8080 <host>")
    print("Ctrl-C to quit.\n")

    try:
        while True:
            if autoplay_gui.value:
                frame_gui.value = (int(frame_gui.value) + 1) % max(T, 1)
            time.sleep(1.0 / max(float(fps_gui.value), 1.0))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
