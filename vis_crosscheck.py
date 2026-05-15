"""
Cross-check visualizer:
  - sceneflow from datasets_500/traj_1  (OpenGL anchor, no flip)
  - cam2world   from test_task/traj_0   (OpenGL c2w, cam_poses.npy)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import viser
import viser.transforms as tf

import sys
sys.path.insert(0, str(Path(__file__).parent / "rbs_sceneflow_scripts/traj2sceneflow"))
from flow_compress import decompress_one_flow

FLOW_DIR = Path("/mnt2/liangzhuowei/Metaworld/datasets_500/door-open-v3/camera_data/traj_1")
CAM_DIR  = Path("/mnt2/liangzhuowei/Metaworld/datasets/test_task/door-open-v3/camera_data/traj_0")
PORT = 8091
DOWNSAMPLE = 8

# --- load sceneflow (OpenGL) ---
anchors = sorted(FLOW_DIR.glob("scene_point_flow_ref*.anchor.npy"))
anchor_path = anchors[0]
anchor = np.load(str(anchor_path))
stem = anchor_path.stem.replace(".anchor", "")
video = sorted(FLOW_DIR.glob(f"{stem}*.mp4"))[0]
flow = decompress_one_flow(video, anchor).astype(np.float32)
T, H, W, _ = flow.shape
print(f"flow: {flow.shape}  from {FLOW_DIR.name}")

# --- load cam2world (OpenGL, from test_task) ---
cam2world = np.load(str(CAM_DIR / "cam_poses.npy")).astype(np.float32)
print(f"cam2world: {cam2world.shape}  from {CAM_DIR.name}")

# Trim to min T
T = min(T, cam2world.shape[0])
flow = flow[:T]
cam2world = cam2world[:T]

ds = DOWNSAMPLE
flow_ds = flow[:, ::ds, ::ds, :]

server = viser.ViserServer(host="0.0.0.0", port=PORT)
print(f"Viser running at http://0.0.0.0:{PORT}")

frame_slider = server.gui.add_slider("Frame", min=0, max=T-1, step=1, initial_value=0)
point_size_slider = server.gui.add_slider("Point size", min=0.001, max=0.05, step=0.001, initial_value=0.005)

pc_handle = None
cam_handle = None

def render_frame(t):
    global pc_handle, cam_handle
    pts_cam = flow_ds[t].reshape(-1, 3)
    valid = np.isfinite(pts_cam).all(1) & (np.abs(pts_cam) < 100).all(1)
    pts_cam = pts_cam[valid]
    c2w = cam2world[t]
    pts = (c2w[:3,:3] @ pts_cam.T).T + c2w[:3,3]
    if pc_handle: pc_handle.remove()
    pc_handle = server.scene.add_point_cloud("/pts", points=pts,
                    colors=np.full((len(pts),3), 180, dtype=np.uint8),
                    point_size=point_size_slider.value)
    if cam_handle: cam_handle.remove()
    cam_handle = server.scene.add_frame("/cam",
                    wxyz=tf.SO3.from_matrix(c2w[:3,:3]).wxyz,
                    position=c2w[:3,3], axes_length=0.05, axes_radius=0.002)

@frame_slider.on_update
def _(_): render_frame(int(frame_slider.value))

render_frame(0)

cam_pos = cam2world[:, :3, 3]
server.scene.add_line_segments("/cam_traj",
    points=np.stack([cam_pos[:-1], cam_pos[1:]], axis=1),
    colors=np.array([50,130,255], dtype=np.uint8))

try:
    while True: time.sleep(1)
except KeyboardInterrupt:
    pass
