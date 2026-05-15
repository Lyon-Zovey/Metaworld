#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import viser


class FlowDecoder:
    def __init__(self, video_path: Path, anchor: np.ndarray, sidecar: dict):
        self.reader = imageio.get_reader(str(video_path))
        self.anchor = anchor.astype(np.float32)
        self.max_scale = float(sidecar.get("max_scale", 1.0))
        self.bits = int(sidecar.get("bits", 8))
        self.orig_w = int(sidecar.get("orig_w", anchor.shape[1]))
        self.qmax = (1 << self.bits) - 1
        self.qmid = self.qmax / 2.0
        self.cache: dict[int, np.ndarray] = {}

    def close(self):
        self.reader.close()

    def get_frame_count(self) -> int:
        meta = self.reader.get_meta_data()
        nframes = meta.get("nframes", None)
        if isinstance(nframes, int) and nframes > 0:
            return nframes
        cnt = 0
        for _ in self.reader:
            cnt += 1
        self.reader.close()
        self.reader = imageio.get_reader(self.reader.request.filename)
        return cnt

    def get_flow(self, t: int) -> np.ndarray:
        if t in self.cache:
            return self.cache[t]

        raw = np.asarray(self.reader.get_data(t), dtype=np.float32)
        rgb = raw[:, :self.orig_w, :]
        a8 = raw[:, self.orig_w:, 0]

        alpha_val = float(a8[0, 0]) / 255.0 * self.qmax
        scale_t = alpha_val / self.qmax * self.max_scale

        norm = rgb / self.qmid - 1.0
        delta = norm * np.float32(scale_t)
        flow = self.anchor + delta
        if alpha_val == 0:
            flow = self.anchor.copy()

        self.cache[t] = flow.astype(np.float32)
        return self.cache[t]


def opengl_to_opencv_points(p: np.ndarray) -> np.ndarray:
    q = p.copy()
    q[..., 1] *= -1.0
    q[..., 2] *= -1.0
    return q


def make_frame_path(T: np.ndarray, axis_len: float = 0.05):
    R = T[:3, :3]
    t = T[:3, 3]
    x = t + R[:, 0] * axis_len
    y = t + R[:, 1] * axis_len
    z = t + R[:, 2] * axis_len
    return t, x, y, z


def add_pose_axes(server: viser.ViserServer, name: str, T: np.ndarray, axis_len: float):
    o, x, y, z = make_frame_path(T, axis_len=axis_len)
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


def pick_flow_bundle(traj_dir: Path, ref: int):
    stem = f"scene_point_flow_ref{ref:05d}"
    anchor = traj_dir / f"{stem}.anchor.npy"

    cands = list(traj_dir.glob(f"{stem}_*.mp4"))
    if len(cands) != 1:
        raise FileNotFoundError(f"Expected exactly 1 flow video for ref={ref}, got {len(cands)}: {cands}")
    video = cands[0]
    sidecar = video.with_suffix(".json")
    if not anchor.exists() or not sidecar.exists():
        raise FileNotFoundError(f"Missing one of: {anchor}, {sidecar}")
    return anchor, video, sidecar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-dir", type=str, required=True)
    ap.add_argument("--ref", type=int, default=0, help="sceneflow reference frame id, e.g. 0/18/35/52/70")
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--step", type=int, default=8, help="subsample stride for flow points")
    ap.add_argument("--axis-len", type=float, default=0.05)
    args = ap.parse_args()

    traj_dir = Path(args.traj_dir)
    meta = json.loads((traj_dir / "meta.json").read_text())

    cam_poses = np.load(traj_dir / "cam_poses.npy").astype(np.float32)
    pose_hammer = np.load(traj_dir / "pose_hammer.npy").astype(np.float32)
    pose_hbody = np.load(traj_dir / "pose_hammerbody.npy").astype(np.float32)

    anchor_path, flow_video, sidecar_path = pick_flow_bundle(traj_dir, args.ref)
    anchor_gl = np.load(anchor_path).astype(np.float32)
    sidecar = json.loads(sidecar_path.read_text())
    dec = FlowDecoder(flow_video, anchor_gl, sidecar)

    T = int(sidecar.get("shape_THW", [cam_poses.shape[0]])[0])
    T = min(T, cam_poses.shape[0], pose_hammer.shape[0], pose_hbody.shape[0])

    anchor = opengl_to_opencv_points(anchor_gl)

    ys = np.arange(0, anchor.shape[0], args.step)
    xs = np.arange(0, anchor.shape[1], args.step)
    YY, XX = np.meshgrid(ys, xs, indexing="ij")
    anchor_sub = anchor[YY, XX]
    valid = np.isfinite(anchor_sub).all(axis=-1) & (np.linalg.norm(anchor_sub, axis=-1) > 1e-6)

    a = anchor_sub.reshape(-1, 3)
    vm = valid.reshape(-1)
    a = a[vm]

    server = viser.ViserServer(host=args.host, port=args.port)

    server.gui.add_markdown(
        f"### traj viewer\n"
        f"- traj: `{traj_dir}`\n"
        f"- flow ref: `{args.ref}`\n"
        f"- camera convention: `{meta.get('camera_convention', 'unknown')}`\n"
        f"- flow convention: `{meta.get('flow_convention', 'unknown')}`"
    )

    frame_gui = server.gui.add_slider("frame", min=0, max=max(T - 1, 0), step=1, initial_value=0)
    autoplay_gui = server.gui.add_checkbox("autoplay", initial_value=True)
    fps_gui = server.gui.add_slider("fps", min=1, max=60, step=1, initial_value=12)
    point_size_gui = server.gui.add_slider("point_size", min=0.001, max=0.03, step=0.001, initial_value=0.004)
    show_anchor_gui = server.gui.add_checkbox("show_anchor", initial_value=True)
    show_flow_gui = server.gui.add_checkbox("show_flow", initial_value=True)
    show_cam_gui = server.gui.add_checkbox("show_cam_poses", initial_value=True)
    show_obj_gui = server.gui.add_checkbox("show_obj_pose", initial_value=True)

    handles = {"anchor": None, "flow": None, "axes": {}}

    cam_centers = cam_poses[:T, :3, 3]
    server.scene.add_point_cloud(
        "/cam/path",
        points=cam_centers,
        colors=np.tile(np.array([[255, 210, 60]], dtype=np.uint8), (cam_centers.shape[0], 1)),
        point_size=0.01,
    )

    def render(t: int):
        if handles["flow"] is not None:
            handles["flow"].remove()
            handles["flow"] = None
        if handles["anchor"] is not None:
            handles["anchor"].remove()
            handles["anchor"] = None

        for k, hs in list(handles["axes"].items()):
            for h in hs:
                h.remove()
            del handles["axes"][k]

        flow_gl_t = dec.get_flow(t)
        flow_cv_t = opengl_to_opencv_points(flow_gl_t)
        p = flow_cv_t[YY, XX].reshape(-1, 3)[vm]

        if show_flow_gui.value:
            d = p - a
            mag = np.linalg.norm(d, axis=1)
            if len(mag) > 0 and float(mag.max()) > float(mag.min()):
                n = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
            else:
                n = np.zeros_like(mag)
            colors = np.stack(
                [
                    (255 * n).astype(np.uint8),
                    (30 + 210 * (1.0 - n)).astype(np.uint8),
                    (255 * (1.0 - n)).astype(np.uint8),
                ],
                axis=1,
            )
            handles["flow"] = server.scene.add_point_cloud(
                "/flow/current",
                points=p,
                colors=colors,
                point_size=float(point_size_gui.value),
            )

        if show_anchor_gui.value:
            gray = np.tile(np.array([[130, 130, 130]], dtype=np.uint8), (a.shape[0], 1))
            handles["anchor"] = server.scene.add_point_cloud(
                "/flow/anchor",
                points=a,
                colors=gray,
                point_size=float(point_size_gui.value),
            )

        if show_cam_gui.value:
            handles["axes"]["cam_t"] = add_pose_axes(server, "cam_t", cam_poses[t], axis_len=args.axis_len)

        if show_obj_gui.value:
            handles["axes"]["obj_hammer"] = add_pose_axes(server, "obj_hammer", pose_hammer[t], axis_len=args.axis_len)
            handles["axes"]["obj_hammerbody"] = add_pose_axes(server, "obj_hammerbody", pose_hbody[t], axis_len=args.axis_len)

    @frame_gui.on_update
    def _(_evt):
        render(int(frame_gui.value))

    @point_size_gui.on_update
    def _(_evt):
        render(int(frame_gui.value))

    @show_anchor_gui.on_update
    def _(_evt):
        render(int(frame_gui.value))

    @show_flow_gui.on_update
    def _(_evt):
        render(int(frame_gui.value))

    @show_cam_gui.on_update
    def _(_evt):
        render(int(frame_gui.value))

    @show_obj_gui.on_update
    def _(_evt):
        render(int(frame_gui.value))

    render(0)

    print(f"Viser running at: http://127.0.0.1:{args.port}")
    print("If remote machine, use: ssh -L 8080:127.0.0.1:8080 <host>")

    try:
        while True:
            if autoplay_gui.value:
                frame_gui.value = (int(frame_gui.value) + 1) % max(T, 1)
            time.sleep(1.0 / max(float(fps_gui.value), 1.0))
    except KeyboardInterrupt:
        pass
    finally:
        dec.close()


if __name__ == "__main__":
    main()
