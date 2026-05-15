"""Quick sanity viz: overlay mask (red) on rgb middle frame."""
import sys
import subprocess
from pathlib import Path
import numpy as np
import imageio_ffmpeg


def read_all_frames(mp4: Path) -> np.ndarray:
    reader = imageio_ffmpeg.read_frames(str(mp4), pix_fmt="rgb24")
    meta = next(reader)
    w, h = meta["size"]
    frames = []
    for raw in reader:
        frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3))
    return np.stack(frames)


def read_mask_npz(npz_path: Path) -> np.ndarray:
    arr = np.load(str(npz_path))["mask"]
    if arr.ndim != 3:
        raise ValueError(f"mask array must be (T,H,W), got shape={arr.shape}")
    return arr


def main() -> int:
    from PIL import Image
    for traj_dir in sys.argv[1:]:
        d = Path(traj_dir)
        rgb = read_all_frames(d / "rgb.mp4")

        meta_path = d / "meta.json"
        if meta_path.exists():
            meta = __import__("json").loads(meta_path.read_text())
            mask_file = meta.get("target_obj_mask", {}).get("file")
            if not mask_file:
                raise ValueError(f"{d}: meta.json missing target_obj_mask.file")
            mask = read_mask_npz(d / mask_file)
        else:
            cands = sorted(d.glob("mask_*.npz"))
            if not cands:
                raise FileNotFoundError(f"{d}: no mask_*.npz found")
            mask = read_mask_npz(cands[0])

        T = min(rgb.shape[0], mask.shape[0])
        mid = T // 2
        r = rgb[mid].copy()
        m = mask[mid] > 127
        r[m] = (0.5 * r[m] + 0.5 * np.array([255, 0, 0], dtype=np.uint8)).astype(np.uint8)
        out = d / "_preview_overlay_mid.png"
        Image.fromarray(r).save(out)
        cov = m.mean() * 100
        print(f"{d}: T={T} mid={mid} mask_coverage={cov:.2f}%  ->  {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
