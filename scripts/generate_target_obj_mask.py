"""Post-process existing datasets_500 to add a binary target-object mask video per trajectory.

For each task under <dataset_root>:
    <dataset_root>/<task>/camera_data/<traj>/
        seg.b2nd                      # (T, H, W) int32 segmentation ids
        traj_task.json (or meta.json) # actors = [{seg_id, name}]
        rgb.mp4                       # reference video (to match size/fps)
    -> writes:
        target_obj_mask.mp4           # (T, H, W) strict binary {0,255} grayscale H.264 (lossless yuv444p)
        meta.json (renamed from traj_task.json if present), with two new fields:
            "target_object": {"body_names": [...], "seg_ids": [...]}
            "target_obj_mask": {
                "file": "target_obj_mask.mp4",
                "format": "h264_lossless_yuv444p",
                "binary_values": [0, 255],
                "fps": <float>, "num_frames": <int>, "height": <int>, "width": <int>
            }

Mapping from task -> target body names is read from a separate JSON
(see scripts/target_objects.json).

Usage:
    python scripts/generate_target_obj_mask.py \
        --dataset-root datasets_500 \
        --mapping scripts/target_objects.json \
        [--tasks basketball-v3 hammer-v3] \
        [--dry-run] [--overwrite] [--limit-trajs N] [--workers N]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import blosc2
import imageio_ffmpeg
import numpy as np


FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
FFPROBE = "ffprobe"


def read_rgb_fps(rgb_mp4: Path) -> float:
    """Read frame rate via ffprobe; falls back to 20.0 only if parsing fails."""
    proc = subprocess.run(
        [
            FFPROBE, "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(rgb_mp4),
        ],
        capture_output=True,
        text=True,
    )
    rate = (proc.stdout or "").strip()
    if "/" in rate:
        num, den = rate.split("/", 1)
        try:
            n, d = float(num), float(den)
            if d > 0 and n > 0:
                return n / d
        except ValueError:
            pass
    try:
        v = float(rate)
        if v > 0:
            return v
    except ValueError:
        pass
    return 20.0


def write_gray_mp4(frames_u8: np.ndarray, dst: Path, fps: float) -> None:
    """Write a strictly binary {0,255} grayscale H.264 video.

    Uses lossless x264 + yuv444p so chroma subsampling can't bleed
    edge pixels into intermediate gray values.
    """
    t, h, w = frames_u8.shape
    assert frames_u8.dtype == np.uint8
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "gray",
        "-r", f"{fps:g}",
        "-i", "-",
        "-an",
        "-vcodec", "libx264",
        "-pix_fmt", "yuv444p",
        "-preset", "veryfast",
        "-qp", "0",
        str(dst),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        proc.stdin.write(frames_u8.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed ({ret}) for {dst}")


def process_one_traj(
    traj_dir: Path,
    target_body_names: list[str],
    child_of_parents: list[str],
    overwrite: bool,
    dry_run: bool,
) -> tuple[str, str]:
    try:
        meta_json = traj_dir / "meta.json"
        legacy_json = traj_dir / "traj_task.json"
        seg_b2nd = traj_dir / "seg.b2nd"
        rgb_mp4 = traj_dir / "rgb.mp4"
        out_mp4 = traj_dir / "target_obj_mask.mp4"

        if meta_json.is_file():
            src_json = meta_json
        elif legacy_json.is_file():
            src_json = legacy_json
        else:
            return (str(traj_dir), "skip:no_meta_or_traj_task_json")
        if not seg_b2nd.is_file():
            return (str(traj_dir), "skip:no_seg_b2nd")
        if not rgb_mp4.is_file():
            return (str(traj_dir), "skip:no_rgb_mp4")
        if out_mp4.is_file() and meta_json.is_file() and not overwrite:
            existing = json.loads(meta_json.read_text())
            if "target_object" in existing and "target_obj_mask" in existing:
                return (str(traj_dir), "skip:exists")

        meta = json.loads(src_json.read_text())
        actors = meta.get("actors", [])

        seg_id_to_idx = {a["seg_id"]: i for i, a in enumerate(actors)}
        wanted = set(target_body_names)
        matched: list[dict] = []
        seen_ids: set[int] = set()

        for i, a in enumerate(actors):
            if not a["name"].startswith("body:"):
                continue
            short = a["name"][len("body:"):]
            if short in wanted:
                if a["seg_id"] not in seen_ids:
                    matched.append(a)
                    seen_ids.add(a["seg_id"])
                if short in child_of_parents and i + 1 < len(actors):
                    nxt = actors[i + 1]
                    if nxt["name"].startswith("body:") and nxt["seg_id"] not in seen_ids:
                        matched.append(nxt)
                        seen_ids.add(nxt["seg_id"])

        if not matched:
            return (str(traj_dir), f"skip:no_matching_body({target_body_names})")

        seg_ids = sorted({int(a["seg_id"]) for a in matched})
        body_names = []
        for a in matched:
            short = a["name"][len("body:"):]
            body_names.append(short if short else f"<child_of:seg_id={a['seg_id']}>")

        if dry_run:
            return (str(traj_dir), f"dry:{body_names}->{seg_ids}")

        seg = blosc2.open(str(seg_b2nd))[:]
        mask = np.isin(seg, seg_ids)
        frames = (mask.astype(np.uint8) * 255)
        T, H, W = frames.shape
        fps = read_rgb_fps(rgb_mp4)
        write_gray_mp4(frames, out_mp4, fps)

        meta["target_object"] = {
            "body_names": body_names,
            "seg_ids": seg_ids,
        }
        meta["target_obj_mask"] = {
            "file": out_mp4.name,
            "format": "h264_lossless_yuv444p",
            "binary_values": [0, 255],
            "fps": float(fps),
            "num_frames": int(T),
            "height": int(H),
            "width": int(W),
        }

        meta_json.write_text(json.dumps(meta, indent=2))
        if legacy_json.exists() and legacy_json != meta_json:
            legacy_json.unlink()

        return (str(traj_dir), f"ok:{body_names}->{seg_ids}")
    except Exception as e:
        return (str(traj_dir), f"err:{type(e).__name__}:{e}")


def discover_jobs(
    dataset_root: Path,
    mapping: dict,
    task_ids: list[str],
    limit: int | None,
) -> list[tuple[Path, list[str], list[str]]]:
    jobs: list[tuple[Path, list[str], list[str]]] = []
    for task_id in task_ids:
        if task_id not in mapping:
            print(f"[WARN] {task_id} not in mapping; skip")
            continue
        task_dir = dataset_root / task_id
        cam = task_dir / "camera_data"
        if not cam.is_dir():
            print(f"[WARN] {cam} missing; skip")
            continue
        bodies = mapping[task_id]["target_bodies"]
        child_of = mapping[task_id].get("child_of", [])
        trajs = sorted(
            (p for p in cam.iterdir() if p.is_dir() and p.name.startswith("traj_")),
            key=lambda p: int(p.name.split("_")[1]) if p.name.split("_")[1].isdigit() else 1 << 30,
        )
        if limit is not None:
            trajs = trajs[:limit]
        for t in trajs:
            jobs.append((t, bodies, child_of))
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=Path("datasets_500"))
    ap.add_argument("--mapping", type=Path, default=Path("scripts/target_objects.json"))
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="subset of task ids; omit to run all tasks in mapping")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit-trajs", type=int, default=None,
                    help="only process first N trajs per task")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 8) // 2),
                    help="global process pool size for parallel traj processing")
    args = ap.parse_args()

    mapping_all = json.loads(args.mapping.read_text())
    mapping = {k: v for k, v in mapping_all.items() if not k.startswith("_")}

    task_ids = args.tasks or sorted(mapping.keys())
    jobs = discover_jobs(args.dataset_root, mapping, task_ids, args.limit_trajs)
    print(f"Discovered {len(jobs)} traj jobs across {len(task_ids)} tasks; workers={args.workers}")
    if not jobs:
        return 0

    t0 = time.time()
    by_task: dict[str, dict[str, int]] = {}
    failures: list[tuple[str, str]] = []
    done = 0
    total = len(jobs)
    last_print = 0.0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(process_one_traj, traj_dir, bodies, child_of, args.overwrite, args.dry_run): (traj_dir, bodies)
            for traj_dir, bodies, child_of in jobs
        }
        for f in as_completed(futs):
            traj_dir, _ = futs[f]
            try:
                p, status = f.result()
            except Exception as e:
                p, status = (str(traj_dir), f"err:executor:{type(e).__name__}:{e}")
            done += 1
            task_id = Path(p).parents[1].name
            d = by_task.setdefault(task_id, {"ok": 0, "skip": 0, "err": 0, "dry": 0})
            kind = status.split(":", 1)[0]
            d[kind] = d.get(kind, 0) + 1
            if not status.startswith(("ok", "skip", "dry")):
                failures.append((p, status))
            now = time.time()
            if now - last_print > 5.0 or done == total:
                rate = done / max(now - t0, 1e-6)
                eta = (total - done) / max(rate, 1e-6)
                print(f"  [{done}/{total}] rate={rate:.1f}/s eta={eta/60:.1f}m  last={Path(p).parent.parent.parent.name}/{Path(p).name}:{kind}")
                last_print = now

    print("\n=== per-task summary ===")
    for tid in sorted(by_task):
        d = by_task[tid]
        print(f"  {tid:32s} ok={d.get('ok',0):4d} skip={d.get('skip',0):4d} err={d.get('err',0):3d} dry={d.get('dry',0):4d}")
    total_ok = sum(d.get("ok", 0) for d in by_task.values())
    total_skip = sum(d.get("skip", 0) for d in by_task.values())
    total_err = sum(d.get("err", 0) for d in by_task.values())
    total_dry = sum(d.get("dry", 0) for d in by_task.values())
    print(f"\n[summary] ok={total_ok} skip={total_skip} err={total_err} dry={total_dry}  elapsed={(time.time()-t0)/60:.1f}m")

    if failures:
        print(f"\n[failures] showing first 20 of {len(failures)}:")
        for p, s in failures[:20]:
            print(f"  · {p}: {s}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
