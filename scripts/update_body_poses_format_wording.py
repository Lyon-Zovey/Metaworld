#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

OLD_SNIPPET = "body-to-world homogeneous transform (column-vec: p_world = R @ p_body + t)"
NEW_SNIPPET = "body-to-cam homogeneous transform (column-vec: p_cam = R @ p_body + t)"


def iter_meta_files(root: Path):
    for p in root.rglob("meta.json"):
        if p.parent.name.startswith("traj_"):
            yield p


def process_meta(meta_path: Path, dry_run: bool) -> bool:
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False

    if meta.get("pose_layout") != "body->cam":
        return False

    bp = meta.get("body_poses")
    if not isinstance(bp, dict):
        return False

    fmt = bp.get("format")
    if not isinstance(fmt, str) or OLD_SNIPPET not in fmt:
        return False

    new_fmt = fmt.replace(OLD_SNIPPET, NEW_SNIPPET)
    if dry_run:
        return True

    bp["format"] = new_fmt
    meta["body_poses"] = bp
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total_meta = 0
    changed = 0
    for root_str in args.roots:
        root = Path(root_str)
        root_total = 0
        root_changed = 0
        for meta_path in iter_meta_files(root):
            total_meta += 1
            root_total += 1
            if process_meta(meta_path, dry_run=args.dry_run):
                changed += 1
                root_changed += 1
        tag = "[DRY RUN] " if args.dry_run else ""
        print(f"{tag}{root}: scanned={root_total}, changed={root_changed}")

    tag = "[DRY RUN] " if args.dry_run else ""
    print(f"{tag}TOTAL scanned={total_meta}, changed={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
