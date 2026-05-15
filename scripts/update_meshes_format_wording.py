#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

OLD_SNIPPET = "p_world = pose_<obj>[t] @ [v; 1]"
NEW_SNIPPET = "p_cam = pose_<obj>[t] @ [v; 1]"


def iter_meta_files(root: Path):
    for p in root.rglob("meta.json"):
        if p.parent.name.startswith("traj_"):
            yield p


def process_meta(meta_path: Path) -> bool:
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False

    meshes = meta.get("meshes")
    if not isinstance(meshes, dict):
        return False

    fmt = meshes.get("format")
    if not isinstance(fmt, str):
        return False

    if OLD_SNIPPET not in fmt:
        return False

    meshes["format"] = fmt.replace(OLD_SNIPPET, NEW_SNIPPET)
    meta["meshes"] = meshes
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True)
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
            if process_meta(meta_path):
                changed += 1
                root_changed += 1
        print(f"{root}: scanned={root_total}, changed={root_changed}")

    print(f"TOTAL scanned={total_meta}, changed={changed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
