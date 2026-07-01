#!/usr/bin/env python3
"""Prepare a tiny underwater frame sequence for D4RT smoke inference."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2


DEFAULT_SOURCE_CANDIDATES = (
    "/media/data/Research/uw_dataset/underwater_caves_sonar/extracted/undistorted_frames",
    "/media/data/Research/uw_dataset/underwater_caves_sonar/extracted/frames",
    "/media/data/Research/uw_dataset/underwater_caves_sonar/extracted/calib_frames",
)


def _frame_sort_key(path: Path) -> tuple[str, int, str]:
    match = re.search(r"(\d+)(?=\D*$)", path.stem)
    frame_id = int(match.group(1)) if match else -1
    prefix = path.stem[: match.start(1)] if match else path.stem
    return prefix, frame_id, path.name


def _find_source_dir(raw: str | None) -> Path:
    if raw:
        path = Path(raw).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Source directory does not exist: {path}")
        return path
    for candidate in DEFAULT_SOURCE_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path
    joined = "\n  ".join(DEFAULT_SOURCE_CANDIDATES)
    raise FileNotFoundError(f"No default underwater source directory found. Checked:\n  {joined}")


def _list_images(source_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    frames = [path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in exts]
    frames.sort(key=_frame_sort_key)
    if not frames:
        raise RuntimeError(f"No image frames found in {source_dir}")
    return frames


def _link_or_copy(src: Path, dst: Path, copy_files: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_files:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def _write_preview_mp4(frame_paths: list[Path], output_path: Path, fps: float) -> bool:
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        return False
    height, width = first.shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        return False
    try:
        for path in frame_paths:
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
    return output_path.exists() and output_path.stat().st_size > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=None, help="Directory containing underwater PNG/JPG frames.")
    parser.add_argument("--output-dir", default="data/aqua_smoke/underwater_caves_sonar_32")
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--copy", action="store_true", help="Copy frames instead of creating symlinks.")
    parser.add_argument("--preview-fps", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = _find_source_dir(args.source_dir)
    frames = _list_images(source_dir)

    start = max(0, int(args.start))
    stride = max(1, int(args.stride))
    num_frames = max(1, int(args.num_frames))
    selected = frames[start : start + num_frames * stride : stride]
    if len(selected) < num_frames:
        raise RuntimeError(
            f"Requested {num_frames} frames from {source_dir}, but only {len(selected)} available "
            f"from start={start}, stride={stride}."
        )

    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    prepared_paths: list[Path] = []
    for idx, src in enumerate(selected):
        dst = frames_dir / f"frame_{idx:06d}{src.suffix.lower()}"
        _link_or_copy(src.resolve(), dst, bool(args.copy))
        prepared_paths.append(dst)

    preview_path = output_dir / "preview.mp4"
    preview_ok = _write_preview_mp4(prepared_paths, preview_path, fps=float(args.preview_fps))

    manifest = {
        "name": output_dir.name,
        "source_dir": str(source_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "frames_dir": str(frames_dir.resolve()),
        "num_frames": len(prepared_paths),
        "start": start,
        "stride": stride,
        "storage": "copy" if args.copy else "symlink",
        "preview_mp4": str(preview_path.resolve()) if preview_ok else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "frames": [str(path.absolute()) for path in prepared_paths],
    }
    manifest_path = output_dir / "manifest.json"
    frames_txt_path = output_dir / "frames.txt"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    frames_txt_path.write_text("\n".join(manifest["frames"]) + "\n", encoding="utf-8")

    print(f"Prepared {len(prepared_paths)} frames")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Manifest: {manifest_path}")
    if preview_ok:
        print(f"Preview: {preview_path}")
    else:
        print("Preview: not written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
