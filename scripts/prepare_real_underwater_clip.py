#!/usr/bin/env python3
"""Prepare a short real underwater frame clip from images or videos."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def _sort_key(path: Path) -> tuple[str, int, str]:
    match = re.search(r"(\d+)(?=\D*$)", path.stem)
    frame_id = int(match.group(1)) if match else -1
    prefix = path.stem[: match.start(1)] if match else path.stem
    return prefix, frame_id, path.name


def _iter_images(path: Path, recursive: bool) -> list[Path]:
    iterator: Iterable[Path] = path.rglob("*") if recursive else path.iterdir()
    frames = [item for item in iterator if item.is_file() and item.suffix.lower() in IMAGE_EXTS]
    frames.sort(key=_sort_key)
    return frames


def _iter_videos(path: Path, recursive: bool) -> list[Path]:
    iterator: Iterable[Path] = path.rglob("*") if recursive else path.iterdir()
    videos = [item for item in iterator if item.is_file() and item.suffix.lower() in VIDEO_EXTS]
    videos.sort(key=_sort_key)
    return videos


def _read_image(path: Path, output_hw: tuple[int, int] | None) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    if output_hw is not None:
        h, w = output_hw
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    return bgr


def _write_preview(frame_paths: list[Path], output_path: Path, fps: float) -> bool:
    if not frame_paths:
        return False
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        return False
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        return False
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is not None:
                writer.write(frame)
    finally:
        writer.release()
    return output_path.exists() and output_path.stat().st_size > 0


def _extract_from_images(
    *,
    image_paths: list[Path],
    output_dir: Path,
    start: int,
    stride: int,
    num_frames: int,
    output_hw: tuple[int, int] | None,
) -> list[Path]:
    selected = image_paths[int(start) :: max(1, int(stride))]
    if num_frames > 0:
        selected = selected[: int(num_frames)]
    if not selected:
        raise RuntimeError("No frames selected from image directory.")
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for idx, src in enumerate(selected):
        frame = _read_image(src, output_hw=output_hw)
        dst = frames_dir / f"frame_{idx:06d}.png"
        cv2.imwrite(str(dst), frame)
        out_paths.append(dst)
    return out_paths


def _extract_from_video(
    *,
    video_path: Path,
    output_dir: Path,
    start: int,
    stride: int,
    num_frames: int,
    output_hw: tuple[int, int] | None,
) -> tuple[list[Path], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 10.0
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    idx = -1
    out_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx < int(start):
                continue
            if (idx - int(start)) % max(1, int(stride)) != 0:
                continue
            if output_hw is not None:
                h, w = output_hw
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            dst = frames_dir / f"frame_{out_idx:06d}.png"
            cv2.imwrite(str(dst), frame)
            out_paths.append(dst)
            out_idx += 1
            if num_frames > 0 and out_idx >= int(num_frames):
                break
    finally:
        cap.release()
    if not out_paths:
        raise RuntimeError(f"No frames extracted from video: {video_path}")
    return out_paths, fps / float(max(1, int(stride)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Image directory, video file, or directory containing videos.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--video-index", type=int, default=0)
    parser.add_argument("--output-width", type=int, default=0)
    parser.add_argument("--output-height", type=int, default=0)
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--clip-name", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_hw = None
    if int(args.output_width) > 0 and int(args.output_height) > 0:
        output_hw = (int(args.output_height), int(args.output_width))

    fps = 10.0
    source_kind = ""
    source_path = input_path
    if input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTS:
        source_kind = "video"
        frame_paths, fps = _extract_from_video(
            video_path=input_path,
            output_dir=output_dir,
            start=int(args.start),
            stride=int(args.stride),
            num_frames=int(args.num_frames),
            output_hw=output_hw,
        )
    elif input_path.is_dir():
        image_paths = _iter_images(input_path, recursive=bool(args.recursive))
        if image_paths:
            source_kind = "image_directory"
            frame_paths = _extract_from_images(
                image_paths=image_paths,
                output_dir=output_dir,
                start=int(args.start),
                stride=int(args.stride),
                num_frames=int(args.num_frames),
                output_hw=output_hw,
            )
        else:
            videos = _iter_videos(input_path, recursive=bool(args.recursive))
            if not videos:
                raise RuntimeError(f"No images or videos found under {input_path}")
            video_idx = int(np.clip(int(args.video_index), 0, len(videos) - 1))
            source_path = videos[video_idx]
            source_kind = "video_in_directory"
            frame_paths, fps = _extract_from_video(
                video_path=source_path,
                output_dir=output_dir,
                start=int(args.start),
                stride=int(args.stride),
                num_frames=int(args.num_frames),
                output_hw=output_hw,
            )
    else:
        raise RuntimeError(f"Unsupported input: {input_path}")

    preview_path = output_dir / "preview.mp4"
    preview_ok = _write_preview(frame_paths, preview_path, fps=max(1.0, min(30.0, float(fps))))
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    h, w = first.shape[:2] if first is not None else (0, 0)
    manifest = {
        "name": args.clip_name or output_dir.name,
        "dataset": args.dataset_name or input_path.name,
        "source_kind": source_kind,
        "source_path": str(source_path.resolve()),
        "frames_dir": str((output_dir / "frames").resolve()),
        "frames": [str(path.resolve()) for path in frame_paths],
        "num_frames": len(frame_paths),
        "height": int(h),
        "width": int(w),
        "fps": float(fps),
        "preview_mp4": str(preview_path.resolve()) if preview_ok else None,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "frames.txt").write_text("\n".join(manifest["frames"]) + "\n", encoding="utf-8")
    print(f"Prepared {len(frame_paths)} frames: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Preview: {preview_path if preview_ok else 'FAILED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
