#!/usr/bin/env python3
"""Download a compact WaterMask/UIIS fish subset from Hugging Face."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download


def _download(repo_id: str, filename: str, output_root: Path) -> Path:
    local_path = output_root / filename
    if local_path.exists() and local_path.stat().st_size > 0:
        return local_path
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=output_root,
        )
    )


def _fish_score(image: dict[str, Any], annotations: list[dict[str, Any]]) -> tuple[int, float, str]:
    total_area = float(sum(float(ann.get("area", 0.0)) for ann in annotations))
    return len(annotations), total_area, str(image.get("file_name", ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="LiamLian0727/UIIS")
    parser.add_argument("--output-root", default="data/watermask_uiis")
    parser.add_argument("--split", default="train", choices=("train", "val"))
    parser.add_argument("--category", default="fish")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--min-ann-area", type=float, default=64.0)
    parser.add_argument(
        "--selection",
        default="largest",
        choices=("largest", "first"),
        help="largest ranks by fish count and total fish area; first follows annotation order.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    readme_path = _download(args.repo_id, "README.md", output_root)
    ann_path = _download(args.repo_id, f"annotations/{args.split}.json", output_root)
    coco = json.loads(ann_path.read_text(encoding="utf-8"))

    cat_by_id = {int(cat["id"]): str(cat["name"]) for cat in coco["categories"]}
    category_ids = {cat_id for cat_id, name in cat_by_id.items() if name == str(args.category)}
    if not category_ids:
        raise RuntimeError(f"Category {args.category!r} not found. Available: {sorted(set(cat_by_id.values()))}")

    image_by_id = {int(image["id"]): image for image in coco["images"]}
    fish_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        if int(ann["category_id"]) not in category_ids:
            continue
        if float(ann.get("area", 0.0)) < float(args.min_ann_area):
            continue
        fish_by_image[int(ann["image_id"])].append(ann)

    candidates = [
        (image_by_id[image_id], anns)
        for image_id, anns in fish_by_image.items()
        if image_id in image_by_id and anns
    ]
    if args.selection == "largest":
        candidates.sort(key=lambda item: _fish_score(item[0], item[1]), reverse=True)
    else:
        candidates.sort(key=lambda item: int(item[0]["id"]))

    selected = candidates[: max(1, int(args.limit))]
    if not selected:
        raise RuntimeError(f"No images selected for category={args.category} split={args.split}")

    print(f"Downloading {len(selected)} {args.category} images from {args.repo_id}/{args.split}", flush=True)
    entries: list[dict[str, Any]] = []
    for idx, (image, anns) in enumerate(selected, start=1):
        filename = str(image["file_name"])
        rel_path = f"{args.split}/{filename}"
        image_path = _download(args.repo_id, rel_path, output_root)
        if idx == 1 or idx == len(selected) or idx % 25 == 0:
            print(f"[{idx:04d}/{len(selected):04d}] {rel_path}", flush=True)
        entries.append(
            {
                "split": args.split,
                "file_name": filename,
                "image_id": int(image["id"]),
                "width": int(image["width"]),
                "height": int(image["height"]),
                "image_path": str(image_path),
                "num_fish_annotations": len(anns),
                "total_fish_area": float(sum(float(ann.get("area", 0.0)) for ann in anns)),
                "annotations": anns,
            }
        )

    manifest = {
        "dataset": "WaterMask/UIIS",
        "repo_id": args.repo_id,
        "split": args.split,
        "category": args.category,
        "limit": int(args.limit),
        "min_ann_area": float(args.min_ann_area),
        "num_images": len(entries),
        "readme_path": str(readme_path),
        "annotation_path": str(ann_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "categories": coco["categories"],
        "entries": entries,
    }
    manifest_path = output_root / f"fish_subset_{args.split}_{len(entries):04d}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
