"""Synthetic underwater transient augmentation wrapper for Aqua-D4RT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .seeding import SeededDatasetMixin


@dataclass
class UnderwaterTransientConfig:
    enabled: bool = False
    train_only: bool = True
    apply_probability: float = 1.0
    supervise_clean_queries: bool = True
    water_color_rgb: tuple[float, float, float] = (0.04, 0.38, 0.48)
    attenuation_min: float = 0.45
    attenuation_max: float = 0.85
    contrast_min: float = 0.75
    contrast_max: float = 1.0
    dynamic_objects_enabled: bool = True
    dynamic_object_probability: float = 0.7
    dynamic_object_count_min: int = 1
    dynamic_object_count_max: int = 3
    dynamic_radius_frac_min: float = 0.035
    dynamic_radius_frac_max: float = 0.12
    dynamic_alpha_min: float = 0.35
    dynamic_alpha_max: float = 0.8
    particles_enabled: bool = True
    particle_probability: float = 0.9
    particles_per_frame_min: int = 40
    particles_per_frame_max: int = 180
    particle_radius_min: float = 0.75
    particle_radius_max: float = 2.25
    particle_alpha_min: float = 0.25
    particle_alpha_max: float = 0.75


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "enabled"}
    return bool(value)


def _float_pair(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return default
    lo = float(value[0])
    hi = float(value[1])
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _int_pair(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    lo, hi = _float_pair(value, (float(default[0]), float(default[1])))
    return max(0, int(round(lo))), max(0, int(round(hi)))


def _rgb_tuple(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return default
    return tuple(float(np.clip(float(v), 0.0, 1.0)) for v in value)  # type: ignore[return-value]


def underwater_transient_cfg_from_train_config(cfg: Any) -> UnderwaterTransientConfig:
    path = "augmentation.underwater_transient"
    dyn_path = f"{path}.dynamic_objects"
    particle_path = f"{path}.particles"
    attenuation = _float_pair(cfg.get_path(f"{path}.attenuation_min_max", [0.45, 0.85]), (0.45, 0.85))
    contrast = _float_pair(cfg.get_path(f"{path}.contrast_min_max", [0.75, 1.0]), (0.75, 1.0))
    dyn_count = _int_pair(cfg.get_path(f"{dyn_path}.count_min_max", [1, 3]), (1, 3))
    dyn_radius = _float_pair(cfg.get_path(f"{dyn_path}.radius_frac_min_max", [0.035, 0.12]), (0.035, 0.12))
    dyn_alpha = _float_pair(cfg.get_path(f"{dyn_path}.alpha_min_max", [0.35, 0.8]), (0.35, 0.8))
    particle_count = _int_pair(cfg.get_path(f"{particle_path}.per_frame_min_max", [40, 180]), (40, 180))
    particle_radius = _float_pair(cfg.get_path(f"{particle_path}.radius_px_min_max", [0.75, 2.25]), (0.75, 2.25))
    particle_alpha = _float_pair(cfg.get_path(f"{particle_path}.alpha_min_max", [0.25, 0.75]), (0.25, 0.75))
    return UnderwaterTransientConfig(
        enabled=_truthy(cfg.get_path(f"{path}.enabled", False)),
        train_only=_truthy(cfg.get_path(f"{path}.train_only", True)),
        apply_probability=float(cfg.get_path(f"{path}.apply_probability", 1.0)),
        supervise_clean_queries=_truthy(cfg.get_path(f"{path}.supervise_clean_queries", True)),
        water_color_rgb=_rgb_tuple(cfg.get_path(f"{path}.water_color_rgb", [0.04, 0.38, 0.48]), (0.04, 0.38, 0.48)),
        attenuation_min=attenuation[0],
        attenuation_max=attenuation[1],
        contrast_min=contrast[0],
        contrast_max=contrast[1],
        dynamic_objects_enabled=_truthy(cfg.get_path(f"{dyn_path}.enabled", True)),
        dynamic_object_probability=float(cfg.get_path(f"{dyn_path}.probability", 0.7)),
        dynamic_object_count_min=dyn_count[0],
        dynamic_object_count_max=max(dyn_count[0], dyn_count[1]),
        dynamic_radius_frac_min=dyn_radius[0],
        dynamic_radius_frac_max=max(dyn_radius[0], dyn_radius[1]),
        dynamic_alpha_min=dyn_alpha[0],
        dynamic_alpha_max=max(dyn_alpha[0], dyn_alpha[1]),
        particles_enabled=_truthy(cfg.get_path(f"{particle_path}.enabled", True)),
        particle_probability=float(cfg.get_path(f"{particle_path}.probability", 0.9)),
        particles_per_frame_min=particle_count[0],
        particles_per_frame_max=max(particle_count[0], particle_count[1]),
        particle_radius_min=particle_radius[0],
        particle_radius_max=max(particle_radius[0], particle_radius[1]),
        particle_alpha_min=particle_alpha[0],
        particle_alpha_max=max(particle_alpha[0], particle_alpha[1]),
    )


def _clone_sample(sample: dict[str, Any]) -> dict[str, Any]:
    out = dict(sample)
    for key in ("query", "target", "mask", "query_stats", "augment_info"):
        value = sample.get(key)
        if isinstance(value, dict):
            out[key] = dict(value)
    return out


def _ellipse_mask(height: int, width: int, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    dist = ((xx - float(cx)) / max(float(rx), 1e-6)) ** 2 + ((yy - float(cy)) / max(float(ry), 1e-6)) ** 2
    return dist <= 1.0


def _disk_mask(height: int, width: int, cx: float, cy: float, radius: float) -> tuple[slice, slice, np.ndarray]:
    r = max(0.5, float(radius))
    x0 = max(0, int(np.floor(cx - r - 1)))
    x1 = min(width, int(np.ceil(cx + r + 2)))
    y0 = max(0, int(np.floor(cy - r - 1)))
    y1 = min(height, int(np.ceil(cy + r + 2)))
    yy, xx = np.ogrid[y0:y1, x0:x1]
    local = (xx - float(cx)) ** 2 + (yy - float(cy)) ** 2 <= r * r
    return slice(y0, y1), slice(x0, x1), local


class UnderwaterTransientDataset(SeededDatasetMixin, Dataset):
    """Apply Aqua-D4RT synthetic transient corruption to any D4RT sample."""

    def __init__(self, dataset: Dataset, config: UnderwaterTransientConfig, *, split: str) -> None:
        self.dataset = dataset
        self.datasets = [dataset]
        self.config = config
        self.split = str(split)
        self._init_dataset_seeding(namespace=f"underwater_transient::{self.split}", default_seed=20260610)

    def __len__(self) -> int:
        return len(self.dataset)

    def _apply_underwater_color(self, video: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        cfg = self.config
        attenuation = float(rng.uniform(cfg.attenuation_min, cfg.attenuation_max))
        contrast = float(rng.uniform(cfg.contrast_min, cfg.contrast_max))
        water = np.asarray(cfg.water_color_rgb, dtype=np.float32).reshape(1, 3, 1, 1)
        out = video * attenuation + water * (1.0 - attenuation)
        out = (out - 0.5) * contrast + 0.5
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def _apply_dynamic_objects(
        self,
        video: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        t, _, h, w = video.shape
        dynamic_mask = np.zeros((t, h, w), dtype=np.bool_)
        if (not cfg.dynamic_objects_enabled) or float(rng.random()) > float(np.clip(cfg.dynamic_object_probability, 0.0, 1.0)):
            return video, dynamic_mask

        count = int(rng.integers(cfg.dynamic_object_count_min, cfg.dynamic_object_count_max + 1))
        fish_palette = np.asarray(
            [
                [0.85, 0.56, 0.18],
                [0.12, 0.18, 0.22],
                [0.65, 0.72, 0.36],
                [0.70, 0.32, 0.22],
            ],
            dtype=np.float32,
        )
        diag = float(min(h, w))
        for _ in range(max(0, count)):
            start = np.asarray([rng.uniform(-0.15 * w, 1.15 * w), rng.uniform(0.1 * h, 0.9 * h)], dtype=np.float32)
            velocity = np.asarray([rng.uniform(-0.45 * w, 0.45 * w), rng.uniform(-0.18 * h, 0.18 * h)], dtype=np.float32)
            rx = float(rng.uniform(cfg.dynamic_radius_frac_min, cfg.dynamic_radius_frac_max) * diag)
            ry = float(rx * rng.uniform(0.35, 0.75))
            alpha = float(rng.uniform(cfg.dynamic_alpha_min, cfg.dynamic_alpha_max))
            color = fish_palette[int(rng.integers(0, len(fish_palette)))].reshape(3, 1)
            for ti in range(t):
                denom = float(max(t - 1, 1))
                frac = float(ti) / denom
                center = start + velocity * frac
                center[1] += float(np.sin(frac * np.pi * 2.0) * 0.04 * h)
                mask = _ellipse_mask(h, w, float(center[0]), float(center[1]), rx, ry)
                if not bool(mask.any()):
                    continue
                dynamic_mask[ti] |= mask
                flat = video[ti, :, mask]
                video[ti, :, mask] = flat * (1.0 - alpha) + color * alpha
        return np.clip(video, 0.0, 1.0).astype(np.float32), dynamic_mask

    def _apply_particles(
        self,
        video: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.config
        t, _, h, w = video.shape
        particle_mask = np.zeros((t, h, w), dtype=np.bool_)
        if (not cfg.particles_enabled) or float(rng.random()) > float(np.clip(cfg.particle_probability, 0.0, 1.0)):
            return video, particle_mask

        snow_color = np.asarray([0.82, 0.95, 1.0], dtype=np.float32).reshape(3, 1, 1)
        for ti in range(t):
            count = int(rng.integers(cfg.particles_per_frame_min, cfg.particles_per_frame_max + 1))
            for _ in range(max(0, count)):
                cx = float(rng.uniform(0, max(w - 1, 1)))
                cy = float(rng.uniform(0, max(h - 1, 1)))
                radius = float(rng.uniform(cfg.particle_radius_min, cfg.particle_radius_max))
                alpha = float(rng.uniform(cfg.particle_alpha_min, cfg.particle_alpha_max))
                ys, xs, local = _disk_mask(h, w, cx, cy, radius)
                if local.size == 0:
                    continue
                particle_mask[ti, ys, xs] |= local
                patch = video[ti, :, ys, xs]
                alpha_map = local.astype(np.float32)[None, :, :] * alpha
                video[ti, :, ys, xs] = patch * (1.0 - alpha_map) + snow_color * alpha_map
        return np.clip(video, 0.0, 1.0).astype(np.float32), particle_mask

    @staticmethod
    def _labels_from_source_mask(mask_t_hw: np.ndarray, query: dict[str, torch.Tensor]) -> torch.Tensor:
        t, h, w = mask_t_hw.shape
        u = query["u"].detach().cpu().numpy().astype(np.float32)
        v = query["v"].detach().cpu().numpy().astype(np.float32)
        tsrc = query["t_src"].detach().cpu().numpy().astype(np.int64)
        x = np.clip(np.rint(u * float(max(w - 1, 1))).astype(np.int64), 0, w - 1)
        y = np.clip(np.rint(v * float(max(h - 1, 1))).astype(np.int64), 0, h - 1)
        tt = np.clip(tsrc, 0, t - 1)
        labels = mask_t_hw[tt, y, x].astype(np.float32)
        return torch.from_numpy(labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        if not isinstance(sample, dict) or "video" not in sample or "query" not in sample:
            return sample

        out = _clone_sample(sample)
        video_tensor = out["video"]
        if not torch.is_tensor(video_tensor) or video_tensor.ndim != 4:
            return out

        rng = self._sample_rng(index=int(index), stream=17)
        cfg = self.config
        apply_aug = float(rng.random()) <= float(np.clip(cfg.apply_probability, 0.0, 1.0))
        supervise = bool(cfg.supervise_clean_queries) or apply_aug

        video_np = video_tensor.detach().cpu().numpy().astype(np.float32, copy=True)
        t, _, h, w = video_np.shape
        dynamic_mask = np.zeros((t, h, w), dtype=np.bool_)
        particle_mask = np.zeros((t, h, w), dtype=np.bool_)
        if apply_aug:
            video_np = self._apply_underwater_color(video_np, rng)
            video_np, dynamic_mask = self._apply_dynamic_objects(video_np, rng)
            video_np, particle_mask = self._apply_particles(video_np, rng)
            out["video"] = torch.from_numpy(video_np).to(dtype=video_tensor.dtype)

        query = out["query"]
        target = out.setdefault("target", {})
        mask = out.setdefault("mask", {})
        query_stats = out.setdefault("query_stats", {})
        if supervise and isinstance(query, dict):
            dyn_labels = self._labels_from_source_mask(dynamic_mask, query).to(dtype=torch.float32)
            particle_labels = self._labels_from_source_mask(particle_mask, query).to(dtype=torch.float32)
            target["dynamic_object"] = dyn_labels
            target["particle"] = particle_labels
            transient_mask = torch.ones_like(dyn_labels, dtype=torch.bool)
            mask["transient"] = transient_mask
            query_stats["synthetic_dynamic_object_query"] = dyn_labels > 0.5
            query_stats["synthetic_particle_query"] = particle_labels > 0.5
        return out


def maybe_wrap_underwater_transient_dataset(dataset: Dataset, *, split: str, cfg: Any) -> Dataset:
    transient_cfg = underwater_transient_cfg_from_train_config(cfg)
    if not transient_cfg.enabled:
        return dataset
    if transient_cfg.train_only and str(split) != "train":
        return dataset
    return UnderwaterTransientDataset(dataset=dataset, config=transient_cfg, split=split)
