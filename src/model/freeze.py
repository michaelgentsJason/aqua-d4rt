"""Fine-tuning freeze helpers shared by training entrypoints."""

from __future__ import annotations

from typing import Any

import torch


def _model_module(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def _freeze_module(module: torch.nn.Module, module_name: str) -> int:
    child = getattr(module, module_name, None)
    if child is None:
        return 0
    count = 0
    for param in child.parameters():
        param.requires_grad_(False)
        count += int(param.numel())
    return count


def apply_fine_tuning_freeze(
    model: torch.nn.Module,
    train_cfg: Any,
    *,
    logger: Any | None = None,
    rank: int = 0,
) -> dict[str, Any]:
    """Apply `fine_tuning.*` parameter-freeze settings once."""

    module = _model_module(model)
    if bool(getattr(module, "_fine_tuning_freeze_applied", False)):
        return {"already_applied": True}

    cfg = train_cfg.get_path("fine_tuning", {}) if hasattr(train_cfg, "get_path") else {}
    if not isinstance(cfg, dict):
        setattr(module, "_fine_tuning_freeze_applied", True)
        return {"already_applied": False, "frozen": {}, "trainable": None, "total": None}

    frozen: dict[str, int] = {}
    freeze_flags = {
        "encoder": bool(cfg.get("freeze_encoder", False)),
        "memory_proj": bool(cfg.get("freeze_memory_proj", False)),
        "query_embedder": bool(cfg.get("freeze_query_embedder", False)),
        "decoder": bool(cfg.get("freeze_decoder", False)),
        "heads": bool(cfg.get("freeze_heads", False)),
    }
    for module_name, enabled in freeze_flags.items():
        if enabled:
            frozen[module_name] = _freeze_module(module, module_name)

    trainable = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    total = sum(int(p.numel()) for p in model.parameters())
    setattr(module, "_fine_tuning_freeze_applied", True)

    if logger is not None and int(rank) == 0 and (frozen or bool(cfg.get("log_parameter_counts", True))):
        frozen_msg = ", ".join(f"{name}={count}" for name, count in frozen.items()) or "none"
        logger.info(
            "Fine-tuning parameter policy: frozen={%s} trainable=%d total=%d trainable_ratio=%.4f",
            frozen_msg,
            trainable,
            total,
            float(trainable) / float(max(1, total)),
        )
    return {"already_applied": False, "frozen": frozen, "trainable": trainable, "total": total}

