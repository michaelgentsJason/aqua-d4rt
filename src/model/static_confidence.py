"""Static-background confidence helpers for Aqua-D4RT outputs."""

from __future__ import annotations

import torch


def compute_static_confidence(
    confidence_logit: torch.Tensor,
    dynamic_object_logit: torch.Tensor,
    particle_logit: torch.Tensor,
) -> torch.Tensor:
    """Combine model confidence with transient probabilities.

    The returned value is a probability-like score in [0, 1] where high values
    indicate geometry that is likely to belong to the static background.
    """

    confidence = torch.sigmoid(confidence_logit)
    dynamic_prob = torch.sigmoid(dynamic_object_logit)
    particle_prob = torch.sigmoid(particle_logit)
    return confidence * (1.0 - dynamic_prob) * (1.0 - particle_prob)


def static_query_mask(outputs: dict[str, torch.Tensor], threshold: float = 0.5) -> torch.Tensor:
    """Return a boolean mask selecting likely static-background queries."""

    if "static_confidence" in outputs:
        score = outputs["static_confidence"]
    elif all(key in outputs for key in ("confidence", "dynamic_object_logit", "particle_logit")):
        score = compute_static_confidence(
            confidence_logit=outputs["confidence"],
            dynamic_object_logit=outputs["dynamic_object_logit"],
            particle_logit=outputs["particle_logit"],
        )
    elif "confidence" in outputs:
        score = torch.sigmoid(outputs["confidence"])
    else:
        raise KeyError("outputs must contain static_confidence or confidence logits")
    return score >= float(threshold)
