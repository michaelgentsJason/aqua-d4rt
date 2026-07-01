"""Task-specific linear heads for D4RT queries."""

from __future__ import annotations

import torch
import torch.nn as nn

from .static_confidence import compute_static_confidence


class D4RTHeads(nn.Module):
    """Predicts query-level outputs for all configured tasks."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.xyz_head = nn.Linear(hidden_dim, 3)
        self.uv_head = nn.Linear(hidden_dim, 2)
        self.visibility_head = nn.Linear(hidden_dim, 1)
        self.displacement_head = nn.Linear(hidden_dim, 3)
        self.normal_head = nn.Linear(hidden_dim, 3)
        self.confidence_head = nn.Linear(hidden_dim, 1)
        self.dynamic_object_head = nn.Linear(hidden_dim, 1)
        self.particle_head = nn.Linear(hidden_dim, 1)

    def forward(self, decoded_queries: torch.Tensor) -> dict[str, torch.Tensor]:
        confidence = self.confidence_head(decoded_queries).squeeze(-1)
        dynamic_object_logit = self.dynamic_object_head(decoded_queries).squeeze(-1)
        particle_logit = self.particle_head(decoded_queries).squeeze(-1)
        return {
            "xyz_3d": self.xyz_head(decoded_queries),
            "uv_2d": self.uv_head(decoded_queries),
            "visibility": self.visibility_head(decoded_queries).squeeze(-1),
            "displacement": self.displacement_head(decoded_queries),
            "normal": self.normal_head(decoded_queries),
            "confidence": confidence,
            "dynamic_object_logit": dynamic_object_logit,
            "particle_logit": particle_logit,
            "static_confidence": compute_static_confidence(
                confidence_logit=confidence,
                dynamic_object_logit=dynamic_object_logit,
                particle_logit=particle_logit,
            ),
        }
