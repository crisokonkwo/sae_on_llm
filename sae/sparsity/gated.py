"""Gated SAE backend (placeholder).

Gated SAE (Rajamanoharan et al., 2024) has a different architecture: a gating
path decides which features are active and a magnitude path decides their
values. This needs more than a sparsity wrapper \u2014 we'll likely override the
encode path in :class:`sae.base.SAE` (or subclass it) when this is wired up.
"""

from __future__ import annotations

from typing import Any

import torch

from .base import SparsityFn


class GatedSAE(SparsityFn):
    def __init__(self, n_features: int, l1_coef: float = 1e-3) -> None:
        super().__init__()
        self.n_features = n_features
        self.l1_coef = l1_coef

    def forward(self, pre: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        raise NotImplementedError(
            "GatedSAE backend not yet implemented \u2014 it requires a custom encode path"
        )
