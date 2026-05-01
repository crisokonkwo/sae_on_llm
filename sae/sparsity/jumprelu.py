"""JumpReLU sparsity backend (placeholder).

JumpReLU SAE (Rajamanoharan et al., 2024): ``z_i = pre_i * 1[pre_i > theta_i]``
with per-feature learnable thresholds ``theta`` trained via a straight-through
estimator and an L0 penalty. To be implemented.
"""

from __future__ import annotations

from typing import Any

import torch

from .base import SparsityFn


class JumpReLU(SparsityFn):
    def __init__(self, n_features: int, l0_coef: float = 1e-3, bandwidth: float = 1e-3) -> None:
        super().__init__()
        self.n_features = n_features
        self.l0_coef = l0_coef
        self.bandwidth = bandwidth

    def forward(self, pre: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        raise NotImplementedError("JumpReLU sparsity backend not yet implemented")
