"""L1 sparsity backend (placeholder).

Standard SAE: ``z = ReLU(pre)`` with an L1 penalty ``lambda * |z|_1`` on the
activations as the sparsity-inducing loss term. To be implemented.
"""

from __future__ import annotations

from typing import Any

import torch

from .base import SparsityFn


class L1(SparsityFn):
    def __init__(self, n_features: int, l1_coef: float = 1e-3) -> None:
        super().__init__()
        self.n_features = n_features
        self.l1_coef = l1_coef

    def forward(self, pre: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        raise NotImplementedError("L1 sparsity backend not yet implemented")
