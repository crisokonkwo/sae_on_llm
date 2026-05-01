"""Base class for SAE sparsity backends.

A *sparsity backend* takes pre-activations of shape ``(..., n_features)`` and
returns sparse feature activations of the same shape, plus an optional
``aux`` dict that downstream loss code can consume (e.g. for a dead-latent
revival term).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from ..base import SAE


class SparsityFn(nn.Module):
    """Subclasses must implement :meth:`forward`.

    They may optionally override :meth:`extra_loss` to contribute a
    mode-specific loss term (e.g. an L1 penalty or aux-k revival loss).
    """

    def forward(self, pre: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        raise NotImplementedError

    # The default contributes nothing.
    def extra_loss(
        self,
        *,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        z: torch.Tensor,
        aux: dict[str, Any],
        sae: "SAE",
    ) -> dict[str, torch.Tensor]:
        return {}
