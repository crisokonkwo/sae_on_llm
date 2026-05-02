"""Top-K sparsity backend.

Implements the Top-K SAE from Gao et al., "Scaling and evaluating sparse
autoencoders" (2024): for each token, keep the ``k`` largest pre-activations
(after ReLU) and zero the rest. We additionally track which features have
fired recently and provide an *aux-k* reconstruction loss that revives dead
features by asking the top-``k_aux`` dead features to reconstruct the
residual ``x - x_hat``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F

from .base import SparsityFn

# The Top-K sparsity backend maintains two buffers to track feature usage over time: a global step counter, and a per-feature "last fired step". 
# When a feature's last fired step is too old, it is considered "dead" and can be revived using the aux-k loss.
if TYPE_CHECKING:
    from ..base import SAE


class TopK(SparsityFn):
    def __init__(
        self,
        n_features: int,
        k: int,
        k_aux: int | None = None,
        aux_coef: float = 1.0 / 32.0,
        dead_steps_threshold: int = 1000,
    ) -> None:
        super().__init__()
        if k > n_features:
            raise ValueError(f"k={k} must be <= n_features={n_features}")
        self.n_features = n_features
        self.k = k
        self.k_aux = k_aux if k_aux is not None else min(2 * k, n_features)
        self.aux_coef = aux_coef
        self.dead_steps_threshold = dead_steps_threshold

        # Step counter and per-feature "last step on which this feature fired".
        self.register_buffer("step", torch.zeros((), dtype=torch.long))
        self.register_buffer("last_fired", torch.zeros(n_features, dtype=torch.long))

    @torch.no_grad()
    def _update_fired(self, z: torch.Tensor) -> None:
        # z: (..., n_features). A feature fired in this batch if any token used it.
        flat = z.reshape(-1, self.n_features)
        fired = (flat > 0).any(dim=0)
        self.last_fired[fired] = self.step
        self.step += 1

    def forward(self, pre: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        relu_pre = F.relu(pre)
        topk_vals, topk_idx = relu_pre.topk(self.k, dim=-1)
        z = torch.zeros_like(relu_pre)
        z.scatter_(-1, topk_idx, topk_vals)
        if self.training:
            self._update_fired(z)
        return z, {"relu_pre": relu_pre}

    def extra_loss(
        self,
        *,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        z: torch.Tensor,
        aux: dict[str, Any],
        sae: "SAE",
    ) -> dict[str, torch.Tensor]:
        """Aux-k loss: reconstruct ``(x - x_hat)`` using the top-``k_aux``
        currently-dead features. Skipped (returns 0) when there are not
        enough dead features yet.
        """
        if not self.training:
            return {}
        dead = (self.step - self.last_fired) > self.dead_steps_threshold  # (n_features,)
        # print(f"[TopK.extra_loss] step={self.step.item()} (self.step - self.last_fired)={self.step - self.last_fired}  dead={dead}  n_dead={(dead.sum().item())}  k_aux={self.k_aux}")
        n_dead = int(dead.sum().item())
        # print(f"[TopK.extra_loss] step={self.step.item()}  n_dead={n_dead}  dead_fraction={n_dead / self.n_features:.4f} k_aux={self.k_aux}")
        if n_dead < self.k_aux:
            return {"aux_k": x.new_zeros(())}

        relu_pre = aux["relu_pre"]
        # Mask out non-dead features, then take top-k_aux among the dead ones.
        masked = relu_pre.masked_fill(~dead, 0.0)
        topk_vals, topk_idx = masked.topk(self.k_aux, dim=-1)
        z_aux = torch.zeros_like(masked)
        z_aux.scatter_(-1, topk_idx, topk_vals)

        # Note: the decoder bias is already absorbed into ``x_hat``; aux only
        # needs to predict the *residual*, so we use W_dec without b_dec.
        residual_hat = z_aux @ sae.W_dec
        residual = (x - x_hat).detach()
        aux_loss = F.mse_loss(residual_hat, residual)
        return {"aux_k": self.aux_coef * aux_loss}

    def extra_repr(self) -> str:
        return f"n_features={self.n_features}, k={self.k}, k_aux={self.k_aux}"
