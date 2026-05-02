"""Sparse Autoencoder core.

Forward path (shared across sparsity backends)::

    x_centered = x - b_dec where ``b_dec`` is the decoder bias, which serves to centre the input distribution.
    pre        = x_centered @ W_enc + b_enc 
    z, aux     = sparsity(pre)
    x_hat      = z @ W_dec + b_dec

Conventions:
    * ``W_enc``: ``(d_model, n_features)``
    * ``W_dec``: ``(n_features, d_model)`` \u2014 each *row* is the decoder
      direction for one feature; rows are kept unit-norm via
      :meth:`SAE.normalize_decoder_` (call after each optimizer step).
    * ``b_enc``: ``(n_features,)``,  ``b_dec``: ``(d_model,)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .sparsity import SparsityFn, build_sparsity


@dataclass
class SAEConfig:
    d_model: int
    n_features: int
    sparsity_mode: str = "topk"
    sparsity_kwargs: dict[str, Any] = field(default_factory=dict)
    tie_decoder_init: bool = True
    normalize_decoder: bool = True


class SAE(nn.Module):
    def __init__(self, cfg: SAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, n = cfg.d_model, cfg.n_features

        # Decoder rows are unit-norm directions in residual-stream space.
        W_dec = torch.randn(n, d)
        W_dec = W_dec / W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.W_dec = nn.Parameter(W_dec)

        # Init encoder as the transpose of the decoder (common SAE init);
        # they're separate parameters so training can break the tie.
        if cfg.tie_decoder_init:
            self.W_enc = nn.Parameter(W_dec.detach().T.contiguous().clone())
        else:
            # Standard random init scaled by 1/sqrt(d) which is good for training
            W_enc = torch.randn(d, n) * (1.0 / d**0.5)
            self.W_enc = nn.Parameter(W_enc)

        self.b_enc = nn.Parameter(torch.zeros(n))
        self.b_dec = nn.Parameter(torch.zeros(d))

        self.sparsity: SparsityFn = build_sparsity(
            cfg.sparsity_mode, n_features=n, **cfg.sparsity_kwargs
        )

    # ----- forward primitives -----------------------------------------------
    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
        pre = self.encode_pre(x)
        z, aux = self.sparsity(pre)
        return z, aux

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        z, aux = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z, aux

    # ----- loss -------------------------------------------------------------
    def loss(
        self, x: torch.Tensor, x_hat: torch.Tensor, z: torch.Tensor, aux: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute total loss = reconstruction MSE + mode-specific terms."""
        recon = ((x - x_hat) ** 2).mean()
        terms: dict[str, torch.Tensor] = {"recon": recon}
        terms.update(self.sparsity.extra_loss(x=x, x_hat=x_hat, z=z, aux=aux, sae=self)) # add any mode-specific loss terms from the sparsity function, e.g. an L1 penalty or aux-k revival loss
        total = sum(terms.values())  # all already scaled by their coefficients
        terms["total"] = total
        return total, terms

    # ----- bookkeeping ------------------------------------------------------
    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        """Project each decoder row back to unit norm. Call after each
        optimizer step when ``cfg.normalize_decoder=True``."""
        if not self.cfg.normalize_decoder:
            return
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.W_dec.data.div_(norms)

    @torch.no_grad()
    def init_b_dec_from_(self, x_sample: torch.Tensor) -> None:
        """Initialise ``b_dec`` to the mean of a batch of activations.
        Recommended before training to centre the input distribution."""
        self.b_dec.data.copy_(x_sample.reshape(-1, x_sample.shape[-1]).mean(dim=0)) # reshape to (num_tokens, d_model) and take mean over tokens

    # ----- diagnostics ------------------------------------------------------
    @torch.no_grad()
    def l0(self, z: torch.Tensor) -> torch.Tensor:
        """Average number of non-zero features per token."""
        return (z != 0).float().sum(dim=-1).mean()

    @torch.no_grad()
    def explained_variance(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        var = x.var(dim=0).sum().clamp_min(1e-8)
        residual_var = (x - x_hat).var(dim=0).sum()
        return 1.0 - residual_var / var
