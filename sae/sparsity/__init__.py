"""Pluggable sparsity backends for the SAE.

Use :func:`build_sparsity` to instantiate a backend by name, e.g.::

    sparsity = build_sparsity("topk", n_features=N, k=64)
"""

from __future__ import annotations

from typing import Any

from .base import SparsityFn
from .gated import GatedSAE
from .jumprelu import JumpReLU
from .l1 import L1
from .topk import TopK

_REGISTRY: dict[str, type[SparsityFn]] = {
    "topk": TopK,
    "l1": L1,
    "jumprelu": JumpReLU,
    "gated": GatedSAE,
}


def build_sparsity(mode: str, n_features: int, **kwargs: Any) -> SparsityFn:
    if mode not in _REGISTRY:
        raise ValueError(f"Unknown sparsity mode '{mode}'. Available: {sorted(_REGISTRY)}")
    # print(f"[build_sparsity] Building sparsity mode '{mode}' with n_features={n_features} and kwargs={kwargs}")
    return _REGISTRY[mode](n_features=n_features, **kwargs)


__all__ = ["SparsityFn", "TopK", "L1", "JumpReLU", "GatedSAE", "build_sparsity"]
