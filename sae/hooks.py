"""Forward-hook utilities for capturing residual-stream activations."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator, List

import torch
import torch.nn as nn


def get_num_layers(model: nn.Module) -> int:
    """Return the number of transformer blocks in a HF causal LM.

    Works for Gemma / Gemma-2 / Llama-style models exposing
    ``model.model.layers``.
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return len(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return len(model.transformer.h)
    raise AttributeError(
        f"Could not determine number of layers for model of type {type(model).__name__}"
    )


def get_residual_block(model: nn.Module, layer_idx: int) -> nn.Module:
    """Return the transformer block whose *output* is the residual stream
    after layer ``layer_idx``."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer_idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer_idx]
    raise AttributeError(f"Unsupported model layout: {type(model).__name__}")


def middle_layer_index(model: nn.Module) -> int:
    """Pick the middle layer (rounded down) of the transformer stack."""
    return get_num_layers(model) // 2


class ResidualStreamCatcher:
    """Captures the residual-stream tensor output by a chosen transformer block (Full residual stream not post-MLP residual stream).

    HF decoder blocks return a tuple whose first element is the hidden state
    of shape ``(batch, seq_len, d_model)`` — i.e. the residual stream after that
    block. We grab it via a forward hook.
    """

    def __init__(self, block: nn.Module):
        self.block = block
        self._handle = None
        self.activations: torch.Tensor | None = None

    def _hook(self, _module, _inputs, output):
        # Block output is typically a tuple (hidden_states, attn_weights, ); grab the hidden states. 
        # Note that for some models (e.g. Llama) the block output is just the hidden states tensor, not a tuple.
        hidden = output[0] if isinstance(output, tuple) else output
        # Detach + move to CPU later in the buffer; here just keep ref.
        self.activations = hidden

    def __enter__(self) -> "ResidualStreamCatcher":
        self._handle = self.block.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


@contextmanager
def capture_residual_stream(model: nn.Module, layer_idx: int) -> Iterator[ResidualStreamCatcher]:
    block = get_residual_block(model, layer_idx)
    catcher = ResidualStreamCatcher(block)
    with catcher:
        yield catcher


class ResidualStreamPatcher:
    """Replaces the residual stream output of a transformer block with
    ``fn(x)`` on the fly.

    ``fn`` receives the hidden state of shape ``(batch, seq, d_model)`` and
    must return a tensor of the same shape and dtype.
    """

    def __init__(self, block: nn.Module, fn: Callable[[torch.Tensor], torch.Tensor]):
        self.block = block
        self.fn = fn
        self._handle = None

    def _hook(self, _module, _inputs, output):
        if isinstance(output, tuple):
            new_h = self.fn(output[0])
            return (new_h,) + output[1:]
        return self.fn(output)

    def __enter__(self) -> "ResidualStreamPatcher":
        self._handle = self.block.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


@contextmanager
def patch_residual_stream(
    model: nn.Module, layer_idx: int, fn: Callable[[torch.Tensor], torch.Tensor]
) -> Iterator[ResidualStreamPatcher]:
    block = get_residual_block(model, layer_idx)
    patcher = ResidualStreamPatcher(block, fn)
    with patcher:
        yield patcher
