"""Concept-suppression utilities.

Two pieces:

* :func:`make_clamp_fn` builds a function suitable for ``patch_residual_stream``
  that overwrites the activation of a chosen set of SAE features at every
  token position. It uses the *delta* trick so non-clamped features pass
  through with **no SAE-reconstruction error**: only the clamped features are
  modified, the rest of the residual is left exactly as the model produced it.

* :func:`find_concept_features` discovers features that fire on a *concept*
  by comparing mean activations on positive vs. (optionally) negative texts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from .base import SAE
from .hooks import capture_residual_stream


# ----------------------------------------------------------------------
# Clamp / suppression hooks
# ----------------------------------------------------------------------
def make_clamp_fn(
    sae: SAE,
    feature_ids: Sequence[int],
    clamp_values: float | Sequence[float] = 0.0,
    sae_dtype: torch.dtype | None = None,
):
    """Build an intervention function ``fn(x) -> x'`` for ``patch_residual_stream``.

    For each chosen feature ``f``, the function forces the SAE activation
    ``z[:, :, f]`` to equal ``clamp_values[f]`` while leaving every other
    direction in ``x`` untouched. Internally::

        z       = sae.encode(x)
        delta   = clamp_values - z[..., feature_ids]                  # (B, T, F)
        x'      = x + delta @ W_dec[feature_ids]                      # (B, T, d)

    With ``clamp_values=0`` this is a clean *feature ablation*: subtract the
    decoded contribution of the chosen features from the residual stream.
    Positive ``clamp_values`` steer the residual *toward* a feature; negative
    values steer *away*.
    """
    feat_idx = torch.as_tensor(list(feature_ids), dtype=torch.long, device=sae.W_dec.device)
    # Handle the case where clamp_values is a single scalar to be applied to all features
    if isinstance(clamp_values, (int, float)):
        clamp = torch.full((feat_idx.numel(),), float(clamp_values),
                           device=sae.W_dec.device, dtype=sae.W_dec.dtype)
    else:
        # Convert the sequence of clamp values to a tensor
        clamp = torch.as_tensor(list(clamp_values), device=sae.W_dec.device, dtype=sae.W_dec.dtype)
    if clamp.shape != feat_idx.shape:
        raise ValueError(f"clamp_values shape {clamp.shape} != feature_ids shape {feat_idx.shape}")
    sae_dtype = sae_dtype or next(sae.parameters()).dtype

    @torch.no_grad()
    def fn(x: torch.Tensor) -> torch.Tensor:
        x_in = x.to(sae_dtype)
        z, _ = sae.encode(x_in)                                # (B, T, n_features)
        current = z[..., feat_idx]                              # (B, T, F)
        delta = clamp.view(1, 1, -1) - current                  # (B, T, F)
        modification = delta @ sae.W_dec[feat_idx]              # (B, T, d_model)
        return (x_in + modification).to(x.dtype)

    return fn


# ----------------------------------------------------------------------
# Concept feature discovery
# ----------------------------------------------------------------------
@dataclass
class ConceptScore:
    feature_id: int
    score: float            # discovery score (mean_act_pos - mean_act_neg, or just mean_act_pos)
    mean_act_pos: float
    mean_act_neg: float | None
    fire_rate_pos: float
    fire_rate_neg: float | None


@torch.no_grad()
def _per_feature_stats(
    sae: SAE,
    model,
    tokenizer,
    layer_idx: int,
    texts: Iterable[str],
    seq_len: int = 256,
    device: str | torch.device = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Returns (sum_activation, fire_count, n_tokens) over the given texts."""
    sae.eval()
    sae_dtype = next(sae.parameters()).dtype
    n_features = sae.cfg.n_features
    sums = torch.zeros(n_features, device=device, dtype=torch.float32)
    fires = torch.zeros(n_features, device=device, dtype=torch.float32)
    n_tokens = 0

    with capture_residual_stream(model, layer_idx) as catcher:
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=seq_len)
            # Skip texts that are too short to produce any activations (e.g. empty or just a BOS token)
            if len(ids) < 2:
                continue
            input_ids = torch.tensor(ids, device=device).unsqueeze(0)
            model(input_ids=input_ids, use_cache=False)
            x = catcher.activations.to(sae_dtype)
            z, _ = sae.encode(x)
            z = z.reshape(-1, n_features).float()
            sums += z.sum(dim=0)
            fires += (z > 0).float().sum(dim=0)
            n_tokens += z.shape[0]
    return sums, fires, n_tokens


@torch.no_grad()
def find_concept_features(
    sae: SAE,
    model,
    tokenizer,
    layer_idx: int,
    positive_texts: Sequence[str],
    negative_texts: Sequence[str] | None = None,
    top_n: int = 8,
    score: str = "mean_act_diff",
    seq_len: int = 256,
    device: str | torch.device = "cuda",
) -> list[ConceptScore]:
    """Rank SAE features by how much more they fire on ``positive_texts``
    than on ``negative_texts``.

    ``score`` ∈ {``"mean_act"``, ``"mean_act_diff"``, ``"fire_rate_diff"``}.
    Without ``negative_texts``, falls back to raw mean activation.
    """
    sums_p, fires_p, n_p = _per_feature_stats(sae, model, tokenizer, layer_idx,
                                              positive_texts, seq_len=seq_len, device=device)
    if n_p == 0:
        raise ValueError("No tokens collected from positive_texts")
    mean_p = sums_p / n_p
    rate_p = fires_p / n_p

    if negative_texts:
        sums_n, fires_n, n_n = _per_feature_stats(sae, model, tokenizer, layer_idx,
                                                  negative_texts, seq_len=seq_len, device=device)
        mean_n = sums_n / max(n_n, 1)
        rate_n = fires_n / max(n_n, 1)
    else:
        mean_n = rate_n = None

    if score == "mean_act":
        score_vec = mean_p
    elif score == "mean_act_diff":
        score_vec = mean_p - (mean_n if mean_n is not None else 0.0)
    elif score == "fire_rate_diff":
        if rate_n is None:
            raise ValueError("fire_rate_diff requires negative_texts")
        score_vec = rate_p - rate_n
    else:
        raise ValueError(f"Unknown score: {score}")

    top_vals, top_idx = score_vec.topk(top_n)
    out: list[ConceptScore] = []
    for v, i in zip(top_vals.tolist(), top_idx.tolist()):
        out.append(ConceptScore(
            feature_id=int(i),
            score=float(v),
            mean_act_pos=float(mean_p[i].item()),
            mean_act_neg=float(mean_n[i].item()) if mean_n is not None else None,
            fire_rate_pos=float(rate_p[i].item()),
            fire_rate_neg=float(rate_n[i].item()) if rate_n is not None else None,
        ))
    return out


# ----------------------------------------------------------------------
# Quantitative check: log-prob delta on a target sequence
# ----------------------------------------------------------------------
@torch.no_grad()
def sequence_logprob(model, tokenizer, text: str, device: str | torch.device = "cuda",
                     max_length: int = 1024) -> tuple[float, int]:
    """Return (sum log P(token_t | token_<t), n_predicted_tokens) for ``text``."""
    ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_length)
    if len(ids) < 2:
        return 0.0, 0
    input_ids = torch.tensor(ids, device=device).unsqueeze(0)
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    sel = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(sel.sum().item()), int(sel.numel())
