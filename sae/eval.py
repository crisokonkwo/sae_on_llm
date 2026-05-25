"""SAE evaluation / validation.

Implement from Do Sparse Autoencoders (SAEs) transfer across base and finetuned language models?
(https://www.alignmentforum.org/posts/bsXPTiAhhwt5nwBW3/do-sparse-autoencoders-saes-transfer-across-base-and)

Three families of metrics:

1. **Activation-space reconstruction** — on a held-out shard set:
   recon MSE, normalised MSE, explained variance, L0 distribution,
   per-feature activation rate (feature density), dead-feature count.

2. **Behavioural fidelity (CE-delta)** — splice ``SAE(x)`` back into the
   residual stream at the trained layer and compare next-token cross-entropy
   to the clean run and a mean-ablation baseline. The "CE recovered"
   fraction (≈1 means the SAE preserves the model's behaviour) is the
   standard summary number.

3. **Interpretability sanity** — for a handful of features, find the
   top-activating tokens with a small left/right context window.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import torch
import torch.nn.functional as F

from .base import SAE
from .dataset import ActivationDataset
from .hooks import (
    capture_residual_stream,
    middle_layer_index,
    patch_residual_stream,
)


# ----------------------------------------------------------------------
# 1. Reconstruction-space metrics
# ----------------------------------------------------------------------
@torch.no_grad()
def reconstruction_metrics(
    sae: SAE,
    dataset: ActivationDataset,
    max_batches: int | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, float | list[float]]:
    """Stream a held-out activation dataset through the SAE and compute
    reconstruction quality + L0 stats.

    The dataset is expected to be *finite* (``infinite=False``).
    """
    sae.eval()
    sae.to(device)

    n_features = sae.cfg.n_features
    sum_se = 0.0          # sum of squared errors
    sum_sq = 0.0          # sum of x**2 (for normalised MSE / EV)
    sum_x = torch.zeros(sae.cfg.d_model, device=device)
    sum_x2 = torch.zeros(sae.cfg.d_model, device=device)
    n_tokens = 0
    l0_sum = 0.0
    fire_count = torch.zeros(n_features, dtype=torch.long, device=device)

    for i, batch in enumerate(dataset):
        if max_batches is not None and i >= max_batches:
            break
        x = batch.to(device, dtype=next(sae.parameters()).dtype)
        x_hat, z, _ = sae(x)
        err = x - x_hat
        sum_se += float((err ** 2).sum().item()) # total squared reconstruction error, accumulated from ((x-\hat{x})^2)
        sum_sq += float((x ** 2).sum().item()) # total squared input, accumulated from (x^2)
        sum_x += x.sum(dim=0)
        sum_x2 += (x ** 2).sum(dim=0) # for explained variance denominator, accumulated from (x^2)
        n_tokens += x.shape[0]
        l0_sum += float((z != 0).sum().item())
        fire_count += (z != 0).sum(dim=0).long()

    if n_tokens == 0:
        raise ValueError("Empty evaluation dataset")

    mean_x = sum_x / n_tokens
    var_x_total = float(((sum_x2 / n_tokens) - mean_x ** 2).sum().item())
    mse = sum_se / (n_tokens * sae.cfg.d_model)
    nmse = sum_se / max(sum_sq, 1e-12)              # ||err||^2 / ||x||^2
    explained_var = 1.0 - (sum_se / n_tokens) / max(var_x_total, 1e-12)
    l0_mean = l0_sum / n_tokens

    fire_rate = fire_count.float() / n_tokens       # per-feature activation rate
    dead = (fire_count == 0)
    # log10 density histogram bucket counts (dead bucket separate)
    log_density = torch.log10(fire_rate.clamp_min(1e-12)).cpu()
    bins = torch.tensor([-12.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0])
    hist = torch.histogram(log_density, bins=bins).hist.tolist()

    return {
        "n_tokens": n_tokens,
        "mse": mse,
        "nmse": nmse,
        "explained_variance": explained_var,
        "l0_mean": l0_mean,
        "n_features": n_features,
        "n_dead": int(dead.sum().item()),
        "dead_fraction": float(dead.float().mean().item()),
        "fire_rate_min": float(fire_rate.min().item()),
        "fire_rate_max": float(fire_rate.max().item()),
        "fire_rate_median": float(fire_rate.median().item()),
        "log10_density_bin_edges": bins.tolist(),
        "log10_density_hist": hist,
    }


# ----------------------------------------------------------------------
# 2. CE-delta (behavioural fidelity)
# ----------------------------------------------------------------------
@torch.no_grad()
def _lm_ce(model, input_ids: torch.Tensor) -> float:
    """Mean next-token cross-entropy over a single batch (no patching)."""
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    return float(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)).item())


@torch.no_grad()
def ce_delta(
    sae: SAE,
    model,
    tokenizer,
    layer_idx: int,
    texts: Iterable[str],
    seq_len: int = 512,
    max_docs: int = 32,
    device: str | torch.device = "cuda",
) -> dict[str, float]:
    """Compare next-token CE with three residual-stream interventions at
    ``layer_idx``:

      * ``clean``  — no intervention.
      * ``sae``    — residual replaced by ``SAE(x)`` (full forward).
      * ``mean``   — residual replaced by its batch-mean (worst-case-baseline).

    Returns means over up to ``max_docs`` documents and a *CE recovered*
    score: ``1 - (ce_sae - ce_clean) / (ce_mean - ce_clean)``.
    """
    sae.eval()
    sae.to(device)
    sae_dtype = next(sae.parameters()).dtype

    def sae_fn(x: torch.Tensor) -> torch.Tensor:
        x_hat, _, _ = sae(x.to(sae_dtype))
        return x_hat.to(x.dtype)

    def mean_fn(x: torch.Tensor) -> torch.Tensor:
        m = x.mean(dim=(0, 1), keepdim=True)
        return m.expand_as(x).contiguous()

    ce_clean: list[float] = []
    ce_sae:   list[float] = []
    ce_mean:  list[float] = []

    for n, text in enumerate(texts):
        if n >= max_docs:
            break
        ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=seq_len)
        if len(ids) < 4:
            continue
        input_ids = torch.tensor(ids, device=device).unsqueeze(0)

        ce_clean.append(_lm_ce(model, input_ids))
        with patch_residual_stream(model, layer_idx, sae_fn):
            ce_sae.append(_lm_ce(model, input_ids))
        with patch_residual_stream(model, layer_idx, mean_fn):
            ce_mean.append(_lm_ce(model, input_ids))

    if not ce_clean:
        raise ValueError("No documents evaluated for CE-delta")

    m_clean = sum(ce_clean) / len(ce_clean)
    m_sae   = sum(ce_sae)   / len(ce_sae)
    m_mean  = sum(ce_mean)  / len(ce_mean)
    denom = m_mean - m_clean
    recovered = 1.0 - ((m_sae - m_clean) / denom) if abs(denom) > 1e-8 else float("nan")

    return {
        "n_docs": len(ce_clean),
        "ce_clean": m_clean,
        "ce_sae": m_sae,
        "ce_mean_ablation": m_mean,
        "ce_delta_sae": m_sae - m_clean,
        "ce_delta_mean_ablation": m_mean - m_clean,
        "ce_recovered": recovered,
    }


# ----------------------------------------------------------------------
# 3. Top-activating tokens (interpretability sanity)
# ----------------------------------------------------------------------
@dataclass(order=True)
class _TokenHit:
    activation: float
    # break ties so heapq doesn't try to compare the dict
    serial: int
    payload: dict = None


@torch.no_grad()
def top_activating_tokens(
    sae: SAE,
    model,
    tokenizer,
    layer_idx: int,
    texts: Iterable[str],
    feature_ids: Sequence[int],
    top_k: int = 8,
    seq_len: int = 256,
    max_docs: int = 256,
    context: int = 8,
    device: str | torch.device = "cuda",
) -> dict[int, list[dict]]:
    """For each feature in ``feature_ids``, find the ``top_k`` tokens (across
    the streamed corpus) with the highest SAE activation, and return a small
    text context window around each."""
    sae.eval().to(device)
    sae_dtype = next(sae.parameters()).dtype

    feat_idx = torch.tensor(list(feature_ids), device=device, dtype=torch.long)
    heaps: dict[int, list[_TokenHit]] = {int(f): [] for f in feature_ids}
    serial = 0

    with capture_residual_stream(model, layer_idx) as catcher:
        for n, text in enumerate(texts):
            if n >= max_docs:
                break
            ids = tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=seq_len)
            if len(ids) < 2:
                continue
            input_ids = torch.tensor(ids, device=device).unsqueeze(0)
            model(input_ids=input_ids, use_cache=False)
            x = catcher.activations.to(sae_dtype)              # (1, T, d)
            z, _ = sae.encode(x)
            z_sel = z[0, :, feat_idx]                           # (T, F)
            T = z_sel.shape[0]

            for fi, f in enumerate(feature_ids):
                acts = z_sel[:, fi]
                # Find candidates above current heap minimum to limit work.
                if heaps[f] and len(heaps[f]) >= top_k:
                    threshold = heaps[f][0].activation
                else:
                    threshold = float("-inf")
                cand_mask = acts > threshold
                if not cand_mask.any():
                    continue
                cand_pos = cand_mask.nonzero(as_tuple=True)[0].tolist()
                for pos in cand_pos:
                    a = float(acts[pos].item())
                    if a <= 0:
                        continue
                    lo, hi = max(0, pos - context), min(T, pos + context + 1)
                    ctx_ids = ids[lo:hi]
                    payload = {
                        "activation": a,
                        "doc_idx": n,
                        "pos": pos,
                        "token": tokenizer.decode([ids[pos]]),
                        "context": tokenizer.decode(ctx_ids),
                    }
                    serial += 1
                    hit = _TokenHit(a, serial, payload)
                    if len(heaps[f]) < top_k:
                        heapq.heappush(heaps[f], hit)
                    else:
                        heapq.heappushpop(heaps[f], hit)

    out: dict[int, list[dict]] = {}
    for f, h in heaps.items():
        out[f] = [hit.payload for hit in sorted(h, key=lambda x: -x.activation)]
    return out
