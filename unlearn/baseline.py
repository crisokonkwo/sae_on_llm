"""Pre-FT baseline evaluation for the SAE-feature-stability unlearning
experiment.

Locks in the "before" snapshot needed to detect feature drift:

    1. Pick concept-associated features via ``find_concept_features``.
    2. Record their per-feature stats and top-activating tokens on the
       concept-positive corpus.
    3. Measure target-answer log-probability on probes, both clean and with
       the chosen features ablated (delta = causal handle strength).
    4. Measure retain-set CE, both clean and intervened (delta = collateral
       damage budget that the unlearning step must beat).
    5. Capture a few greedy generations per probe for qualitative review.

All four numbers feed directly into U4 comparison metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

from sae.base import SAE
from sae.eval import top_activating_tokens
from sae.hooks import patch_residual_stream
from sae.intervene import find_concept_features, make_clamp_fn
from unlearn.concept_data import ConceptProbe, CorpusSentence


# ---------------------------------------------------------------------------
# Per-probe log-probability (clean and intervened)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _completion_logprob(
    model, tokenizer,
    prompt: str, completion: str,
    *,
    device: str | torch.device,
    max_length: int = 512,
) -> tuple[float, int]:
    """Return ``(sum log P(completion[t] | prompt + completion[<t]), n_tokens)``.

    Uses the standard prompt-masking trick: we tokenise ``prompt + completion``
    together (so BPE merges across the boundary are honoured), then slice the
    log-probabilities to keep only positions where the model is predicting
    completion tokens.
    """
    full = prompt + completion
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True,
                                  truncation=True, max_length=max_length)
    full_ids = tokenizer.encode(full, add_special_tokens=True,
                                truncation=True, max_length=max_length)
    if len(full_ids) <= len(prompt_ids):
        return 0.0, 0
    input_ids = torch.tensor(full_ids, device=device).unsqueeze(0)
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    sel = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (1, T-1)
    # Position t in logits predicts input_ids[t+1]; so completion log-probs
    # live at indices [n_prompt-1 ..] of `sel`.
    n_prompt = len(prompt_ids)
    completion_logprobs = sel[0, n_prompt - 1:]
    return float(completion_logprobs.sum().item()), int(completion_logprobs.numel())


@dataclass
class ProbeLogprobs:
    """Per-prompt log-prob of the expected completion."""
    prompts: list[str]
    logprob_total: list[float]
    n_tokens: list[int]

    @property
    def logprob_per_token(self) -> list[float]:
        return [lp / max(n, 1) for lp, n in zip(self.logprob_total, self.n_tokens)]

    @property
    def mean_logprob_per_token(self) -> float:
        per_tok = self.logprob_per_token
        return sum(per_tok) / len(per_tok) if per_tok else 0.0


@torch.no_grad()
def probe_logprobs_clean(
    model, tokenizer,
    probes: Sequence[ConceptProbe],
    *,
    device: str | torch.device,
    max_length: int = 512,
) -> list[ProbeLogprobs | None]:
    """For each probe with ``expected_completion``, compute log-probs over the
    base prompt and all paraphrases. Returns ``None`` for probes without an
    expected completion.
    """
    out: list[ProbeLogprobs | None] = []
    for p in probes:
        if p.expected_completion is None:
            out.append(None)
            continue
        prompts = p.all_prompts()
        lps, ntoks = [], []
        for q in prompts:
            lp, nt = _completion_logprob(model, tokenizer, q, p.expected_completion,
                                         device=device, max_length=max_length)
            lps.append(lp); ntoks.append(nt)
        out.append(ProbeLogprobs(prompts=prompts, logprob_total=lps, n_tokens=ntoks))
    return out


@torch.no_grad()
def probe_logprobs_intervened(
    model, tokenizer,
    probes: Sequence[ConceptProbe],
    *,
    sae: SAE,
    layer_idx: int,
    feature_ids: Sequence[int],
    clamp_value: float = 0.0,
    device: str | torch.device,
    max_length: int = 512,
) -> list[ProbeLogprobs | None]:
    """Same as :func:`probe_logprobs_clean` but with the chosen features
    clamped to ``clamp_value`` during the model forward pass.
    """
    clamp_fn = make_clamp_fn(sae, feature_ids, clamp_values=clamp_value)
    out: list[ProbeLogprobs | None] = []
    for p in probes:
        if p.expected_completion is None:
            out.append(None)
            continue
        prompts = p.all_prompts()
        lps, ntoks = [], []
        with patch_residual_stream(model, layer_idx, clamp_fn):
            for q in prompts:
                lp, nt = _completion_logprob(model, tokenizer, q, p.expected_completion,
                                             device=device, max_length=max_length)
                lps.append(lp); ntoks.append(nt)
        out.append(ProbeLogprobs(prompts=prompts, logprob_total=lps, n_tokens=ntoks))
    return out


# ---------------------------------------------------------------------------
# Greedy generation samples (clean vs intervened)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _greedy_generate(model, tokenizer, prompt: str, *, max_new_tokens: int, device) -> str:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    out = model.generate(
        ids, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()


@torch.no_grad()
def probe_generations(
    model, tokenizer,
    probes: Sequence[ConceptProbe],
    *,
    max_new_tokens: int,
    device: str | torch.device,
    n_probes_to_sample: int = 5,
    sae: SAE | None = None,
    layer_idx: int | None = None,
    feature_ids: Sequence[int] | None = None,
    clamp_value: float = 0.0,
) -> list[dict]:
    """Run greedy generation on the first ``n_probes_to_sample`` probes
    (using the base prompt of each). Both clean and intervened generations
    are produced when ``sae``/``feature_ids``/``layer_idx`` are provided.
    """
    samples: list[dict] = []
    intervene = sae is not None and feature_ids is not None and layer_idx is not None
    clamp_fn = make_clamp_fn(sae, feature_ids, clamp_values=clamp_value) if intervene else None
    for p in probes[:n_probes_to_sample]:
        clean = _greedy_generate(model, tokenizer, p.prompt,
                                 max_new_tokens=max_new_tokens, device=device)
        item = {"prompt": p.prompt, "expected_completion": p.expected_completion, "clean": clean}
        if intervene:
            with patch_residual_stream(model, layer_idx, clamp_fn):
                item["intervened"] = _greedy_generate(model, tokenizer, p.prompt,
                                                      max_new_tokens=max_new_tokens, device=device)
        samples.append(item)
    return samples


# ---------------------------------------------------------------------------
# Retain-set CE (collateral damage budget)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _next_token_ce(
    model, tokenizer,
    sentences: Iterable[str],
    *,
    device: str | torch.device,
    max_sentences: int,
    max_length: int,
) -> tuple[float, int, int]:
    """Mean next-token CE (in nats per token) over up to ``max_sentences`` items.

    Returns ``(mean_ce_per_token, total_tokens, n_sentences_eval'd)``.
    """
    total_loss = 0.0
    total_tokens = 0
    n_sent = 0
    for sent in sentences:
        if n_sent >= max_sentences:
            break
        ids = tokenizer.encode(sent, add_special_tokens=True,
                               truncation=True, max_length=max_length)
        if len(ids) < 2:
            continue
        input_ids = torch.tensor(ids, device=device).unsqueeze(0)
        out = model(input_ids=input_ids, use_cache=False)
        logits = out.logits[:, :-1, :].float()
        targets = input_ids[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="sum",
        )
        total_loss += float(loss.item())
        total_tokens += int(targets.numel())
        n_sent += 1
    mean_ce = total_loss / max(total_tokens, 1)
    return mean_ce, total_tokens, n_sent


@torch.no_grad()
def retain_set_ce(
    model, tokenizer,
    retain_sentences: Sequence[CorpusSentence] | Sequence[str],
    *,
    device: str | torch.device,
    max_sentences: int = 200,
    max_length: int = 256,
    sae: SAE | None = None,
    layer_idx: int | None = None,
    feature_ids: Sequence[int] | None = None,
    clamp_value: float = 0.0,
) -> dict:
    """Compute next-token CE on the retain corpus.

    With ``sae`` + ``feature_ids`` + ``layer_idx`` set, also compute the
    intervened CE so the report records the baseline collateral damage
    caused purely by the clamp (before any unlearning training).
    """
    def _texts() -> list[str]:
        return [s.text if isinstance(s, CorpusSentence) else s for s in retain_sentences]
    texts = _texts()

    clean_ce, clean_tokens, n_sent = _next_token_ce(
        model, tokenizer, texts,
        device=device, max_sentences=max_sentences, max_length=max_length,
    )
    result: dict = {
        "n_sentences_eval": n_sent,
        "n_tokens_eval": clean_tokens,
        "clean": clean_ce,
    }
    if sae is not None and feature_ids is not None and layer_idx is not None:
        clamp_fn = make_clamp_fn(sae, feature_ids, clamp_values=clamp_value)
        with patch_residual_stream(model, layer_idx, clamp_fn):
            int_ce, _, _ = _next_token_ce(
                model, tokenizer, texts,
                device=device, max_sentences=max_sentences, max_length=max_length,
            )
        result["intervened"] = int_ce
        result["delta"] = int_ce - clean_ce
    return result


# ---------------------------------------------------------------------------
# Feature picking + top-activating examples
# ---------------------------------------------------------------------------
def pick_concept_features(
    sae: SAE,
    model,
    tokenizer,
    *,
    layer_idx: int,
    positive_texts: Sequence[str],
    negative_texts: Sequence[str],
    top_n: int,
    seq_len: int = 256,
    device: str | torch.device,
) -> list[dict]:
    """Wrap ``find_concept_features`` and serialise the result as JSON-able
    dicts (one per feature). Sorted by score, descending.
    """
    feats = find_concept_features(
        sae, model, tokenizer, layer_idx,
        positive_texts=list(positive_texts),
        negative_texts=list(negative_texts),
        top_n=top_n,
        score="mean_act_diff",
        seq_len=seq_len,
        device=device,
    )
    return [asdict(f) for f in feats]


@torch.no_grad()
def feature_top_examples(
    sae: SAE,
    model,
    tokenizer,
    *,
    layer_idx: int,
    feature_ids: Sequence[int],
    texts: Sequence[str],
    top_k_tokens: int = 8,
    max_docs: int = 256,
    seq_len: int = 256,
    context: int = 8,
    device: str | torch.device,
) -> dict[int, list[dict]]:
    """For each feature in ``feature_ids``, return its top-K activating
    tokens with short context windows, drawn from ``texts``. Thin wrapper
    around ``sae.eval.top_activating_tokens``.
    """
    return top_activating_tokens(
        sae, model, tokenizer, layer_idx,
        texts=list(texts),
        feature_ids=list(feature_ids),
        top_k=top_k_tokens,
        seq_len=seq_len,
        max_docs=max_docs,
        context=context,
        device=device,
    )


# ---------------------------------------------------------------------------
# Aggregation helpers for the report
# ---------------------------------------------------------------------------
@dataclass
class ProbeReportRow:
    prompt: str
    expected_completion: str | None
    all_prompts: list[str]
    clean_logprob_total: list[float] = field(default_factory=list)
    clean_n_tokens: list[int] = field(default_factory=list)
    clean_mean_lp_per_token: float = 0.0
    intervened_logprob_total: list[float] = field(default_factory=list)
    intervened_n_tokens: list[int] = field(default_factory=list)
    intervened_mean_lp_per_token: float = 0.0
    delta_per_token: float = 0.0    # clean - intervened, mean per token


def build_probe_rows(
    probes: Sequence[ConceptProbe],
    clean: list[ProbeLogprobs | None],
    intervened: list[ProbeLogprobs | None],
) -> list[ProbeReportRow]:
    rows: list[ProbeReportRow] = []
    for p, c, i in zip(probes, clean, intervened):
        row = ProbeReportRow(
            prompt=p.prompt,
            expected_completion=p.expected_completion,
            all_prompts=p.all_prompts(),
        )
        if c is not None:
            row.clean_logprob_total = c.logprob_total
            row.clean_n_tokens = c.n_tokens
            row.clean_mean_lp_per_token = c.mean_logprob_per_token
        if i is not None:
            row.intervened_logprob_total = i.logprob_total
            row.intervened_n_tokens = i.n_tokens
            row.intervened_mean_lp_per_token = i.mean_logprob_per_token
        if c is not None and i is not None:
            row.delta_per_token = row.clean_mean_lp_per_token - row.intervened_mean_lp_per_token
        rows.append(row)
    return rows


def aggregate_probe_rows(rows: Sequence[ProbeReportRow]) -> dict:
    scoreable = [r for r in rows if r.expected_completion is not None]
    if not scoreable:
        return {"n_probes_with_expected": 0}
    mean_clean = sum(r.clean_mean_lp_per_token for r in scoreable) / len(scoreable)
    mean_int = sum(r.intervened_mean_lp_per_token for r in scoreable) / len(scoreable)
    return {
        "n_probes_with_expected": len(scoreable),
        "mean_logprob_clean_per_token": mean_clean,
        "mean_logprob_intervened_per_token": mean_int,
        "mean_delta_per_token": mean_clean - mean_int,
    }
