"""Lightweight TOFU evaluation.

Implements the core TOFU metrics adequate for M3 (establish the FT baseline):

* ``answer_prob``        — mean P(answer | prompt) per example (geometric mean
  over answer tokens). On forget set it should be high after FT; on retain
  set it should also be high after FT. After unlearning, the forget-set
  number should drop while retain stays high.
* ``answer_logprob``     — mean log-prob per answer token (same info, more
  numerically stable to read).
* ``rouge_l``            — surface-form ROUGE-L between greedy generation
  and reference answer.
* generation samples     — qualitative sanity check (clean strings).

The full TOFU paper adds *truth ratio* (using perturbed alternative answers)
and aggregate forget quality scores. Those need the ``*_perturbed`` configs
and are easy to add later; we keep the M3 harness focused.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F

from .tofu_data import DEFAULT_PROMPT_TEMPLATE, load_tofu_split


@dataclass
class TOFUEvalResult:
    config: str
    n: int
    mean_answer_logprob: float       # average log P(answer | prompt) per token
    mean_answer_prob: float          # geometric-mean per-token prob
    mean_rouge_l: float
    samples: list[dict] = field(default_factory=list)


# ----------------------------------------------------------------------
# Probability of the reference answer
# ----------------------------------------------------------------------
@torch.no_grad()
def _answer_logprob(model, tokenizer, prompt: str, answer: str, device, max_length: int) -> tuple[float, int]:
    """Return (sum log-prob of answer tokens, number of answer tokens)."""
    eos = tokenizer.eos_token or ""
    full = f"{prompt} {answer}{eos}"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True, truncation=True, max_length=max_length)
    full_ids = tokenizer.encode(full, add_special_tokens=True, truncation=True, max_length=max_length)
    if len(full_ids) <= len(prompt_ids):
        return 0.0, 0
    input_ids = torch.tensor(full_ids, device=device).unsqueeze(0)
    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[:, :-1, :].float()
    targets = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    sel = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (1, T-1)
    n_prompt = len(prompt_ids)
    ans_logprobs = sel[0, n_prompt - 1:]                            # answer tokens only
    return float(ans_logprobs.sum().item()), int(ans_logprobs.numel())


# ----------------------------------------------------------------------
# Surface-form ROUGE-L (no external deps)
# ----------------------------------------------------------------------
def _lcs_length(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            tmp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[len(b)]


def rouge_l(ref: str, hyp: str) -> float:
    """Token-level ROUGE-L F1 (whitespace tokenization, lowercased)."""
    r = ref.lower().split()
    h = hyp.lower().split()
    if not r or not h:
        return 0.0
    lcs = _lcs_length(r, h)
    if lcs == 0:
        return 0.0
    p = lcs / len(h)
    rec = lcs / len(r)
    return 2 * p * rec / (p + rec)


# ----------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------
@torch.no_grad()
def _generate(model, tokenizer, prompt: str, max_new_tokens: int, device) -> str:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    # Strip the prompt prefix if echoed.
    if full.startswith(prompt):
        return full[len(prompt):].strip()
    return full.strip()


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate_tofu_config(
    model,
    tokenizer,
    config: str,
    split: str = "train",
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    max_examples: int = 100,
    max_length: int = 512,
    max_new_tokens: int = 64,
    n_generation_samples: int = 5,
    device: str | torch.device = "cuda",
) -> TOFUEvalResult:
    """Evaluate the (possibly LoRA-wrapped) model on one TOFU config."""
    ds = load_tofu_split(config, split=split)
    n = min(max_examples, len(ds))
    model.eval()

    total_lp = 0.0
    total_tok = 0
    rouge_scores: list[float] = []
    samples: list[dict] = []

    for i in range(n):
        row = ds[i]
        q, a = row["question"], row["answer"]
        prompt = prompt_template.format(question=q)

        # 1. answer log-prob
        lp, ntok = _answer_logprob(model, tokenizer, prompt, a, device, max_length=max_length)
        total_lp += lp
        total_tok += ntok

        # 2. generation + ROUGE (only on a subset to keep eval fast)
        if i < n_generation_samples or (i < n and len(rouge_scores) < min(n, 32)):
            gen = _generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens, device=device)
            r = rouge_l(a, gen)
            rouge_scores.append(r)
            if len(samples) < n_generation_samples:
                samples.append({
                    "question": q,
                    "reference": a,
                    "generated": gen,
                    "rouge_l": r,
                    "answer_logprob_per_token": (lp / ntok) if ntok else 0.0,
                })

    mean_lp_per_tok = (total_lp / total_tok) if total_tok else 0.0
    return TOFUEvalResult(
        config=config,
        n=n,
        mean_answer_logprob=mean_lp_per_tok,
        mean_answer_prob=float(torch.exp(torch.tensor(mean_lp_per_tok)).item()),
        mean_rouge_l=float(sum(rouge_scores) / len(rouge_scores)) if rouge_scores else 0.0,
        samples=samples,
    )


def write_eval_report(out_path: Path, results: Sequence[TOFUEvalResult]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {r.config: {
        "n": r.n,
        "mean_answer_logprob_per_token": r.mean_answer_logprob,
        "mean_answer_prob_per_token": r.mean_answer_prob,
        "mean_rouge_l": r.mean_rouge_l,
        "samples": r.samples,
    } for r in results}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
