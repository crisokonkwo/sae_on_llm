"""TOFU dataset loading and Q&A prompt formatting.

The TOFU benchmark (Maini et al., 2024) consists of fictional author profiles
with question/answer pairs. The HF dataset ``locuslab/TOFU`` provides several
configurations:

* ``full``                  — all 4000 Q&A pairs (200 authors x 20 Qs).
* ``forget01`` / ``forget05`` / ``forget10`` — 1/5/10 percent designated as
  the "forget set" (the rest is the corresponding ``retain99/95/90``).
* ``real_authors_perturbed``, ``world_facts_perturbed``,
  ``forget*_perturbed``, ``retain_perturbed`` — perturbed eval splits used to
  compute truth-ratio metrics.

Each row has fields ``question`` and ``answer`` (plus ``paraphrased_answer``,
``perturbed_answer`` etc. in eval configs).

We treat the model as a question-answering completion model and train with
causal-LM loss on the answer tokens only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


TOFU_DATASET_NAME = "locuslab/TOFU"

# Known config names. Not exhaustive — the dataset may add more — but covers
# everything needed for M3/M4.
TOFU_KNOWN_CONFIGS = (
    # train sets
    "full",
    "forget01", "forget05", "forget10",
    "retain90", "retain95", "retain99",
    # eval sets
    "world_facts", "real_authors",
    "forget01_perturbed", "forget05_perturbed", "forget10_perturbed",
    "retain_perturbed",
    "real_authors_perturbed", "world_facts_perturbed",
)

DEFAULT_PROMPT_TEMPLATE = "Question: {question}\nAnswer:"


def format_qa(
    question: str,
    answer: str,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    eos_token: str = "",
) -> tuple[str, str]:
    """Return (prompt, full_text). ``full_text = prompt + ' ' + answer + eos``.

    The trainer tokenizes ``full_text`` and masks the prompt portion with
    ``-100`` so only answer tokens contribute to the loss.
    """
    prompt = prompt_template.format(question=question)
    full = f"{prompt} {answer}{eos_token}"
    return prompt, full


def load_tofu_split(config: str, split: str = "train"):
    """Lazy-import ``datasets`` and load a TOFU configuration."""
    from datasets import load_dataset

    if config not in TOFU_KNOWN_CONFIGS:
        # Don't hard-fail; new configs may exist. Just warn.
        print(f"[tofu] note: '{config}' not in known configs {TOFU_KNOWN_CONFIGS}")
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    ds = load_dataset(TOFU_DATASET_NAME, config, split=split, token=hf_token)
    return ds


# ----------------------------------------------------------------------
# Causal-LM tokenization with prompt masking
# ----------------------------------------------------------------------
@dataclass
class TokenizedExample:
    input_ids: list[int]
    labels: list[int]
    attention_mask: list[int]


def _tokenize_example(
    question: str,
    answer: str,
    tokenizer,
    prompt_template: str,
    max_length: int,
    mask_prompt: bool,
) -> TokenizedExample:
    eos = tokenizer.eos_token or ""
    prompt, full = format_qa(question, answer, prompt_template, eos_token=eos)

    full_ids = tokenizer.encode(full, add_special_tokens=True, truncation=True, max_length=max_length)
    if mask_prompt:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True, truncation=True, max_length=max_length)
        n_prompt = min(len(prompt_ids), len(full_ids))
        labels = [-100] * n_prompt + full_ids[n_prompt:]
    else:
        labels = list(full_ids)
    return TokenizedExample(
        input_ids=full_ids,
        labels=labels,
        attention_mask=[1] * len(full_ids),
    )


def build_tofu_train_dataset(
    tokenizer,
    config: str = "full",
    split: str = "train",
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    max_length: int = 512,
    mask_prompt: bool = True,
):
    """Return a HF Dataset of tokenized Q&A pairs ready for causal-LM loss.

    Rows: ``input_ids``, ``labels``, ``attention_mask``. Labels are
    ``-100`` over the prompt portion when ``mask_prompt=True``.
    """
    raw = load_tofu_split(config, split=split)

    def _map_one(row):
        ex = _tokenize_example(
            row["question"], row["answer"], tokenizer,
            prompt_template=prompt_template,
            max_length=max_length,
            mask_prompt=mask_prompt,
        )
        return {
            "input_ids": ex.input_ids,
            "labels": ex.labels,
            "attention_mask": ex.attention_mask,
        }

    # Keep the originals around for eval / inspection.
    cols_to_remove = [c for c in raw.column_names if c not in ("question", "answer")]
    tokenized = raw.map(
        _map_one,
        remove_columns=cols_to_remove,
        desc="[tofu] tokenizing",
    )
    return tokenized


# ----------------------------------------------------------------------
# Collator (right-pads to max length in batch, pads labels with -100)
# ----------------------------------------------------------------------
@dataclass
class CausalLMCollator:
    pad_token_id: int

    def __call__(self, batch: Sequence[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(b["input_ids"]) for b in batch)
        input_ids = torch.full((len(batch), max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            input_ids[i, :n] = torch.tensor(b["input_ids"], dtype=torch.long)
            labels[i, :n] = torch.tensor(b["labels"], dtype=torch.long)
            attn[i, :n] = torch.tensor(b["attention_mask"], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attn}
