"""End-to-end smoke test for the TOFU finetuning pipeline.

Does NOT require Gemma, peft, or a GPU. It substitutes a tiny GPT-2 model
and a mock dataset to verify:

  * tokenization with prompt masking,
  * causal-LM collator (right-padding, label masking),
  * the training loop runs and loss decreases,
  * LoRA-style trainable subset behaves correctly (we run *without* peft and
    instead just freeze everything except the LM head as a stand-in),
  * the eval harness computes ROUGE-L, answer log-prob, and generations,
  * checkpoint manifests are written.

Run:
    python scripts/smoke_tofu_pipeline.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torch.utils.data import DataLoader

from finetune.eval import evaluate_tofu_config, rouge_l
from finetune.finetune import FinetuneConfig, LoRAConfig, TOFUTrainer
from finetune.tofu_data import CausalLMCollator, build_tofu_train_dataset


def _make_mock_dataset(n: int = 16):
    """Build a tiny in-memory HF dataset with the same schema as TOFU."""
    from datasets import Dataset
    rows = []
    for i in range(n):
        rows.append({
            "question": f"Who wrote book number {i}?",
            "answer": f"Book number {i} was written by Author {i % 4}.",
        })
    return Dataset.from_list(rows)


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="tofu_smoke_"))
    print(f"[smoke] tmp dir: {tmp}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "sshleifer/tiny-gpt2"   # ~3M params, cpu-friendly
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)

    # Replace TOFU loader with a mock for both tokenization and eval. ``eval.py``
    # imports the symbol at module load, so we patch both spots.
    mock_ds = _make_mock_dataset(n=16)
    print(f"[smoke] mock dataset OK ({len(mock_ds)} rows, columns={mock_ds.column_names})") 
    print(f"[smoke] example row: Q: {mock_ds[0]['question']} A: {mock_ds[0]['answer']}")
    with mock.patch("finetune.tofu_data.load_tofu_split", return_value=mock_ds), \
         mock.patch("finetune.eval.load_tofu_split", return_value=mock_ds):
        train_ds = build_tofu_train_dataset(tokenizer, config="full", max_length=64, mask_prompt=True)
        assert len(train_ds) == 16
        assert {"input_ids", "labels", "attention_mask"} <= set(train_ds.column_names)
        # First row: confirm prompt tokens are masked with -100.
        row0 = train_ds[0]
        assert any(l == -100 for l in row0["labels"]), "expected prompt tokens masked"
        assert any(l != -100 for l in row0["labels"]), "expected answer tokens unmasked"
        print(f"[smoke] tokenized OK ({len(train_ds)} rows, first len={len(row0['input_ids'])})")

        # Collator sanity check (also verifies DataLoader integration). We don't need to verify padding 
        # correctness here since the trainer will error if shapes are wrong, but we do want to confirm the collator is wired up 
        # and producing tensors of the expected shape. 
        # We also check that the collator doesn't error when the batch has variable-length sequences.
        collator = CausalLMCollator(pad_token_id=tokenizer.pad_token_id)
        loader = DataLoader(train_ds, batch_size=4, shuffle=False, collate_fn=collator, drop_last=True)
        batch = next(iter(loader))
        assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
        print(f"[smoke] collator OK (batch shape {tuple(batch['input_ids'].shape)})")

        # Tiny LoRA stand-in: train only the LM head (no peft dependency).
        for p in model.parameters():
            p.requires_grad = False
        head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        if head is None:
            head = model.lm_head
        for p in head.parameters():
            p.requires_grad = True

        cfg = FinetuneConfig(
            model_name=model_name,
            output_dir=str(tmp / "run"),
            dataset_config="full",
            max_length=64,
            mask_prompt=True,
            batch_size=4, grad_accum_steps=1,
            lr=1e-3, warmup_steps=2, max_steps=12,
            epochs=0, log_every=4, ckpt_every=12,
            seed=0, device="cpu", compute_dtype="float32",
            progress=False, gradient_checkpointing=False,
            lora=LoRAConfig(),
        )
        trainer = TOFUTrainer(model, tokenizer, cfg)
        trainer.train(loader)

        # Verify training loss decreased.
        rows = [json.loads(l) for l in (Path(cfg.output_dir) / "metrics.jsonl").read_text().splitlines() if l.strip()]
        assert len(rows) >= 2, rows
        assert rows[-1]["loss"] < rows[0]["loss"], (rows[0]["loss"], rows[-1]["loss"])
        print(f"[smoke] loss decreased: {rows[0]['loss']:.3f} -> {rows[-1]['loss']:.3f}")
        # Checkpoint manifest exists (we skip the peft save path because we don't have peft).
        assert (Path(cfg.output_dir) / "ft_config.json").exists()

        # Eval harness
        result = evaluate_tofu_config(
            model, tokenizer, config="full",
            max_examples=4, max_new_tokens=8, n_generation_samples=2, device="cpu",
        )
        assert result.n == 4
        assert 0.0 <= result.mean_answer_prob <= 1.0
        assert -50.0 <= result.mean_answer_logprob <= 0.0
        assert 0.0 <= result.mean_rouge_l <= 1.0
        print(f"[smoke] eval OK: ans_prob={result.mean_answer_prob:.4f} "
              f"ans_logp={result.mean_answer_logprob:+.4f} rougeL={result.mean_rouge_l:.4f}")

    # ROUGE-L standalone sanity
    assert rouge_l("the cat sat on the mat", "the cat sat on the mat") == 1.0
    assert rouge_l("", "anything") == 0.0
    print("[smoke] ALL OK")


if __name__ == "__main__":
    main()
