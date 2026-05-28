"""Evaluate a TOFU-finetuned model on retain/forget/utility splits.

Reports answer log-prob, surface ROUGE-L, and a few greedy generations.

Examples:

    # Evaluate a LoRA checkpoint on multiple splits in one go
    python scripts/eval_tofu.py \
        --ckpt runs/gemma2b_tofu_ft/run_1/ckpt_final \
        --configs forget01 retain99 real_authors world_facts \
        --output runs/gemma2b_tofu_ft/run_1/tofu_eval.json

    # Evaluate the base model (no finetuning) for comparison
    python scripts/eval_tofu.py \
        --base-model google/gemma-2-2b \
        --configs forget01 retain99 \
        --output runs/baseline_tofu_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from finetune.eval import evaluate_tofu_config, write_eval_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", default=None, help="PEFT adapter directory (ckpt_final).")
    src.add_argument("--base-model", default=None, help="Plain HF model name (no FT).")
    p.add_argument("--base-model-name", default=None,
                   help="Override base model name when loading a LoRA ckpt.")
    p.add_argument("--merge-adapter", action="store_true",
                   help="Merge LoRA into base weights before eval (faster, larger memory).")
    # eval
    p.add_argument("--configs", nargs="+", default=["forget01", "retain99", "real_authors", "world_facts"])
    p.add_argument("--split", default="train",
                   help="HF split. TOFU only ships 'train' for most configs.")
    p.add_argument("--max-examples", type=int, default=100)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--n-generation-samples", type=int, default=5)
    p.add_argument("--prompt-template", default="Question: {question}\nAnswer:")
    # device
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.ckpt is not None:
        from finetune.load_ft import load_ft_model
        print(f"[eval] loading LoRA ckpt {args.ckpt}")
        model, tokenizer = load_ft_model(
            args.ckpt,
            base_model_name=args.base_model_name,
            device=args.device,
            dtype=args.compute_dtype,
            merge_adapter=args.merge_adapter,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[eval] loading base model {args.base_model}")
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.compute_dtype]
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, device_map=args.device)
        model.eval()

    results = []
    for cfg in args.configs:
        print(f"\n[eval] === {cfg} ===")
        r = evaluate_tofu_config(
            model, tokenizer,
            config=cfg,
            split=args.split,
            prompt_template=args.prompt_template,
            max_examples=args.max_examples,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            n_generation_samples=args.n_generation_samples,
            device=args.device,
        )
        print(f"        n={r.n}")
        print(f"        mean answer logprob per token: {r.mean_answer_logprob:+.4f}")
        print(f"        mean answer prob per token  : {r.mean_answer_prob:.4f}")
        print(f"        mean ROUGE-L                : {r.mean_rouge_l:.4f}")
        for s in r.samples[:2]:
            print(f"        Q: {s['question'][:80]}")
            print(f"        A_ref: {s['reference'][:80]}")
            print(f"        A_gen: {s['generated'][:80]}")
        results.append(r)

    write_eval_report(Path(args.output), results)
    print(f"\n[eval] wrote {args.output}")


if __name__ == "__main__":
    main()
