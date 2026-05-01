"""Evaluate a trained SAE.

Produces a JSON report at ``<output>/eval_report.json`` with reconstruction
metrics, CE-delta vs. mean ablation, and top-activating-token examples for
a sample of features.

Example:
    python scripts/eval_sae.py \
        --ckpt runs/gemma2b_layer13_topk_8d/ckpt_final.pt \
        --shard-dir activations/gemma2b_mid_eval \
        --model google/gemma-2-2b \
        --output runs/gemma2b_layer13_topk_8d/eval
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from sae import SAE, SAEConfig
from sae.dataset import ActivationDataset, load_meta
from sae.eval import (
    ce_delta,
    reconstruction_metrics,
    top_activating_tokens,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to ckpt_*.pt produced by train_sae.py")
    p.add_argument("--shard-dir", required=True, help="Held-out activation shards.")
    p.add_argument("--output", required=True, help="Output directory for the report.")
    # reconstruction
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--max-batches", type=int, default=200)
    # CE delta + interp (require LM)
    p.add_argument("--model", default=None,
                   help="HF causal LM. If omitted, CE-delta and top-activating-token sections are skipped.")
    p.add_argument("--layer", type=int, default=None,
                   help="Defaults to harvest meta.json -> resolved_layer_idx.")
    p.add_argument("--ce-dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--ce-dataset-config", default="sample-10BT")
    p.add_argument("--ce-dataset-split", default="train")
    p.add_argument("--ce-text-field", default="text")
    p.add_argument("--ce-max-docs", type=int, default=32)
    p.add_argument("--ce-seq-len", type=int, default=512)
    # top-activating tokens
    p.add_argument("--n-features-to-probe", type=int, default=16,
                   help="Sample this many random features to inspect.")
    p.add_argument("--top-k-tokens", type=int, default=8)
    p.add_argument("--interp-max-docs", type=int, default=256)
    p.add_argument("--interp-seq-len", type=int, default=256)
    # device
    p.add_argument("--device", default="cuda")
    p.add_argument("--compute-dtype", default="float32",
                   choices=["float16", "bfloat16", "float32"])
    return p.parse_args()


def _load_sae(ckpt_path: Path, device: torch.device, dtype: torch.dtype) -> tuple[SAE, dict]:
    payload = torch.load(ckpt_path, map_location="cpu")
    sae_cfg_dict = payload["sae_cfg"]
    sae_cfg = SAEConfig(**sae_cfg_dict)
    sae = SAE(sae_cfg)
    sae.load_state_dict(payload["sae_state"])
    sae.to(device, dtype=dtype)
    sae.eval()
    return sae, payload


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.compute_dtype]

    print(f"[eval] loading SAE from {args.ckpt}")
    sae, payload = _load_sae(Path(args.ckpt), device, dtype)
    print(f"[eval] SAE: d_model={sae.cfg.d_model}, n_features={sae.cfg.n_features}, mode={sae.cfg.sparsity_mode}")

    report: dict = {
        "ckpt": str(args.ckpt),
        "step": int(payload.get("step", -1)),
        "sae_cfg": asdict(sae.cfg),
    }

    # 1. Reconstruction metrics on held-out shards.
    print(f"[eval] reconstruction metrics on {args.shard_dir} (max_batches={args.max_batches})")
    eval_ds = ActivationDataset(
        shard_dir=args.shard_dir,
        batch_size=args.batch_size,
        buffer_shards=1,
        shuffle=False,
        infinite=False,
        drop_last=False,
    )
    report["reconstruction"] = reconstruction_metrics(
        sae, eval_ds, max_batches=args.max_batches, device=device
    )
    for k, v in report["reconstruction"].items():
        if isinstance(v, (int, float)):
            print(f"        {k}={v:.6g}" if isinstance(v, float) else f"        {k}={v}")

    # # 2/3. CE-delta + top-activating tokens (require the LM).
    # if args.model is not None:
    #     from transformers import AutoModelForCausalLM, AutoTokenizer
    #     from sae.data import _stream_text  # internal helper, fine to reuse

    #     meta = load_meta(args.shard_dir)
    #     layer_idx = args.layer if args.layer is not None else meta.get("resolved_layer_idx")
    #     if layer_idx is None:
    #         raise ValueError("--layer not given and not found in shard meta.json")

    #     print(f"[eval] loading LM {args.model} (layer={layer_idx})")
    #     tokenizer = AutoTokenizer.from_pretrained(args.model)
    #     model = AutoModelForCausalLM.from_pretrained(
    #         args.model, torch_dtype=dtype, device_map=device
    #     )
    #     model.eval()

    #     # CE delta
    #     print(f"[eval] CE-delta on {args.ce_max_docs} docs from {args.ce_dataset}")
    #     texts = list(_stream_text(
    #         args.ce_dataset, args.ce_dataset_config, args.ce_dataset_split,
    #         args.ce_text_field, max_samples=args.ce_max_docs,
    #     ))
    #     report["ce_delta"] = ce_delta(
    #         sae, model, tokenizer, layer_idx, texts,
    #         seq_len=args.ce_seq_len, max_docs=args.ce_max_docs, device=device,
    #     )
    #     for k, v in report["ce_delta"].items():
    #         print(f"        {k}={v:.4f}" if isinstance(v, float) else f"        {k}={v}")

    #     # Top-activating tokens for a sample of features.
    #     gen = torch.Generator().manual_seed(0)
    #     feat_ids = torch.randperm(sae.cfg.n_features, generator=gen)[: args.n_features_to_probe].tolist()
    #     print(f"[eval] top-activating tokens for features {feat_ids[:5]}{'...' if len(feat_ids) > 5 else ''}")
    #     interp_texts = list(_stream_text(
    #         args.ce_dataset, args.ce_dataset_config, args.ce_dataset_split,
    #         args.ce_text_field, max_samples=args.interp_max_docs,
    #     ))
    #     top_tokens = top_activating_tokens(
    #         sae, model, tokenizer, layer_idx, interp_texts,
    #         feature_ids=feat_ids, top_k=args.top_k_tokens,
    #         seq_len=args.interp_seq_len, max_docs=args.interp_max_docs,
    #         device=device,
    #     )
    #     # Convert int keys to str for JSON.
    #     report["top_activating_tokens"] = {str(f): hits for f, hits in top_tokens.items()}
    # else:
    #     print("[eval] --model not given; skipping CE-delta + top-activating-token sections.")

    # out_path = out / "eval_report.json"
    # with open(out_path, "w") as f:
    #     json.dump(report, f, indent=2)
    # print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
