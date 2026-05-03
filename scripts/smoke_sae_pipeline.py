"""End-to-end smoke test for the reproducible SAE training pipeline.

Creates synthetic activation shards, trains a small Top-K SAE from a generated
YAML config via ``scripts/train_sae.py --config ...``, verifies:
  * training loss decreases,
  * L0 ~= configured top-k,
  * final checkpoint exists,
  * validation_report.json and training_curves.png are produced.

Run:
    python scripts/smoke_sae_pipeline.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml


def _make_shards(root: Path, d_model: int, n_features: int, k: int) -> None:
    torch.manual_seed(0)
    root.mkdir(parents=True, exist_ok=True)
    true_dict = torch.randn(n_features, d_model) / (d_model ** 0.5)

    def make_shard(n_tokens: int) -> torch.Tensor:
        code = torch.zeros(n_tokens, n_features)
        idx = torch.randint(0, n_features, (n_tokens, k))
        code.scatter_(1, idx, torch.rand(n_tokens, k))
        return code @ true_dict + 0.05 * torch.randn(n_tokens, d_model)

    for split, n_shards, n_tokens in [("train", 4, 2000), ("val", 2, 1000)]:
        d = root / split
        d.mkdir(parents=True)
        for i in range(n_shards):
            torch.save(make_shard(n_tokens), d / f"shard_{i:05d}.pt")
        with open(d / "meta.json", "w") as f:
            json.dump({
                "resolved_d_model": d_model,
                "resolved_layer_idx": 0,
                "tokens_written": n_shards * n_tokens,
                "dataset_name": "synthetic",
            }, f)


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="sae_smoke_"))
    d_model, n_features, k = 32, 128, 8
    shard_root = tmp / "activations"
    out_dir = tmp / "run"
    _make_shards(shard_root, d_model, n_features, k)

    cfg = {
        "output_dir": str(out_dir),
        "model": {"name": "synthetic", "layer_idx": 0, "hook_point": "resid_post"},
        "data": {"shard_dir": str(shard_root), "batch_size": 256, "buffer_shards": 2},
        "sae": {
            "d_model": None,
            "n_features": n_features,
            "expansion_factor": 4,
            "sparsity_mode": "topk",
            "topk": {"k": k, "k_aux": None, "aux_coef": 1/32, "dead_steps_threshold": 100},
        },
        "train": {
            "lr": 1e-3,
            "warmup_steps": 20,
            "max_steps": 200,
            "grad_clip": 1.0,
            "log_every": 50,
            "ckpt_every": 100,
            "seed": 0,
            "device": "cpu",
            "compute_dtype": "float32",
            "progress": False,
        },
        "validation": {"validate_after": True, "max_batches": 4, "batch_size": 256, "plot_after": True},
    }
    cfg_path = tmp / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    cmd = [sys.executable, "scripts/train_sae.py", "--config", str(cfg_path)]
    print("[smoke] running:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent,
                         text=True, capture_output=True)
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise SystemExit(res.returncode)

    metrics_path = out_dir / "metrics.jsonl"
    rows = [json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()]
    assert len(rows) >= 2, "expected at least two metric rows"
    assert rows[-1]["recon"] < rows[0]["recon"], (rows[0]["recon"], rows[-1]["recon"])
    assert abs(rows[-1]["l0"] - k) < 0.25, rows[-1]["l0"]
    assert (out_dir / "ckpt_final.pt").exists()
    assert (out_dir / "validation_report.json").exists()
    assert (out_dir / "training_curves.png").exists()

    val = json.loads((out_dir / "validation_report.json").read_text())
    assert "reconstruction" in val
    assert val["reconstruction"]["l0_mean"] == k

    print("[smoke] OK")
    print(f"[smoke] temp run dir: {out_dir}")
    print(f"[smoke] first recon={rows[0]['recon']:.4f} last recon={rows[-1]['recon']:.4f}")
    print(f"[smoke] val EV={val['reconstruction']['explained_variance']:.4f}")


if __name__ == "__main__":
    main()
