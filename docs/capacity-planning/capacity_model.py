"""Capacity and cost model driven by MEASURED single-GPU benchmarks.

Inputs
  - model shape (preset or HF config.json)       -> KV bytes per token (exact)
  - GPU memory, weight memory                    -> KV capacity, max concurrent sequences (computed)
  - measured output tok/s per GPU (aggregate.json from a loadgen run on that GPU) (measured)
  - target traffic: req/s, mean in/out tokens, headroom                      (assumed)
  - $/GPU-hour                                                              (assumed)
Outputs
  - GPUs required, cost per 1M output tokens, and a large-cluster table that is
    explicitly labelled EXTRAPOLATED (linear scaling + stated efficiency factor).

python docs/capacity-planning/capacity_model.py --preset qwen2.5-1.5b --gpu a100-40gb \
    --measured benchmarks/results/baseline/<run>/direct/aggregate.json --rps 50 --in-tokens 1000 --out-tokens 200
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

PRESETS = {  # layers, kv_heads, head_dim, params (B)
    "qwen2.5-0.5b": (24, 2, 64, 0.49),
    "qwen2.5-1.5b": (28, 2, 128, 1.54),
    "qwen2.5-7b": (28, 4, 128, 7.6),
    "llama-3.1-8b": (32, 8, 128, 8.0),
    "llama-3.1-70b": (80, 8, 128, 70.6),
}
GPUS = {  # HBM GiB, assumed on-demand $/hr (EDIT for your provider)
    "a100-40gb": (40, 3.0), "a100-80gb": (80, 4.0), "h100-80gb": (80, 7.0), "l4-24gb": (24, 0.9),
}


def kv_bytes_per_token(layers, kv_heads, head_dim, dtype_bytes=2) -> int:
    return 2 * layers * kv_heads * head_dim * dtype_bytes


def plan(preset: str, gpu: str, measured_tok_s: float | None, rps: float, in_tok: int, out_tok: int,
         gpu_mem_util: float = 0.9, headroom: float = 0.7, tp: int = 1, dollars_per_hr: float | None = None,
         scale_efficiency: float = 0.9) -> dict:
    L, H, D, params_b = PRESETS[preset]
    hbm_gib, price = GPUS[gpu]
    price = dollars_per_hr or price
    weights_gib = params_b * 2 / 1.073741824  # bf16, GiB
    kv_tok = kv_bytes_per_token(L, H, D)
    usable_gib = hbm_gib * gpu_mem_util * tp - weights_gib - 1.5 * tp  # ~1.5 GiB activations/graphs per GPU (assumed)
    kv_capacity_tokens = max(0, int(usable_gib * 2**30 / kv_tok))
    ctx = in_tok + out_tok
    out = {
        "inputs": {"preset": preset, "gpu": gpu, "tp": tp, "rps": rps, "in_tokens": in_tok, "out_tokens": out_tok,
                   "headroom": headroom, "dollars_per_gpu_hr": price},
        "computed": {"kv_bytes_per_token": kv_tok, "weights_gib": round(weights_gib, 2),
                     "kv_capacity_tokens_per_replica": kv_capacity_tokens,
                     "max_concurrent_full_context_seqs_per_replica": kv_capacity_tokens // ctx if ctx else None},
    }
    if measured_tok_s:
        need_tok_s = rps * out_tok
        replicas = math.ceil(need_tok_s / (measured_tok_s * headroom))
        gpus = replicas * tp
        cost_per_m = (price * tp) / (measured_tok_s * headroom * 3600) * 1e6
        out["measured_basis"] = {"output_tok_s_per_replica": measured_tok_s}
        out["plan"] = {"required_output_tok_s": need_tok_s, "replicas": replicas, "gpus": gpus,
                       "monthly_cost_usd": gpus * price * 730, "cost_per_1M_output_tokens_usd": cost_per_m}
        out["extrapolated_cluster_table"] = {
            "label": "EXTRAPOLATED - assumes near-linear data-parallel scaling at the stated efficiency; NOT measured",
            "scale_efficiency": scale_efficiency,
            "rows": [{"gpus": g, "est_output_tok_s": g / tp * measured_tok_s * scale_efficiency * headroom,
                      "est_rps_at_out_tokens": g / tp * measured_tok_s * scale_efficiency * headroom / out_tok}
                     for g in (8, 64, 512, 4096)],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="qwen2.5-1.5b", choices=PRESETS)
    ap.add_argument("--gpu", default="a100-40gb", choices=GPUS)
    ap.add_argument("--measured", default=None, help="aggregate.json from loadgen (uses output_tok_per_s mean)")
    ap.add_argument("--measured-tok-s", type=float, default=None)
    ap.add_argument("--rps", type=float, default=20)
    ap.add_argument("--in-tokens", type=int, default=1000)
    ap.add_argument("--out-tokens", type=int, default=200)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--price", type=float, default=None)
    a = ap.parse_args()
    tok_s = a.measured_tok_s
    if a.measured:
        tok_s = json.loads(Path(a.measured).read_text())["aggregate"]["output_tok_per_s"]["mean"]
    print(json.dumps(plan(a.preset, a.gpu, tok_s, a.rps, a.in_tokens, a.out_tokens, tp=a.tp, dollars_per_hr=a.price),
                     indent=2))


if __name__ == "__main__":
    main()
