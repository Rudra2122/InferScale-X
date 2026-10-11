"""Pillar 2 (single-GPU stand-in for Dynamo routing): routing policy comparison.

Two vLLM replicas share one A100 (gpu_memory_utilization 0.42 each). The gateway
routes with round_robin (baseline), least_outstanding and prefix_affinity
(KV-cache-aware). Prefix caches are reset before every run.

Caveat recorded in the report: both replicas share one GPU's SMs and HBM
bandwidth, so absolute throughput is NOT what two GPUs would give. The
comparison between policies is still controlled: same hardware, same load.

python -m benchmarks.runners.exp_routing --repeats 3
"""
from __future__ import annotations

import argparse

from benchmarks.analysis.stats import compare
from benchmarks.runners.common import (GATEWAY_URL, KEYS, MODEL, rep_values, reset_prefix_cache, result_dir,
                                       save_json, start_gateway, start_vllm)
from benchmarks.runners.loadgen import Workload, run_and_save
from scripts import serverctl

REPLICAS = {"vllm-a": "http://127.0.0.1:8001", "vllm-b": "http://127.0.0.1:8002"}
POLICIES = ["round_robin", "least_outstanding", "prefix_affinity"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workloads", default="benchmarks/workloads/prefix_heavy.yaml,benchmarks/workloads/chat_short.yaml")
    ap.add_argument("--num-requests", type=int, default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.42)
    a = ap.parse_args()
    out = result_dir("routing")
    report = {"experiment": "routing", "model": MODEL, "replicas": 2, "shared_gpu": True,
              "caveat": "Both replicas share one A100; policy deltas are valid, absolute scale is not.", "runs": {}}
    try:
        start_vllm("vllm-a", 8001, gpu_mem=a.gpu_mem, extra="--enable-prompt-tokens-details")
        start_vllm("vllm-b", 8002, gpu_mem=a.gpu_mem, extra="--enable-prompt-tokens-details")
        for wl in a.workloads.split(","):
            w = Workload.from_yaml(wl, model=MODEL, num_requests=a.num_requests)
            report["runs"][w.name] = {}
            for policy in POLICIES:
                serverctl.stop("gateway")
                start_gateway(REPLICAS, policy=policy)
                for url in REPLICAS.values():
                    reset_prefix_cache(url)
                d = out / w.name / policy
                agg = run_and_save(GATEWAY_URL, KEYS, w, str(d), a.repeats, f"{w.name}/{policy}",
                                   scrape_metrics_urls=list(REPLICAS.values()))
                report["runs"][w.name][policy] = agg["aggregate"]
            base = out / w.name / "round_robin"
            report["runs"][w.name]["vs_round_robin"] = {
                p: {
                    "ttft_p50": compare(rep_values(base, "ttft_s", "p50"), rep_values(out / w.name / p, "ttft_s", "p50")),
                    "ttft_p95": compare(rep_values(base, "ttft_s", "p95"), rep_values(out / w.name / p, "ttft_s", "p95")),
                    "throughput": compare(rep_values(base, "output_tok_per_s"),
                                          rep_values(out / w.name / p, "output_tok_per_s"), lower_is_better=False),
                    "cached_token_fraction": {
                        "baseline": rep_values(base, "cached_token_fraction"),
                        "candidate": rep_values(out / w.name / p, "cached_token_fraction")},
                } for p in POLICIES if p != "round_robin"
            }
        save_json(out / "report.json", report)
        print(f"[routing] wrote {out}/report.json")
    finally:
        serverctl.stopall()


if __name__ == "__main__":
    main()
