"""Pillar 1 baseline: one vLLM replica, measured directly and through the gateway.

Produces the gateway overhead (TTFT / E2E / throughput) with repeats so later
experiments have a controlled reference point. Starts and stops its own servers.

python -m benchmarks.runners.exp_baseline --repeats 3
"""
from __future__ import annotations

import argparse

from benchmarks.analysis.stats import compare
from benchmarks.runners.common import (GATEWAY_URL, KEYS, MODEL, rep_values, result_dir, save_json,
                                       start_gateway, start_vllm)
from benchmarks.runners.loadgen import Workload, run_and_save
from scripts import serverctl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workload", default="benchmarks/workloads/chat_short.yaml")
    ap.add_argument("--num-requests", type=int, default=None)
    a = ap.parse_args()
    out = result_dir("baseline")
    try:
        info = start_vllm("vllm-a", 8001, extra="--enable-prompt-tokens-details")
        start_gateway({"vllm-a": "http://127.0.0.1:8001"}, policy="round_robin")
        w = Workload.from_yaml(a.workload, model=MODEL, num_requests=a.num_requests)
        scrape = ["http://127.0.0.1:8001"]
        direct = run_and_save("http://127.0.0.1:8001", KEYS, w, str(out / "direct"), a.repeats, "direct",
                              extra_env={"vllm_startup": info}, scrape_metrics_urls=scrape)
        via_gw = run_and_save(GATEWAY_URL, KEYS, w, str(out / "gateway"), a.repeats, "gateway",
                              scrape_metrics_urls=scrape)
        d, g = out / "direct", out / "gateway"
        report = {
            "experiment": "baseline", "model": MODEL, "workload": w.name, "repeats": a.repeats,
            "vllm_ready_after_s": info.get("ready_after_s"), "model_load_gib": info.get("model_load_gib"),
            "gateway_overhead": {
                "ttft_p50": compare(rep_values(d, "ttft_s", "p50"), rep_values(g, "ttft_s", "p50")),
                "ttft_p99": compare(rep_values(d, "ttft_s", "p99"), rep_values(g, "ttft_s", "p99")),
                "e2e_p50": compare(rep_values(d, "e2e_s", "p50"), rep_values(g, "e2e_s", "p50")),
                "output_tok_per_s": compare(rep_values(d, "output_tok_per_s"), rep_values(g, "output_tok_per_s"),
                                            lower_is_better=False),
            },
            "direct": direct["aggregate"], "gateway": via_gw["aggregate"],
        }
        save_json(out / "report.json", report)
        print(f"[baseline] wrote {out}/report.json")
    finally:
        serverctl.stopall()


if __name__ == "__main__":
    main()
