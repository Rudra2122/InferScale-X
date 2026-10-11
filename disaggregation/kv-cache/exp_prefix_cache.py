"""Pillar 4: KV-cache reuse, HBM pressure and cross-tenant cache isolation on one A100.

Part A  reuse      : prefix_heavy workload with vLLM prefix caching ON vs OFF.
Part B  pressure   : prefix caching ON at gpu_memory_utilization 0.85 / 0.5 / 0.3
                     -> hit rate, preemptions and TTFT as KV capacity shrinks.
Part C  isolation  : tenant A warms a long prefix; tenant B sends the same prefix.
                     Without per-tenant cache_salt, B gets cached tokens (and a
                     faster TTFT: a timing side channel). With the gateway
                     injecting cache_salt, B must get zero cached tokens.

Requires a vLLM version that supports `cache_salt` (V1 engine). If the request
is rejected, Part C records the failure instead of guessing.

python -m disaggregation.kv-cache.exp_prefix_cache   (module path has a dash; use the file path:)
python disaggregation/kv-cache/exp_prefix_cache.py --repeats 3
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx  # noqa: E402

from benchmarks.analysis.stats import compare, dist  # noqa: E402
from benchmarks.runners.common import (GATEWAY_URL, KEYS, MODEL, rep_values, reset_prefix_cache,  # noqa: E402
                                       result_dir, save_json, start_gateway, start_vllm)
from benchmarks.runners.loadgen import VOCAB, Workload, run_and_save  # noqa: E402
from scripts import serverctl  # noqa: E402

VLLM = "http://127.0.0.1:8001"
EXTRA = "--enable-prompt-tokens-details"


def part_reuse(out: Path, repeats: int, n: int | None) -> dict:
    w = Workload.from_yaml("benchmarks/workloads/prefix_heavy.yaml", model=MODEL, num_requests=n)
    res = {}
    for mode, flag in (("cache_off", "--no-enable-prefix-caching"), ("cache_on", "--enable-prefix-caching")):
        serverctl.stopall()
        start_vllm("vllm-a", 8001, extra=f"{EXTRA} {flag}")
        res[mode] = run_and_save(VLLM, KEYS, w, str(out / "reuse" / mode), repeats, f"reuse/{mode}",
                                 scrape_metrics_urls=[VLLM])["aggregate"]
    off, on = out / "reuse" / "cache_off", out / "reuse" / "cache_on"
    res["ttft_p50"] = compare(rep_values(off, "ttft_s", "p50"), rep_values(on, "ttft_s", "p50"))
    res["ttft_p95"] = compare(rep_values(off, "ttft_s", "p95"), rep_values(on, "ttft_s", "p95"))
    res["throughput"] = compare(rep_values(off, "output_tok_per_s"), rep_values(on, "output_tok_per_s"), False)
    return res


def part_pressure(out: Path, repeats: int, n: int | None, levels=(8000, 3000, 1200)) -> dict:
    """KV capacity sweep via --num-gpu-blocks-override (16 tokens/block). Working set = 32 prefixes x 2000
    tokens = 4000 blocks, so 8000 fits, 3000 thrashes a little, 1200 thrashes hard. (v1 swept
    gpu_memory_utilization 0.85/0.5/0.3, but even 0.3 of a 40 GB A100 leaves ~9 GB of KV for a 1.5B model,
    far more than the working set: no pressure, no difference - kept as a documented design bug.)"""
    w = Workload.from_yaml("benchmarks/workloads/prefix_heavy.yaml", model=MODEL, num_requests=n)
    w.num_shared_prefixes = 32
    res = {"working_set_blocks": 32 * 2000 // 16}
    base = None
    for blocks in levels:
        serverctl.stopall()
        try:
            start_vllm("vllm-a", 8001, extra=f"{EXTRA} --enable-prefix-caching --num-gpu-blocks-override {blocks}")
        except RuntimeError as e:
            res[str(blocks)] = {"error": str(e), "log_tail": serverctl.tail("vllm-a", 30)}
            continue
        d = out / "pressure" / f"kv_blocks_{blocks}"
        agg = run_and_save(VLLM, KEYS, w, str(d), repeats, f"pressure/{blocks}blk", scrape_metrics_urls=[VLLM])
        entry = {"kv_capacity_tokens": blocks * 16, "aggregate": agg["aggregate"],
                 "cached_token_fraction_per_rep": rep_values(d, "cached_token_fraction"),
                 "kv_log_lines": kv_capacity_lines("vllm-a")}
        if base is None:
            base = d
        else:
            entry["vs_largest_cache"] = {
                "ttft_p50": compare(rep_values(base, "ttft_s", "p50"), rep_values(d, "ttft_s", "p50")),
                "throughput": compare(rep_values(base, "output_tok_per_s"), rep_values(d, "output_tok_per_s"), False)}
        res[str(blocks)] = entry
    return res


def kv_capacity_lines(name: str) -> list[str]:
    log = serverctl.LOGS / f"{name}.log"
    if not log.exists():
        return []
    keys = ("KV cache", "GPU KV cache size", "Maximum concurrency", "num_gpu_blocks")
    return [ln.strip()[-200:] for ln in log.read_text(errors="replace").splitlines() if any(k in ln for k in keys)][:6]


def timed_request(url: str, key: str, prompt: str, extra: dict | None = None) -> dict:
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1,
            "temperature": 0.0, "stream": False, **(extra or {})}
    t = time.perf_counter()
    r = httpx.post(f"{url}/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"}, timeout=120)
    dt = time.perf_counter() - t
    out = {"status": r.status_code, "latency_s": dt}
    if r.status_code == 200:
        u = r.json().get("usage") or {}
        out["prompt_tokens"] = u.get("prompt_tokens")
        out["cached_tokens"] = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    else:
        out["error"] = r.text[:300]
    return out


def part_isolation(out: Path, trials: int) -> dict:
    import random

    serverctl.stopall()
    start_vllm("vllm-a", 8001, extra=f"{EXTRA} --enable-prefix-caching")
    res = {}
    for salt in (False, True):
        serverctl.stop("gateway")
        start_gateway({"vllm-a": VLLM}, policy="round_robin", cache_salt=salt)
        rows = []
        for i in range(trials):
            reset_prefix_cache(VLLM)
            rng = random.Random(i)
            secret_doc = " ".join(rng.choice(VOCAB) for _ in range(1500))
            a = timed_request(GATEWAY_URL, KEYS["acme"], secret_doc + " Summarize.")
            b = timed_request(GATEWAY_URL, KEYS["globex"], secret_doc + " Summarize.")
            a2 = timed_request(GATEWAY_URL, KEYS["acme"], secret_doc + " Summarize.")  # same tenant should hit
            rows.append({"trial": i, "acme_cold": a, "globex_same_prefix": b, "acme_warm": a2})
        key = "with_cache_salt" if salt else "without_cache_salt"
        res[key] = {
            "rows": rows,
            "globex_cached_tokens": dist(r["globex_same_prefix"].get("cached_tokens") for r in rows),
            "acme_warm_cached_tokens": dist(r["acme_warm"].get("cached_tokens") for r in rows),
            "acme_cold_latency_s": dist(r["acme_cold"]["latency_s"] for r in rows),
            "globex_latency_s": dist(r["globex_same_prefix"]["latency_s"] for r in rows),
            "acme_warm_latency_s": dist(r["acme_warm"]["latency_s"] for r in rows),
            "errors": [r for r in rows if r["globex_same_prefix"]["status"] != 200][:3],
        }
    ws = res["with_cache_salt"]
    res["isolation_enforced"] = (not ws["errors"] and ws["globex_cached_tokens"].get("max") == 0
                                 and (ws["acme_warm_cached_tokens"].get("min") or 0) > 0)
    res["leak_without_salt"] = (res["without_cache_salt"]["globex_cached_tokens"].get("min") or 0) > 0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--num-requests", type=int, default=None)
    ap.add_argument("--parts", default="reuse,pressure,isolation")
    ap.add_argument("--isolation-trials", type=int, default=10)
    a = ap.parse_args()
    parts = a.parts.split(",")
    # a pressure-only rerun must not become the "latest kv_cache report" (it would hide reuse/isolation evidence)
    out = result_dir("kv_pressure" if parts == ["pressure"] else "kv_cache")
    report = {"experiment": "kv_cache", "model": MODEL}
    try:
        if "reuse" in parts:
            report["reuse"] = part_reuse(out, a.repeats, a.num_requests)
            save_json(out / "report.json", report)
        if "pressure" in parts:
            report["pressure"] = part_pressure(out, a.repeats, a.num_requests)
            save_json(out / "report.json", report)
        if "isolation" in parts:
            report["isolation"] = part_isolation(out, a.isolation_trials)
        save_json(out / "report.json", report)
        print(f"[kv_cache] wrote {out}/report.json")
    finally:
        serverctl.stopall()


if __name__ == "__main__":
    main()
