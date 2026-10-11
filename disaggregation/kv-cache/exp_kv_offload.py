"""Pillar 4: REAL multi-tier KV cache on one A100 — HBM vs HBM + CPU DRAM (+ optional localhost remote tier).

Setup forces a working set larger than HBM KV capacity:
  - GPU KV capped with --num-gpu-blocks-override (default 1500 blocks x 16 tok = 24K tokens)
  - workload reuses 32 distinct 2,000-token prefixes (64K tokens of reusable KV)
so an HBM-only prefix cache thrashes, and a CPU tier can serve evicted prefixes.

Variants
  no_cache        prefix caching disabled                                (floor)
  hbm_only        vLLM prefix cache, capped HBM                          (baseline)
  hbm_cpu         + CPU DRAM tier. Tries, in order, and records which worked:
                    1. vLLM native offloading  (--kv-offloading-size / --kv-offloading-backend native)
                    2. vLLM OffloadingConnector via --kv-transfer-config
                    3. LMCache LMCacheConnectorV1 with LMCACHE_LOCAL_CPU=True   (pip install lmcache)
  hbm_cpu_remote  (--remote) LMCache CPU + remote tier served by `lmcache_server` on localhost.
                  Labelled "remote tier on localhost": it exercises the code path, not a network.

Flags and connector names change between vLLM releases; a variant that fails to start is
recorded with its log tail. Measured metrics: TTFT p50/p95, throughput, cached-token fraction
(where the engine reports it), vLLM prefix-cache counters, host RSS of the server process.

python disaggregation/kv-cache/exp_kv_offload.py --repeats 3 [--remote]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.analysis.stats import compare  # noqa: E402
from benchmarks.runners.common import KEYS, MODEL, rep_values, result_dir, save_json  # noqa: E402
from benchmarks.runners.loadgen import Workload, run_and_save  # noqa: E402
from scripts import serverctl  # noqa: E402

URL = "http://127.0.0.1:8001"


def cpu_offload_candidates(cpu_gib: float, blocks_cpu: int) -> list[tuple[str, str, dict]]:
    lm_env = {"LMCACHE_LOCAL_CPU": "True", "LMCACHE_MAX_LOCAL_CPU_SIZE": str(cpu_gib), "LMCACHE_CHUNK_SIZE": "256",
              "LMCACHE_USE_EXPERIMENTAL": "True"}
    return [
        ("vllm_native_offloading", f"--kv-offloading-size {cpu_gib} --kv-offloading-backend native", {}),
        ("vllm_offloading_connector",
         "--kv-transfer-config '" + json.dumps({"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
                                                "kv_connector_extra_config": {"num_cpu_blocks": blocks_cpu}}) + "'", {}),
        ("lmcache_cpu", "--kv-transfer-config '" + json.dumps({"kv_connector": "LMCacheConnectorV1",
                                                               "kv_role": "kv_both"}) + "'", lm_env),
    ]


def start(name_extra: str, extra: str, env: dict, gpu_blocks: int, timeout: float = 1200) -> dict:
    serverctl.stopall()
    cmd = serverctl.vllm_cmd(MODEL, 8001, 0.85, 4096,
                             f"--enable-prompt-tokens-details --num-gpu-blocks-override {gpu_blocks} {extra}")
    info = serverctl.start("vllm-a", cmd, 8001, timeout=timeout, env={"VLLM_SERVER_DEV_MODE": "1", **env})
    info["variant_args"] = extra
    if not info.get("healthy"):
        info["log_tail"] = serverctl.tail("vllm-a", 40)
    return info


def server_rss_gib() -> float | None:
    try:
        info = json.loads((serverctl.RUN / "vllm-a.json").read_text())
        out = subprocess.run(["ps", "-o", "rss=", "-g", str(info["pid"])], capture_output=True, text=True).stdout
        return sum(int(x) for x in out.split() if x.isdigit()) / 2**20
    except Exception:
        return None


def workload(n: int | None) -> Workload:
    return Workload(name="offload_working_set", model=MODEL, num_requests=n or 256, warmup_requests=32,
                    concurrency=8, input_tokens=2100, input_tokens_jitter=0.02, output_tokens=32,
                    prefix_reuse=1.0, shared_prefix_tokens=2000, num_shared_prefixes=32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--num-requests", type=int, default=None)
    ap.add_argument("--gpu-blocks", type=int, default=1500)
    ap.add_argument("--cpu-gib", type=float, default=8.0)
    ap.add_argument("--remote", action="store_true", help="also try LMCache with a localhost remote tier")
    a = ap.parse_args()
    out = result_dir("kv_offload")
    w = workload(a.num_requests)
    kv_bytes_tok = 2 * 28 * 2 * 128 * 2  # Qwen2.5-1.5B bf16
    blocks_cpu = int(a.cpu_gib * 2**30 / (kv_bytes_tok * 16))
    report = {"experiment": "kv_offload_multitier", "model": MODEL, "gpu_blocks": a.gpu_blocks,
              "hbm_kv_capacity_tokens": a.gpu_blocks * 16, "working_set_tokens": 32 * 2000,
              "cpu_tier_gib": a.cpu_gib, "variants": {},
              "labels": {"hbm_cpu": "measured", "hbm_cpu_remote": "measured, remote tier on localhost"}}
    try:
        plan = [("no_cache", [("no_cache", "--no-enable-prefix-caching", {})]),
                ("hbm_only", [("hbm_only", "--enable-prefix-caching", {})]),
                ("hbm_cpu", cpu_offload_candidates(a.cpu_gib, blocks_cpu))]
        lm_server = None
        if a.remote:
            if shutil.which("lmcache_server"):
                lm_server = subprocess.Popen(["lmcache_server", "localhost", "65432"], stdout=subprocess.DEVNULL,
                                             stderr=subprocess.STDOUT)
                time.sleep(3)
                env = {"LMCACHE_LOCAL_CPU": "True", "LMCACHE_MAX_LOCAL_CPU_SIZE": "2.0", "LMCACHE_CHUNK_SIZE": "256",
                       "LMCACHE_REMOTE_URL": "lm://localhost:65432", "LMCACHE_REMOTE_SERDE": "naive"}
                plan.append(("hbm_cpu_remote", [("lmcache_cpu_remote", "--kv-transfer-config '" + json.dumps(
                    {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}) + "'", env)]))
            else:
                report["variants"]["hbm_cpu_remote"] = {"error": "lmcache_server not found (pip install lmcache)"}
        for variant, candidates in plan:
            attempts = []
            for impl, extra, env in candidates:
                info = start(variant, extra, env, a.gpu_blocks)
                attempts.append({"impl": impl, "healthy": info.get("healthy"), "args": extra,
                                 "log_tail": info.get("log_tail")})
                if info.get("healthy"):
                    d = out / variant
                    agg = run_and_save(URL, KEYS, w, str(d), a.repeats, f"{variant}/{impl}", scrape_metrics_urls=[URL])
                    report["variants"][variant] = {"impl": impl, "aggregate": agg["aggregate"],
                                                   "server_rss_gib_after": server_rss_gib(), "attempts": attempts}
                    break
            else:
                report["variants"][variant] = {"error": "no implementation started", "attempts": attempts}
            save_json(out / "report.json", report)
        if lm_server:
            lm_server.terminate()
        base = out / "hbm_only"
        for v in ("no_cache", "hbm_cpu", "hbm_cpu_remote"):
            if (out / v).exists() and base.exists():
                report["variants"][v]["vs_hbm_only"] = {
                    "ttft_p50": compare(rep_values(base, "ttft_s", "p50"), rep_values(out / v, "ttft_s", "p50")),
                    "ttft_p95": compare(rep_values(base, "ttft_s", "p95"), rep_values(out / v, "ttft_s", "p95")),
                    "throughput": compare(rep_values(base, "output_tok_per_s"), rep_values(out / v, "output_tok_per_s"),
                                          lower_is_better=False)}
        report["cpu_tier_working"] = "aggregate" in report["variants"].get("hbm_cpu", {})
        save_json(out / "report.json", report)
        print(f"[kv_offload] cpu tier working={report['cpu_tier_working']} -> {out}/report.json")
    finally:
        serverctl.stopall()


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
