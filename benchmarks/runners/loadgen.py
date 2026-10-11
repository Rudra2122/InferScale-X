"""Async load generator for OpenAI-compatible endpoints (gateway, vLLM, SGLang, mock).

Measures per request: TTFT (first non-empty content chunk), inter-token latencies,
TPOT, end-to-end latency, output tokens (from `usage`), status, backend.
Writes raw JSONL + summary.json + env.json so every number is traceable.

Examples
--------
python -m benchmarks.runners.loadgen --url http://127.0.0.1:9000 --api-key sk-dev-acme \
    --workload benchmarks/workloads/chat_short.yaml --out benchmarks/results/baseline --repeats 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import yaml

from benchmarks.analysis.stats import dist, jain_fairness, mean_ci
from benchmarks.runners.env_capture import capture
from benchmarks.runners.power import GpuSampler

VOCAB = ("system data model request token cache latency throughput memory kernel tensor batch decode "
         "prefill queue router tenant policy metric trace cluster node replica stream window budget "
         "signal vector matrix compute network storage service gateway engine worker region shard").split()


@dataclass
class Workload:
    name: str = "default"
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    num_requests: int = 200
    warmup_requests: int = 10
    concurrency: int | None = 16  # closed loop; set None and request_rate for open loop
    request_rate: float | None = None  # req/s Poisson (open loop)
    input_tokens: int = 256
    input_tokens_jitter: float = 0.25  # +- fraction, uniform
    output_tokens: int = 128
    ignore_eos: bool = True  # vLLM/SGLang extension: force exact output length for fair TPOT
    prefix_reuse: float = 0.0  # fraction of requests that start with a shared prefix
    shared_prefix_tokens: int = 0
    num_shared_prefixes: int = 1
    tenants: dict[str, float] = field(default_factory=lambda: {"default": 1.0})  # tenant -> weight
    temperature: float = 0.0
    seed: int = 0
    timeout_s: float = 600.0

    @classmethod
    def from_yaml(cls, path: str, **overrides) -> "Workload":
        d = yaml.safe_load(Path(path).read_text()) or {}
        d.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**d)


def words(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(VOCAB) for _ in range(max(1, n)))


def build_prompts(w: Workload, n: int, rng: random.Random) -> list[tuple[str, str]]:
    prefixes = [f"[doc {i}] " + words(random.Random(10_000 + i + w.seed), w.shared_prefix_tokens)
                for i in range(max(1, w.num_shared_prefixes))]
    tenants, weights = list(w.tenants.keys()), list(w.tenants.values())
    out = []
    for i in range(n):
        n_in = int(w.input_tokens * (1 + rng.uniform(-w.input_tokens_jitter, w.input_tokens_jitter)))
        if w.shared_prefix_tokens and rng.random() < w.prefix_reuse:
            p = rng.choice(prefixes)
            body = p + " | question " + str(i) + ": " + words(rng, max(1, n_in - w.shared_prefix_tokens))
        else:
            body = f"unique {uuid.uuid4().hex[:8]} " + words(rng, n_in)
        out.append((rng.choices(tenants, weights)[0], body))
    return out


async def one_request(client: httpx.AsyncClient, url: str, keys: dict[str, str], w: Workload,
                      tenant: str, prompt: str) -> dict:
    payload = {
        "model": w.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": w.output_tokens,
        "temperature": w.temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if w.ignore_eos:
        payload["ignore_eos"] = True
    rec = {"tenant": tenant, "prompt_chars": len(prompt), "ok": False, "status": None, "error": None,
           "ttft_s": None, "e2e_s": None, "itl_s": [], "completion_tokens": None, "prompt_tokens": None,
           "cached_tokens": None, "backend": None}
    headers = {"Authorization": f"Bearer {keys.get(tenant, keys.get('default', 'none'))}"}
    t0 = time.perf_counter()
    rec["t_start"] = time.time()
    last = None
    chunks = 0
    try:
        async with client.stream("POST", f"{url}/v1/chat/completions", json=payload, headers=headers,
                                 timeout=w.timeout_s) as r:
            rec["status"] = r.status_code
            rec["backend"] = r.headers.get("x-backend")
            if r.status_code != 200:
                rec["error"] = (await r.aread()).decode(errors="replace")[:300]
                rec["e2e_s"] = time.perf_counter() - t0
                return rec
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data.strip() == "[DONE]":
                    break
                ev = json.loads(data)
                if "error" in ev:
                    rec["error"] = str(ev["error"])[:300]
                    break
                if ev.get("usage"):
                    u = ev["usage"]
                    rec["completion_tokens"] = u.get("completion_tokens")
                    rec["prompt_tokens"] = u.get("prompt_tokens")
                    rec["cached_tokens"] = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
                for ch in ev.get("choices") or []:
                    piece = (ch.get("delta") or {}).get("content") or ch.get("text")
                    if piece:
                        now = time.perf_counter()
                        if rec["ttft_s"] is None:
                            rec["ttft_s"] = now - t0
                        elif last is not None:
                            rec["itl_s"].append(now - last)
                        last = now
                        chunks += 1
        rec["e2e_s"] = time.perf_counter() - t0
        rec["ok"] = rec["error"] is None and rec["ttft_s"] is not None
        if rec["completion_tokens"] is None:
            rec["completion_tokens"] = chunks  # fallback; chunks ~ tokens for vLLM
        if rec["ok"] and rec["completion_tokens"] and rec["completion_tokens"] > 1:
            rec["tpot_s"] = (rec["e2e_s"] - rec["ttft_s"]) / (rec["completion_tokens"] - 1)
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
        rec["e2e_s"] = time.perf_counter() - t0
    return rec


async def run_load(url: str, keys: dict[str, str], w: Workload, on_progress=None) -> tuple[list[dict], dict]:
    rng = random.Random(w.seed)
    prompts = build_prompts(w, w.num_requests + w.warmup_requests, rng)
    limits = httpx.Limits(max_connections=2048, max_keepalive_connections=512)
    async with httpx.AsyncClient(limits=limits, timeout=w.timeout_s) as client:
        # warmup (sequential-ish, not recorded)
        if w.warmup_requests:
            await asyncio.gather(*(one_request(client, url, keys, w, t, p) for t, p in prompts[:w.warmup_requests]))
        todo = prompts[w.warmup_requests:]
        results: list[dict] = []
        t_begin = time.perf_counter()
        if w.request_rate:  # open loop, Poisson arrivals
            arr_rng = random.Random(w.seed + 1)
            tasks = []
            for t, p in todo:
                tasks.append(asyncio.create_task(one_request(client, url, keys, w, t, p)))
                await asyncio.sleep(arr_rng.expovariate(w.request_rate))
            results = list(await asyncio.gather(*tasks))
        else:  # closed loop with fixed concurrency
            q: asyncio.Queue = asyncio.Queue()
            for item in todo:
                q.put_nowait(item)

            async def worker():
                while True:
                    try:
                        t, p = q.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    results.append(await one_request(client, url, keys, w, t, p))
                    if on_progress:
                        on_progress(len(results))

            await asyncio.gather(*(worker() for _ in range(w.concurrency or 1)))
        wall = time.perf_counter() - t_begin
    return results, summarize(results, wall, w)


def summarize(results: list[dict], wall_s: float, w: Workload) -> dict:
    ok = [r for r in results if r["ok"]]
    out_tok = sum(r["completion_tokens"] or 0 for r in ok)
    itl = [x for r in ok for x in r["itl_s"]]
    per_tenant = {}
    for t, weight in w.tenants.items():
        rs = [r for r in ok if r["tenant"] == t]
        toks = sum(r["completion_tokens"] or 0 for r in rs)
        per_tenant[t] = {"ok": len(rs), "output_tokens": toks, "tok_per_s": toks / wall_s if wall_s else 0,
                         "ttft_s": dist(r["ttft_s"] for r in rs), "weight": weight}
    norm = [v["tok_per_s"] / v["weight"] for v in per_tenant.values() if v["weight"]]
    cached = [r["cached_tokens"] for r in ok if r.get("cached_tokens") is not None]
    prompt = [r["prompt_tokens"] for r in ok if r.get("prompt_tokens")]
    by_backend: dict[str, int] = {}
    for r in results:
        by_backend[str(r.get("backend"))] = by_backend.get(str(r.get("backend")), 0) + 1
    statuses: dict[str, int] = {}
    for r in results:
        statuses[str(r["status"])] = statuses.get(str(r["status"]), 0) + 1
    return {
        "workload": w.name,
        "n_requests": len(results), "n_ok": len(ok), "error_rate": 1 - len(ok) / len(results) if results else None,
        "statuses": statuses, "requests_by_backend": by_backend,
        "wall_s": wall_s, "req_per_s": len(ok) / wall_s if wall_s else None,
        "output_tok_per_s": out_tok / wall_s if wall_s else None,
        "ttft_s": dist(r["ttft_s"] for r in ok),
        "tpot_s": dist(r.get("tpot_s") for r in ok if r.get("tpot_s") is not None),
        "itl_s": dist(itl),
        "e2e_s": dist(r["e2e_s"] for r in ok),
        "mean_prompt_tokens": sum(prompt) / len(prompt) if prompt else None,
        "cached_token_fraction": (sum(cached) / sum(prompt)) if cached and prompt else None,
        "per_tenant": per_tenant,
        "jain_fairness_weighted": jain_fairness(norm) if len(norm) > 1 else None,
        "errors_sample": [r["error"] for r in results if r["error"]][:5],
    }


def aggregate(summaries: list[dict]) -> dict:
    keys = {
        "output_tok_per_s": lambda s: s["output_tok_per_s"],
        "req_per_s": lambda s: s["req_per_s"],
        "ttft_p50_s": lambda s: s["ttft_s"].get("p50"),
        "ttft_p95_s": lambda s: s["ttft_s"].get("p95"),
        "ttft_p99_s": lambda s: s["ttft_s"].get("p99"),
        "tpot_p50_s": lambda s: s["tpot_s"].get("p50"),
        "tpot_p95_s": lambda s: s["tpot_s"].get("p95"),
        "e2e_p50_s": lambda s: s["e2e_s"].get("p50"),
        "e2e_p99_s": lambda s: s["e2e_s"].get("p99"),
        "error_rate": lambda s: s["error_rate"],
        "cached_token_fraction": lambda s: s.get("cached_token_fraction"),
        "jain_fairness_weighted": lambda s: s.get("jain_fairness_weighted"),
        "joules_per_output_token": lambda s: (s.get("gpu") or {}).get("joules_per_output_token"),
        "avg_gpu_util_pct": lambda s: (s.get("gpu") or {}).get("avg_util_pct"),
    }
    agg = {}
    for k, f in keys.items():
        vals = []
        for s in summaries:
            try:
                v = f(s)
            except Exception:
                v = None
            if v is not None:
                vals.append(v)
        agg[k] = mean_ci(vals)
    return agg


def run_and_save(url: str, keys: dict[str, str], w: Workload, out_dir: str, repeats: int = 1,
                 label: str | None = None, extra_env: dict | None = None,
                 scrape_metrics_urls: list[str] | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "workload.yaml").write_text(yaml.safe_dump(asdict(w), sort_keys=False))
    (out / "env.json").write_text(json.dumps(capture(extra_env), indent=2))
    summaries = []
    for rep in range(repeats):
        w_rep = Workload(**{**asdict(w), "seed": w.seed + rep * 1000})
        before = scrape_all(scrape_metrics_urls)
        with GpuSampler() as gpu:
            results, summary = asyncio.run(run_load(url, keys, w_rep))
        after = scrape_all(scrape_metrics_urls)
        summary["backend_metric_deltas"] = metric_deltas(before, after)
        out_tokens = sum(r["completion_tokens"] or 0 for r in results if r["ok"])
        summary["gpu"] = gpu.summary(out_tokens)  # includes warmup time; see power.py caveats
        summary["repeat"] = rep
        rep_dir = out / f"rep{rep}"
        rep_dir.mkdir(exist_ok=True)
        with open(rep_dir / "requests.jsonl", "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        (rep_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        summaries.append(summary)
        print(f"[{label or w.name}] rep {rep}: ok={summary['n_ok']}/{summary['n_requests']} "
              f"tok/s={summary['output_tok_per_s']:.1f} ttft_p50={summary['ttft_s'].get('p50')} "
              f"tpot_p50={summary['tpot_s'].get('p50')}")
    agg = {"label": label or w.name, "url": url, "repeats": repeats, "aggregate": aggregate(summaries)}
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2))
    return agg


# ------------- backend metric scraping (vLLM exposes Prometheus text) -------------
WANTED_PREFIXES = ("vllm:prefix_cache_hits", "vllm:prefix_cache_queries", "vllm:gpu_prefix_cache_hits",
                   "vllm:gpu_prefix_cache_queries", "vllm:request_queue_time_seconds_sum",
                   "vllm:request_queue_time_seconds_count", "vllm:num_preemptions", "vllm:prompt_tokens_total",
                   "vllm:generation_tokens_total", "vllm:spec_decode")


def scrape(url: str) -> dict[str, float]:
    try:
        text = httpx.get(f"{url}/metrics", timeout=5).text
    except Exception:
        return {}
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name = line.split("{")[0].split(" ")[0]
        if any(name.startswith(p) for p in WANTED_PREFIXES):
            try:
                out[name] = out.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
            except ValueError:
                pass
    return out


def scrape_all(urls: list[str] | None) -> dict[str, dict[str, float]]:
    return {u: scrape(u) for u in (urls or [])}


def metric_deltas(before: dict, after: dict) -> dict:
    out = {}
    for u, a in after.items():
        b = before.get(u, {})
        out[u] = {k: a[k] - b.get(k, 0.0) for k in a}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--api-key", default="none", help="key for tenant 'default' (or all tenants)")
    ap.add_argument("--tenant-keys", default=None, help="tenant=key,tenant2=key2")
    ap.add_argument("--workload", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--model", default=None)
    ap.add_argument("--num-requests", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--scrape", default="", help="comma-separated backend URLs to scrape /metrics deltas")
    a = ap.parse_args()
    w = Workload.from_yaml(a.workload, model=a.model, num_requests=a.num_requests, concurrency=a.concurrency)
    keys = {"default": a.api_key}
    for t in w.tenants:
        keys.setdefault(t, a.api_key)
    if a.tenant_keys:
        for kv in a.tenant_keys.split(","):
            t, k = kv.split("=", 1)
            keys[t] = k
    agg = run_and_save(a.url, keys, w, a.out, a.repeats, a.label,
                       scrape_metrics_urls=[u for u in a.scrape.split(",") if u])
    print(json.dumps(agg["aggregate"], indent=2))


if __name__ == "__main__":
    main()
