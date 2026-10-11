"""Why disaggregate? Measure prefill->decode interference on a colocated vLLM server.

Probe  : 8 decode-heavy streams (64-token prompt, 512-token output), measured TPOT.
Noise  : concurrent long-prefill requests (~3000-token prompts, 32-token outputs).
Sweep  : vLLM chunked-prefill budget --max-num-batched-tokens in {512, 2048, 8192}.

Reports TPOT(probe alone) vs TPOT(probe + prefill noise) per budget, and the noise
requests' TTFT (the price of smaller chunks). delta_tpot feeds the break-even
analysis in disaggregation/nixl-benchmarks/kv_transfer_bench.py.

python disaggregation/prefill/exp_interference.py --repeats 3
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.analysis.stats import compare, mean_ci  # noqa: E402
from benchmarks.runners.common import KEYS, MODEL, result_dir, save_json, start_vllm  # noqa: E402
from benchmarks.runners.loadgen import Workload, run_load, summarize  # noqa: E402
from scripts import serverctl  # noqa: E402

URL = "http://127.0.0.1:8001"


def probe_wl(seed):
    return Workload(name="probe", model=MODEL, num_requests=32, warmup_requests=4, concurrency=8,
                    input_tokens=64, output_tokens=512, seed=seed)


def noise_wl(seed, n: int = 8):
    return Workload(name="noise", model=MODEL, num_requests=n, warmup_requests=0, concurrency=4,
                    input_tokens=3000, input_tokens_jitter=0.05, output_tokens=32, seed=seed + 7)


async def with_noise(seed):
    """Long-prefill noise runs in back-to-back batches until the probe finishes, so every probe token is
    generated while prefills compete for the GPU. (v1 sent a fixed 48 noise requests that finished in the
    first ~20% of the probe, which is why it measured no interference - kept as a documented design bug.)"""
    import time

    p = probe_wl(seed)
    stop = asyncio.Event()
    noise_results: list[dict] = []

    async def noise_loop():
        i = 0
        while not stop.is_set():
            res, _ = await run_load(URL, KEYS, noise_wl(seed * 100 + i))
            noise_results.extend(res)
            i += 1

    t = time.perf_counter()
    task = asyncio.create_task(noise_loop())
    pr, _ = await run_load(URL, KEYS, p)
    stop.set()
    await task
    wall = time.perf_counter() - t
    return summarize(pr, wall, p), summarize(noise_results, wall, noise_wl(seed, len(noise_results)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--budgets", default="512,2048,8192")
    a = ap.parse_args()
    out = result_dir("interference")
    report = {"experiment": "prefill_decode_interference", "model": MODEL, "budgets": {}}
    try:
        for budget in [int(x) for x in a.budgets.split(",")]:
            serverctl.stopall()
            start_vllm("vllm-a", 8001, max_len=4096, extra=f"--max-num-batched-tokens {budget}")
            alone, noisy, noise_ttft, itl99_alone, itl99_noisy, n_noise = [], [], [], [], [], []
            for rep in range(a.repeats):
                _, s_alone = asyncio.run(run_load(URL, KEYS, probe_wl(rep)))
                s_probe, s_noise = asyncio.run(with_noise(rep))
                alone.append(s_alone["tpot_s"]["p50"])
                noisy.append(s_probe["tpot_s"]["p50"])
                noise_ttft.append(s_noise["ttft_s"].get("p50"))
                itl99_alone.append(s_alone["itl_s"].get("p99"))
                itl99_noisy.append(s_probe["itl_s"].get("p99"))
                n_noise.append(s_noise["n_requests"])
                save_json(out / f"budget_{budget}" / f"rep{rep}.json",
                          {"alone": s_alone, "with_noise_probe": s_probe, "noise": s_noise})
            report["budgets"][budget] = {
                "tpot_p50_alone": mean_ci(alone), "tpot_p50_with_prefill_noise": mean_ci(noisy),
                "interference": compare(alone, noisy),
                "noise_ttft_p50": mean_ci([x for x in noise_ttft if x is not None]),
                "itl_p99_alone": mean_ci([x for x in itl99_alone if x is not None]),
                "itl_p99_with_prefill_noise": mean_ci([x for x in itl99_noisy if x is not None]),
                "interference_itl_p99": compare([x for x in itl99_alone if x is not None],
                                                [x for x in itl99_noisy if x is not None]),
                "noise_requests_per_rep": n_noise,
            }
            b = report["budgets"][budget]
            print(f"[interference] budget={budget} tpot p50 alone={b['tpot_p50_alone']['mean']*1e3:.2f}ms "
                  f"noisy={b['tpot_p50_with_prefill_noise']['mean']*1e3:.2f}ms | itl p99 alone="
                  f"{b['itl_p99_alone']['mean']*1e3:.1f}ms noisy={b['itl_p99_with_prefill_noise']['mean']*1e3:.1f}ms "
                  f"| noise reqs/rep={n_noise}")
        # headline delta for break-even analysis: largest budget = default-like behaviour
        top = report["budgets"][max(report["budgets"])]
        report["delta_tpot_p50_s"] = top["tpot_p50_with_prefill_noise"]["mean"] - top["tpot_p50_alone"]["mean"]
        save_json(out / "report.json", report)
        print(f"[interference] wrote {out}/report.json")
    finally:
        serverctl.stopall()


if __name__ == "__main__":
    main()
