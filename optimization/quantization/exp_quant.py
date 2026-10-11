"""Pillar 3: quantization vs BF16 on vLLM (A100 / Ampere).

Variants (all official Qwen checkpoints, auto-detected by vLLM):
  bf16       Qwen/Qwen2.5-1.5B-Instruct
  awq_int4   Qwen/Qwen2.5-1.5B-Instruct-AWQ
  gptq_int8  Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8
Optional:  fp8 weight-only via `--quantization fp8` on the BF16 model (on Ampere vLLM
           uses a weight-only Marlin path; A100 has no FP8 tensor cores, so this is
           expected to save memory but NOT speed up compute - a likely negative result).

Measures: weight memory (from vLLM's load log), throughput/latency at concurrency
1 and 32, and quality (arithmetic accuracy + exact agreement with BF16 greedy output).

python optimization/quantization/exp_quant.py --repeats 3 --variants bf16,awq_int4,gptq_int8,fp8
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx  # noqa: E402

from benchmarks.analysis.arith_eval import eval_set, messages, score  # noqa: E402
from benchmarks.analysis.stats import compare  # noqa: E402
from benchmarks.runners.common import KEYS, rep_values, result_dir, save_json, start_vllm  # noqa: E402
from benchmarks.runners.loadgen import Workload, run_and_save  # noqa: E402
from scripts import serverctl  # noqa: E402

URL = "http://127.0.0.1:8001"
VARIANTS = {
    "bf16": ("Qwen/Qwen2.5-1.5B-Instruct", ""),
    "awq_int4": ("Qwen/Qwen2.5-1.5B-Instruct-AWQ", ""),
    "gptq_int8": ("Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8", ""),
    "fp8": ("Qwen/Qwen2.5-1.5B-Instruct", "--quantization fp8"),
}


async def generate_all(model: str, problems: list[dict], max_tokens=256) -> list[str]:
    sem = asyncio.Semaphore(32)
    async with httpx.AsyncClient(timeout=300) as c:
        async def one(p):
            async with sem:
                r = await c.post(f"{URL}/v1/chat/completions", json={
                    "model": model, "messages": messages(p), "max_tokens": max_tokens, "temperature": 0.0, "seed": 0})
                return r.json()["choices"][0]["message"]["content"] if r.status_code == 200 else ""
        return await asyncio.gather(*(one(p) for p in problems))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="bf16,awq_int4,gptq_int8,fp8")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--num-requests", type=int, default=200)
    a = ap.parse_args()
    out = result_dir("quantization")
    problems = eval_set(a.eval_n)
    report = {"experiment": "quantization", "gpu_note": "A100 (sm80): no FP8 tensor cores", "variants": {}}
    outputs = {}
    try:
        for v in a.variants.split(","):
            model, extra = VARIANTS[v]
            serverctl.stopall()
            try:
                info = start_vllm("vllm-a", 8001, model=model, extra=extra)
            except RuntimeError as e:
                report["variants"][v] = {"model": model, "error": str(e), "log_tail": serverctl.tail("vllm-a", 30)}
                save_json(out / "report.json", report)
                continue
            outs = asyncio.run(generate_all(model, problems))
            outputs[v] = outs
            entry = {"model": model, "extra_args": extra, "model_load_gib": info.get("model_load_gib"),
                     "quality": score(outs, problems)}
            for conc in (1, 32):
                w = Workload(name=f"c{conc}", model=model, num_requests=a.num_requests if conc > 1 else 40,
                             warmup_requests=4, concurrency=conc, input_tokens=512, output_tokens=128)
                entry[f"perf_c{conc}"] = run_and_save(URL, KEYS, w, str(out / v / f"c{conc}"), a.repeats,
                                                      f"{v}/c{conc}")["aggregate"]
            report["variants"][v] = entry
            save_json(out / "report.json", report)
        if "bf16" in outputs:
            for v, outs in outputs.items():
                same = sum(1 for x, y in zip(outs, outputs["bf16"]) if x == y)
                report["variants"][v]["exact_agreement_with_bf16"] = same / len(outs)
            for v in outputs:
                if v == "bf16":
                    continue
                report["variants"][v]["vs_bf16"] = {
                    c: {"throughput": compare(rep_values(out / "bf16" / c, "output_tok_per_s"),
                                              rep_values(out / v / c, "output_tok_per_s"), lower_is_better=False),
                        "tpot_p50": compare(rep_values(out / "bf16" / c, "tpot_s", "p50"),
                                            rep_values(out / v / c, "tpot_s", "p50"))}
                    for c in ("c1", "c32")}
        save_json(out / "report.json", report)
        print(f"[quant] wrote {out}/report.json")
    finally:
        serverctl.stopall()


if __name__ == "__main__":
    main()
