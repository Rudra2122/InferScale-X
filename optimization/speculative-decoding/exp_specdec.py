"""Pillar 3: speculative decoding vs standard decoding.

Part A (HF, controlled, batch=1):
  target Qwen2.5-7B-Instruct (bf16, ~15 GB), draft Qwen2.5-0.5B-Instruct (same tokenizer).
  Greedy decoding with and without `assistant_model`. Measures TPOT, tokens/s,
  target forward passes (=> mean tokens accepted per target step), and checks the
  outputs are IDENTICAL (greedy speculative decoding must be lossless).

Part B (vLLM serving, optional --vllm):
  same target with --speculative-config (draft model, and n-gram) at concurrency 1 and 16.
  Expectation to test, not assume: speculation helps at low concurrency and can
  hurt at high concurrency when the GPU is already compute-bound.

python optimization/speculative-decoding/exp_specdec.py --n 30
python optimization/speculative-decoding/exp_specdec.py --vllm --skip-hf
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.analysis.arith_eval import eval_set, messages  # noqa: E402
from benchmarks.analysis.stats import compare, mean_ci  # noqa: E402
from benchmarks.runners.common import KEYS, rep_values, result_dir, save_json, start_vllm  # noqa: E402

# Qwen2.5-0.5B/1.5B/3B share a 151,936-token vocab; 7B+ use 152,064. transformers refuses a draft whose
# vocab differs ("different tokenizers"), which is exactly what v1 (7B target) hit. 3B keeps a real draft/target
# size ratio (~6x) with an identical vocab.
TARGET = "Qwen/Qwen2.5-3B-Instruct"
DRAFT = "Qwen/Qwen2.5-0.5B-Instruct"


def part_hf(n: int, max_new: int, out: Path, target: str, draft: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(target)
    tgt = AutoModelForCausalLM.from_pretrained(target, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    drf = AutoModelForCausalLM.from_pretrained(draft, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    dtok = AutoTokenizer.from_pretrained(draft)
    calls = {"n": 0}
    orig_forward = tgt.forward

    def counting_forward(*a, **k):
        calls["n"] += 1
        return orig_forward(*a, **k)

    tgt.forward = counting_forward
    prompts = [tok.apply_chat_template(messages(p), tokenize=False, add_generation_prompt=True)
               for p in eval_set(n)]
    rows = []
    for i, prompt in enumerate(prompts):
        ids = tok(prompt, return_tensors="pt").to("cuda")
        rec = {"i": i}
        for mode in ("standard", "speculative"):
            kw = dict(max_new_tokens=max_new, do_sample=False, min_new_tokens=max_new)  # fixed length = fair TPOT
            if mode == "speculative":
                kw["assistant_model"] = drf
                if getattr(tgt.config, "vocab_size", None) != getattr(drf.config, "vocab_size", None):
                    kw.update(tokenizer=tok, assistant_tokenizer=dtok)  # universal assisted decoding fallback
            torch.cuda.synchronize()
            calls["n"] = 0
            t0 = time.perf_counter()
            with torch.inference_mode():
                o = tgt.generate(**ids, **kw)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            new = o[0, ids["input_ids"].shape[1]:]
            rec[mode] = {"s": dt, "new_tokens": int(new.numel()), "target_forwards": calls["n"],
                         "tok_per_s": new.numel() / dt, "text": tok.decode(new, skip_special_tokens=True)}
        rec["identical_output"] = rec["standard"]["text"] == rec["speculative"]["text"]
        rec["tokens_per_target_forward"] = rec["speculative"]["new_tokens"] / max(1, rec["speculative"]["target_forwards"])
        rows.append(rec)
        if i < 3 or i % 10 == 0:
            print(f"[spec-hf] {i}: std {rec['standard']['tok_per_s']:.1f} tok/s, spec "
                  f"{rec['speculative']['tok_per_s']:.1f} tok/s, tokens/target-fwd "
                  f"{rec['tokens_per_target_forward']:.2f}, identical={rec['identical_output']}")
    rows_warm = rows[1:] if len(rows) > 1 else rows  # first prompt includes CUDA/graph warmup
    res = {
        "target": target, "draft": draft, "max_new_tokens": max_new, "n_prompts": len(rows),
        "tok_per_s_standard": mean_ci([r["standard"]["tok_per_s"] for r in rows_warm]),
        "tok_per_s_speculative": mean_ci([r["speculative"]["tok_per_s"] for r in rows_warm]),
        "speedup": compare([r["standard"]["s"] for r in rows_warm], [r["speculative"]["s"] for r in rows_warm]),
        "mean_tokens_per_target_forward": statistics.mean(r["tokens_per_target_forward"] for r in rows_warm),
        "identical_output_rate": sum(r["identical_output"] for r in rows) / len(rows),
        "note": "identical_output_rate < 1.0 can come from bf16 numerics differing between batch shapes; inspect rows.",
    }
    with open(out / "hf_rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tgt.forward = orig_forward  # drop the counting wrapper: its closure kept the model alive
    del tgt, drf, orig_forward, counting_forward
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return res


def part_vllm(out: Path, target: str, draft: str, repeats: int) -> dict:
    from benchmarks.runners.loadgen import Workload, run_and_save
    from scripts import serverctl

    configs = {
        "standard": "",
        "draft_model_k4": f"--speculative-config '{json.dumps({'model': draft, 'num_speculative_tokens': 4})}'",
        "ngram_k4": "--speculative-config '" + json.dumps({"method": "ngram", "num_speculative_tokens": 4,
                                                           "prompt_lookup_max": 4}) + "'",
    }
    res = {}
    url = "http://127.0.0.1:8001"
    import torch

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        res["gpu_free_before_vllm_gib"] = round(free / 2**30, 2)
        if free / total < 0.85:
            print(f"[spec] WARNING only {free / 2**30:.1f} GiB free before vLLM; run with --skip-hf in a separate process")
    for name, extra in configs.items():
        serverctl.stopall()
        try:
            start_vllm("vllm-a", 8001, model=target, gpu_mem=0.9, extra=extra)
        except RuntimeError as e:
            res[name] = {"error": str(e), "log_tail": serverctl.tail("vllm-a", 30)}
            continue
        res[name] = {}
        for conc in (1, 16):
            w = Workload(name=f"c{conc}", model=target, num_requests=24 if conc == 1 else 96, warmup_requests=2,
                         concurrency=conc, input_tokens=256, output_tokens=256)
            res[name][f"c{conc}"] = run_and_save(url, KEYS, w, str(out / "vllm" / name / f"c{conc}"), repeats,
                                                 f"{name}/c{conc}", scrape_metrics_urls=[url])["aggregate"]
    serverctl.stopall()
    for name in configs:
        if name == "standard" or "error" in res.get(name, {}):
            continue
        res[name]["vs_standard"] = {c: compare(rep_values(out / "vllm" / "standard" / c, "tpot_s", "p50"),
                                               rep_values(out / "vllm" / name / c, "tpot_s", "p50"))
                                    for c in ("c1", "c16")}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--draft", default=DRAFT)
    ap.add_argument("--skip-hf", action="store_true")
    ap.add_argument("--vllm", action="store_true")
    ap.add_argument("--repeats", type=int, default=2)
    a = ap.parse_args()
    out = result_dir("speculative_decoding")
    report = {"experiment": "speculative_decoding"}
    if not a.skip_hf:
        try:
            report["hf"] = part_hf(a.n, a.max_new, out, a.target, a.draft)
        except Exception as e:  # record and continue to the vLLM part
            import traceback
            report["hf"] = {"error": f"{type(e).__name__}: {e}"[:500], "traceback": traceback.format_exc()[-2000:]}
            print(f"[spec] HF part failed: {report['hf']['error']}")
        save_json(out / "report.json", report)
    if a.vllm:
        report["vllm"] = part_vllm(out, a.target, a.draft, a.repeats)
    save_json(out / "report.json", report)
    print(f"[spec] wrote {out}/report.json")


if __name__ == "__main__":
    main()
