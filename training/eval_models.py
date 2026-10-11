"""Held-out evaluation of base vs customised models + inference-cost implications.

For each model (base, or base+LoRA adapter, optionally merged on top of a prior adapter)
runs batched greedy generation on training/data/eval.jsonl and reports accuracy,
format compliance, mean output tokens and generation throughput. Longer outputs after
RL mean higher serving cost per query; the report makes that explicit.

python training/eval_models.py --adapters base,training/runs/<sft>/adapter,training/runs/<rlvr>/adapter
python training/eval_models.py --adapters training/runs/<dpo-from-sft>/adapter --merge-first training/runs/<sft>/adapter
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.analysis.arith_eval import extract_answer, score  # noqa: E402
from training.common import BASE_MODEL, load_jsonl  # noqa: E402


def load(model_name: str, adapter: str | None, merge_first: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    if merge_first:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, merge_first).merge_and_unload()
    if adapter and adapter != "base":
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return tok, model.eval()


def evaluate(tok, model, rows, batch: int = 64, max_new: int = 160) -> dict:
    import torch

    preds, out_tokens = [], []
    t0 = time.perf_counter()
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        texts = [tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True) for r in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        for row in new:
            ids = row[row != tok.pad_token_id]
            out_tokens.append(int(ids.numel()))
            preds.append(tok.decode(ids, skip_special_tokens=True))
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    problems = [{"answer": r["answer"]} for r in rows]
    s = score(preds, problems)
    s.update({"mean_output_tokens": sum(out_tokens) / len(out_tokens), "gen_wall_s": wall,
              "gen_tok_per_s": sum(out_tokens) / wall,
              "samples": [{"id": r["id"], "answer": r["answer"], "pred": extract_answer(p), "text": p[:300]}
                          for r, p in list(zip(rows, preds))[:8]]})
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--adapters", default="base")
    ap.add_argument("--merge-first", default=None, help="adapter merged before evaluating each listed adapter")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default="training/runs/eval_report.json")
    a = ap.parse_args()
    import torch

    rows = load_jsonl("eval.jsonl")[:a.n]
    report = {"base_model": a.model, "n_eval": len(rows), "results": {}}
    for ad in a.adapters.split(","):
        tok, model = load(a.model, ad, a.merge_first if ad != "base" else None)
        r = evaluate(tok, model, rows)
        meta = Path(ad).parent / "run_meta.json"
        r["run_meta"] = json.loads(meta.read_text()).get("config") if meta.exists() else None
        report["results"][ad] = r
        print(f"[eval] {ad}: acc={r['accuracy']:.3f} format={r['format_rate']:.3f} "
              f"out_tok={r['mean_output_tokens']:.1f}")
        del model
        torch.cuda.empty_cache()
    base = report["results"].get("base")
    if base:
        for k, r in report["results"].items():
            r["delta_accuracy_vs_base"] = r["accuracy"] - base["accuracy"]
            r["relative_serving_tokens_vs_base"] = r["mean_output_tokens"] / base["mean_output_tokens"]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(f"[eval] wrote {a.out}")


if __name__ == "__main__":
    main()
