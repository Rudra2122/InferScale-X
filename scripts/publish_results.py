"""Build docs/RESULTS.md: the public results page (GitHub renders it), generated only from result files.

python scripts/publish_results.py      (run after evidence_report.py and make_plots.py)
"""
from __future__ import annotations

import glob
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "benchmarks" / "results"


def latest(pattern):
    fs = sorted(glob.glob(str(R / pattern)), key=lambda p: Path(p).stat().st_mtime)
    return Path(fs[-1]) if fs else None


def load(p):
    return json.loads(p.read_text()) if p else None


def fmt(m, scale=1.0, unit="", digits=1):
    if not isinstance(m, dict) or "mean" not in m:
        return "-"
    s = f"{m['mean'] * scale:.{digits}f}{unit}"
    if m.get("ci95") and m.get("n", 1) > 1:
        s += f" ± {(m['ci95'][1] - m['ci95'][0]) / 2 * scale:.{digits}f}"
    return s


def verdict(c):
    if not isinstance(c, dict) or "relative_change" not in c:
        return "-"
    return f"{c['relative_change']:+.1%} ({c['verdict']})"


def rel(p):
    return str(p.relative_to(ROOT)) if p else "-"


def main():
    out = ["# InferScale-X results", "",
           f"_Generated {time.strftime('%Y-%m-%d')} by scripts/publish_results.py. Every number links to its source file; "
           "± is the 95% CI across repeats. Labels: measured unless stated._", ""]
    env = load(latest("*/*/env.json") or latest("*/*/*/env.json"))
    if env:
        g = (env.get("gpus") or [{}])[0]
        pk = env.get("packages", {})
        out += [f"**Hardware/software:** {g.get('name', '?')} ({g.get('memory.total', '?')}), driver {g.get('driver_version', '?')}, "
                f"CUDA {(env.get('cuda') or {}).get('torch_cuda', '?')}, torch {pk.get('torch', '?')}, vLLM {pk.get('vllm', '?')}", ""]
    sections = []
    p = latest("baseline/*/report.json")
    if (b := load(p)):
        sections += ["## Gateway overhead (vLLM direct vs through gateway)", "",
                     "| | direct | gateway |", "|---|---|---|",
                     f"| output tok/s | {fmt(b['direct']['output_tok_per_s'], digits=0)} | {fmt(b['gateway']['output_tok_per_s'], digits=0)} |",
                     f"| TTFT p50 (ms) | {fmt(b['direct']['ttft_p50_s'], 1e3)} | {fmt(b['gateway']['ttft_p50_s'], 1e3)} |",
                     f"| TTFT change | | {verdict(b['gateway_overhead']['ttft_p50'])} |", "", f"source: `{rel(p)}`", ""]
    p = latest("routing/*/report.json")
    if (r := load(p)):
        sections += ["## Routing policies (2 replicas sharing one A100)", "", "| workload | policy | TTFT p50 (ms) | tok/s | vs round-robin (TTFT p50) |", "|---|---|---|---|---|"]
        for wl, runs in r["runs"].items():
            for pol, agg in runs.items():
                if pol == "vs_round_robin":
                    continue
                vs = runs.get("vs_round_robin", {}).get(pol, {}).get("ttft_p50")
                sections.append(f"| {wl} | {pol} | {fmt(agg['ttft_p50_s'], 1e3)} | {fmt(agg['output_tok_per_s'], digits=0)} | {verdict(vs)} |")
        sections += ["", f"source: `{rel(p)}`", ""]
    p = latest("kv_cache/*/report.json")
    if (k := load(p)):
        sections += ["## KV cache", ""]
        if "reuse" in k:
            sections.append(f"- Prefix caching on vs off, TTFT p50: {verdict(k['reuse'].get('ttft_p50'))}")
        if "isolation" in k:
            sections.append(f"- Cross-tenant cache leak without salt: {k['isolation'].get('leak_without_salt')}; "
                            f"isolation enforced with gateway salt: {k['isolation'].get('isolation_enforced')}")
        sections += ["", f"source: `{rel(p)}`", ""]
    p = latest("kv_offload/*/report.json")
    if (o := load(p)):
        sections += ["## Multi-tier KV (HBM capped below working set)", "", "| variant | impl | TTFT p50 (ms) | vs HBM-only |", "|---|---|---|---|"]
        for v, d in o["variants"].items():
            if "aggregate" in d:
                sections.append(f"| {v} | {d.get('impl')} | {fmt(d['aggregate']['ttft_p50_s'], 1e3)} | {verdict(d.get('vs_hbm_only', {}).get('ttft_p50'))} |")
            else:
                sections.append(f"| {v} | failed to start | - | - |")
        sections += ["", f"source: `{rel(p)}`", ""]
    for name, pat, key in (("Triton fused add+RMSNorm", "triton_kernel/*/report.json", "all_correct"),
                           ("Triton FlashAttention-style forward", "triton_attention/*/report.json", "all_correct")):
        p = latest(pat)
        if (t := load(p)):
            sections += [f"## {name}", "", f"- correctness: {sum(r['pass'] for r in t['correctness'])}/{len(t['correctness'])} cases pass",
                         f"- source: `{rel(p)}`", ""]
    p = latest("retrieval/*/report.json")
    if (rt := load(p)):
        sections += ["## Retrieval quality", "", "| retriever | hit@1 | recall@5 | MRR@10 | nDCG@5 |", "|---|---|---|---|---|"]
        for n, d in rt["retrievers"].items():
            a = d["aggregate"]
            sections.append(f"| {n} | {a['hit@1']:.2f} | {a['recall@5']:.2f} | {a['mrr@10']:.2f} | {a['ndcg@5']:.2f} |")
        sections += ["", f"source: `{rel(p)}`", ""]
    p = latest("k8s_kind/*/report.json")
    if (kk := load(p)):
        rec = kk.get("recovery", {})
        sections += ["## Kubernetes (kind, CPU mock backends — control-plane behaviour only)", "",
                     f"- scheduling across nodes: {kk.get('scheduling', {}).get('pass')}",
                     f"- mock pod recreated + Ready after kill: {rec.get('mock_pod_recreated_ready_s')} s; gateway detected in {rec.get('gateway_detect_s')} s",
                     f"- HPA scale-up detected after: {kk.get('hpa', {}).get('scale_up_detected_after_s')} s",
                     f"- NetworkPolicy enforced: {kk.get('netpol', {}).get('pass')}", "", f"source: `{rel(p)}`", ""]
    if not sections:
        sections = ["_No results yet. Run the notebooks / scripts, then re-run this script._", ""]
    plots = sorted(glob.glob(str(R / "plots" / "*.png")))
    if plots:
        sections += ["## Plots", ""] + [f"![{Path(x).stem}](../{Path(x).relative_to(ROOT)})" for x in plots]
    sections += ["", "See also: [READINESS](READINESS.md) · [JD coverage](jd-coverage.md) · [claims](CLAIMS.md) · "
                 "[negative results](negative-results.md)"]
    (ROOT / "docs" / "RESULTS.md").write_text("\n".join(out + sections) + "\n")
    print("wrote docs/RESULTS.md")


if __name__ == "__main__":
    main()
