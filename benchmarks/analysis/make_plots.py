"""Render plots from the latest result files into benchmarks/results/plots/.

Each plot is drawn only if its evidence exists; nothing is synthesised.
python benchmarks/analysis/make_plots.py
"""
from __future__ import annotations

import glob
import json
import sys
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "benchmarks" / "results"
OUT = R / "plots"


def latest(pattern):
    fs = sorted(glob.glob(str(R / pattern)), key=lambda p: Path(p).stat().st_mtime)
    return Path(fs[-1]) if fs else None


def ci_err(m):
    if not m or "ci95" not in m:
        return 0
    return (m["ci95"][1] - m["ci95"][0]) / 2


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=140)
    plt.close(fig)
    print(f"[plots] {name}")


def plot_baseline():
    p = latest("baseline/*/report.json")
    if not p:
        return
    d = p.parent
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for label in ("direct", "gateway"):
        ttft = [json.loads(line)["ttft_s"] for line in open(d / label / "rep0" / "requests.jsonl")
                if json.loads(line)["ok"]]
        ax[0].hist([t * 1e3 for t in ttft], bins=40, alpha=0.6, label=label)
    ax[0].set_xlabel("TTFT (ms)"), ax[0].set_ylabel("requests"), ax[0].legend(), ax[0].set_title("TTFT distribution (rep0)")
    rep = json.loads(p.read_text())
    names = ["direct", "gateway"]
    vals = [rep[n]["output_tok_per_s"]["mean"] for n in names]
    ax[1].bar(names, vals, yerr=[ci_err(rep[n]["output_tok_per_s"]) for n in names], capsize=4)
    ax[1].set_ylabel("output tok/s (mean ± 95% CI)"), ax[1].set_title("Throughput")
    save(fig, "baseline.png")


def plot_routing():
    p = latest("routing/*/report.json")
    if not p:
        return
    rep = json.loads(p.read_text())
    wls = list(rep["runs"])
    fig, axes = plt.subplots(1, len(wls), figsize=(5 * len(wls), 3.6), squeeze=False)
    for ax, wl in zip(axes[0], wls):
        pols = [k for k in rep["runs"][wl] if k != "vs_round_robin"]
        m = [rep["runs"][wl][k]["ttft_p50_s"] for k in pols]
        ax.bar(pols, [x["mean"] * 1e3 for x in m], yerr=[ci_err(x) * 1e3 for x in m], capsize=4)
        ax.set_title(f"{wl}: p50 TTFT"), ax.set_ylabel("ms"), ax.tick_params(axis="x", rotation=15)
    fig.suptitle("Routing policies (2 replicas, one shared A100)", fontsize=10)
    save(fig, "routing.png")


def plot_kv_cache():
    p = latest("kv_cache/*/report.json")
    if not p:
        return
    rep = json.loads(p.read_text())
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    if "reuse" in rep:
        m = [rep["reuse"]["cache_off"]["ttft_p50_s"], rep["reuse"]["cache_on"]["ttft_p50_s"]]
        ax[0].bar(["prefix cache off", "on"], [x["mean"] * 1e3 for x in m], yerr=[ci_err(x) * 1e3 for x in m], capsize=4)
        ax[0].set_ylabel("p50 TTFT (ms)"), ax[0].set_title("Prefix reuse")
    if "isolation" in rep:
        iso = rep["isolation"]
        keys = ["without_cache_salt", "with_cache_salt"]
        ax[1].bar(["no salt", "per-tenant salt"], [iso[k]["globex_cached_tokens"].get("mean", 0) for k in keys])
        ax[1].set_ylabel("cached tokens for OTHER tenant"), ax[1].set_title("Cross-tenant cache isolation")
    save(fig, "kv_cache.png")


def plot_kv_transfer():
    p = latest("kv_transfer/*/report.json")
    if not p:
        return
    rep = json.loads(p.read_text())
    rows = rep["kv"]["rows"]
    mb = [r["kv_bytes_measured"] / 2**20 for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for path in rows[0]["transfer_s"]:
        ax[0].plot(mb, [r["transfer_s"][path]["median"] * 1e3 for r in rows], "o-", label=f"measured {path}")
    for path in rows[0]["predicted_transfer_s"]:
        ax[0].plot(mb, [r["predicted_transfer_s"][path] * 1e3 for r in rows], "--", label=f"model {path}")
    ax[0].set_xscale("log"), ax[0].set_yscale("log"), ax[0].set_xlabel("KV size (MiB)"), ax[0].set_ylabel("ms")
    ax[0].legend(fontsize=7), ax[0].set_title("KV transfer: measured vs cost model")
    L = [r["seq_len"] for r in rows]
    ax[1].plot(L, [r["ttft_colocated_s"] * 1e3 for r in rows], "o-", label="colocated")
    for path in ("d2d", "host_roundtrip_pinned"):
        ax[1].plot(L, [r["ttft_disagg_s"][path] * 1e3 for r in rows], "o--", label=f"disagg via {path}")
    ax[1].set_xlabel("prompt tokens"), ax[1].set_ylabel("TTFT (ms)"), ax[1].legend(fontsize=7)
    ax[1].set_title("Emulated disaggregation (1 GPU)")
    save(fig, "kv_transfer.png")


def plot_interference():
    p = latest("interference/*/report.json")
    if not p:
        return
    rep = json.loads(p.read_text())
    bs = sorted(rep["budgets"], key=int)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = range(len(bs))
    a = [rep["budgets"][b]["tpot_p50_alone"] for b in bs]
    n = [rep["budgets"][b]["tpot_p50_with_prefill_noise"] for b in bs]
    ax.bar([i - 0.2 for i in x], [m["mean"] * 1e3 for m in a], 0.4, yerr=[ci_err(m) * 1e3 for m in a], label="decode alone")
    ax.bar([i + 0.2 for i in x], [m["mean"] * 1e3 for m in n], 0.4, yerr=[ci_err(m) * 1e3 for m in n], label="+ long prefills")
    ax.set_xticks(list(x)), ax.set_xticklabels([f"budget {b}" for b in bs]), ax.set_ylabel("p50 TPOT (ms)")
    ax.legend(), ax.set_title("Prefill→decode interference (colocated vLLM)")
    save(fig, "interference.png")


def plot_triton():
    p = latest("triton_kernel/*/report.json")
    if not p:
        return
    rep = json.loads(p.read_text())
    rows = rep["performance"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    labels = [f"{r['M']}x{r['N']}" for r in rows]
    for impl in rows[0]["impls"]:
        ax.plot(labels, [r["impls"][impl].get("effective_GBps", float("nan")) for r in rows], "o-", label=impl)
    ax.axhline(rows[0]["hbm_peak_GBps"], ls=":", c="gray", label="HBM peak")
    ax.set_ylabel("effective GB/s"), ax.tick_params(axis="x", rotation=30), ax.legend(fontsize=7)
    ax.set_title("Fused add+RMSNorm")
    save(fig, "triton_kernel.png")


def plot_quant():
    p = latest("quantization/*/report.json")
    if not p:
        return
    rep = json.loads(p.read_text())["variants"]
    vs = [v for v in rep if "error" not in rep[v]]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    ax[0].bar(vs, [rep[v].get("model_load_gib") or 0 for v in vs]), ax[0].set_title("weight memory (GiB)")
    ax[1].bar(vs, [rep[v]["perf_c32"]["output_tok_per_s"]["mean"] for v in vs],
              yerr=[ci_err(rep[v]["perf_c32"]["output_tok_per_s"]) for v in vs], capsize=4)
    ax[1].set_title("tok/s @ concurrency 32")
    ax[2].bar(vs, [rep[v]["quality"]["accuracy"] for v in vs]), ax[2].set_title("arith accuracy")
    save(fig, "quantization.png")


def plot_failure():
    p = latest("failure_injection/*vllm/crash_report.json") or latest("failure_injection/*/crash_report.json")
    if not p:
        return
    rows = [json.loads(line) for line in open(p.parent / "crash_requests.jsonl")]
    rep = json.loads(p.read_text())
    t0 = min(r["t_start"] for r in rows)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ok = [(r["t_start"] - t0, (r["ttft_s"] or 0) * 1e3) for r in rows if r["ok"]]
    bad = [(r["t_start"] - t0, 0) for r in rows if not r["ok"]]
    ax.scatter(*zip(*ok), s=6, label="ok (TTFT ms)")
    if bad:
        ax.scatter(*zip(*bad), s=20, c="red", marker="x", label="failed")
    ev = rep["events_relative_s"]
    shift = rep.get("t0_epoch", t0) - t0  # align experiment clock with first measured request
    for k in ("kill", "restart_healthy", "readmitted"):
        if ev.get(k) is not None:
            ax.axvline(ev[k] + shift, ls="--", lw=1)
            ax.text(ev[k] + shift, ax.get_ylim()[1] * 0.9, k, fontsize=7, rotation=90)
    ax.set_xlabel("seconds since first measured request"), ax.legend(fontsize=7), ax.set_title(f"Replica crash ({rep['kind']})")
    save(fig, "failure_injection.png")


def plot_attention():
    p = latest("triton_attention/*/report.json")
    if not p:
        return
    rows = json.loads(p.read_text())["performance"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    labels = [f"N{r['N']} D{r['D']}{' c' if r['causal'] else ''}" for r in rows]
    for impl in ("triton_flash", "torch_sdpa", "torch_naive"):
        ax.plot(labels, [r["impls"].get(impl, {}).get("tflops", float("nan")) for r in rows], "o-", label=impl)
    ax.set_ylabel("TFLOP/s (bf16)"), ax.tick_params(axis="x", rotation=30), ax.legend(fontsize=7)
    ax.set_title("Attention forward on A100 (c = causal)")
    save(fig, "triton_attention.png")


def plot_offload():
    p = latest("kv_offload/*/report.json")
    if not p:
        return
    rep = json.loads(p.read_text())
    vs = [v for v, d in rep["variants"].items() if "aggregate" in d]
    m = [rep["variants"][v]["aggregate"]["ttft_p50_s"] for v in vs]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(vs, [x["mean"] * 1e3 for x in m], yerr=[ci_err(x) * 1e3 for x in m], capsize=4)
    ax.set_ylabel("p50 TTFT (ms)")
    ax.set_title(f"KV tiers: working set {rep['working_set_tokens']} tok > HBM KV {rep['hbm_kv_capacity_tokens']} tok")
    save(fig, "kv_offload.png")


def main():
    for f in (plot_attention, plot_offload, plot_baseline, plot_routing, plot_kv_cache, plot_kv_transfer, plot_interference, plot_triton,
              plot_quant, plot_failure):
        try:
            f()
        except Exception:
            print(f"[plots] {f.__name__} failed:\n{traceback.format_exc(limit=2)}", file=sys.stderr)


if __name__ == "__main__":
    main()
