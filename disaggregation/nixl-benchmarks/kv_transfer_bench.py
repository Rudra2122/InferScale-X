"""KV movement measurement + cost-model reconciliation + emulated disaggregation (one A100).

What is real here
-----------------
1. Raw copy bandwidth on this machine: D2D, pinned H2D/D2H, pageable H2D/D2H.
2. Real KV caches from a real model (HF transformers, Qwen2.5-1.5B by default)
   for prompts of 256..8192 tokens: their exact byte size, and the measured time
   to move them over each path (and over NIXL if the `nixl` package works here).
3. Theoretical model  t = alpha + bytes / beta  using the measured bandwidths;
   we report predicted vs measured error for every size and path.
4. Emulated disaggregation: TTFT_colocated = prefill + first decode step, vs
   TTFT_disagg = prefill + KV transfer (path) + first decode step on a separate
   "decode worker" (same GPU, different CUDA stream / a copied cache).

What is NOT real
----------------
There is one GPU, so there is no NVLink / RDMA / InfiniBand hop. The pinned
host round trip (D2H + H2D) is used as a proxy for a cross-node transfer; the
report states this explicitly. Use the measured alpha/beta in the cost model
with your target fabric's bandwidth to extrapolate (labelled 'extrapolated').

python disaggregation/nixl-benchmarks/kv_transfer_bench.py --model Qwen/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from benchmarks.runners.env_capture import capture  # noqa: E402


def cuda_time(fn, iters: int = 10, warmup: int = 3) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / 1000.0)
    return ts


def bandwidth_sweep(sizes_mb=(1, 4, 16, 64, 256, 1024)) -> dict:
    out = {"points": []}
    for mb in sizes_mb:
        n = mb * 1024 * 1024 // 2
        g = torch.empty(n, dtype=torch.bfloat16, device="cuda")
        g2 = torch.empty_like(g)
        hp = torch.empty(n, dtype=torch.bfloat16, pin_memory=True)
        hu = torch.empty(n, dtype=torch.bfloat16)
        paths = {
            "d2d": lambda: g2.copy_(g),
            "h2d_pinned": lambda: g.copy_(hp, non_blocking=True),
            "d2h_pinned": lambda: hp.copy_(g, non_blocking=True),
            "h2d_pageable": lambda: g.copy_(hu),
            "d2h_pageable": lambda: hu.copy_(g),
        }
        for name, fn in paths.items():
            t = statistics.median(cuda_time(fn))
            out["points"].append({"path": name, "bytes": n * 2, "median_s": t, "GBps": n * 2 / t / 1e9})
        g = g2 = hp = hu = None  # release before next size
        torch.cuda.empty_cache()
    # fit t = alpha + bytes/beta per path (least squares)
    out["fit"] = {}
    for path in {p["path"] for p in out["points"]}:
        pts = [p for p in out["points"] if p["path"] == path]
        xs = [p["bytes"] for p in pts]
        ys = [p["median_s"] for p in pts]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        alpha = my - slope * mx
        out["fit"][path] = {"alpha_s": max(alpha, 0.0), "beta_GBps": 1 / slope / 1e9 if slope > 0 else None}
    largest = {p["path"]: p["GBps"] for p in out["points"] if p["bytes"] == max(x["bytes"] for x in out["points"])}
    out["peak_GBps"] = largest
    return out


def kv_bytes_theory(cfg, seq_len: int, dtype_bytes: int = 2) -> int:
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return 2 * cfg.num_hidden_layers * n_kv * head_dim * seq_len * dtype_bytes


def cache_tensors(past) -> list[torch.Tensor]:
    """Flatten an HF cache object (DynamicCache or legacy tuple) into its K/V tensors."""
    if hasattr(past, "layers"):  # transformers >= 4.56 DynamicCache
        ts = []
        for layer in past.layers:
            ts += [layer.keys, layer.values]
        return ts
    if hasattr(past, "key_cache"):
        return list(past.key_cache) + list(past.value_cache)
    return [t for kv in past for t in kv]


def try_nixl_roundtrip(tensors: list[torch.Tensor]) -> dict:
    """Attempt a NIXL loopback transfer (VRAM->DRAM) of the KV tensors. Records failure honestly."""
    try:
        from nixl._api import nixl_agent, nixl_agent_config  # type: ignore
    except Exception as e:
        return {"available": False, "reason": f"import failed: {type(e).__name__}: {e}"}
    try:
        agent = nixl_agent("inferscale", nixl_agent_config(backends=["UCX"]))
        dst = [torch.empty_like(t, device="cpu").pin_memory() for t in tensors]
        src_reg = agent.register_memory(tensors)
        dst_reg = agent.register_memory(dst)
        src_desc = src_reg.trim()
        dst_desc = dst_reg.trim()
        agent_name = "inferscale"
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            h = agent.initialize_xfer("WRITE", src_desc, dst_desc, agent_name)
            agent.transfer(h)
            while agent.check_xfer_state(h) not in ("DONE", "ERR"):
                pass
            state = agent.check_xfer_state(h)
            times.append(time.perf_counter() - t0)
            agent.release_xfer_handle(h)
            if state == "ERR":
                return {"available": True, "ok": False, "reason": "transfer state ERR"}
        ok = all(torch.equal(a.cpu(), b) for a, b in zip(tensors[:2], dst[:2]))
        return {"available": True, "ok": ok, "median_s": statistics.median(times),
                "note": "loopback within one process/GPU: measures NIXL+UCX overhead, not a network hop"}
    except Exception as e:
        return {"available": True, "ok": False, "reason": f"{type(e).__name__}: {e}"}


@torch.inference_mode()
def kv_experiment(model_name: str, seq_lens: list[int], bw: dict, iters: int, use_nixl: bool) -> dict:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    cfg = model.config
    rows = []
    for L in seq_lens:
        ids = torch.randint(100, cfg.vocab_size - 100, (1, L), device="cuda")

        def prefill():
            return model(input_ids=ids, use_cache=True)

        # prefill timing (repeated)
        pre_t = cuda_time(lambda: prefill(), iters=iters, warmup=2)
        out = prefill()
        past = out.past_key_values
        tensors = cache_tensors(past)
        nbytes = sum(t.numel() * t.element_size() for t in tensors)
        next_tok = out.logits[:, -1:].argmax(-1)

        # NOTE: a decode step mutates DynamicCache (appends one token). Re-prefill per measurement is too slow,
        # so we time decode on a fresh cache each iteration by cropping back after the call.
        def decode_once():
            o = model(input_ids=next_tok, past_key_values=past, use_cache=True)
            if hasattr(past, "crop"):
                past.crop(L)
            return o

        dec_t = cuda_time(decode_once, iters=iters, warmup=2)

        paths = {}
        # D2D copy (same GPU, models a NVLink-class hop with ~zero distance)
        dst = [torch.empty_like(t) for t in tensors]
        paths["d2d"] = cuda_time(lambda: [d.copy_(s) for d, s in zip(dst, tensors)], iters=iters)
        dst = None
        # pinned host round trip (proxy for cross-node: GPU -> host -> GPU)
        host = [torch.empty(t.shape, dtype=t.dtype, pin_memory=True) for t in tensors]
        back = [torch.empty_like(t) for t in tensors]

        def roundtrip():
            for h, s in zip(host, tensors):
                h.copy_(s, non_blocking=True)
            for b, h in zip(back, host):
                b.copy_(h, non_blocking=True)

        paths["host_roundtrip_pinned"] = cuda_time(roundtrip, iters=iters)
        host = back = None
        # serialized (pickle-like) path: what a naive RPC transfer would cost
        import io

        def serialize():
            buf = io.BytesIO()
            torch.save([t.cpu() for t in tensors], buf)
            buf.seek(0)
            return [t.cuda() for t in torch.load(buf)]

        paths["serialize_torch_save"] = cuda_time(serialize, iters=max(3, iters // 3), warmup=1)
        torch.cuda.empty_cache()

        fit = bw["fit"]
        pred = {
            "d2d": fit["d2d"]["alpha_s"] * len(tensors) + nbytes / (fit["d2d"]["beta_GBps"] * 1e9),
            "host_roundtrip_pinned": (fit["d2h_pinned"]["alpha_s"] + fit["h2d_pinned"]["alpha_s"]) * len(tensors)
            + nbytes / (fit["d2h_pinned"]["beta_GBps"] * 1e9) + nbytes / (fit["h2d_pinned"]["beta_GBps"] * 1e9),
        }
        row = {
            "seq_len": L, "kv_bytes_measured": nbytes, "kv_bytes_theory": kv_bytes_theory(cfg, L),
            "n_tensors": len(tensors),
            "prefill_s": {"median": statistics.median(pre_t), "stdev": statistics.pstdev(pre_t)},
            "decode_step_s": {"median": statistics.median(dec_t), "stdev": statistics.pstdev(dec_t)},
            "transfer_s": {k: {"median": statistics.median(v), "stdev": statistics.pstdev(v)} for k, v in paths.items()},
            "predicted_transfer_s": pred,
            "prediction_error_pct": {k: 100 * (statistics.median(paths[k]) - v) / v for k, v in pred.items()},
        }
        med = lambda k: row["transfer_s"][k]["median"]  # noqa: E731
        ttft_col = row["prefill_s"]["median"] + row["decode_step_s"]["median"]
        row["ttft_colocated_s"] = ttft_col
        row["ttft_disagg_s"] = {k: row["prefill_s"]["median"] + med(k) + row["decode_step_s"]["median"]
                                for k in paths}
        row["transfer_overhead_pct_of_ttft"] = {k: 100 * med(k) / ttft_col for k in paths}
        if use_nixl and L == seq_lens[0]:
            row["nixl"] = try_nixl_roundtrip(tensors)
        rows.append(row)
        print(f"[kv] L={L} kv={nbytes/2**20:.1f}MiB prefill={row['prefill_s']['median']*1e3:.1f}ms "
              f"d2d={med('d2d')*1e3:.2f}ms host_rt={med('host_roundtrip_pinned')*1e3:.2f}ms "
              f"pred_err_host={row['prediction_error_pct']['host_roundtrip_pinned']:.1f}%")
        past = out = tensors = None
        torch.cuda.empty_cache()
    return {"model": model_name, "config": {"layers": cfg.num_hidden_layers,
            "kv_heads": getattr(cfg, "num_key_value_heads", None), "hidden": cfg.hidden_size}, "rows": rows}


def break_even(rows: list[dict], interference_report: str | None, fabric_GBps: dict) -> dict:
    """Disaggregation pays off when the decode slowdown it removes exceeds the KV transfer it adds.

    saved per request  ~= delta_TPOT_from_prefill_interference * output_tokens   (measured by exp_interference)
    cost per request   ~= kv_bytes / fabric_bandwidth + alpha                     (measured here / extrapolated)
    """
    out = {"label": "analytic; transfer costs for other fabrics are EXTRAPOLATED from bytes, not measured",
           "fabrics_GBps_assumed": fabric_GBps, "by_seq_len": []}
    delta_tpot = None
    if interference_report and Path(interference_report).exists():
        rep = json.loads(Path(interference_report).read_text())
        delta_tpot = rep.get("delta_tpot_p50_s")
    out["measured_delta_tpot_s"] = delta_tpot
    for r in rows:
        entry = {"seq_len": r["seq_len"], "kv_bytes": r["kv_bytes_measured"],
                 "measured_host_roundtrip_s": r["transfer_s"]["host_roundtrip_pinned"]["median"],
                 "extrapolated_transfer_s": {f: r["kv_bytes_measured"] / (bw * 1e9) for f, bw in fabric_GBps.items()}}
        if delta_tpot:
            entry["min_output_tokens_to_break_even"] = {
                f: (t / delta_tpot) for f, t in entry["extrapolated_transfer_s"].items()}
        out["by_seq_len"].append(entry)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--seq-lens", default="256,1024,2048,4096,8192")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--nixl", action="store_true", help="also attempt a NIXL loopback transfer")
    ap.add_argument("--interference-report", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    from benchmarks.runners.common import result_dir, save_json

    out = Path(a.out) if a.out else result_dir("kv_transfer")
    bw = bandwidth_sweep()
    print("[kv] peak GB/s:", {k: round(v, 1) for k, v in bw["peak_GBps"].items()})
    kv = kv_experiment(a.model, [int(x) for x in a.seq_lens.split(",")], bw, a.iters, a.nixl)
    fabrics = {"pcie4_x16_measured_h2d": bw["peak_GBps"]["h2d_pinned"], "nvlink3_a100_unidir": 300.0,
               "ib_hdr_200g": 25.0, "ib_ndr_400g": 50.0, "ethernet_100g": 12.5}
    report = {"experiment": "kv_transfer", "env": capture(), "bandwidth": {
        **bw, "h2d_pinned_GBps": bw["peak_GBps"]["h2d_pinned"]}, "kv": kv,
        "break_even": break_even(kv["rows"], a.interference_report, fabrics),
        "caveat": "Single GPU. 'host_roundtrip_pinned' is a proxy for a cross-node hop. No physical disaggregation."}
    save_json(out / "report.json", report)
    print(f"[kv] wrote {out}/report.json")


if __name__ == "__main__":
    main()
