"""Custom Triton kernel: fused residual-add + RMSNorm (the op at every transformer block boundary).

    residual_out = x + residual
    y            = residual_out * rsqrt(mean(residual_out^2) + eps) * weight

Unfused PyTorch reads/writes the hidden state several times; the fused kernel
reads x, residual, weight once and writes y and residual_out once, so it is
memory-bound and judged by effective bandwidth vs the A100's HBM peak.

Baselines: torch eager, torch.compile, and vLLM's CUDA fused_add_rms_norm if installed.
Correctness: max abs/rel error vs an fp32 reference across shapes and dtypes.

python optimization/triton-kernels/fused_add_rmsnorm.py --profile
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@triton.jit
def _fused_add_rmsnorm_kernel(X, R, W, Y, ROUT, stride_row, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(X + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(R + row * stride_row + cols, mask=mask, other=0.0).to(tl.float32)
    h = x + r
    tl.store(ROUT + row * stride_row + cols, h.to(ROUT.dtype.element_ty), mask=mask)
    var = tl.sum(h * h, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * stride_row + cols, (h * rstd * w).to(Y.dtype.element_ty), mask=mask)


def fused_add_rmsnorm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    assert (x.is_cuda or os.environ.get("TRITON_INTERPRET") == "1") and x.shape == residual.shape and x.shape[-1] == weight.shape[0]
    x2, r2 = x.reshape(-1, x.shape[-1]).contiguous(), residual.reshape(-1, x.shape[-1]).contiguous()
    M, N = x2.shape
    y, rout = torch.empty_like(x2), torch.empty_like(x2)
    BLOCK = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK <= 2048 else (8 if BLOCK <= 8192 else 16)
    kw = {} if os.environ.get("TRITON_INTERPRET") == "1" else {"num_warps": num_warps}
    _fused_add_rmsnorm_kernel[(M,)](x2, r2, weight, y, rout, x2.stride(0), N, eps, BLOCK=BLOCK, **kw)
    return y.view_as(x), rout.view_as(x)


def torch_ref(x, residual, weight, eps=1e-6):
    h = x + residual
    var = h.float().pow(2).mean(-1, keepdim=True)
    return (h.float() * torch.rsqrt(var + eps) * weight.float()).to(x.dtype), h


def fp32_ref(x, residual, weight, eps=1e-6):
    h = x.float() + residual.float()
    return h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * weight.float(), h


def vllm_op():
    try:
        from vllm import _custom_ops as ops  # type: ignore

        def f(x, residual, weight, eps=1e-6):
            xc, rc = x.clone(), residual.clone()  # vLLM op is in-place; clone cost excluded below via pre-cloning
            ops.fused_add_rms_norm(xc, rc, weight, eps)
            return xc, rc
        return f
    except Exception:
        return None


def bench(fn, *args, iters=100):
    for _ in range(10):
        fn(*args)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn(*args)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) / 1e3)
    return ts


def hbm_peak_GBps() -> float:
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    if "A100" in name:
        return 2039.0 if total > 60 else 1555.0
    if "H100" in name:
        return 3350.0
    if "L4" in name:
        return 300.0
    if "T4" in name:
        return 320.0
    return float("nan")


def correctness(device: str = "cuda") -> list[dict]:
    rows = []
    cpu = device == "cpu"
    for dtype in ((torch.float32,) if cpu else (torch.float16, torch.bfloat16, torch.float32)):
        for M, N in (((3, 1000), (5, 64)) if cpu else
                     ((1, 1536), (7, 1536), (128, 3584), (4096, 4096), (333, 5120), (16, 8192), (3, 1000))):
            torch.manual_seed(M * N)
            x = torch.randn(M, N, device=device, dtype=dtype)
            r = torch.randn(M, N, device=device, dtype=dtype)
            w = torch.randn(N, device=device, dtype=dtype)
            y, h = fused_add_rmsnorm(x, r, w)
            y32, h32 = fp32_ref(x, r, w)
            tol = {torch.float32: 1e-4, torch.float16: 2e-2, torch.bfloat16: 6e-2}[dtype]
            err = (y.float() - y32).abs().max().item()
            herr = (h.float() - h32).abs().max().item()
            ok = torch.allclose(y.float(), y32, atol=tol, rtol=tol) and torch.allclose(h.float(), h32, atol=tol, rtol=tol)
            rows.append({"dtype": str(dtype), "M": M, "N": N, "max_abs_err_y": err, "max_abs_err_residual": herr,
                         "tol": tol, "pass": bool(ok)})
    return rows


def performance(iters: int) -> list[dict]:
    peak = hbm_peak_GBps()
    compiled = torch.compile(torch_ref)
    vop = vllm_op()
    rows = []
    for M, N in ((1, 1536), (32, 1536), (512, 1536), (4096, 1536), (16384, 1536), (4096, 4096), (8192, 8192)):
        x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        r = torch.randn_like(x)
        w = torch.randn(N, device="cuda", dtype=torch.bfloat16)
        bytes_moved = (3 * M * N + N + M * N) * 2  # read x, r, w; write y, residual_out (minimum traffic)
        impls = {"triton_fused": fused_add_rmsnorm, "torch_eager": torch_ref, "torch_compile": compiled}
        if vop:
            impls["vllm_cuda_fused"] = vop
        row = {"M": M, "N": N, "bytes_min": bytes_moved, "hbm_peak_GBps": peak, "impls": {}}
        for name, fn in impls.items():
            try:
                ts = bench(fn, x, r, w, iters=iters)
                med = statistics.median(ts)
                row["impls"][name] = {"median_us": med * 1e6, "p90_us": sorted(ts)[int(0.9 * len(ts))] * 1e6,
                                      "effective_GBps": bytes_moved / med / 1e9,
                                      "pct_of_peak": 100 * bytes_moved / med / 1e9 / peak}
            except Exception as e:
                row["impls"][name] = {"error": f"{type(e).__name__}: {e}"[:200]}
        if vop:
            row["note"] = "vllm op timing includes 2 clones (it is in-place); compare with care"
        base = row["impls"].get("torch_compile", {}).get("median_us")
        tri = row["impls"]["triton_fused"].get("median_us")
        row["triton_speedup_vs_torch_compile"] = base / tri if base and tri else None
        rows.append(row)
        print(f"[triton] M={M} N={N} " + " ".join(f"{k}={v.get('median_us', float('nan')):.1f}us"
                                                    for k, v in row["impls"].items()))
    return rows


def profile(out: Path):
    from torch.profiler import ProfilerActivity, profile as tprof

    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    r, w = torch.randn_like(x), torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    with tprof(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        for _ in range(20):
            fused_add_rmsnorm(x, r, w)
            torch_ref(x, r, w)
        torch.cuda.synchronize()
    p.export_chrome_trace(str(out / "trace_fused_vs_eager.json"))
    (out / "profile_table.txt").write_text(p.key_averages().table(sort_by="cuda_time_total", row_limit=20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--check-only", action="store_true", help="correctness only (works on CPU with TRITON_INTERPRET=1)")
    a = ap.parse_args()
    if a.check_only:
        corr = correctness("cpu" if os.environ.get("TRITON_INTERPRET") == "1" else "cuda")
        print(f"[triton] correctness: {sum(r['pass'] for r in corr)}/{len(corr)} pass")
        sys.exit(0 if all(r["pass"] for r in corr) else 1)
    from benchmarks.runners.common import result_dir, save_json
    from benchmarks.runners.env_capture import capture

    out = result_dir("triton_kernel")
    corr = correctness()
    print(f"[triton] correctness: {sum(r['pass'] for r in corr)}/{len(corr)} pass")
    perf = performance(a.iters)
    if a.profile:
        profile(out)
    report = {"experiment": "triton_fused_add_rmsnorm", "env": capture(), "correctness": corr,
              "all_correct": all(r["pass"] for r in corr), "performance": perf,
              "interpretation_hint": "Small M is launch-bound (all impls ~equal); large M is HBM-bound. "
                                     "If torch.compile matches Triton, report that as a negative result."}
    save_json(out / "report.json", report)
    print(f"[triton] wrote {out}/report.json")
    print(json.dumps({"all_correct": report["all_correct"]}))


if __name__ == "__main__":
    main()
