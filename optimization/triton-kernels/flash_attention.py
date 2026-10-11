"""Custom Triton kernel: FlashAttention-style fused attention forward (causal or full).

One program computes a BLOCK_M x HEAD_DIM tile of the output for one (batch, head),
streaming K/V in BLOCK_N tiles with the online-softmax recurrence
    m_new = max(m, rowmax(S));  l = l*2^(m-m_new) + rowsum(2^(S-m_new));  acc = acc*2^(m-m_new) + P V
so the N x N score matrix never touches HBM. Causal programs stop at their diagonal block.

Baselines:
  torch_sdpa   F.scaled_dot_product_attention (dispatches to FlashAttention-2 / mem-efficient kernels on A100)
  torch_naive  softmax(QK^T)V materialising the full score matrix (the thing FlashAttention avoids)
Correctness: max error vs an fp32 reference across shapes, head dims, dtypes, causal/non-causal.
CPU check (no GPU): TRITON_INTERPRET=1 python optimization/triton-kernels/flash_attention.py --check-only

python optimization/triton-kernels/flash_attention.py            # correctness + benchmark on A100
"""
from __future__ import annotations

import argparse
import math
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


@triton.jit
def _attn_fwd(Q, K, V, O, sm_scale,
              sqb, sqh, sqn, sqd, skb, skh, skn, skd, svb, svh, svn, svd, sob, soh, son, sod,
              H, N_CTX,
              HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr):
    start_m = tl.program_id(0)
    off_bh = tl.program_id(1)
    b = off_bh // H
    h = off_bh % H
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + b * sqb + h * sqh + offs_m[:, None] * sqn + offs_d[None, :] * sqd
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.4426950408889634  # log2(e): use exp2

    if CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
    else:
        hi = N_CTX

    for start_n in range(0, hi, BLOCK_N):
        cols = start_n + offs_n
        k = tl.load(K + b * skb + h * skh + cols[None, :] * skn + offs_d[:, None] * skd,
                    mask=cols[None, :] < N_CTX, other=0.0)  # [HEAD_DIM, BLOCK_N]
        s = tl.dot(q, k) * qk_scale
        valid = cols[None, :] < N_CTX
        if CAUSAL:
            valid = valid & (offs_m[:, None] >= cols[None, :])
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.math.exp2(s - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + b * svb + h * svh + cols[:, None] * svn + offs_d[None, :] * svd,
                    mask=cols[:, None] < N_CTX, other=0.0)  # [BLOCK_N, HEAD_DIM]
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = O + b * sob + h * soh + offs_m[:, None] * son + offs_d[None, :] * sod
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = True,
                    sm_scale: float | None = None, block_m: int | None = None, block_n: int | None = None):
    """q, k, v: [batch, heads, seq, head_dim] (same seq for q and k; head_dim in {16,32,64,128})."""
    B, H, N, D = q.shape
    assert k.shape == q.shape and v.shape == q.shape, "self-attention shapes only"
    assert D in (16, 32, 64, 128), "head_dim must be 16/32/64/128"
    assert q.is_cuda or INTERPRET
    sm_scale = sm_scale or 1.0 / math.sqrt(D)
    o = torch.empty_like(q)
    if block_m is None:
        block_m = 16 if INTERPRET else (128 if D <= 64 else 64)
    if block_n is None:
        block_n = 16 if INTERPRET else 64
    grid = (triton.cdiv(N, block_m), B * H)
    kwargs = {} if INTERPRET else {"num_warps": 4 if D <= 64 else 8, "num_stages": 2}
    _attn_fwd[grid](q, k, v, o, sm_scale, *q.stride(), *k.stride(), *v.stride(), *o.stride(), H, N,
                    HEAD_DIM=D, BLOCK_M=block_m, BLOCK_N=block_n, CAUSAL=causal, **kwargs)
    return o


def reference_fp32(q, k, v, causal):
    s = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if causal:
        n = q.shape[-2]
        s = s.masked_fill(torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    return torch.softmax(s, -1) @ v.float()


def naive(q, k, v, causal):
    s = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(q.shape[-1]))
    if causal:
        n = q.shape[-2]
        s = s.masked_fill(torch.ones(n, n, dtype=torch.bool, device=q.device).triu(1), float("-inf"))
    return torch.softmax(s.float(), -1).to(q.dtype) @ v


def sdpa(q, k, v, causal):
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def correctness(device: str) -> list[dict]:
    rows = []
    dtypes = (torch.float32,) if device == "cpu" else (torch.float16, torch.bfloat16)
    shapes = ((1, 2, 37, 64), (2, 3, 64, 32)) if device == "cpu" else \
        ((1, 1, 1, 64), (2, 4, 127, 64), (1, 8, 512, 128), (2, 12, 1024, 64), (1, 4, 2049, 128), (4, 16, 333, 32))
    for dtype in dtypes:
        for shape in shapes:
            for causal in (True, False):
                torch.manual_seed(sum(shape))
                q, k, v = (torch.randn(shape, device=device, dtype=dtype) for _ in range(3))
                out = flash_attention(q, k, v, causal=causal)
                ref = reference_fp32(q, k, v, causal)
                err = (out.float() - ref).abs().max().item()
                tol = {torch.float32: 2e-3, torch.float16: 1e-2, torch.bfloat16: 3e-2}[dtype]
                rows.append({"shape": list(shape), "dtype": str(dtype), "causal": causal, "max_abs_err": err,
                             "tol": tol, "pass": bool(err <= tol and torch.isfinite(out).all())})
    return rows


def bench(fn, *args, iters=50):
    for _ in range(5):
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
    return statistics.median(ts)


def performance(iters: int) -> list[dict]:
    peak_tflops = 312.0  # A100 dense BF16/FP16 tensor-core peak
    rows = []
    for B, H, N, D in ((1, 16, 1024, 64), (4, 16, 2048, 64), (2, 16, 4096, 64), (1, 32, 8192, 128), (8, 12, 512, 64)):
        for causal in (True, False):
            q, k, v = (torch.randn(B, H, N, D, device="cuda", dtype=torch.bfloat16) for _ in range(3))
            flops = 4 * B * H * N * N * D * (0.5 if causal else 1.0)
            row = {"B": B, "H": H, "N": N, "D": D, "causal": causal, "flops": flops, "impls": {}}
            for name, fn in (("triton_flash", flash_attention), ("torch_sdpa", sdpa), ("torch_naive", naive)):
                try:
                    t = bench(fn, q, k, v, causal, iters=iters)
                    row["impls"][name] = {"median_ms": t * 1e3, "tflops": flops / t / 1e12,
                                          "pct_of_peak": 100 * flops / t / 1e12 / peak_tflops}
                except torch.cuda.OutOfMemoryError:
                    row["impls"][name] = {"error": "OOM (expected for naive at long N: N^2 score matrix)"}
                    torch.cuda.empty_cache()
                except Exception as e:
                    row["impls"][name] = {"error": f"{type(e).__name__}: {e}"[:200]}
            tri, sd = row["impls"]["triton_flash"].get("median_ms"), row["impls"]["torch_sdpa"].get("median_ms")
            row["triton_vs_sdpa_speedup"] = sd / tri if tri and sd else None
            rows.append(row)
            print(f"[attn] B={B} H={H} N={N} D={D} causal={causal} " +
                  " ".join(f"{k}={v.get('tflops', float('nan')):.1f}TF" for k, v in row["impls"].items()))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    device = "cpu" if INTERPRET else "cuda"
    corr = correctness(device)
    ok = all(r["pass"] for r in corr)
    print(f"[attn] correctness on {device}: {sum(r['pass'] for r in corr)}/{len(corr)} pass")
    if a.check_only:
        sys.exit(0 if ok else 1)
    from benchmarks.runners.common import result_dir, save_json
    from benchmarks.runners.env_capture import capture

    out = result_dir("triton_attention")
    perf = performance(a.iters)
    save_json(out / "report.json", {
        "experiment": "triton_flash_attention_fwd", "env": capture(), "correctness": corr, "all_correct": ok,
        "performance": perf,
        "interpretation_hint": "SDPA on A100 uses a tuned FlashAttention-2 kernel; matching it is a good result, "
                               "losing to it is expected and is a valid negative result. Naive OOM/slowdown at long "
                               "N shows why tiling matters."})
    print(f"[attn] wrote {out}/report.json")


if __name__ == "__main__":
    main()
