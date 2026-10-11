# Evidence-backed claims

_Rendered from docs/claims.yaml with numbers read from result files._

- **gateway_overhead**: Built an OpenAI-compatible multi-tenant gateway (auth, rate limits, admission control, SSE streaming) over vLLM on an A100; measured p50 TTFT change vs direct vLLM: +6.3% (no_significant_difference, n=3 runs); baseline 5078 output tok/s.  
  evidence: `benchmarks/results/baseline/20261010-092402/report.json`
- **routing**: Implemented KV-cache-aware (prefix-affinity) routing; on a RAG-style prefix-heavy workload across 2 vLLM replicas (sharing one A100) p50 TTFT changed +51.7% vs round-robin (no_significant_difference).  
  evidence: `benchmarks/results/routing/20261010-092925/report.json`
- **prefix_cache**: Measured vLLM prefix caching on a prefix-heavy workload: p50 TTFT -34.2% vs caching disabled (no_significant_difference).  
  evidence: `benchmarks/results/kv_cache/20261010-093938/report.json`
- **cache_isolation**: Demonstrated a cross-tenant prefix-cache timing/side-channel (leak observed without salting: True) and closed it with per-tenant cache salts injected at the gateway (isolation enforced: True).  
  evidence: `benchmarks/results/kv_cache/20261010-093938/report.json`
- **failure_recovery**: Fault-injected a vLLM replica crash under load: detected in 0.23s, replica back in 56s; 3/720 requests failed (in-flight streams on the killed replica).  
  evidence: `benchmarks/results/failure_injection/20261010-093547-vllm/crash_report.json`
- **kv_transfer**: Measured KV-cache movement for real model caches (256-8K tokens) and fit a latency/bandwidth cost model (pinned H2D 12.4 GB/s) used for a disaggregation break-even analysis (single GPU, emulated).  
  evidence: `benchmarks/results/kv_transfer/20261010-103616/report.json`
- **triton_kernel**: Wrote a fused residual-add + RMSNorm Triton kernel, verified against an fp32 reference across fp16/bf16/fp32 shapes (all correct: True), and benchmarked effective HBM bandwidth vs torch.compile.  
  evidence: `benchmarks/results/triton_kernel/20261010-100951/report.json`
- **agents_security**: Built a governed agent runtime (versioned skills, sandboxed tools, tenant-scoped memory, hash-chained audit log): task success 83% vs 83% for single-shot RAG; all injection / unauthorized-tool attacks blocked: True.  
  evidence: `benchmarks/results/agents/20261010-101332-live/report.json`
- **model_customization**: Ran SFT, DPO, GRPO (heuristic reward) and RLVR (verifiable reward) LoRA experiments on a 0.5B model with dataset lineage and leakage checks; see training/runs/eval_report.json for held-out accuracy and serving-token cost per method.  
  evidence: `training/runs/eval_report.json`
- **triton_attention**: Implemented a FlashAttention-style fused attention forward pass in Triton (online softmax, causal block skipping); correct vs an fp32 reference on all tested shapes (True); at B=4,H=16,N=2048,D=64 causal it runs at 0.76x the speed of PyTorch SDPA (FlashAttention-2) on an A100.  
  evidence: `benchmarks/results/triton_attention/20261010-101024/report.json`
- **kv_offload**: Added a CPU-DRAM KV-cache tier (vllm_native_offloading) behind vLLM's HBM prefix cache; with a working set 2.7x larger than HBM KV capacity, p50 TTFT changed -34.5% vs HBM-only (improved).  
  evidence: `benchmarks/results/kv_offload/20261010-101543/report.json`
- **retrieval**: Built a labelled retrieval benchmark (32 queries, document-level qrels): BM25 MRR@10 0.94, recall@5 0.98 (dense and hybrid RRF results in the same report when run).  
  evidence: `benchmarks/results/retrieval/20261010-101058/report.json`
- **kubernetes**: Ran the gateway and replicas on a 3-node Kubernetes cluster (kind, Calico) with anti-affinity across nodes. Under ~25 req/s, a force-killed replica pod was back Ready in 2.7s (2/131 requests failed in that window) and a killed gateway pod cost 4/677; 1119/1125 succeeded overall. HPA scaled the gateway to 6 replicas 18s into a load spike, and a Calico NetworkPolicy blocked non-gateway pods from the replicas (True). CPU mock backends; no GPU scheduling claimed.  
  evidence: `benchmarks/results/k8s_kind/20261010-034938/report.json`
