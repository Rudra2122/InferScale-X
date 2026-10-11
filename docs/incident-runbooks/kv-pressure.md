# Runbook: KV-cache pressure

**Alert:** `KVCachePressure` (`vllm:num_preemptions_total` rate > 0.5/s for 10 min)

Preemption means vLLM evicted running sequences because KV blocks ran out; they are recomputed later, wasting GPU time and spiking latency.

1. Check context lengths in traffic vs the capacity in the engine log ("GPU KV cache size: N tokens", also captured in the kv_cache pressure report).
2. Short term: lower `--max-num-seqs`, cap `max_tokens` per tenant (gateway), or route long-context traffic to dedicated replicas.
3. Medium term: quantised weights free HBM for KV (see quantization report → model_load_gib), larger-memory GPUs, or tensor parallelism.
4. The pressure sweep (gpu_memory_utilization 0.85 → 0.3) shows how hit rate and TTFT degrade as KV capacity shrinks — use it to explain the trade-off to customers.
