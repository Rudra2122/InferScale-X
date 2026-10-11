# Runbook: latency regression (TTFT / TPOT)

**Alert:** `HighTTFT` (p95 TTFT > 2 s for 5 min)

1. Is it queueing or compute? Compare gateway TTFT with engine metrics: rising `vllm:num_requests_waiting` = queueing (capacity); flat queue but higher per-token time = compute/interference.
2. Prefix-cache hit rate dropped? (`vllm:prefix_cache_hits/queries`). Causes: routing policy changed to round_robin, a replica restart (cold cache), prompt template change breaking shared prefixes, cache_salt rotation.
3. Long-prompt traffic surge? Prefill interference raises TPOT for everyone (measured in `interference` report). Mitigation: lower `--max-num-batched-tokens`, route long prompts to dedicated replicas, or consider disaggregation (ADR-003).
4. Recent deploy? Compare against the last benchmark run with the same workload YAML — the verdict field tells you whether the change is outside the CI.
