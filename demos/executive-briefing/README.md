# Executive briefing (one page, filled from evidence)

Fill every bracket from `docs/CLAIMS.md` / result files. If a bracket has no evidence, delete the sentence.

**What we can serve today, per A100:** [output tok/s ± CI] at [workload], p95 TTFT [ms]. (baseline report)

**Cost:** ≈ $[cost per 1M output tokens] at $[price]/GPU-h with 30% headroom. (capacity_model.py with measured throughput)

**Reliability:** a replica failure is detected in [s]; capacity returns in [s]; requests in flight on that replica are lost ([n]/[N] in the drill). (failure report)

**Multi-tenant safety:** tenants are rate-limited independently; a cross-tenant cache timing leak was demonstrated and closed. (kv_cache isolation)

**Agents:** governed agents complete [x]% of benchmark tasks vs [y]% for plain RAG; all tested injection / unauthorised-tool attacks were blocked by policy. (agents report)

**What we have not proven:** multi-GPU disaggregation and cluster-scale numbers. The scaling table is an extrapolation from one GPU and is labelled as such.

**Decision asked:** [e.g. pilot on 8× A100 to validate disaggregation break-even for long-prompt traffic]
