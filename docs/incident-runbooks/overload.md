# Runbook: overload / rejections

**Alerts:** `RejectionsSpike` (global_overload) or tenant 429 complaints.

1. Break down `inferscale_rejections_total` by `reason`:
   - `tenant_rate_limit` / `tenant_concurrency` → the tenant exceeds its contract. Expected; confirm other tenants' latency is unaffected (the overload experiment checks exactly this).
   - `global_overload` → the platform is out of capacity.
2. For global overload: check engine queue depth (`vllm:num_requests_waiting`) and GPU util. If GPUs are saturated, scale replicas (capacity model gives N). If GPUs are idle but the gateway rejects, `global_max_inflight` is too low for the replica count.
3. Never "fix" overload by removing limits — queueing moves the pain into TTFT for everyone.
