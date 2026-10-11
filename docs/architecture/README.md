# Architecture

## Request path (as implemented and measured on one A100)

```
client ──HTTPS/SSE──▶ Gateway (FastAPI, gateway/api/app.py)
                       │ 1 authenticate: Bearer key → sha256 → tenant          (authentication/auth.py)
                       │ 2 validate body, authorise model, clamp max_tokens     (tenant_policies/policies.py)
                       │ 3 strip client cache_salt; inject per-tenant salt      (KV-cache isolation)
                       │ 4 admission: global in-flight cap → 503, tenant
                       │   concurrency / token bucket → 429 + Retry-After
                       │ 5 route: round_robin | least_outstanding | prefix_affinity   (api/router.py)
                       │ 6 send; on connect error / 5xx before first byte → retry other replica
                       │ 7 stream bytes through unchanged; sniff usage; metrics + JSON log
                       ▼
             vLLM replica A (:8001)      vLLM replica B (:8002)      ← same A100 in the routing experiment
             paged KV cache + prefix cache (salted per tenant), chunked prefill
```

Health: a background loop probes `/health` on each replica; N consecutive failures (probe or
request) eject a replica, a successful probe re-admits it. Mid-stream failures cannot be
retried transparently (tokens already reached the client) — the gateway emits an explicit SSE
error event; the failure-injection experiment counts these as lost requests.

## Components and the evidence each produces

| Component | Code | Evidence |
|---|---|---|
| Gateway | `gateway/` | unit + integration tests, `baseline` (overhead) |
| Routing | `gateway/api/router.py` | `routing` report |
| Reliability | gateway retries/health + `serverctl.kill_hard` | `failure_injection/*vllm` |
| KV cache | vLLM prefix cache + gateway salt | `kv_cache` (reuse, pressure, isolation) |
| KV movement | `disaggregation/nixl-benchmarks` | `kv_transfer` (bandwidth fit, per-size error) |
| Disaggregation | interference + emulation + connector attempt | `interference`, `kv_transfer.break_even`, `disagg_attempt` |
| Kernels | `optimization/triton-kernels` | `triton_kernel` |
| Model opt | `optimization/quantization`, `speculative-decoding` | `quantization`, `speculative_decoding` |
| Agents | `agents/` | `agents/*live`, `agents/*scripted`, security tests |
| Customisation | `training/` | `training/runs/*`, `eval_report.json` |

## Target production topology (reference design — NOT tested)

```
             ┌─────────── L7 LB ───────────┐
             ▼                             ▼
      gateway pods (HPA on CPU)      gateway pods          stateless; config from Secret
             │  prefix-affinity routing via StatefulSet DNS (vllm-0, vllm-1, …)
   ┌─────────┴───────────┬────────────────────┐
   ▼                     ▼                    ▼
 prefill pool        decode pool         decode pool       (Dynamo / KV connector; NIXL over RDMA/NVLink)
 (compute-bound)     (bandwidth-bound)                     sized from measured ΔTPOT and KV bytes/token
   └─────── KV transfer (NIXL: GPU→GPU RDMA, fallback host-staged) ───────┘
 Prometheus + DCGM exporter + Grafana; alerts in infrastructure/monitoring/alerts.yml
```

Why this split (and when not to do it) is argued in `docs/decisions/ADR-003-disaggregation.md`
using the break-even computed from the A100 measurements.
