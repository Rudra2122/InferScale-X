# ADR-002: Thin streaming gateway with explicit admission control

Status: accepted

## Decision
- Proxy bytes, don't re-serialise SSE: lower overhead, engine-agnostic (vLLM/SGLang/TRT-LLM
  all speak the OpenAI protocol). Usage is sniffed from the stream without modifying it.
- Reject early instead of queueing in the gateway: 503 for global overload, 429 for tenant
  limits, always with Retry-After. Queueing belongs in the engine scheduler, which knows KV
  capacity; a second queue in the gateway only hides latency.
- Retry only before the first byte. After that, emit an explicit SSE error event.
- Keys are stored as sha256; constant-time compare.
- Per-tenant `cache_salt` injected by the gateway; client-supplied salts are removed.

## Alternatives considered
- Envoy/LLM-aware proxies: less code, but tenant-aware salting, routing experiments and
  evidence capture would still need custom code.
- Gateway-side request queue with priorities: rejected for now; measured overload behaviour
  (failure experiment phase 2) is the basis to revisit.
