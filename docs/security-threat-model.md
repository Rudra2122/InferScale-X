# Security threat model

Scope: gateway, vLLM replicas, agent runtime. Method: STRIDE per trust boundary. "Test" names
the evidence that the control works; "Gap" is what a production deployment must add.

## Trust boundaries
1. Internet → gateway (untrusted clients)  2. gateway → engines (internal network)
3. model output → tool runtime (untrusted text that can drive actions)  4. retrieved/user documents → model context

| # | Threat | Control | Test | Gap |
|---|---|---|---|---|
| T1 | Stolen / guessed API key | sha256-hashed keys, constant-time compare, per-tenant rate limit | unit `test_auth_paths` | key rotation, mTLS, OIDC |
| T2 | Tenant uses forbidden model or unbounded output | model allowlist, `max_tokens` clamp | integration `test_model_permission`, `test_max_tokens_clamped_for_tenant` | |
| T3 | Noisy neighbour / DoS | token bucket, tenant concurrency cap, global in-flight cap (429/503) | failure experiment phase 2, `test_rate_limit_returns_429_with_retry_after` | per-tenant token-based (not request-based) budgets |
| T4 | **Cross-tenant prefix-cache side channel** (TTFT reveals another tenant's prompt prefix) | gateway-injected per-tenant `cache_salt`; client salt stripped | `kv_cache` isolation report (leak shown without salt, closed with salt); `test_cache_salt_isolates_tenants…` | constant-time padding not attempted |
| T5 | Direct access to engines bypassing gateway | engines bind 127.0.0.1 on Colab; NetworkPolicy in K8s reference | — (reference) | |
| T6 | Prompt injection via retrieved docs → tool abuse | untrusted wrapping, heuristic screening, **taint disables high-risk tools**, authorisation outside model | security S2 (scripted + live) | semantic injection that evades regexes still can't run high-risk tools, but can mislead answers |
| T7 | Model calls tool not granted | tenant ∩ skill allowlist, budget | S1, `test_policy_decisions_and_audit_chain` | |
| T8 | Model spoofs tenant in tool args | runtime-bound tenant; args with tenant/namespace rejected | S4 | |
| T9 | Cross-tenant memory read | tenant-keyed SQL, no tenant parameter exposed | S3, `test_memory_isolation` | encryption at rest |
| T10 | Sandbox escape (network, exec, fs, resources) | separate process, `-I`, scrubbed env, rlimits (each applied independently), parent-side RSS watchdog (macOS ignores RLIMIT_AS), PEP 578 audit hook | S5 cases, `test_sandbox_*`, `test_memory_limit_enforced_without_rlimit_as` | **Audit hooks are bypassable by a determined attacker** (e.g. via C extensions already loaded). Production: gVisor/Firecracker/nsjail + no network namespace |
| T11 | Tampering with audit trail | SHA-256 hash chain | `test_policy_decisions_and_audit_chain` (tamper detected) | append-only external store (WORM) |
| T12 | Silent skill change | per-version content hash + lockfile | `test_skill_registry_versions_and_lock` | signed manifests |
| T13 | Prompt/response leakage in logs | prompts not logged by default (`log_prompts: false`) | code review | PII redaction |
