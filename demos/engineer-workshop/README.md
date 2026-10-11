# Engineer workshop (90 minutes, one A100)

Goal: participants leave able to explain TTFT vs TPOT, prefix caching, routing, and why
"faster" claims need baselines and confidence intervals.

| Min | Exercise | Command | Discussion |
|---|---|---|---|
| 0–10 | Architecture walk-through | `docs/architecture/README.md` | where latency is spent |
| 10–25 | Baseline & gateway overhead | `python -m benchmarks.runners.exp_baseline --repeats 3 --num-requests 120` | read mean ± CI; why repeats matter |
| 25–40 | Break the cache | `exp_prefix_cache.py --parts reuse --repeats 2` | why RAG traffic loves prefix caching |
| 40–55 | Routing | `exp_routing.py --workloads benchmarks/workloads/prefix_heavy.yaml --repeats 2` | KV-aware routing vs round-robin |
| 55–70 | Kill a replica | `python -m tests.failure_injection.exp_failure --kind vllm --duration 60` | in-flight stream loss; detection time |
| 70–85 | Side channel | `exp_prefix_cache.py --parts isolation` | timing leak, and the salt that closes it |
| 85–90 | Negative result | pick one "regressed" verdict from READINESS.md | how to present it honestly |

No-GPU fallback: every step except KV/vLLM metrics runs against mock backends (`make demo`).
