# Readiness checklist - 11/12 verified

_Generated 2026-10-10 18:49 by scripts/evidence_report.py from files in benchmarks/results/ and tests/results/. Do not edit by hand._

| # | Test | Status | Evidence | Note |
|---|---|---|---|---|
| 1 | Reproducibility | **VERIFIED** | benchmarks/results/dynamo_attempt/20261010-232036/bench/env.json<br>requirements/lock-colab.txt<br>benchmarks/results/baseline/20261010-092402/report.json | Needs env capture + pinned lockfile from the run + a baseline report. Fresh-runtime rerun is a manual step. |
| 2 | Correctness | **VERIFIED** | tests/results/junit-unit.xml (73 tests, 0 failed, 10 skipped)<br>tests/results/junit-live.xml (8 tests, 0 failed, 0 skipped) | Live = streaming/non-streaming parity, determinism, usage, limits against real vLLM. |
| 3 | Performance | **VERIFIED** | benchmarks/results/baseline/20261010-092402/report.json | repeats=3; every experiment reports mean, std and 95% CI across repeats. |
| 4 | Disaggregation | **PARTIAL** | benchmarks/results/interference/20261010-102752/report.json<br>benchmarks/results/kv_transfer/20261010-103616/report.json<br>benchmarks/results/disagg_attempt/20261010-095606/attempt.json | Single GPU: interference measured + emulated disaggregated TTFT; real vLLM P/D connector attempt ok=False. Cannot be VERIFIED without separate prefill/decode GPUs. |
| 5 | KV movement | **VERIFIED** | benchmarks/results/kv_transfer/20261010-103616/report.json | cost-model error per size (host round trip, %): [-47.6, -21.1, -12.1, -6.5, -3.4] |
| 6 | Reliability | **VERIFIED** | benchmarks/results/failure_injection/20261010-093547-vllm/crash_report.json<br>benchmarks/results/failure_injection/20261010-093547-vllm/overload_report.json | Crash/restart timings + overload shedding with real vLLM. |
| 7 | Isolation | **VERIFIED** | benchmarks/results/kv_cache/20261010-093938/report.json<br>benchmarks/results/agents/20261010-101332-live/report.json<br>tests/unit/test_gateway.py | KV cache salt enforced=True; memory cross-tenant blocked=True |
| 8 | Security | **VERIFIED** | benchmarks/results/agents/20261010-101332-live/report.json<br>benchmarks/results/agents/20261010-101332-live/report.json | Prompt-injection + unauthorized-tool + sandbox-escape suites (scripted adversary and live model). |
| 9 | Scaling | **VERIFIED** | benchmarks/results/routing/20261010-092925/report.json<br>docs/capacity-planning/README.md | Measured: 1 GPU, 2 replicas. Everything above that is labelled EXTRAPOLATED by capacity_model.py. |
| 10 | Customer readiness | **VERIFIED** | docs/partner-deployment-guide.md<br>docs/capacity-planning/capacity_model.py<br>docs/incident-runbooks/backend-down.md<br>docs/architecture/README.md | Docs exist; a real partner walkthrough is still a human activity. |
| 11 | Traceability | **VERIFIED** | docs/CLAIMS.md | 13/13 claims backed by evidence files. |
| 12 | Negative results | **VERIFIED** | docs/negative-results.md<br>benchmarks/results/speculative_decoding/20261010-105841/report.json<br>benchmarks/results/speculative_decoding/20261010-105841/report.json<br>benchmarks/results/speculative_decoding/20261010-105841/report.json<br>benchmarks/results/speculative_decoding/20261010-105841/report.json<br>benchmarks/results/speculative_decoding/20261010-104217/report.json | 41 non-improving comparisons found in reports; explain at least one in docs/negative-results.md. |

## Negative-result candidates (auto-detected)

- `benchmarks/results/speculative_decoding/20261010-105841/report.json` → `vllm.draft_model_k4.vs_standard.c1`: regressed (+241.8%)
- `benchmarks/results/speculative_decoding/20261010-105841/report.json` → `vllm.draft_model_k4.vs_standard.c16`: regressed (+245.7%)
- `benchmarks/results/speculative_decoding/20261010-105841/report.json` → `vllm.ngram_k4.vs_standard.c1`: regressed (+94.2%)
- `benchmarks/results/speculative_decoding/20261010-105841/report.json` → `vllm.ngram_k4.vs_standard.c16`: regressed (+98.9%)
- `benchmarks/results/speculative_decoding/20261010-104217/report.json` → `hf.speedup`: regressed (+17.2%)
- `benchmarks/results/kv_offload/20261010-101543/report.json` → `variants.no_cache.vs_hbm_only.ttft_p50`: regressed (+34.1%)
- `benchmarks/results/kv_offload/20261010-101543/report.json` → `variants.no_cache.vs_hbm_only.ttft_p95`: regressed (+6.4%)
- `benchmarks/results/kv_offload/20261010-101543/report.json` → `variants.no_cache.vs_hbm_only.throughput`: regressed (-17.2%)
- `benchmarks/results/baseline/20261010-092402/report.json` → `gateway_overhead.ttft_p50`: no_significant_difference (+6.3%)
- `benchmarks/results/baseline/20261010-092402/report.json` → `gateway_overhead.ttft_p99`: no_significant_difference (+8.0%)
- `benchmarks/results/baseline/20261010-092402/report.json` → `gateway_overhead.e2e_p50`: no_significant_difference (+0.3%)
- `benchmarks/results/baseline/20261010-092402/report.json` → `gateway_overhead.output_tok_per_s`: no_significant_difference (-0.4%)
- `benchmarks/results/kv_pressure/20261010-103719/report.json` → `pressure.3000.vs_largest_cache.ttft_p50`: regressed (+108.5%)
- `benchmarks/results/kv_pressure/20261010-103719/report.json` → `pressure.3000.vs_largest_cache.throughput`: regressed (-29.2%)
- `benchmarks/results/kv_pressure/20261010-103719/report.json` → `pressure.1200.vs_largest_cache.ttft_p50`: regressed (+800.3%)
- `benchmarks/results/kv_pressure/20261010-103719/report.json` → `pressure.1200.vs_largest_cache.throughput`: regressed (-56.1%)
- `benchmarks/results/quantization/20261010-095855/report.json` → `variants.gptq_int8.vs_bf16.c32.throughput`: no_significant_difference (-0.9%)
- `benchmarks/results/quantization/20261010-095855/report.json` → `variants.gptq_int8.vs_bf16.c32.tpot_p50`: no_significant_difference (-5.0%)
- `benchmarks/results/quantization/20261010-095855/report.json` → `variants.fp8.vs_bf16.c32.throughput`: no_significant_difference (-0.4%)
- `benchmarks/results/kv_cache/20261010-093938/report.json` → `reuse.ttft_p50`: no_significant_difference (-34.2%)
- `benchmarks/results/interference/20261010-102752/report.json` → `budgets.512.interference`: regressed (+80.2%)
- `benchmarks/results/interference/20261010-102752/report.json` → `budgets.512.interference_itl_p99`: regressed (+377.6%)
- `benchmarks/results/interference/20261010-102752/report.json` → `budgets.2048.interference`: regressed (+93.6%)
- `benchmarks/results/interference/20261010-102752/report.json` → `budgets.2048.interference_itl_p99`: regressed (+828.9%)
- `benchmarks/results/interference/20261010-102752/report.json` → `budgets.8192.interference`: regressed (+98.3%)
- `benchmarks/results/interference/20261010-102752/report.json` → `budgets.8192.interference_itl_p99`: regressed (+2210.8%)
- `benchmarks/results/interference/20261010-094747/report.json` → `budgets.512.interference`: no_significant_difference (+0.0%)
- `benchmarks/results/interference/20261010-094747/report.json` → `budgets.2048.interference`: no_significant_difference (+0.1%)
- `benchmarks/results/interference/20261010-094747/report.json` → `budgets.8192.interference`: no_significant_difference (+0.1%)
- `benchmarks/results/routing/20261010-092925/report.json` → `runs.prefix_heavy.vs_round_robin.least_outstanding.ttft_p50`: no_significant_difference (+64.5%)
