# Partner deployment guide

Audience: a customer platform team deploying the InferScale-X pattern on their own GPUs.

## 1. Pre-flight
- GPUs with NVIDIA driver + container toolkit, Kubernetes ≥ 1.28 with the NVIDIA GPU Operator.
- Decide per model: GPU type, tensor-parallel size, max context. Run `capacity_model.py`
  with throughput **measured on their hardware** (step 3), not ours.
- Network: engines reachable only from gateways (NetworkPolicy provided).

## 2. Install (reference — validate in a staging cluster first)
```bash
helm install isx infrastructure/helm/inferscale \
  --set vllm.model=Qwen/Qwen2.5-7B-Instruct --set vllm.replicas=2 \
  --set-json 'gateway.tenants={"acme":{"api_key_sha256":"<sha256>","max_concurrent":64,"requests_per_minute":600,"burst":20}}'
kubectl -n default rollout status statefulset/isx-vllm
```
Pin image digests; record them in `docs/compatibility-matrix.md`.

## 3. Acceptance on the customer's hardware (same scripts as ours)
1. Smoke + correctness: `INFERSCALE_LIVE_URL=<gateway> pytest tests/integration/test_live_endpoint.py`
2. Baseline load at their traffic shape: copy a workload YAML, set token lengths and tenants, run `loadgen` with `--repeats 3`.
3. Failure drill: delete one vLLM pod under load; compare with `docs/incident-runbooks/backend-down.md` expectations.
4. Isolation: run the KV isolation part against their gateway.

## 4. Sizing conversation (template)
"At your mix of X req/s, Y input / Z output tokens, one replica sustains T tok/s (measured,
CI ±c). With 30% headroom you need N replicas = N×TP GPUs ≈ $M/month; cost per 1M output
tokens ≈ $C. Beyond N GPUs the numbers are extrapolated at 90% scaling efficiency."

## 5. Troubleshooting index
| Symptom | First check | Runbook |
|---|---|---|
| 503s from gateway | `/admin/backends`, engine `/health` | backend-down.md |
| Rising TTFT, normal throughput | queue (`vllm:num_requests_waiting`), prefix hit rate | latency-regression.md |
| 429 spikes for one tenant | tenant limits vs traffic | overload.md |
| Throughput collapse at long contexts | `vllm:num_preemptions_total` | kv-pressure.md |
