# Capacity planning and the large-cluster reference design

## What is measured vs. assumed
| Quantity | Source | Label |
|---|---|---|
| KV bytes / token | model config (exact arithmetic) | computed |
| output tok/s per replica | `loadgen` on the A100 at your target workload | **measured** |
| KV transfer α, β on PCIe | `kv_transfer_bench.py` | **measured** |
| NVLink / InfiniBand bandwidth | vendor spec | assumed |
| Scaling efficiency beyond 1 GPU | parameter (default 0.9) | **EXTRAPOLATED** |
| $/GPU-hour | your provider | assumed |

```bash
python docs/capacity-planning/capacity_model.py --preset qwen2.5-1.5b --gpu a100-40gb \
  --measured benchmarks/results/baseline/<run>/direct/aggregate.json --rps 50 --in-tokens 1000 --out-tokens 200
```
Worked example of the KV arithmetic (exact): Qwen2.5-1.5B has 28 layers, 2 KV heads, head dim
128 → 2 × 28 × 2 × 128 × 2 B = **28,672 B/token**; a 4K-token context is ≈ 112 MiB of KV.

## Large-cluster reference design (NOT tested — design reasoning only)
- **Unit of scale:** a rack/NVLink domain (e.g. 8 GPUs). Tensor parallelism stays inside it;
  data-parallel replicas and prefill/decode pools span racks.
- **Prefill vs decode pools:** size from measured ΔTPOT (interference) and KV bytes/token;
  prefill pool is compute-bound, decode pool bandwidth/KV-capacity bound.
- **KV transfer fabric:** NIXL over RDMA (IB/RoCE). Per-request transfer = KV bytes / link
  bandwidth (+α). With the break-even table, choose which traffic classes disaggregate
  (long prompts, long outputs) and which stay colocated.
- **Failure domains:** a GPU/node failure loses in-flight streams on that worker (measured on
  1 GPU: in-flight streams on a killed replica fail; new requests reroute). At scale: spread
  replicas across racks, PodDisruptionBudgets, drain-before-maintenance, autoscale on queue depth.
- **Routing:** KV-aware routing per model shard group; fall back to least-loaded under
  imbalance (same rule as `prefix_affinity` here).
- **Operational cost:** two pools double the capacity-planning surface and add a transfer path
  that can degrade independently (alert on transfer latency p99).
- **Bottlenecks to expect:** KV capacity (preemptions), cross-rack bandwidth for KV transfer,
  router hot-spotting on popular prefixes, model-load time during scale-up (startup measured
  here: `vllm_ready_after_s` in the baseline report).

None of the above has been run on more than one GPU. Do not quote cluster-scale numbers as results.
