# Compatibility matrix

Status values: **tested** (evidence file exists from a run), **attempted** (script ran, outcome
recorded, may have failed), **reference** (config/manifests only, never executed here),
**untested**. Update this table from `benchmarks/results/` after each Colab run — the
"Version" column must come from `env.json` / `lock-colab*.txt`, never from memory.

| Component | Environment | Status in this repo | Version (fill from env.json) | Evidence / notes |
|---|---|---|---|---|
| Gateway (FastAPI) | CPU, Python 3.10–3.13 | tested (CPU suite) | | tests/results/junit-unit.xml |
| vLLM OpenAI server | Colab A100, runtime 1 | untested until notebook 01 runs | | baseline, routing, kv_cache… |
| vLLM prefix-cache `cache_salt` | vLLM V1 | untested | | kv_cache isolation part |
| vLLM `--speculative-config` | vLLM V1 | untested | | speculative_decoding/vllm |
| vLLM KV connectors (NixlConnector / P2pNccl), 1 GPU | Colab | attempted | | disagg_attempt/attempt.json |
| NIXL Python API loopback | Colab | attempted | | kv_transfer report → `nixl` |
| NVIDIA Dynamo (frontend + vLLM worker) | Colab, etcd/NATS as binaries | attempted | | dynamo_attempt/status.json |
| SGLang server | Colab A100, separate runtime | untested | | engines/*-sglang |
| TensorRT-LLM `trtllm-serve` | Colab A100, separate runtime | attempted (experimental) | | engines/*-trtllm |
| Triton Inference Server | needs Docker | reference | | inference/README.md |
| Triton (language) kernels | A100 sm80 | correctness verified on CPU via TRITON_INTERPRET (Triton 3.8, torch 2.14 in the build env); GPU perf untested until run | | triton_kernel, triton_attention |
| vLLM KV CPU offloading (native / OffloadingConnector) | Colab A100 | attempted | | kv_offload |
| LMCache (CPU + localhost remote tier) | Colab A100, `--no-deps` install | attempted | | kv_offload |
| sentence-transformers dense retrieval | any | untested here (no model download in build env) | | retrieval |
| kind 3-node cluster + Calico + metrics-server | Mac (Docker Desktop, kind v0.33.0, k8s v1.37.0) | **tested**: scheduling, recovery, HPA, NetworkPolicy all pass. Docker Desktop did not forward kind's `extraPortMappings` to the Mac host (host→:30900 hung while container→node:30900 worked); worked around with a socat relay container on `127.0.0.1:30901` | kind v0.33.0 | k8s_kind |
| docker-compose stack | Mac (Docker Desktop) | **tested**: gateway + 2 mocks + Prometheus + Grafana; gateway needs ~2 s after `up` before accepting connections (Makefile now waits) | | |
| TRL SFT/DPO/GRPO + PEFT | Colab A100, runtime 2 | untested until notebook 02 | | training/runs |
| Kubernetes GPU manifests / Helm | any K8s + GPU Operator | reference (YAML-validated) | | infrastructure/kubernetes, helm |
| Grove | K8s | reference (not executed) | | infrastructure/grove-experiments |
| DCGM exporter | K8s / bare metal | reference; Colab uses gpu_exporter.py | | |
| NCCL / RDMA / InfiniBand | multi-GPU / multi-node | not applicable on 1 GPU | | capacity-planning (extrapolated) |

Known conflicts: vLLM, SGLang and TensorRT-LLM pin different torch builds → one engine per
Colab runtime. Training (TRL) also runs in its own runtime.
