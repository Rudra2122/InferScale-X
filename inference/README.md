# Inference engines

Each engine is a separate configuration tested independently (ADR-001); none is treated as
interchangeable. All are driven through the same OpenAI-compatible protocol so the same
`loadgen` workloads compare them.

| Dir | How it runs here | Launch |
|---|---|---|
| `vllm/` | Colab runtime 1 (primary backend) | `python scripts/serverctl.py vllm --name vllm-a --port 8001 --model Qwen/Qwen2.5-1.5B-Instruct` |
| `sglang/` | Colab, **separate runtime** | `bash scripts/colab_setup.sh sglang && python inference/exp_engines.py --engine sglang` |
| `tensorrt-llm/` | Colab, separate runtime, experimental | `bash scripts/colab_setup.sh trtllm && python inference/exp_engines.py --engine trtllm` |
| `triton-server/` | **reference only** — needs Docker | see below |
| `dynamo/` | Colab attempt with etcd/NATS binaries | `bash inference/dynamo/try_dynamo.sh` |
| `mock/` | CPU fake for CI/laptop | `python scripts/serverctl.py mock --name m --port 8001` |

## vLLM flags used by experiments
`--enable-prompt-tokens-details` (cached-token counts in `usage`), `--enable-prefix-caching` /
`--no-enable-prefix-caching`, `--max-num-batched-tokens` (chunked-prefill budget),
`--speculative-config`, `--quantization fp8`, `--kv-transfer-config`. `VLLM_SERVER_DEV_MODE=1`
is set so experiments can call `/reset_prefix_cache` between runs. Flags evolve between
releases — if one is rejected, the server log tail is printed and the change belongs in
`docs/compatibility-matrix.md`.

## Triton Inference Server (reference)
Not runnable on Colab (requires the NGC container). Reference model repository for its vLLM
backend, for a partner environment with Docker:

```
model_repository/
└── qwen25_1p5b/
    ├── 1/model.json      {"model": "Qwen/Qwen2.5-1.5B-Instruct", "gpu_memory_utilization": 0.85,
    │                      "enable_prefix_caching": true}
    └── config.pbtxt      backend: "vllm"
                          instance_group [{ count: 1, kind: KIND_MODEL }]
```
`docker run --gpus all -p 8000:8000 -v $PWD/model_repository:/models nvcr.io/nvidia/tritonserver:<xx.yy>-vllm-python-py3 tritonserver --model-repository=/models`
Triton's OpenAI-compatible frontend is a separate component; without it, `loadgen` cannot
target Triton directly. Status: **reference**.
