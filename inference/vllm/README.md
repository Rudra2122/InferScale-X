# vLLM (primary backend)
Started by `scripts/serverctl.py vllm` / `benchmarks/runners/common.start_vllm`. Defaults: Qwen2.5-1.5B-Instruct,
gpu_memory_utilization 0.85 (0.42 per replica when two share the A100), max_model_len 4096.
Experiments that use it: baseline, routing, failure, kv_cache, interference, quantization, specdec, agents, engines.
