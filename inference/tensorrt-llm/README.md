# TensorRT-LLM (experimental on Colab)
Fresh runtime: `bash scripts/colab_setup.sh trtllm && python inference/exp_engines.py --engine trtllm`.
Uses `trtllm-serve <hf-model>` (PyTorch backend in recent releases, no manual engine build). The wheel is large and
tied to specific CUDA/driver versions; if install or startup fails, `engines/*-trtllm/report.json` records the log
tail — that is the result to report (status "attempted" in docs/compatibility-matrix.md).
`ignore_eos` is disabled for this engine, so its output lengths may differ: compare tok/s, not e2e latency.
