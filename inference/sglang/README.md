# SGLang
Fresh Colab runtime: `bash scripts/colab_setup.sh sglang && python inference/exp_engines.py --engine sglang`.
Launch command built in `scripts/serverctl.py sglang_cmd` (`--mem-fraction-static`, `--context-length`).
RadixAttention prefix caching is on by default, so compare its `prefix_heavy` result with vLLM's cache-on run.
