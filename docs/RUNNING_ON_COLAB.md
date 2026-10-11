# Running InferScale-X on a Colab A100

## 0. Before you start
- Runtime → Change runtime type → **A100 GPU** (Colab Pro/Pro+ or compute units). Check `nvidia-smi` says A100 (40 GB or 80 GB both work; defaults assume 40 GB).
- Budget: core notebook ≈ 3–5 h, training ≈ 1.5–3 h, engines/Dynamo ≈ 1–2 h each. Colab can disconnect; every section saves to Drive so you can resume.
- Models (all ungated on Hugging Face, no token needed): Qwen2.5-0.5B/1.5B/7B-Instruct, plus official AWQ and GPTQ-Int8 variants of 1.5B. ≈ 25 GB of downloads total.

## 1. Get the code into Colab
```python
from google.colab import drive; drive.mount('/content/drive')
!cp /content/drive/MyDrive/InferScale-X.zip /content/ && cd /content && unzip -q -o InferScale-X.zip
%cd /content/InferScale-X
```

## 2. Why every experiment is a `!python ...` subprocess
vLLM, SGLang and TRT-LLM each pin different torch/CUDA builds. The notebook kernel never
imports torch, so installing an engine never fights the kernel's already-imported libraries,
and a crashed experiment never kills the notebook. Servers are started/stopped by
`scripts/serverctl.py` (logs in `logs/<name>.log`).

## 3. Order of work (notebook 01)

| Step | Command | Time | Produces |
|---|---|---|---|
| setup | `bash scripts/colab_setup.sh core` | 5–8 min | lockfile, env.json |
| CPU tests | `pytest ... --junitxml tests/results/junit-unit.xml` | 1 min | correctness evidence (unit) |
| baseline | `python -m benchmarks.runners.exp_baseline --repeats 3` | 15 min | gateway overhead |
| live correctness | vLLM + gateway + `tests/integration/test_live_endpoint.py` | 5 min | junit-live.xml |
| routing | `python -m benchmarks.runners.exp_routing --repeats 3` | 40 min | policy comparison |
| failure | `python -m tests.failure_injection.exp_failure --kind vllm` | 10 min | crash/overload report |
| KV cache | `python disaggregation/kv-cache/exp_prefix_cache.py --repeats 3` | 45 min | reuse / pressure / isolation |
| interference | `python disaggregation/prefill/exp_interference.py` | 25 min | ΔTPOT per chunk budget |
| KV transfer | `python disaggregation/nixl-benchmarks/kv_transfer_bench.py --nixl --interference-report …` | 10 min | cost model + break-even |
| disagg attempt | `python disaggregation/decode/vllm_disagg_attempt.py` | 10 min | works / doesn't (both are results) |
| quantisation | `python optimization/quantization/exp_quant.py` | 40 min | memory/speed/quality |
| Triton kernels | `fused_add_rmsnorm.py --profile`, `flash_attention.py` | 8 min | correctness + GB/s / TFLOP/s + trace |
| retrieval eval | `python agents/eval/retrieval_eval.py` | 3 min | recall@k / MRR / nDCG, BM25 vs dense vs hybrid |
| KV tiers (last) | `python disaggregation/kv-cache/exp_kv_offload.py --remote` | 40 min | HBM vs +CPU vs +remote tier |
| spec decoding | `python optimization/speculative-decoding/exp_specdec.py --vllm` | 40 min | speedup, acceptance, lossless check |
| agents | 7B vLLM + gateway + `python agents/eval/run_eval.py` | 20 min | task success + security |
| report | `make_plots.py`, `evidence_report.py`, `publish_results.py`, `save_results.sh` | 1 min | plots, READINESS.md, jd-coverage.md, CLAIMS.md, RESULTS.md |
| demo (optional) | `bash scripts/colab_public_endpoint.sh` | 2 min | temporary public HTTPS endpoint + fresh key |

`bash scripts/run_all_core.sh 3` runs all of the above unattended and logs OK/FAILED per step.

## 4. Things that commonly go wrong
- **vLLM flag changed** (e.g. `--enable-prompt-tokens-details`, `--no-enable-prefix-caching`): the server log tail is printed. Edit the `extra=` string in that experiment; record the change in `docs/compatibility-matrix.md`.
- **OOM starting a second replica**: lower `--gpu-mem` (routing uses 0.42 each on 40 GB).
- **`cache_salt` rejected**: your vLLM predates it; Part C of the KV experiment records the error. That is a compatibility finding, not something to hide.
- **Colab disconnect**: re-mount Drive, `cd`, and re-run from the failed step; results already saved stay in `benchmarks/results/`.

## 5. After the run
Open `docs/READINESS.md` (generated) and `docs/CLAIMS.md`. Fill in `docs/negative-results.md`
using the auto-detected candidates, then commit results (or the evidence zip) to the repo.
