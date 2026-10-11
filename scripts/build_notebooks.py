"""Generate the Colab notebooks in notebooks/ (keeps them diff-able and in sync with scripts)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SETUP = [
    ("md", "## 0 · Get the code\nUpload `InferScale-X.zip` to your Drive root (or to this session), then run:"),
    ("code", "from google.colab import drive\ndrive.mount('/content/drive')\n"
             "import os\nif not os.path.exists('/content/InferScale-X'):\n"
             "    !cp /content/drive/MyDrive/InferScale-X.zip /content/ 2>/dev/null || echo 'upload InferScale-X.zip to /content'\n"
             "    !cd /content && unzip -q -o InferScale-X.zip\n"
             "%cd /content/InferScale-X\n!nvidia-smi"),
]
SAVE = ("code", "!bash scripts/save_results.sh /content/drive/MyDrive/InferScale-X-results")


def nb(cells):
    out = []
    for kind, src in cells:
        lines = src.splitlines(keepends=True)
        if kind == "md":
            out.append({"cell_type": "markdown", "metadata": {}, "source": lines})
        else:
            out.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines})
    return {"cells": out, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


CORE = [
    ("md", "# InferScale-X · 01 Core platform on one A100\n"
           "Runtime → **A100 GPU**. Every step runs as a subprocess (the kernel never imports torch), writes raw results to "
           "`benchmarks/results/`, and is safe to re-run. Approximate times are for an A100 40GB.\n\n"
           "**Labels you will see in reports:** measured · emulated (single-GPU stand-in) · simulated · extrapolated."),
    *SETUP,
    ("md", "## 1 · Install (fresh runtime, ~6 min) — writes `requirements/lock-colab.txt`"),
    ("code", "!bash scripts/colab_setup.sh core"),
    ("md", "## 2 · CPU test suite → `tests/results/junit-unit.xml` (~2 min)"),
    ("code", "!python -m pytest tests/unit tests/integration/test_gateway_mock.py tests/security "
             "tests/failure_injection/test_failure_mock.py --junitxml tests/results/junit-unit.xml"),
    ("md", "## 3 · Pillar 1 baseline: vLLM direct vs through the gateway (~15 min)"),
    ("code", "!python -m benchmarks.runners.exp_baseline --repeats 3"),
    ("md", "## 4 · Live correctness tests against real vLLM → `junit-live.xml` (~5 min)"),
    ("code", "!python scripts/serverctl.py vllm --name vllm-a --port 8001 --model Qwen/Qwen2.5-1.5B-Instruct --gpu-mem 0.85\n"
             "!python scripts/serverctl.py gateway --backends vllm-a=http://127.0.0.1:8001\n"
             "!INFERSCALE_LIVE_URL=http://127.0.0.1:9000 python -m pytest tests/integration/test_live_endpoint.py "
             "--junitxml tests/results/junit-live.xml\n!python scripts/serverctl.py stopall"),
    SAVE,
    ("md", "## 5 · Pillar 2 routing: round-robin vs least-outstanding vs KV-aware prefix affinity (~40 min)\n"
           "Two vLLM replicas share the A100 (0.42 of memory each). Policy deltas are valid; absolute scale is not 2 GPUs."),
    ("code", "!python -m benchmarks.runners.exp_routing --repeats 3"),
    ("md", "## 6 · Reliability: kill a replica under load, restart it; tenant flood (~10 min)"),
    ("code", "!python -m tests.failure_injection.exp_failure --kind vllm"),
    SAVE,
    ("md", "## 7 · Pillar 4 KV cache: reuse, HBM pressure, cross-tenant isolation (~45 min)"),
    ("code", "!python disaggregation/kv-cache/exp_prefix_cache.py --repeats 3"),
    ("md", "## 8 · Why disaggregate: prefill→decode interference vs chunk budget (~25 min)"),
    ("code", "!python disaggregation/prefill/exp_interference.py --repeats 3"),
    ("md", "## 9 · KV movement: bandwidth, real KV caches, cost model, emulated disaggregation, NIXL attempt (~10 min)\n"
           "Optional first: `!pip install nixl` (the script records whether NIXL works here)."),
    ("code", "import glob, os\ninterf = sorted(glob.glob('benchmarks/results/interference/*/report.json'))\n"
             "arg = f'--interference-report {interf[-1]}' if interf else ''\n"
             "!python disaggregation/nixl-benchmarks/kv_transfer_bench.py --nixl {arg}"),
    ("code", "import glob\nkvt = sorted(glob.glob('benchmarks/results/kv_transfer/*/report.json'))\n"
             "arg = f'--transfer-report {kvt[-1]}' if kvt else ''\n"
             "!python disaggregation/kv-cache/kv_sim.py {arg} --out benchmarks/results/kv_sim_results.json"),
    ("md", "## 9b · REAL multi-tier KV: HBM (capped below the working set) vs + CPU DRAM tier vs + localhost remote tier (~40 min)\n"
           "Tries vLLM native offloading, then OffloadingConnector, then LMCache; records which worked. "
           "LMCache is installed with --no-deps so it cannot replace the torch/vLLM build the other sections use; "
           "if its compiled extension doesn't match, that variant fails and is recorded."),
    ("code", "!pip install -q --no-deps lmcache || echo 'lmcache install failed - native vLLM offloading will still be tried'\n"
             "!python disaggregation/kv-cache/exp_kv_offload.py --repeats 3 --remote"),
    ("md", "## 10 · EXPERIMENTAL: real vLLM prefill/decode split via KV connector on one GPU (~10 min)\nFailure is a valid, recorded result."),
    ("code", "!python disaggregation/decode/vllm_disagg_attempt.py"),
    SAVE,
    ("md", "## 11 · Pillar 3: quantization BF16 vs AWQ-INT4 vs GPTQ-INT8 vs FP8 weight-only (~40 min)"),
    ("code", "!python optimization/quantization/exp_quant.py --repeats 3"),
    ("md", "## 12 · Custom Triton kernel: fused residual-add + RMSNorm (~3 min)"),
    ("code", "!python optimization/triton-kernels/fused_add_rmsnorm.py --profile\n"
             "!which nsys && nsys profile -o benchmarks/results/triton_kernel/nsys_fused --force-overwrite true "
             "python optimization/triton-kernels/fused_add_rmsnorm.py --iters 20 || echo 'nsys not available on this runtime - torch.profiler trace saved instead'"),
    ("md", "## 12b · Custom Triton FlashAttention-style forward vs PyTorch SDPA (~5 min)"),
    ("code", "!python optimization/triton-kernels/flash_attention.py"),
    ("md", "## 13 · Speculative decoding: HF lossless check + vLLM draft/n-gram at concurrency 1 and 16 (~40 min)"),
    ("code", "!python optimization/speculative-decoding/exp_specdec.py --n 30 --vllm --repeats 2"),
    SAVE,
    ("md", "## 14 · Pillar 5 agents: Qwen2.5-7B through the gateway; agent vs RAG; security suites (~20 min)"),
    ("code", "!python scripts/serverctl.py vllm --name vllm-a --port 8001 --model Qwen/Qwen2.5-7B-Instruct --gpu-mem 0.9 --max-len 8192\n"
             "!python scripts/serverctl.py gateway --backends vllm-a=http://127.0.0.1:8001\n"
             "!python agents/eval/run_eval.py --url http://127.0.0.1:9000 --model Qwen/Qwen2.5-7B-Instruct\n"
             "!python scripts/serverctl.py stopall"),
    ("md", "## 14b · Retrieval quality: BM25 vs dense vs hybrid (recall@k, MRR, nDCG) (~3 min)"),
    ("code", "!pip install -q sentence-transformers\n!python agents/eval/retrieval_eval.py"),
    ("md", "## 15 · vLLM entry for the engine comparison (SGLang / TRT-LLM run in notebook 03) (~20 min)"),
    ("code", "!python inference/exp_engines.py --engine vllm --repeats 3"),
    ("md", "## 16 · Optional: live monitoring (Prometheus + GPU exporter) while a load test runs"),
    ("code", "!mkdir -p .bin && cd .bin && [ -x prometheus ] || (wget -q https://github.com/prometheus/prometheus/releases/download/v2.53.2/prometheus-2.53.2.linux-amd64.tar.gz && tar xzf prometheus-2.53.2.linux-amd64.tar.gz && cp prometheus-2.53.2.linux-amd64/prometheus .)\n"
             "!nohup python infrastructure/monitoring/gpu_exporter.py --port 9401 > logs/gpu_exporter.log 2>&1 &\n"
             "!nohup .bin/prometheus --config.file=infrastructure/monitoring/prometheus.yml --storage.tsdb.path=/tmp/prom > logs/prometheus.log 2>&1 &\n"
             "!python scripts/serverctl.py vllm --name vllm-a --port 8001 --model Qwen/Qwen2.5-1.5B-Instruct\n"
             "!python scripts/serverctl.py gateway --backends vllm-a=http://127.0.0.1:8001\n"
             "!python -m benchmarks.runners.loadgen --url http://127.0.0.1:9000 --api-key sk-dev-acme --workload benchmarks/workloads/multitenant.yaml "
             "--tenant-keys acme=sk-dev-acme,globex=sk-dev-globex,freetier=sk-dev-free --out benchmarks/results/multitenant --repeats 1\n"
             "!curl -s 'http://127.0.0.1:9090/api/v1/query?query=histogram_quantile(0.95,sum(rate(inferscale_ttft_seconds_bucket[5m]))by(le))' | head -c 600\n"
             "!python scripts/serverctl.py stopall; pkill -f prometheus; pkill -f gpu_exporter"),
    ("md", "## 17 · Plots, readiness checklist, JD coverage, claims, results page, capacity model, save"),
    ("code", "!python benchmarks/analysis/make_plots.py\n!python scripts/evidence_report.py\n!python scripts/publish_results.py\n"
             "!cat docs/CLAIMS.md\n!head -30 docs/jd-coverage.md"),
    ("code", "import glob\nbase = sorted(glob.glob('benchmarks/results/baseline/*/direct/aggregate.json'))\n"
             "if base:\n    !python docs/capacity-planning/capacity_model.py --preset qwen2.5-1.5b --gpu a100-40gb --measured {base[-1]} --rps 50 --in-tokens 1000 --out-tokens 200"),
    ("code", "import glob\nfrom IPython.display import Image, display\nfor p in sorted(glob.glob('benchmarks/results/plots/*.png')):\n    print(p); display(Image(p))"),
    SAVE,
    ("md", "## 18 · Optional: live public demo endpoint (real model on this A100, fresh API key, Cloudflare quick tunnel)\n"
           "Prints a temporary HTTPS URL + key you can hit from your Mac or share for a demo. Stop it when done."),
    ("code", "!bash scripts/colab_public_endpoint.sh Qwen/Qwen2.5-1.5B-Instruct"),
    ("code", "!bash scripts/colab_public_endpoint.sh stop"),
]

TRAIN = [
    ("md", "# InferScale-X · 02 Model customisation (SFT · DPO · GRPO · RLVR) on one A100\n"
           "Use a **fresh runtime** (TRL's torch pin differs from vLLM's). LoRA on Qwen2.5-0.5B-Instruct; "
           "verifiable arithmetic data with lineage and a leakage check. ~1.5–3 h total."),
    *SETUP,
    ("code", "!bash scripts/colab_setup.sh training"),
    ("md", "## Data curation → `training/data/lineage.json`"),
    ("code", "!python training/data-curation/build_datasets.py && cat training/data/lineage.json"),
    ("md", "## SFT (~15 min)"),
    ("code", "!python training/sft/train_sft.py --max-steps 300"),
    ("md", "## DPO from base, and SFT→DPO (~20 min each)"),
    ("code", "import glob\nsft = sorted(glob.glob('training/runs/*-sft'))[-1]\nprint(sft)\n"
             "!python training/dpo/train_dpo.py --max-steps 200\n"
             "!python training/dpo/train_dpo.py --max-steps 200 --init-adapter {sft}/adapter"),
    ("md", "## GRPO with a heuristic (format) reward vs RLVR with a verifiable reward — identical hyper-parameters (~30 min each)"),
    ("code", "!python training/grpo/train_grpo.py --reward heuristic --max-steps 150\n!python training/rlvr/train_rlvr.py --max-steps 150"),
    ("md", "## Held-out evaluation: accuracy, format compliance, output length (= serving cost)"),
    ("code", "last = lambda pat: sorted(glob.glob(f'training/runs/*-{pat}'))[-1]\n"
             "ads = ','.join(['base', last('sft')+'/adapter', last('dpo')+'/adapter', last('grpo')+'/adapter', last('rlvr')+'/adapter'])\n"
             "!python training/eval_models.py --adapters {ads}\n"
             "chain = last('dpo-from-sft')\n"
             "!python training/eval_models.py --adapters {chain}/adapter --merge-first {last('sft')}/adapter --out training/runs/eval_report_dpo_from_sft.json"),
    ("code", "import json\nr = json.load(open('training/runs/eval_report.json'))\n"
             "for k, v in r['results'].items():\n    print(f\"{k[-40:]:40s} acc={v['accuracy']:.3f} fmt={v['format_rate']:.3f} out_tok={v['mean_output_tokens']:.1f} d_acc={v.get('delta_accuracy_vs_base', 0):+.3f}\")"),
    SAVE,
]

ENGINES = [
    ("md", "# InferScale-X · 03 Alternative engines + NVIDIA Dynamo attempt\n"
           "Each section needs a **fresh runtime** (Runtime → Disconnect and delete runtime), then re-run section 0. "
           "Results accumulate in Drive via `save_results.sh`; compare at the end."),
    *SETUP,
    ("md", "## A · SGLang"),
    ("code", "!bash scripts/colab_setup.sh sglang\n!python inference/exp_engines.py --engine sglang --repeats 3"),
    SAVE,
    ("md", "## B · TensorRT-LLM (experimental; fresh runtime)"),
    ("code", "!bash scripts/colab_setup.sh trtllm\n!python inference/exp_engines.py --engine trtllm --repeats 3"),
    SAVE,
    ("md", "## C · NVIDIA Dynamo single-node attempt (fresh runtime). `status.json` says whether it actually served."),
    ("code", "!pip install -q -r requirements/dev.txt\n!bash inference/dynamo/try_dynamo.sh"),
    SAVE,
    ("md", "## D · Compare (after copying every runtime's results back from Drive)"),
    ("code", "!cp -rn /content/drive/MyDrive/InferScale-X-results/results/* benchmarks/results/ 2>/dev/null; "
             "python benchmarks/analysis/compare_engines.py"),
]


def main():
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    for name, cells in (("01_core_platform_A100.ipynb", CORE), ("02_model_customization_A100.ipynb", TRAIN),
                        ("03_alt_engines_and_dynamo.ipynb", ENGINES)):
        (out / name).write_text(json.dumps(nb(cells), indent=1))
        print("wrote", out / name)


if __name__ == "__main__":
    main()
