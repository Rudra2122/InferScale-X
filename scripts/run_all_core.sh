#!/usr/bin/env bash
# Run every core A100 experiment in sequence (≈3-5 h on a Colab A100 40GB).
# Each step is independent; a failure is logged and the script moves on, so one broken
# experiment never hides the others. Results land in benchmarks/results/.
#   bash scripts/run_all_core.sh [repeats]
set -uo pipefail
cd "$(dirname "$0")/.."
REPEATS="${1:-3}"
mkdir -p tests/results logs
STATUS=logs/run_all_status.txt
: > "$STATUS"
run() {
  local name="$1"; shift
  echo "================ $name ================"
  local t0=$(date +%s)
  if "$@" 2>&1 | tee "logs/run_$name.log"; then r=OK; else r=FAILED; fi
  python scripts/serverctl.py stopall >/dev/null 2>&1
  echo "$name $r $(( $(date +%s) - t0 ))s" | tee -a "$STATUS"
}

run unit_tests python -m pytest tests/unit tests/integration/test_gateway_mock.py tests/security \
    tests/failure_injection/test_failure_mock.py --junitxml tests/results/junit-unit.xml
run baseline python -m benchmarks.runners.exp_baseline --repeats "$REPEATS"
run live_correctness bash -c '
  python scripts/serverctl.py vllm --name vllm-a --port 8001 --model Qwen/Qwen2.5-1.5B-Instruct --gpu-mem 0.85 &&
  python scripts/serverctl.py gateway --backends vllm-a=http://127.0.0.1:8001 &&
  INFERSCALE_LIVE_URL=http://127.0.0.1:9000 python -m pytest tests/integration/test_live_endpoint.py \
    --junitxml tests/results/junit-live.xml'
run routing python -m benchmarks.runners.exp_routing --repeats "$REPEATS"
run failure python -m tests.failure_injection.exp_failure --kind vllm
run kv_cache python disaggregation/kv-cache/exp_prefix_cache.py --repeats "$REPEATS"
run interference python disaggregation/prefill/exp_interference.py --repeats "$REPEATS"
INTERF=$(ls -td benchmarks/results/interference/*/ 2>/dev/null | head -1)
run kv_transfer python disaggregation/nixl-benchmarks/kv_transfer_bench.py --nixl \
    ${INTERF:+--interference-report ${INTERF}report.json}
KVT=$(ls -td benchmarks/results/kv_transfer/*/ 2>/dev/null | head -1)
run kv_sim python disaggregation/kv-cache/kv_sim.py ${KVT:+--transfer-report ${KVT}report.json} \
    --out benchmarks/results/kv_sim_results.json
run disagg_attempt python disaggregation/decode/vllm_disagg_attempt.py
run quantization python optimization/quantization/exp_quant.py --repeats "$REPEATS"
run triton_kernel python optimization/triton-kernels/fused_add_rmsnorm.py --profile
run triton_attention python optimization/triton-kernels/flash_attention.py
run retrieval bash -c 'pip install -q sentence-transformers || true; python agents/eval/retrieval_eval.py'
run specdec python optimization/speculative-decoding/exp_specdec.py --n 30 --vllm --repeats 2
run agents bash -c '
  python scripts/serverctl.py vllm --name vllm-a --port 8001 --model Qwen/Qwen2.5-7B-Instruct --gpu-mem 0.9 --max-len 8192 &&
  python scripts/serverctl.py gateway --backends vllm-a=http://127.0.0.1:8001 &&
  python agents/eval/run_eval.py --url http://127.0.0.1:9000 --model Qwen/Qwen2.5-7B-Instruct'
run engines_vllm python inference/exp_engines.py --engine vllm --repeats "$REPEATS"
# last GPU step: installing LMCache must never disturb the vLLM/torch stack used above (--no-deps)
run kv_offload bash -c 'pip install -q --no-deps lmcache || true; python disaggregation/kv-cache/exp_kv_offload.py --repeats "$0" --remote' "$REPEATS"
run plots python benchmarks/analysis/make_plots.py
run evidence python scripts/evidence_report.py
run results_page python scripts/publish_results.py
echo; echo "Summary:"; cat "$STATUS"
