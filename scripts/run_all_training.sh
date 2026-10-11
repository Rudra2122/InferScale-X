#!/usr/bin/env bash
# Model-customisation track on one A100 (≈1.5-3 h). Run after: bash scripts/colab_setup.sh training
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
STATUS=logs/run_training_status.txt
: > "$STATUS"
run() { local n="$1"; shift; echo "======== $n"; if "$@" 2>&1 | tee "logs/train_$n.log"; then r=OK; else r=FAILED; fi; echo "$n $r" | tee -a "$STATUS"; }
latest() { ls -td training/runs/*-"$1" 2>/dev/null | head -1; }

run data python training/data-curation/build_datasets.py
run sft python training/sft/train_sft.py --max-steps "${SFT_STEPS:-300}"
SFT=$(latest sft)
run dpo python training/dpo/train_dpo.py --max-steps "${DPO_STEPS:-200}"
run dpo_from_sft python training/dpo/train_dpo.py --max-steps "${DPO_STEPS:-200}" --init-adapter "$SFT/adapter"
run grpo python training/grpo/train_grpo.py --reward heuristic --max-steps "${RL_STEPS:-150}"
run rlvr python training/rlvr/train_rlvr.py --max-steps "${RL_STEPS:-150}"
DPO=$(latest dpo); GRPO=$(latest grpo); RLVR=$(latest rlvr); DPOSFT=$(latest dpo-from-sft)
run eval python training/eval_models.py --adapters "base,$SFT/adapter,$DPO/adapter,$GRPO/adapter,$RLVR/adapter"
run eval_chain python training/eval_models.py --adapters "$DPOSFT/adapter" --merge-first "$SFT/adapter" \
    --out training/runs/eval_report_dpo_from_sft.json
cat "$STATUS"
