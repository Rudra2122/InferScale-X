#!/usr/bin/env bash
# Publish the repo (with your evidence) to GitHub. Run on your Mac after copying results back from Drive.
#   bash scripts/publish_repo.sh [repo-name] [--private]
# Needs: git, gh (brew install gh; gh auth login).
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="${1:-InferScale-X}"
VIS="--public"; [ "${2:-}" = "--private" ] && VIS="--private"

python -m pytest -q                       # never publish a red tree
python scripts/evidence_report.py
python benchmarks/analysis/make_plots.py || true
python scripts/publish_results.py

# guard: refuse to publish if a generated public-endpoint config (fresh secret) is present and tracked
if git ls-files --error-unmatch configs/gateway.public.yaml >/dev/null 2>&1; then
  echo "configs/gateway.public.yaml is tracked - remove it first"; exit 1
fi
# large raw files: keep JSONL under 50 MB each (GitHub hard limit is 100 MB)
find benchmarks/results -name "*.jsonl" -size +50M -print -exec gzip {} \;

[ -d .git ] || git init -b main
git add -A
git commit -m "InferScale-X: code + evidence $(date +%Y-%m-%d)" || echo "nothing to commit"
if ! git remote get-url origin >/dev/null 2>&1; then
  gh repo create "$NAME" $VIS --source . --remote origin --description \
    "Evidence-first LLM inference reference platform: vLLM gateway, KV-aware routing, KV tiers, Triton kernels, secure agents, SFT/DPO/GRPO/RLVR - single A100 + kind"
fi
git push -u origin main
echo "Published. Pin it on your profile; docs/RESULTS.md is the landing page for reviewers."
