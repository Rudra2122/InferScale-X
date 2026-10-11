#!/usr/bin/env bash
# Copy evidence (results, test reports, logs, training eval) to Google Drive so a Colab
# disconnect doesn't lose it, and build a zip you can download.
#   bash scripts/save_results.sh /content/drive/MyDrive/InferScale-X-results
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-/content/drive/MyDrive/InferScale-X-results}"
mkdir -p "$DEST"
cd "$ROOT"
python scripts/evidence_report.py >/dev/null 2>&1 || true
STAMP=$(date +%Y%m%d-%H%M%S)
zip -qr "/tmp/inferscale-evidence-$STAMP.zip" benchmarks/results tests/results logs docs/READINESS.md docs/CLAIMS.md \
  requirements/lock-colab*.txt training/runs/eval_report*.json training/runs/*/run_meta.json training/runs/*/log_history.json \
  training/data/lineage.json 2>/dev/null
cp "/tmp/inferscale-evidence-$STAMP.zip" "$DEST/" && echo "saved $DEST/inferscale-evidence-$STAMP.zip"
# rsync-style merge so multiple runtimes (engines, training) accumulate in one place
cp -rn benchmarks/results "$DEST/" 2>/dev/null || true
