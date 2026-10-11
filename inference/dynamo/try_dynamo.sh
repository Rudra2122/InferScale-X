#!/usr/bin/env bash
# EXPERIMENTAL: attempt NVIDIA Dynamo (single node, single A100) on Colab.
#
# Dynamo's control plane normally uses etcd + NATS (docker compose). Colab has no
# Docker, so this script downloads the static etcd and nats-server binaries and
# runs them as plain processes, then tries the Dynamo frontend + a vLLM worker.
#
# Package names / CLI entry points have changed across Dynamo releases. Every step's
# outcome is written to benchmarks/results/dynamo_attempt/attempt.log and
# status.json. A failed attempt is a valid, documentable result - do NOT claim
# Dynamo works unless status.json says "frontend_serving": true.
#
# Run in a FRESH Colab runtime:   bash inference/dynamo/try_dynamo.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/benchmarks/results/dynamo_attempt/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT" "$ROOT/.bin"
LOG="$OUT/attempt.log"
MODEL="${INFERSCALE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
ETCD_VER="${ETCD_VER:-v3.5.17}"
NATS_VER="${NATS_VER:-v2.10.24}"
declare -A S

step() { echo "=== $1" | tee -a "$LOG"; }
record() { S[$1]=$2; }

step "install ai-dynamo[vllm]"
if pip install -q "ai-dynamo[vllm]" >>"$LOG" 2>&1; then record pip_install true; else record pip_install false; fi
pip show ai-dynamo >>"$LOG" 2>&1 || true

step "fetch etcd $ETCD_VER and nats-server $NATS_VER"
cd "$ROOT/.bin"
if [ ! -x etcd ]; then
  curl -sSL "https://github.com/etcd-io/etcd/releases/download/${ETCD_VER}/etcd-${ETCD_VER}-linux-amd64.tar.gz" -o etcd.tgz >>"$LOG" 2>&1 \
    && tar xzf etcd.tgz && cp "etcd-${ETCD_VER}-linux-amd64/etcd" . && record etcd_download true || record etcd_download false
fi
if [ ! -x nats-server ]; then
  curl -sSL "https://github.com/nats-io/nats-server/releases/download/${NATS_VER}/nats-server-${NATS_VER}-linux-amd64.tar.gz" -o nats.tgz >>"$LOG" 2>&1 \
    && tar xzf nats.tgz && cp "nats-server-${NATS_VER}-linux-amd64/nats-server" . && record nats_download true || record nats_download false
fi
cd "$ROOT"

step "start etcd + nats"
nohup "$ROOT/.bin/etcd" --data-dir /tmp/etcd-data --listen-client-urls http://127.0.0.1:2379 \
  --advertise-client-urls http://127.0.0.1:2379 >"$OUT/etcd.log" 2>&1 &
nohup "$ROOT/.bin/nats-server" -js -p 4222 >"$OUT/nats.log" 2>&1 &
sleep 5
curl -s http://127.0.0.1:2379/health >>"$LOG" 2>&1 && record etcd_up true || record etcd_up false
export ETCD_ENDPOINTS=http://127.0.0.1:2379 NATS_SERVER=nats://127.0.0.1:4222

step "start dynamo frontend (python -m dynamo.frontend)"
nohup python -m dynamo.frontend --http-port 8000 >"$OUT/frontend.log" 2>&1 &
sleep 10
step "start dynamo vllm worker (python -m dynamo.vllm)"
nohup python -m dynamo.vllm --model "$MODEL" --gpu-memory-utilization 0.85 >"$OUT/worker.log" 2>&1 &

record frontend_serving false
for i in $(seq 1 180); do
  if curl -sf http://127.0.0.1:8000/v1/models | grep -q "$MODEL"; then record frontend_serving true; break; fi
  sleep 5
done

if [ "${S[frontend_serving]}" = "true" ]; then
  step "smoke request through Dynamo"
  curl -s http://127.0.0.1:8000/v1/chat/completions -H 'content-type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi\"}],\"max_tokens\":16}" \
    >"$OUT/smoke.json" 2>>"$LOG" && record smoke_ok true || record smoke_ok false
  step "benchmark through Dynamo (same workload as vLLM baseline)"
  python -m benchmarks.runners.loadgen --url http://127.0.0.1:8000 --workload benchmarks/workloads/chat_short.yaml \
    --model "$MODEL" --out "$OUT/bench" --repeats 2 >>"$LOG" 2>&1 && record bench_ok true || record bench_ok false
fi

{
  echo "{"
  first=1
  for k in "${!S[@]}"; do
    [ $first -eq 0 ] && echo ","
    printf '  "%s": %s' "$k" "${S[$k]}"
    first=0
  done
  printf ',\n  "model": "%s",\n  "label": "EXPERIMENTAL single-node Dynamo attempt on Colab"\n}\n' "$MODEL"
} >"$OUT/status.json"
tail -n 40 "$OUT/frontend.log" "$OUT/worker.log" >>"$LOG" 2>/dev/null
cat "$OUT/status.json"
echo "Logs: $OUT"
pkill -f dynamo.frontend; pkill -f dynamo.vllm; pkill -f nats-server; pkill -f "/etcd"
