#!/usr/bin/env bash
# Expose the gateway (+ real vLLM on the Colab A100) on a temporary public HTTPS URL for a live demo.
# Uses a Cloudflare quick tunnel (no account). The URL dies when the runtime stops.
#
#   bash scripts/colab_public_endpoint.sh [model]
#
# Security: a fresh random API key is generated for this session; the repo's dev keys are NOT
# accepted (they are public). Rate limits are low on purpose. Stop with: bash scripts/colab_public_endpoint.sh stop
set -euo pipefail
cd "$(dirname "$0")/.."
if [ "${1:-}" = "stop" ]; then
  pkill -f "cloudflared tunnel" || true; python scripts/serverctl.py stopall; exit 0
fi
MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
mkdir -p .bin logs
if [ ! -x .bin/cloudflared ]; then
  curl -sSL -o .bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x .bin/cloudflared
fi
KEY="sk-demo-$(python -c 'import secrets;print(secrets.token_urlsafe(24))')"
python - "$KEY" "$MODEL" <<'EOF'
import hashlib, sys, yaml
key, model = sys.argv[1], sys.argv[2]
cfg = {"gateway": {"routing_policy": "least_outstanding", "global_max_inflight": 64, "inject_cache_salt": True,
                   "cache_salt_secret": hashlib.sha256(key.encode()).hexdigest()},
       "backends": [{"name": "vllm-a", "url": "http://127.0.0.1:8001", "model": model}],
       "tenants": {"demo": {"api_key_sha256": hashlib.sha256(key.encode()).hexdigest(), "requests_per_minute": 120,
                            "burst": 10, "max_concurrent": 8, "max_output_tokens": 512}}}
open("configs/gateway.public.yaml", "w").write(yaml.safe_dump(cfg))
EOF
python scripts/serverctl.py vllm --name vllm-a --port 8001 --model "$MODEL" --gpu-mem 0.85
python scripts/serverctl.py gateway --config configs/gateway.public.yaml --port 9000
nohup .bin/cloudflared tunnel --no-autoupdate --url http://127.0.0.1:9000 > logs/cloudflared.log 2>&1 &
URL=""
for i in $(seq 1 60); do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' logs/cloudflared.log | head -1 || true)
  [ -n "$URL" ] && break; sleep 1
done
[ -z "$URL" ] && { echo "tunnel did not come up; see logs/cloudflared.log"; exit 1; }
cat <<EOM

Public endpoint (ephemeral): $URL
API key (this session only):  $KEY

curl $URL/v1/chat/completions -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \\
  -d '{"model":"$MODEL","messages":[{"role":"user","content":"Hello from InferScale-X"}],"max_tokens":64}'

From your Mac:  python -m benchmarks.runners.loadgen --url $URL --api-key $KEY \\
  --workload benchmarks/workloads/smoke.yaml --model $MODEL --out benchmarks/results/remote_smoke
Stop:  bash scripts/colab_public_endpoint.sh stop
EOM
