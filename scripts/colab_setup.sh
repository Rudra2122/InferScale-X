#!/usr/bin/env bash
# Install a dependency profile on Colab, capture the environment, and write a lockfile.
#   bash scripts/colab_setup.sh core       # vLLM + gateway + HF (notebook 01)
#   bash scripts/colab_setup.sh training   # TRL/PEFT (notebook 02)
#   bash scripts/colab_setup.sh sglang     # SGLang (notebook 03, fresh runtime)
#   bash scripts/colab_setup.sh trtllm     # TensorRT-LLM (notebook 03, fresh runtime, experimental)
set -euo pipefail
PROFILE="${1:-core}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || { echo "No NVIDIA GPU visible"; exit 1; }
if [ -f "requirements/lock-colab-$PROFILE.txt" ] && [ "${USE_LOCK:-0}" = "1" ]; then
  echo ">> installing from lockfile requirements/lock-colab-$PROFILE.txt"
  pip install -q -r "requirements/lock-colab-$PROFILE.txt"
else
  if [ "$PROFILE" = "sglang" ] && ! command -v cargo >/dev/null; then
    # sglang pins outlines==0.1.11 -> outlines_core==0.1.26, which has no Python 3.13 wheel (Colab's Python),
    # so pip builds it from source and that needs a Rust toolchain.
    curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null
  fi
  export PATH="$HOME/.cargo/bin:$PATH"
  pip install -q -r "requirements/colab-$PROFILE.txt"
fi
if [ "$PROFILE" = "training" ]; then
  # Colab ships torchao 0.10; peft >= 0.18 refuses to build LoRA layers if any torchao < 0.16 is importable.
  # LoRA here doesn't use torchao, so remove it rather than chase a torch-compatible newer build.
  pip uninstall -y -q torchao || true
fi
mkdir -p benchmarks/results tests/results logs
pip freeze > "requirements/lock-colab-$PROFILE.txt"
[ "$PROFILE" = "core" ] && cp requirements/lock-colab-core.txt requirements/lock-colab.txt
python -m benchmarks.runners.env_capture > "benchmarks/results/env-$PROFILE.json"
echo ">> installed profile '$PROFILE'. Key versions:"
python - <<'EOF'
import importlib.metadata as md
for p in ("torch", "vllm", "transformers", "triton", "trl", "peft", "sglang", "tensorrt_llm"):
    try:
        print(f"   {p}=={md.version(p)}")
    except md.PackageNotFoundError:
        pass
EOF
