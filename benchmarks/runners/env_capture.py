"""Capture environment metadata next to every result (hardware, drivers, versions)."""
from __future__ import annotations

import importlib.metadata as md
import json
import os
import platform
import subprocess
import sys
import time

PKGS = ["torch", "vllm", "sglang", "tensorrt_llm", "tensorrt-llm", "transformers", "triton", "trl", "peft",
        "accelerate", "flashinfer-python", "xformers", "fastapi", "httpx", "numpy", "nixl", "ai-dynamo",
        "bitsandbytes", "autoawq"]


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:
        return None


def gpu_info() -> list[dict]:
    q = "name,driver_version,memory.total,pcie.link.gen.current,pcie.link.width.current,clocks.max.sm,power.limit"
    out = _run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"])
    if not out:
        return []
    keys = q.split(",")
    return [dict(zip(keys, [x.strip() for x in line.split(",")])) for line in out.splitlines()]


def capture(extra: dict | None = None) -> dict:
    versions = {}
    for p in PKGS:
        try:
            versions[p] = md.version(p)
        except md.PackageNotFoundError:
            pass
    cuda = None
    try:
        import torch  # noqa

        cuda = {"torch_cuda": torch.version.cuda, "available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count(),
                "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None}
    except Exception:
        pass
    topo = _run(["nvidia-smi", "topo", "-m"])
    env = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "colab": "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ,
        "gpus": gpu_info(),
        "cuda": cuda,
        "nvidia_smi_topology": topo,
        "packages": versions,
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
    }
    if extra:
        env.update(extra)
    return env


if __name__ == "__main__":
    print(json.dumps(capture(), indent=2))
