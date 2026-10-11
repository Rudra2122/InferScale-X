"""Shared helpers for A100 experiments: server lifecycle, result paths, defaults."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import httpx

from scripts import serverctl

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "benchmarks" / "results"
MODEL = os.environ.get("INFERSCALE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
GATEWAY_URL = "http://127.0.0.1:9000"
KEYS = {"default": "sk-dev-acme", "acme": "sk-dev-acme", "globex": "sk-dev-globex", "freetier": "sk-dev-free"}


def result_dir(experiment: str, tag: str | None = None) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    d = RESULTS / experiment / (f"{stamp}-{tag}" if tag else stamp)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def start_vllm(name: str, port: int, model: str = MODEL, gpu_mem: float = 0.85, max_len: int = 4096,
               extra: str = "", timeout: float = 1200) -> dict:
    info = serverctl.start(name, serverctl.vllm_cmd(model, port, gpu_mem, max_len, extra), port,
                           timeout=timeout, env={"VLLM_SERVER_DEV_MODE": "1"})
    if not info.get("healthy"):
        raise RuntimeError(f"vLLM {name} failed to start; see logs/{name}.log")
    info["model_load_gib"] = parse_model_load_gib(name)
    return info


def parse_model_load_gib(name: str) -> float | None:
    log = (serverctl.LOGS / f"{name}.log")
    if not log.exists():
        return None
    m = re.findall(r"Model loading took ([\d.]+) ?GiB", log.read_text(errors="replace"))
    return float(m[-1]) if m else None


def start_gateway(backends: dict[str, str], policy: str = "least_outstanding", cache_salt: bool = False,
                  port: int = 9000, config: str = "configs/gateway.colab.yaml") -> None:
    env = {"INFERSCALE_GATEWAY_CONFIG": config,
           "INFERSCALE_BACKENDS": ",".join(f"{k}={v}" for k, v in backends.items()),
           "INFERSCALE_ROUTING_POLICY": policy,
           "INFERSCALE_INJECT_CACHE_SALT": "true" if cache_salt else "false"}
    import sys

    info = serverctl.start("gateway", [sys.executable, "-m", "uvicorn", "gateway.api.app:app", "--host", "127.0.0.1",
                                       "--port", str(port), "--log-level", "warning"], port, env=env, timeout=60)
    if not info.get("healthy"):
        raise RuntimeError("gateway failed to start; see logs/gateway.log")
    # wait until the gateway sees at least one healthy backend
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/ready", timeout=2).status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError("gateway has no healthy backends")


def rep_values(run_dir: Path | str, *path: str) -> list[float]:
    """Collect one metric from every rep*/summary.json, e.g. rep_values(d, 'ttft_s', 'p50')."""
    xs = []
    for p in sorted(Path(run_dir).glob("rep*/summary.json")):
        v = json.loads(p.read_text())
        for k in path:
            v = v.get(k) if isinstance(v, dict) else None
        if v is not None:
            xs.append(float(v))
    return xs


def reset_prefix_cache(url: str) -> bool:
    try:
        return httpx.post(f"{url}/reset_prefix_cache", timeout=30).status_code == 200
    except Exception:
        return False


def chat(url: str, prompt: str, model: str = MODEL, max_tokens: int = 64, key: str = "sk-dev-acme",
         temperature: float = 0.0, **extra) -> dict:
    r = httpx.post(f"{url}/v1/chat/completions", headers={"Authorization": f"Bearer {key}"}, timeout=300,
                   json={"model": model, "messages": [{"role": "user", "content": prompt}],
                         "max_tokens": max_tokens, "temperature": temperature, **extra})
    r.raise_for_status()
    return r.json()
