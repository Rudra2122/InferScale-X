"""Start/stop/health-check long-running servers from Colab cells or a shell.

Each server runs in its own process group; logs go to logs/<name>.log, state to .run/<name>.json.

Examples
--------
python scripts/serverctl.py vllm  --name vllm-a --port 8001 --model Qwen/Qwen2.5-1.5B-Instruct --gpu-mem 0.42
python scripts/serverctl.py sglang --name sgl --port 8003 --model Qwen/Qwen2.5-1.5B-Instruct --gpu-mem 0.8
python scripts/serverctl.py mock  --name mock-a --port 8001
python scripts/serverctl.py gateway --port 9000 --config configs/gateway.colab.yaml --backends vllm-a=http://127.0.0.1:8001
python scripts/serverctl.py stop vllm-a      |  stopall  |  status
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / ".run"
LOGS = ROOT / "logs"
_PROCS: dict[int, subprocess.Popen] = {}  # children started by this interpreter (so we can reap them)


def _alive(pid: int) -> bool:
    p = _PROCS.get(pid)
    if p is not None:
        return p.poll() is None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def _state(name: str) -> Path:
    return RUN / f"{name}.json"


def start(name: str, cmd: list[str], port: int, env: dict | None = None, timeout: float = 900,
          health_path: str = "/health") -> dict:
    RUN.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    if _state(name).exists():
        stop(name)
    log = open(LOGS / f"{name}.log", "w")
    full_env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                **(env or {})}
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, env=full_env, start_new_session=True)
    _PROCS[p.pid] = p
    info = {"name": name, "pid": p.pid, "port": port, "cmd": cmd, "started": t0, "log": str(LOGS / f"{name}.log")}
    _state(name).write_text(json.dumps(info))
    print(f"[serverctl] started {name} pid={p.pid}: {shlex.join(cmd)}")
    ok = wait_healthy(f"http://127.0.0.1:{port}", timeout=timeout, proc=p, path=health_path,
                      log_path=LOGS / f"{name}.log")
    info["ready_after_s"] = round(time.time() - t0, 2)
    info["healthy"] = ok
    _state(name).write_text(json.dumps(info))
    if not ok:
        print(f"[serverctl] {name} NOT healthy after {timeout}s; last log lines:")
        print(tail(name, 40))
    else:
        print(f"[serverctl] {name} healthy after {info['ready_after_s']}s")
    return info


FATAL_MARKERS = ("Engine core initialization failed", "pydantic_core._pydantic_core.ValidationError",
                 "CUDA out of memory", "No available memory for the cache blocks", "Unsupported connector type")


def wait_healthy(url: str, timeout: float = 600, proc: subprocess.Popen | None = None, path: str = "/health",
                 log_path: Path | None = None) -> bool:
    """Poll /health. Fail fast if the process exits OR its log shows a fatal startup error (vLLM's API server
    can stay alive for the whole timeout after its engine core has already died)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        if log_path is not None and log_path.exists():
            try:
                tail = log_path.read_text(errors="replace")[-20000:]
                if any(m in tail for m in FATAL_MARKERS):
                    return False
            except OSError:
                pass
        try:
            if httpx.get(url + path, timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


def stop(name: str, sig=signal.SIGTERM, wait: float = 30) -> None:
    st = _state(name)
    if not st.exists():
        return
    info = json.loads(st.read_text())
    try:
        os.killpg(info["pid"], sig)
        deadline = time.time() + wait
        while time.time() < deadline:
            if not _alive(info["pid"]):
                break
            time.sleep(0.2)
        else:
            os.killpg(info["pid"], signal.SIGKILL)
    except ProcessLookupError:
        pass
    st.unlink(missing_ok=True)
    print(f"[serverctl] stopped {name}")


def kill_hard(name: str) -> float:
    """SIGKILL without cleanup - used by failure injection. Returns kill timestamp."""
    info = json.loads(_state(name).read_text())
    t = time.time()
    try:
        os.killpg(info["pid"], signal.SIGKILL)
    except ProcessLookupError:
        pass
    p = _PROCS.get(info["pid"])
    if p is not None:
        p.wait(timeout=30)
    return t


def stopall() -> None:
    if RUN.exists():
        for f in RUN.glob("*.json"):
            stop(f.stem)


def status() -> list[dict]:
    out = []
    if RUN.exists():
        for f in RUN.glob("*.json"):
            info = json.loads(f.read_text())
            info["alive"] = _alive(info["pid"])
            out.append(info)
    return out


def tail(name: str, n: int = 30) -> str:
    p = LOGS / f"{name}.log"
    return "\n".join(p.read_text(errors="replace").splitlines()[-n:]) if p.exists() else "(no log)"


def vllm_cmd(model, port, gpu_mem, max_len, extra) -> list[str]:
    cmd = ["vllm", "serve", model, "--port", str(port), "--host", "127.0.0.1",
           "--gpu-memory-utilization", str(gpu_mem), "--max-model-len", str(max_len)]
    return cmd + (shlex.split(extra) if extra else [])


def sglang_cmd(model, port, gpu_mem, max_len, extra) -> list[str]:
    cmd = [sys.executable, "-m", "sglang.launch_server", "--model-path", model, "--port", str(port),
           "--host", "127.0.0.1", "--mem-fraction-static", str(gpu_mem), "--context-length", str(max_len)]
    return cmd + (shlex.split(extra) if extra else [])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for eng in ("vllm", "sglang", "trtllm"):
        s = sub.add_parser(eng)
        s.add_argument("--name", required=True)
        s.add_argument("--port", type=int, required=True)
        s.add_argument("--model", required=True)
        s.add_argument("--gpu-mem", type=float, default=0.85)
        s.add_argument("--max-len", type=int, default=4096)
        s.add_argument("--extra", default="")
        s.add_argument("--timeout", type=float, default=1200)
    m = sub.add_parser("mock")
    m.add_argument("--name", required=True)
    m.add_argument("--port", type=int, required=True)
    g = sub.add_parser("gateway")
    g.add_argument("--name", default="gateway")
    g.add_argument("--port", type=int, default=9000)
    g.add_argument("--config", default="configs/gateway.colab.yaml")
    g.add_argument("--backends", default=None, help="name=url,name=url")
    g.add_argument("--policy", default=None)
    g.add_argument("--cache-salt", default=None, choices=["true", "false"])
    s = sub.add_parser("stop")
    s.add_argument("name")
    sub.add_parser("stopall")
    sub.add_parser("status")
    t = sub.add_parser("tail")
    t.add_argument("name")
    t.add_argument("-n", type=int, default=40)
    a = ap.parse_args()

    if a.cmd == "vllm":
        start(a.name, vllm_cmd(a.model, a.port, a.gpu_mem, a.max_len, a.extra), a.port, timeout=a.timeout,
              env={"VLLM_SERVER_DEV_MODE": "1"})  # exposes /reset_prefix_cache for clean experiments
    elif a.cmd == "sglang":
        start(a.name, sglang_cmd(a.model, a.port, a.gpu_mem, a.max_len, a.extra), a.port, timeout=a.timeout)
    elif a.cmd == "trtllm":
        cmd = ["trtllm-serve", a.model, "--port", str(a.port), "--host", "127.0.0.1",
               "--max_seq_len", str(a.max_len)] + (shlex.split(a.extra) if a.extra else [])
        start(a.name, cmd, a.port, timeout=a.timeout)
    elif a.cmd == "mock":
        start(a.name, [sys.executable, "-m", "inference.mock.mock_server", "--port", str(a.port)], a.port, timeout=60)
    elif a.cmd == "gateway":
        env = {"INFERSCALE_GATEWAY_CONFIG": a.config}
        if a.backends:
            env["INFERSCALE_BACKENDS"] = a.backends
        if a.policy:
            env["INFERSCALE_ROUTING_POLICY"] = a.policy
        if a.cache_salt:
            env["INFERSCALE_INJECT_CACHE_SALT"] = a.cache_salt
        start(a.name, [sys.executable, "-m", "uvicorn", "gateway.api.app:app", "--host", "127.0.0.1",
                       "--port", str(a.port), "--log-level", "warning"], a.port, env=env, timeout=60)
    elif a.cmd == "stop":
        stop(a.name)
    elif a.cmd == "stopall":
        stopall()
    elif a.cmd == "status":
        print(json.dumps(status(), indent=2))
    elif a.cmd == "tail":
        print(tail(a.name, a.n))


if __name__ == "__main__":
    main()
