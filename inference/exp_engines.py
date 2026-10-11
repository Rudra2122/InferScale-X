"""Pillar 3: engine comparison (vLLM vs SGLang vs TensorRT-LLM) on identical workloads.

Each engine runs in its OWN Colab runtime (their pinned torch/CUDA stacks conflict).
Run this once per engine, save results to Drive, then compare with
benchmarks/analysis/compare_engines.py.

python inference/exp_engines.py --engine vllm
python inference/exp_engines.py --engine sglang     (fresh runtime, pip install "sglang[all]")
python inference/exp_engines.py --engine trtllm     (fresh runtime, pip install tensorrt_llm; experimental)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.runners.common import KEYS, MODEL, result_dir, save_json  # noqa: E402
from benchmarks.runners.loadgen import Workload, run_and_save  # noqa: E402
from scripts import serverctl  # noqa: E402

PORT = 8001
WORKLOADS = ["benchmarks/workloads/chat_short.yaml", "benchmarks/workloads/prefix_heavy.yaml"]


def launch(engine: str, model: str, gpu_mem: float) -> dict:
    if engine == "vllm":
        return serverctl.start("engine", serverctl.vllm_cmd(model, PORT, gpu_mem, 4096, ""), PORT, timeout=1200)
    if engine == "sglang":
        return serverctl.start("engine", serverctl.sglang_cmd(model, PORT, gpu_mem, 4096, ""), PORT, timeout=1200)
    if engine == "trtllm":
        cmd = ["trtllm-serve", model, "--port", str(PORT), "--host", "127.0.0.1", "--max_seq_len", "4096",
               "--kv_cache_free_gpu_memory_fraction", str(gpu_mem)]
        return serverctl.start("engine", cmd, PORT, timeout=2400)
    raise ValueError(engine)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["vllm", "sglang", "trtllm"], required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--num-requests", type=int, default=None)
    a = ap.parse_args()
    out = result_dir("engines", a.engine)
    report = {"experiment": "engine_comparison", "engine": a.engine, "model": a.model, "workloads": {}}
    try:
        try:
            info = launch(a.engine, a.model, a.gpu_mem)
        except OSError as e:  # e.g. trtllm-serve not installed: record it instead of crashing with no report
            info = {"healthy": False, "cmd": None}
            report["launch_exception"] = f"{type(e).__name__}: {e}"
        report["startup"] = {k: info.get(k) for k in ("healthy", "ready_after_s", "cmd")}
        if not info.get("healthy"):
            report["error"] = "engine failed to start"
            report["log_tail"] = serverctl.tail("engine", 60)
        else:
            for wl in WORKLOADS:
                w = Workload.from_yaml(wl, model=a.model, num_requests=a.num_requests)
                # ignore_eos is supported by vLLM and SGLang; TRT-LLM serve may reject unknown fields
                if a.engine == "trtllm":
                    w.ignore_eos = False
                report["workloads"][w.name] = run_and_save(f"http://127.0.0.1:{PORT}", KEYS, w,
                                                           str(out / w.name), a.repeats,
                                                           f"{a.engine}/{w.name}")["aggregate"]
        save_json(out / "report.json", report)
        print(f"[engines] wrote {out}/report.json")
    finally:
        serverctl.stopall()


if __name__ == "__main__":
    main()
