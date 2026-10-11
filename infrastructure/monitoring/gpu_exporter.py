"""Minimal Prometheus exporter for GPU metrics via nvidia-smi (Colab-friendly DCGM stand-in).

python infrastructure/monitoring/gpu_exporter.py --port 9401
Metric names mirror DCGM exporter where possible so dashboards work with either.
"""
from __future__ import annotations

import argparse
import subprocess
import time

from prometheus_client import Gauge, start_http_server

FIELDS = {
    "utilization.gpu": ("DCGM_FI_DEV_GPU_UTIL", "GPU utilisation %"),
    "utilization.memory": ("DCGM_FI_DEV_MEM_COPY_UTIL", "Memory controller utilisation %"),
    "memory.used": ("DCGM_FI_DEV_FB_USED", "Framebuffer used MiB"),
    "memory.free": ("DCGM_FI_DEV_FB_FREE", "Framebuffer free MiB"),
    "power.draw": ("DCGM_FI_DEV_POWER_USAGE", "Power draw W"),
    "temperature.gpu": ("DCGM_FI_DEV_GPU_TEMP", "GPU temperature C"),
    "clocks.sm": ("DCGM_FI_DEV_SM_CLOCK", "SM clock MHz"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9401)
    ap.add_argument("--interval", type=float, default=2.0)
    a = ap.parse_args()
    gauges = {k: Gauge(n, h, ["gpu"]) for k, (n, h) in FIELDS.items()}
    start_http_server(a.port)
    q = ",".join(["index"] + list(FIELDS))
    while True:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        for line in out.strip().splitlines():
            vals = [v.strip() for v in line.split(",")]
            for (k, g), v in zip(gauges.items(), vals[1:]):
                try:
                    g.labels(vals[0]).set(float(v))
                except ValueError:
                    pass
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
