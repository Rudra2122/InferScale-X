"""GPU power / utilisation sampler (nvidia-smi streaming) for energy-per-token.

Board power from nvidia-smi is averaged by the driver over ~100 ms windows; energy
is the trapezoidal integral of samples. Good enough for comparisons on the same GPU,
not a calibrated measurement. Silently disabled when nvidia-smi is absent.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time


class GpuSampler:
    def __init__(self, interval_ms: int = 100, gpu_index: int = 0):
        self.interval_ms, self.gpu = interval_ms, gpu_index
        self.samples: list[tuple[float, float, float, float]] = []  # t, watts, util%, mem MiB
        self.proc = None
        self.thread = None
        self.enabled = shutil.which("nvidia-smi") is not None

    def _reader(self):
        for line in self.proc.stdout:
            try:
                w, u, m = (float(x) for x in line.strip().split(","))
                self.samples.append((time.time(), w, u, m))
            except ValueError:
                continue

    def __enter__(self):
        if self.enabled:
            self.proc = subprocess.Popen(
                ["nvidia-smi", "-i", str(self.gpu), "--query-gpu=power.draw,utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits", f"-lms={self.interval_ms}"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            self.thread = threading.Thread(target=self._reader, daemon=True)
            self.thread.start()
        return self

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()

    def summary(self, output_tokens: int | None = None) -> dict:
        s = self.samples
        if len(s) < 2:
            return {"available": False}
        energy = sum((s[i][0] - s[i - 1][0]) * (s[i][1] + s[i - 1][1]) / 2 for i in range(1, len(s)))
        dur = s[-1][0] - s[0][0]
        out = {"available": True, "samples": len(s), "duration_s": dur, "energy_j": energy,
               "avg_power_w": energy / dur if dur else None, "max_power_w": max(x[1] for x in s),
               "avg_util_pct": sum(x[2] for x in s) / len(s), "max_mem_mib": max(x[3] for x in s)}
        if output_tokens:
            out["joules_per_output_token"] = energy / output_tokens
        return out
