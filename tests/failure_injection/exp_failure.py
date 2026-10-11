"""Reliability experiment: replica crash + recovery under load, and tenant overload.

Phase 1 (crash): 2 replicas behind the gateway, open-loop Poisson load. At 30% of
the run one replica is SIGKILLed; at 60% it is restarted. We record:
  - time for the gateway to mark it unhealthy (detection)
  - backend restart time and time until the gateway re-admits it
  - failed / mid-stream-broken requests per window (before / outage / after)
Phase 2 (overload): the rate-limited 'freetier' tenant floods the gateway while
'acme' runs normal load; we check freetier gets 429s (not 5xx) and acme's latency.

Runs with real vLLM on Colab (--kind vllm) or with CPU mock backends (--kind mock, used in CI).
python -m tests.failure_injection.exp_failure --kind vllm
"""
from __future__ import annotations

import argparse
import asyncio
import threading
import time

import httpx

from benchmarks.analysis.stats import dist
from benchmarks.runners.common import KEYS, MODEL, result_dir, save_json, start_gateway, start_vllm
from benchmarks.runners.loadgen import Workload, run_load, summarize
from scripts import serverctl

REPLICAS = {"replica-a": ("http://127.0.0.1:8001", 8001), "replica-b": ("http://127.0.0.1:8002", 8002)}


def start_backend(kind: str, name: str, port: int, gpu_mem: float):
    if kind == "mock":
        import sys

        info = serverctl.start(name, [sys.executable, "-m", "inference.mock.mock_server", "--port", str(port)],
                               port, timeout=60)
        if not info.get("healthy"):
            raise RuntimeError(f"mock {name} failed")
        return info
    return start_vllm(name, port, gpu_mem=gpu_mem)


class Monitor(threading.Thread):
    def __init__(self, url: str, key: str, interval: float = 0.25):
        super().__init__(daemon=True)
        self.url, self.key, self.interval = url, key, interval
        self.timeline: list[tuple[float, dict]] = []
        self._stop = threading.Event()

    def run(self):
        with httpx.Client() as c:
            while not self._stop.is_set():
                try:
                    r = c.get(f"{self.url}/admin/backends", headers={"Authorization": f"Bearer {self.key}"}, timeout=2)
                    snap = {b["name"]: b["healthy"] for b in r.json()["backends"]}
                    self.timeline.append((time.time(), snap))
                except Exception:
                    pass
                time.sleep(self.interval)

    def stop(self):
        self._stop.set()

    def first_time(self, name: str, healthy: bool, after: float) -> float | None:
        for t, snap in self.timeline:
            if t >= after and snap.get(name) is healthy:
                return t
        return None


def crash_phase(kind: str, model: str, rate: float, duration: float, gpu_mem: float, out) -> dict:
    victim = "replica-b"
    w = Workload(name="crash", model=model, num_requests=int(rate * duration), warmup_requests=4,
                 concurrency=None, request_rate=rate, input_tokens=256, output_tokens=64,
                 ignore_eos=True, tenants={"acme": 1.0})
    mon = Monitor("http://127.0.0.1:9000", KEYS["acme"])
    mon.start()
    events: dict = {}
    result: dict = {}

    def load():
        res, summ = asyncio.run(run_load("http://127.0.0.1:9000", KEYS, w))
        result["res"], result["summary"] = res, summ

    th = threading.Thread(target=load)
    t0 = time.time()
    th.start()
    time.sleep(duration * 0.3)
    events["kill"] = serverctl.kill_hard(victim)
    time.sleep(duration * 0.3)
    events["restart_begin"] = time.time()
    url, port = REPLICAS[victim]
    info = start_backend(kind, victim, port, gpu_mem)
    events["restart_healthy"] = time.time()
    th.join()
    time.sleep(2)
    mon.stop()
    events["detected_unhealthy"] = mon.first_time(victim, False, events["kill"])
    events["readmitted"] = mon.first_time(victim, True, events["restart_begin"])

    res = result["res"]
    windows = {"before_kill": (0, events["kill"]), "outage": (events["kill"], events["readmitted"] or 1e18),
               "after_readmit": (events["readmitted"] or 1e18, 1e18)}
    per_window = {}
    for wname, (lo, hi) in windows.items():
        rs = [r for r in res if lo <= r["t_start"] < hi]
        failed = [r for r in rs if not r["ok"]]
        per_window[wname] = {
            "requests": len(rs), "failed": len(failed),
            "midstream_broken": sum(1 for r in failed if r["error"] and "interrupted" in r["error"]),
            "status_counts": {s: sum(1 for r in rs if str(r["status"]) == s) for s in {str(r["status"]) for r in rs}},
            "ttft_s": dist(r["ttft_s"] for r in rs if r["ok"]),
            "served_by": {b: sum(1 for r in rs if r.get("backend") == b) for b in REPLICAS},
        }
    report = {
        "phase": "crash", "kind": kind, "rate_rps": rate, "duration_s": duration, "victim": victim,
        "time_to_detect_s": (events["detected_unhealthy"] - events["kill"]) if events["detected_unhealthy"] else None,
        "backend_restart_s": events["restart_healthy"] - events["restart_begin"],
        "time_to_readmit_from_restart_s": (events["readmitted"] - events["restart_begin"]) if events["readmitted"] else None,
        "total_failed": sum(1 for r in res if not r["ok"]), "total_requests": len(res),
        "per_window": per_window, "summary": result["summary"],
        "events_relative_s": {k: (v - t0 if v else None) for k, v in events.items()},
        "t0_epoch": t0,
        "health_timeline": [(round(t - t0, 2), s) for t, s in mon.timeline],
        "backend_restart_info": info,
    }
    import json

    with open(out / "crash_requests.jsonl", "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    return report


def overload_phase(model: str, out) -> dict:
    flood = Workload(name="flood", model=model, num_requests=200, warmup_requests=0, concurrency=40,
                     input_tokens=64, output_tokens=16, tenants={"freetier": 1.0})
    normal = Workload(name="normal", model=model, num_requests=100, warmup_requests=4, concurrency=8,
                      input_tokens=256, output_tokens=64, tenants={"acme": 1.0})

    async def both():
        t = time.perf_counter()
        (fr, _), (nr, _) = await asyncio.gather(run_load("http://127.0.0.1:9000", KEYS, flood),
                                                run_load("http://127.0.0.1:9000", KEYS, normal))
        return fr, nr, time.perf_counter() - t

    fr, nr, wall = asyncio.run(both())
    statuses = {}
    for r in fr:
        statuses[str(r["status"])] = statuses.get(str(r["status"]), 0) + 1
    five_xx = sum(v for k, v in statuses.items() if k.startswith("5"))
    return {"phase": "overload", "freetier_statuses": statuses, "freetier_5xx": five_xx,
            "freetier_429_fraction": statuses.get("429", 0) / len(fr) if fr else None,
            "acme_during_flood": summarize(nr, wall, normal),
            "pass": five_xx == 0 and statuses.get("429", 0) > 0 and all(r["ok"] for r in nr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["vllm", "mock"], default="vllm")
    ap.add_argument("--model", default=None)
    ap.add_argument("--rate", type=float, default=None)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--gpu-mem", type=float, default=0.42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    model = a.model or (MODEL if a.kind == "vllm" else "mock-model")
    rate = a.rate or (8.0 if a.kind == "vllm" else 40.0)
    duration = a.duration or (90.0 if a.kind == "vllm" else 12.0)
    from pathlib import Path

    out = Path(a.out) if a.out else result_dir("failure_injection", a.kind)
    out.mkdir(parents=True, exist_ok=True)
    try:
        for name, (url, port) in REPLICAS.items():
            start_backend(a.kind, name, port, a.gpu_mem)
        cfg = "configs/gateway.example.yaml" if a.kind == "mock" else "configs/gateway.colab.yaml"
        start_gateway({k: v[0] for k, v in REPLICAS.items()}, policy="least_outstanding", config=cfg)
        crash = crash_phase(a.kind, model, rate, duration, a.gpu_mem, out)
        save_json(out / "crash_report.json", crash)
        over = overload_phase(model, out)
        save_json(out / "overload_report.json", over)
        print(f"[failure] detect={crash['time_to_detect_s']} restart={crash['backend_restart_s']:.1f}s "
              f"failed={crash['total_failed']}/{crash['total_requests']} overload_pass={over['pass']}")
        print(f"[failure] wrote {out}")
    finally:
        serverctl.stopall()


if __name__ == "__main__":
    main()
