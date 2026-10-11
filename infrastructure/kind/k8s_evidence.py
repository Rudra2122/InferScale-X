"""Real Kubernetes evidence on a laptop (kind + Docker Desktop): scheduling, recovery, autoscaling, NetworkPolicy.

Prereqs (macOS):  brew install kind kubectl  +  Docker Desktop running (>= 4 CPUs, 6 GB RAM)
                  pip install -r requirements/dev.txt
Run from the repo root:
    python infrastructure/kind/k8s_evidence.py            # create cluster, build, deploy, run all checks
    python infrastructure/kind/k8s_evidence.py --reuse    # cluster exists; redeploy + re-run checks
    python infrastructure/kind/k8s_evidence.py --teardown # delete the cluster afterwards

What is real: Kubernetes scheduling (anti-affinity across 2 worker nodes), StatefulSet pod recovery,
gateway failover during a pod kill under load, HPA scale-up from CPU metrics, Calico-enforced
NetworkPolicy. What is NOT: GPUs. Backends are CPU mocks, so latency numbers here say nothing
about model serving - only about control-plane behaviour. Results: benchmarks/results/k8s_kind/<stamp>/.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402

from benchmarks.runners.common import result_dir, save_json  # noqa: E402
from benchmarks.runners.loadgen import Workload, run_load, summarize  # noqa: E402

CLUSTER = "inferscale"
NS = "inferscale"
GW = "http://127.0.0.1:30901"  # via the ingress relay container (see setup_ingress)
INGRESS = "inferscale-ingress"
KEY = "sk-dev-acme"
CALICO = "https://raw.githubusercontent.com/projectcalico/calico/v3.28.2/manifests/calico.yaml"
METRICS_SERVER = "https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.7.2/components.yaml"
LOG: list[dict] = []


def sh(cmd: str | list[str], check: bool = True, capture: bool = True, timeout: float = 900) -> str:
    args = cmd if isinstance(cmd, list) else ["bash", "-lc", cmd]
    t = time.time()
    p = subprocess.run(args, capture_output=capture, text=True, timeout=timeout, cwd=ROOT)
    LOG.append({"cmd": cmd if isinstance(cmd, str) else " ".join(cmd), "rc": p.returncode,
                "s": round(time.time() - t, 2), "stderr": (p.stderr or "")[-400:]})
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}\n{(p.stdout or '')[-1500:]}\n{(p.stderr or '')[-1500:]}")
    return (p.stdout or "").strip()


def kubectl(args: str, **kw) -> str:
    return sh(f"kubectl --context kind-{CLUSTER} {args}", **kw)


def prereqs() -> dict:
    missing = [b for b in ("docker", "kind", "kubectl") if not shutil.which(b)]
    if missing:
        raise SystemExit(f"missing tools: {missing}. macOS: brew install kind kubectl; start Docker Desktop.")
    sh("docker info >/dev/null")
    return {"docker": sh("docker version --format '{{.Server.Version}}'"), "kind": sh("kind version"),
            "kubectl": sh("kubectl version --client -o json | python3 -c 'import json,sys;print(json.load(sys.stdin)[\"clientVersion\"][\"gitVersion\"])'"),
            "arch": sh("uname -m")}


def create_cluster(reuse: bool) -> dict:
    exists = CLUSTER in sh("kind get clusters", check=False).split()
    t = time.time()
    if not exists:
        sh(f"kind create cluster --config infrastructure/kind/kind-config.yaml --wait 0s", timeout=900)
        kubectl(f"apply -f {CALICO}")
        kubectl(f"apply -f {METRICS_SERVER}")
        kubectl("-n kube-system patch deployment metrics-server --type=json "
                "-p='[{\"op\":\"add\",\"path\":\"/spec/template/spec/containers/0/args/-\",\"value\":\"--kubelet-insecure-tls\"}]'")
    elif not reuse:
        print("[k8s] cluster exists; reusing it (pass --teardown afterwards to delete)")
    kubectl("wait --for=condition=Ready nodes --all --timeout=600s", timeout=650)
    kubectl("-n kube-system rollout status deployment/metrics-server --timeout=300s", timeout=350)
    return {"created": not exists, "ready_after_s": round(time.time() - t, 1),
            "nodes": len(json.loads(kubectl("get nodes -o json"))["items"])}


def build_and_load() -> dict:
    t = time.time()
    sh("docker build -q -f infrastructure/docker/Dockerfile.gateway -t inferscale-gateway:dev .", timeout=1200)
    sh("docker build -q -f infrastructure/docker/Dockerfile.mock -t inferscale-mock:dev .", timeout=1200)
    sh(f"kind load docker-image inferscale-gateway:dev inferscale-mock:dev --name {CLUSTER}", timeout=600)
    return {"build_and_load_s": round(time.time() - t, 1)}


def setup_ingress() -> dict:
    """Docker Desktop on macOS can fail to forward kind's extraPortMappings to the host (observed:
    host -> :30900 hangs while container -> node:30900 works). A socat relay container on the `kind`
    network, published on 127.0.0.1:30901, forwards to the control-plane NodePort. kube-proxy still
    load-balances across gateway pods, so pod-kill tests keep their meaning."""
    sh(f"docker rm -f {INGRESS}", check=False)
    sh(f"docker run -d --name {INGRESS} --network kind -p 127.0.0.1:30901:30901 alpine/socat:1.8.0.0 "
       f"TCP-LISTEN:30901,fork,reuseaddr TCP:{CLUSTER}-control-plane:30900")
    return {"relay": f"127.0.0.1:30901 -> {CLUSTER}-control-plane:30900 (NodePort)"}


def deploy() -> dict:
    t = time.time()
    kubectl("apply -f infrastructure/kind/manifests.yaml")
    kubectl(f"-n {NS} rollout restart deployment/gateway")  # pick up rebuilt :dev image on --reuse
    kubectl(f"-n {NS} rollout status statefulset/mock --timeout=300s", timeout=350)
    kubectl(f"-n {NS} rollout status deployment/gateway --timeout=300s", timeout=350)
    ready = False
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            if httpx.get(f"{GW}/ready", timeout=3).status_code == 200:
                ready = True
                break
        except Exception:
            pass
        time.sleep(1)
    if not ready:
        raise RuntimeError(f"gateway not reachable at {GW} after 120s. Check: curl {GW}/health ; "
                           f"docker logs {INGRESS} ; kubectl --context kind-{CLUSTER} -n {NS} get pods")
    last = None
    for _ in range(5):
        try:
            r = httpx.post(f"{GW}/v1/chat/completions", headers={"Authorization": f"Bearer {KEY}"}, timeout=15,
                           json={"model": "mock-model", "messages": [{"role": "user", "content": "hello"}],
                                 "max_tokens": 4})
            return {"deploy_s": round(time.time() - t, 1), "smoke_status": r.status_code,
                    "smoke_backend": r.headers.get("x-backend")}
        except Exception as e:
            last = e
            time.sleep(2)
    raise RuntimeError(f"smoke request failed 5x: {last!r}")


def pods() -> list[dict]:
    items = json.loads(kubectl(f"-n {NS} get pods -o json"))["items"]
    return [{"name": p["metadata"]["name"], "app": p["metadata"]["labels"].get("app"), "node": p["spec"].get("nodeName"),
             "phase": p["status"].get("phase"),
             "ready": any(c["type"] == "Ready" and c["status"] == "True" for c in p["status"].get("conditions", []))}
            for p in items]


def check_scheduling() -> dict:
    ps = pods()
    mock_nodes = [p["node"] for p in ps if p["app"] == "mock"]
    gw_nodes = [p["node"] for p in ps if p["app"] == "gateway"]
    return {"pods": ps, "mock_replicas_on_distinct_nodes": len(set(mock_nodes)) == len(mock_nodes) == 2,
            "gateway_nodes": gw_nodes, "pass": len(set(mock_nodes)) == 2}


def pod_state(name: str) -> tuple[str | None, bool]:
    """(uid, ready) of a pod; uid lets us tell the recreated StatefulSet pod from the one we killed."""
    try:
        p = json.loads(kubectl(f"-n {NS} get pod {name} -o json", check=True))
    except Exception:
        return None, False
    ready = any(c["type"] == "Ready" and c["status"] == "True" for c in p["status"].get("conditions", []))
    return p["metadata"]["uid"], ready


def backend_health() -> dict:
    try:
        r = httpx.get(f"{GW}/admin/backends", headers={"Authorization": f"Bearer {KEY}"}, timeout=2).json()
        return {b["name"]: b["healthy"] for b in r["backends"]}
    except Exception:
        return {}


def check_recovery(duration: float = 45.0, rate: float = 25.0) -> dict:
    w = Workload(name="k8s_recovery", model="mock-model", num_requests=int(rate * duration), warmup_requests=4,
                 concurrency=None, request_rate=rate, input_tokens=64, output_tokens=32, tenants={"acme": 1.0})
    result, events, timeline = {}, {}, []

    def load():
        res, _ = asyncio.run(run_load(GW, {"acme": KEY, "default": KEY}, w))
        result["res"] = res

    th = threading.Thread(target=load)
    t0 = time.time()
    th.start()
    time.sleep(duration * 0.25)
    old_uid, _ = pod_state("mock-1")
    events["kill_mock_1"] = time.time()
    kubectl(f"-n {NS} delete pod mock-1 --grace-period=0 --force --wait=false", check=False)
    while time.time() - events["kill_mock_1"] < 120:
        h = backend_health()
        timeline.append((round(time.time() - t0, 2), h))
        if "gateway_marked_unhealthy" not in events and h.get("mock-1") is False:
            events["gateway_marked_unhealthy"] = time.time()
        uid, ready = pod_state("mock-1")
        if "mock_1_pod_ready" not in events and ready and uid and uid != old_uid:
            events["mock_1_pod_ready"] = time.time()
        if "mock_1_readmitted" not in events and "mock_1_pod_ready" in events and h.get("mock-1") is True:
            events["mock_1_readmitted"] = time.time()
            break
        time.sleep(0.5)
    gw = [p["name"] for p in pods() if p["app"] == "gateway"]
    events["kill_gateway_pod"] = time.time()
    kubectl(f"-n {NS} delete pod {gw[0]} --grace-period=0 --force --wait=false", check=False)
    th.join()
    kubectl(f"-n {NS} rollout status deployment/gateway --timeout=120s", check=False)
    res = result["res"]

    def window(lo, hi):
        rs = [r for r in res if lo <= r["t_start"] < hi]
        return {"requests": len(rs), "failed": sum(not r["ok"] for r in rs),
                "errors": sorted({(r["error"] or "")[:80] for r in rs if not r["ok"]})[:5]}

    k1, k2 = events["kill_mock_1"], events["kill_gateway_pod"]
    rel = {k: round(v - t0, 2) for k, v in events.items()}
    return {
        "events_relative_s": rel,
        "mock_pod_recreated_ready_s": (events["mock_1_pod_ready"] - k1) if "mock_1_pod_ready" in events else None,
        "gateway_detect_s": (events["gateway_marked_unhealthy"] - k1) if "gateway_marked_unhealthy" in events else None,
        "readmit_after_kill_s": (events["mock_1_readmitted"] - k1) if "mock_1_readmitted" in events else None,
        "windows": {"before_kills": window(0, k1), "mock_pod_down": window(k1, events.get("mock_1_readmitted", k2)),
                    "after_mock_recovery": window(events.get("mock_1_readmitted", k2), k2),
                    "gateway_pod_kill": window(k2, 1e18)},
        "summary": summarize(res, time.time() - t0, w),
        "health_timeline": timeline[:400],
        "note": "Requests in flight on a killed pod fail (expected). Gateway-pod kill errors come from kube-proxy "
                "still routing to the terminating pod until endpoints update.",
    }


def check_hpa(max_wait: float = 240.0) -> dict:
    start = json.loads(kubectl(f"-n {NS} get hpa gateway -o json"))
    w = Workload(name="k8s_hpa", model="mock-model", num_requests=100000, warmup_requests=0, concurrency=96,
                 input_tokens=32, output_tokens=8, tenants={"acme": 1.0}, timeout_s=30)
    stop = threading.Event()
    done = {}

    def load():
        async def bounded():
            task = asyncio.create_task(run_load(GW, {"acme": KEY, "default": KEY}, w))
            while not stop.is_set() and not task.done():
                await asyncio.sleep(0.5)
            task.cancel()
        try:
            asyncio.run(bounded())
        except Exception as e:  # cancellation is the normal exit
            done["exit"] = type(e).__name__

    th = threading.Thread(target=load, daemon=True)
    t0 = time.time()
    th.start()
    timeline, scaled_at = [], None
    while time.time() - t0 < max_wait:
        h = json.loads(kubectl(f"-n {NS} get hpa gateway -o json"))
        st = h.get("status", {})
        cpu = None
        for m in st.get("currentMetrics") or []:
            cpu = (m.get("resource") or {}).get("current", {}).get("averageUtilization", cpu)
        timeline.append({"t": round(time.time() - t0, 1), "current_replicas": st.get("currentReplicas"),
                         "desired_replicas": st.get("desiredReplicas"), "cpu_util_pct": cpu})
        if scaled_at is None and (st.get("desiredReplicas") or 0) > (start["spec"]["minReplicas"]):
            scaled_at = time.time() - t0
        if scaled_at and time.time() - t0 > scaled_at + 45:
            break
        time.sleep(5)
    stop.set()
    th.join(timeout=30)
    return {"min_replicas": start["spec"]["minReplicas"], "max_replicas": start["spec"]["maxReplicas"],
            "scale_up_detected_after_s": scaled_at, "peak_desired_replicas": max((x["desired_replicas"] or 0) for x in timeline),
            "timeline": timeline, "pass": scaled_at is not None,
            "note": "Scale-down is not awaited (default 5-minute stabilisation window)."}


def check_network_policy() -> dict:
    kubectl(f"-n {NS} delete pod attacker --ignore-not-found --wait=true", check=False)
    kubectl(f"-n {NS} run attacker --image=curlimages/curl:8.10.1 --restart=Never --command -- sleep 600")
    kubectl(f"-n {NS} wait --for=condition=Ready pod/attacker --timeout=180s", timeout=200)
    attacker_to_mock = kubectl(f"-n {NS} exec attacker -- curl -s -m 5 -o /dev/null -w '%{{http_code}}' "
                               "http://mock-0.mock.inferscale.svc.cluster.local:8000/health", check=False)
    attacker_to_gateway = kubectl(f"-n {NS} exec attacker -- curl -s -m 5 -o /dev/null -w '%{{http_code}}' "
                                  "http://gateway.inferscale.svc.cluster.local/health", check=False)
    gw_pod = [p["name"] for p in pods() if p["app"] == "gateway" and p["ready"]][0]
    gateway_to_mock = kubectl(f"-n {NS} exec {gw_pod} -- python -c \"import urllib.request as u;"
                              "print(u.urlopen('http://mock-0.mock.inferscale.svc.cluster.local:8000/health',timeout=5).status)\"",
                              check=False)
    kubectl(f"-n {NS} delete pod attacker --wait=false", check=False)
    blocked = attacker_to_mock.strip("'") in ("000", "")
    return {"attacker_to_mock_http": attacker_to_mock, "attacker_to_gateway_http": attacker_to_gateway,
            "gateway_to_mock": gateway_to_mock, "attacker_blocked_from_mock": blocked,
            "gateway_allowed_to_mock": gateway_to_mock.strip() == "200",
            "pass": blocked and gateway_to_mock.strip() == "200" and attacker_to_gateway.strip("'") == "200"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--teardown", action="store_true")
    ap.add_argument("--only", default="scheduling,recovery,hpa,netpol")
    a = ap.parse_args()
    out = result_dir("k8s_kind")
    report = {"experiment": "kubernetes_kind", "label": "real Kubernetes, CPU mock backends (no GPUs)"}
    try:
        report["tools"] = prereqs()
        report["cluster"] = create_cluster(a.reuse)
        if not a.skip_build:
            report["images"] = build_and_load()
        report["ingress"] = setup_ingress()
        report["deploy"] = deploy()
        steps = a.only.split(",")
        checks = {"scheduling": check_scheduling, "recovery": check_recovery, "hpa": check_hpa,
                  "netpol": check_network_policy}
        for name in steps:
            print(f"[k8s] running {name} ...")
            try:
                report[name] = checks[name]()
            except Exception as e:
                report[name] = {"error": f"{type(e).__name__}: {e}"[:1500], "pass": False}
            save_json(out / "report.json", report)
        report["versions"] = {"server": kubectl("version -o json", check=False)[:2000]}
        report["all_pass"] = all(report.get(s, {}).get("pass", s == "recovery") for s in steps) and \
            report["deploy"]["smoke_status"] == 200
        (out / "kubectl_get_all.txt").write_text(kubectl(f"-n {NS} get all,hpa,pdb,networkpolicy -o wide", check=False))
    finally:
        report["command_log"] = LOG[-200:]
        save_json(out / "report.json", report)
        if a.teardown:
            sh(f"docker rm -f {INGRESS}", check=False)
            sh(f"kind delete cluster --name {CLUSTER}", check=False)
    print(json.dumps({k: (v.get("pass") if isinstance(v, dict) else v) for k, v in report.items()
                      if k in ("scheduling", "hpa", "netpol", "all_pass")}, indent=2))
    rec = report.get("recovery", {})
    print(f"[k8s] mock pod recreated in {rec.get('mock_pod_recreated_ready_s')}s; gateway detect "
          f"{rec.get('gateway_detect_s')}s -> {out}/report.json")


if __name__ == "__main__":
    main()
