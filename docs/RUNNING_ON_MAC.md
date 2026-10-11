# Running and deploying on a Mac (Apple Silicon)

The Mac has no NVIDIA GPU, so it runs everything that is **not** model serving: the gateway, CPU mock
replicas, the full CPU test suite, retrieval eval, the agent security suite, docker-compose with
Prometheus + Grafana, and a real 3-node Kubernetes cluster (kind). GPU experiments stay on Colab;
you can still drive a Colab-hosted model from the Mac via the public endpoint (§5).

## 0. One-time setup
```bash
brew install python@3.11 kind kubectl gh        # plus Docker Desktop (Settings → Resources: ≥4 CPU, ≥6 GB RAM)
cd InferScale-X && python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements/dev.txt
```
Triton has no macOS wheels: `make kernels-cpu` (kernel correctness via the Triton interpreter) runs on
Linux, in CI, or on Colab — not on the Mac. Its tests skip automatically there.

## 1. Tests + local demo (2 min)
```bash
make test            # CPU suite incl. gateway, routing, failure injection, security, retrieval
make demo            # gateway + 2 mock replicas + load test; `make demo-stop`
make retrieval       # BM25 (+ dense/hybrid if `pip install sentence-transformers`)
```

## 2. docker-compose deployment with monitoring (5 min)
```bash
make compose-up
curl -H 'Authorization: Bearer sk-dev-acme' localhost:9000/v1/models
python -m benchmarks.runners.loadgen --url http://127.0.0.1:9000 --api-key sk-dev-acme \
  --workload benchmarks/workloads/smoke.yaml --model mock-model --out benchmarks/results/compose_smoke
open http://localhost:3000     # Grafana, dashboard "InferScale-X Serving"
make compose-down
```

## 3. Real Kubernetes evidence on kind (10–20 min first time)
```bash
make k8s             # creates 3-node cluster with Calico + metrics-server, builds/loads images, deploys, then:
                     #   scheduling  - mock replicas forced onto different worker nodes (anti-affinity)
                     #   recovery    - kills a replica pod and a gateway pod under load; measures recreate/detect/readmit
                     #   hpa         - drives gateway CPU until the HPA scales out
                     #   netpol      - attacker pod blocked from replicas; gateway allowed
cat benchmarks/results/k8s_kind/*/report.json | python -m json.tool | head -60
python infrastructure/kind/k8s_evidence.py --reuse --only recovery    # re-run one check
docker rm -f inferscale-ingress && make k8s-down
```
Mac notes: first Calico start takes a few minutes; if `hpa` shows `cpu_util_pct: null`, metrics-server
isn't ready yet — re-run with `--reuse --only hpa`. The script reaches the gateway through a small relay container
(`inferscale-ingress`, `127.0.0.1:30901` → NodePort 30900) because Docker Desktop on macOS did not forward kind's
port mapping to the host in testing. Port 30901 must be free; `docker rm -f inferscale-ingress` cleans it up.

CI also runs this on GitHub's Linux runners on every push (`kind-k8s` job), so the repo carries a
second, independent K8s evidence artifact.

## 4. Merge Colab results and publish
```bash
# download the evidence zip(s) from Drive (InferScale-X-results/), then from the repo root:
unzip -o ~/Downloads/inferscale-evidence-*.zip
make results         # READINESS.md, jd-coverage.md, CLAIMS.md, plots, RESULTS.md
bash scripts/publish_repo.sh InferScale-X     # tests -> regenerate docs -> git init/commit -> gh repo create -> push
```

## 5. Hit the real model on Colab from the Mac
In Colab (notebook 01, §18): `!bash scripts/colab_public_endpoint.sh` prints an `https://*.trycloudflare.com`
URL and a fresh key. From the Mac:
```bash
python -m benchmarks.runners.loadgen --url https://<id>.trycloudflare.com --api-key <key> \
  --workload benchmarks/workloads/smoke.yaml --model Qwen/Qwen2.5-1.5B-Instruct --out benchmarks/results/remote_smoke
```
Latency over the tunnel includes internet RTT — label such numbers "remote client", never compare them with on-box results.
