.PHONY: install test lint demo demo-stop evidence plots results kernels-cpu retrieval compose-up compose-down k8s k8s-down publish colab-core colab-training

install:
	pip install -r requirements/dev.txt

test:
	python -m pytest --junitxml tests/results/junit-unit.xml

lint:
	ruff check --select E9,F63,F7,F82,F401,F841 .

demo:
	python scripts/serverctl.py mock --name mock-a --port 8001
	python scripts/serverctl.py mock --name mock-b --port 8002
	python scripts/serverctl.py gateway --config configs/gateway.example.yaml --port 9000
	python -m benchmarks.runners.loadgen --url http://127.0.0.1:9000 --api-key sk-dev-acme \
	  --workload benchmarks/workloads/smoke.yaml --model mock-model --out benchmarks/results/demo-mock --repeats 2
	@echo "gateway still running on :9000 - 'make demo-stop' to stop"

demo-stop:
	python scripts/serverctl.py stopall

evidence:
	python scripts/evidence_report.py

results:
	python scripts/evidence_report.py && python benchmarks/analysis/make_plots.py && python scripts/publish_results.py

kernels-cpu:   ## Triton kernels correctness on CPU via the interpreter (needs torch + triton)
	TRITON_INTERPRET=1 python optimization/triton-kernels/flash_attention.py --check-only
	TRITON_INTERPRET=1 python optimization/triton-kernels/fused_add_rmsnorm.py --check-only

retrieval:
	python agents/eval/retrieval_eval.py

compose-up:    ## gateway + 2 mock replicas + Prometheus + Grafana on your laptop
	docker compose -f infrastructure/docker/docker-compose.yml up --build -d
	@for i in $$(seq 1 30); do curl -sf localhost:9000/health >/dev/null && echo 'gateway ready on :9000' && exit 0; sleep 1; done; echo 'gateway not ready - check: docker compose -f infrastructure/docker/docker-compose.yml logs gateway'; exit 1

compose-down:
	docker compose -f infrastructure/docker/docker-compose.yml down

k8s:           ## kind cluster: scheduling, pod-kill recovery, HPA, NetworkPolicy evidence
	python infrastructure/kind/k8s_evidence.py

k8s-down:
	-docker rm -f inferscale-ingress
	kind delete cluster --name inferscale

publish:
	bash scripts/publish_repo.sh

plots:
	python benchmarks/analysis/make_plots.py

colab-core:
	bash scripts/colab_setup.sh core && bash scripts/run_all_core.sh 3

colab-training:
	bash scripts/colab_setup.sh training && bash scripts/run_all_training.sh
