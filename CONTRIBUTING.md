# Contributing

Thanks for helping. This repo has one rule above all: **no number without a result file.**

## Dev setup (CPU, any OS)
```bash
pip install -r requirements/dev.txt
python -m pytest          # must pass
make lint
```
GPU experiments run on an A100 (see `docs/RUNNING_ON_COLAB.md`); Kubernetes checks run on kind (`docs/RUNNING_ON_MAC.md`).

## Submitting benchmark results
1. Use an existing workload YAML (or add one) and `--repeats >= 3`.
2. Include the whole result directory (`env.json`, `workload.yaml`, `rep*/requests.jsonl`, `aggregate.json`/`report.json`).
3. Open an issue with the **Benchmark result** template: GPU, driver, CUDA, engine versions come from `env.json`, not memory.
4. Results that show an optimisation *losing* are welcome — add them to `docs/negative-results.md`.

## Code changes
- New experiments write to `benchmarks/results/<experiment>/<stamp>/report.json`, label measured / emulated / simulated / extrapolated, and use `benchmarks.analysis.stats.compare` for verdicts.
- Anything security-relevant (gateway auth, governance, sandbox) needs a test in `tests/security` or `tests/unit`.
- Keep engine-specific flags in the experiment scripts and note version changes in `docs/compatibility-matrix.md`.

## Commit / PR
Small PRs, one concern each. CI (`.github/workflows/ci.yml`) runs lint, the CPU suite and the security regression gate.
