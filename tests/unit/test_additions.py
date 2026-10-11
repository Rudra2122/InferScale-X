import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agents" / "eval"))

import retrieval_eval as rev  # noqa: E402


# ---------- retrieval evaluation ----------
def test_metrics_math():
    m = rev.metrics(["a", "b", "c", "d"], {"b", "x"})
    assert m["hit@1"] == 0 and m["recall@3"] == 0.5 and m["mrr@10"] == 0.5
    assert rev.metrics(["a"], {"a"})["ndcg@5"] == pytest.approx(1.0)
    assert rev.doc_ranking(["a", "a", "b", "a", "c"]) == ["a", "b", "c"]


def test_qrels_are_consistent():
    spec = yaml.safe_load(rev.QRELS.read_text())
    corpus = set(spec["corpus"])
    assert all((ROOT / p).exists() for p in corpus)
    ids = [q["id"] for q in spec["queries"]]
    assert len(ids) == len(set(ids)) >= 30
    for q in spec["queries"]:
        assert q["relevant"] and set(q["relevant"]) <= corpus, q["id"]


def test_bm25_eval_runs():
    rep = rev.run(no_dense=True)
    agg = rep["retrievers"]["bm25"]["aggregate"]
    assert rep["n_queries"] >= 30 and 0 <= agg["mrr@10"] <= 1


# ---------- JD coverage ----------
def test_jd_coverage_schema_and_code_paths():
    spec = yaml.safe_load((ROOT / "docs/jd_coverage.yaml").read_text())["keywords"]
    assert len(spec) >= 22
    for k in spec:
        assert k["scope"] in {"built", "attempted", "emulated", "partial", "design_only", "not_done"}, k["keyword"]
        for p in k.get("code", []):
            assert (ROOT / p).exists(), f"{k['keyword']}: {p} missing"
        if k["scope"] == "design_only":
            assert not k.get("evidence"), "design-only items must not pretend to have evidence"


def test_evidence_report_renders_jd(tmp_path):
    spec = importlib.util.spec_from_file_location("evr", ROOT / "scripts/evidence_report.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["evr"] = m
    spec.loader.exec_module(m)
    rows = m.jd_coverage()
    assert {r["status"] for r in rows} <= {"EVIDENCED", "PARTIALLY EVIDENCED", "CODE ONLY (not yet run)", "DESIGN-ONLY"}


# ---------- Triton kernels in interpreter mode (CPU) ----------
def _have_triton():
    try:
        import torch  # noqa: F401
        import triton  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_triton(), reason="torch/triton not installed")
@pytest.mark.parametrize("script", ["optimization/triton-kernels/flash_attention.py",
                                    "optimization/triton-kernels/fused_add_rmsnorm.py"])
def test_triton_kernels_interpreter(script):
    p = subprocess.run([sys.executable, script, "--check-only"], cwd=ROOT, capture_output=True, text=True,
                       timeout=600, env={**os.environ, "TRITON_INTERPRET": "1"})
    assert p.returncode == 0, p.stdout[-1500:] + p.stderr[-1500:]


# ---------- deployment files ----------
def test_kind_and_compose_files_parse():
    docs = [d for d in yaml.safe_load_all((ROOT / "infrastructure/kind/manifests.yaml").read_text()) if d]
    kinds = {d["kind"] for d in docs}
    assert {"StatefulSet", "Deployment", "HorizontalPodAutoscaler", "NetworkPolicy", "Service"} <= kinds
    cfg = yaml.safe_load(next(d for d in docs if d["kind"] == "Secret")["stringData"]["gateway.yaml"])
    from gateway.config import config_from_dict

    assert len(config_from_dict(cfg).backends) == 2
    kc = yaml.safe_load((ROOT / "infrastructure/kind/kind-config.yaml").read_text())
    assert kc["networking"]["disableDefaultCNI"] is True and len(kc["nodes"]) == 3
    comp = yaml.safe_load((ROOT / "infrastructure/docker/docker-compose.yml").read_text())
    assert {"gateway", "mock-a", "mock-b", "prometheus", "grafana"} <= set(comp["services"])


def test_offload_candidates_are_well_formed():
    spec = importlib.util.spec_from_file_location("off", ROOT / "disaggregation/kv-cache/exp_kv_offload.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["off"] = m
    spec.loader.exec_module(m)
    import shlex

    for name, extra, env in m.cpu_offload_candidates(8.0, 1000):
        shlex.split(extra)  # must survive the same parsing serverctl applies
        assert name and isinstance(env, dict)
