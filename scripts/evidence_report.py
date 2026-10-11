"""Generate docs/READINESS.md: the 12 acceptance tests, judged ONLY from evidence files.

Status per test: VERIFIED (all evidence present and passing), PARTIAL (some evidence,
or evidence that is explicitly emulated/simulated), MISSING (no evidence). Also renders
docs/claims.yaml with real numbers, and lists negative-result candidates found in reports.

python scripts/evidence_report.py
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "benchmarks" / "results"


def latest(pattern: str) -> Path | None:
    files = sorted(glob.glob(str(ROOT / pattern)), key=lambda p: Path(p).stat().st_mtime)
    return Path(files[-1]) if files else None


def load(p: Path | None):
    try:
        return json.loads(p.read_text()) if p else None
    except Exception:
        return None


def dig(obj, dotted: str):
    for k in dotted.split("."):
        if isinstance(obj, dict) and k in obj:
            obj = obj[k]
        elif isinstance(obj, list) and k.isdigit() and int(k) < len(obj):
            obj = obj[int(k)]
        else:
            return None
    return obj


def rel(p: Path | None) -> str:
    return str(p.relative_to(ROOT)) if p else "-"


def junit_ok(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    import xml.etree.ElementTree as ET

    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root)
    t = sum(int(s.get("tests", 0)) for s in suites)
    f = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
    sk = sum(int(s.get("skipped", 0)) for s in suites)
    return (t > 0 and f == 0 and sk < t), f"{t} tests, {f} failed, {sk} skipped"


def check_all() -> list[dict]:
    rows = []

    def add(name, status, evidence, note=""):
        rows.append({"test": name, "status": status, "evidence": evidence, "note": note})

    env = latest("benchmarks/results/*/*/env.json") or latest("benchmarks/results/*/*/*/env.json")
    lock = ROOT / "requirements" / "lock-colab.txt"
    base = latest("benchmarks/results/baseline/*/report.json")
    add("Reproducibility", "VERIFIED" if (env and lock.exists() and base) else ("PARTIAL" if env or base else "MISSING"),
        [rel(env), rel(lock) if lock.exists() else "requirements/lock-colab.txt (missing)", rel(base)],
        "Needs env capture + pinned lockfile from the run + a baseline report. Fresh-runtime rerun is a manual step.")

    ok_u, msg_u = junit_ok(ROOT / "tests" / "results" / "junit-unit.xml")
    ok_l, msg_l = junit_ok(ROOT / "tests" / "results" / "junit-live.xml")
    add("Correctness", "VERIFIED" if ok_u and ok_l else ("PARTIAL" if ok_u else "MISSING"),
        [f"tests/results/junit-unit.xml ({msg_u})", f"tests/results/junit-live.xml ({msg_l})"],
        "Live = streaming/non-streaming parity, determinism, usage, limits against real vLLM.")

    b = load(base)
    reps = dig(b, "repeats") or 0
    add("Performance", "VERIFIED" if b and reps >= 3 else ("PARTIAL" if b else "MISSING"), [rel(base)],
        f"repeats={reps}; every experiment reports mean, std and 95% CI across repeats.")

    inter = latest("benchmarks/results/interference/*/report.json")
    kvt = latest("benchmarks/results/kv_transfer/*/report.json")
    dis = latest("benchmarks/results/disagg_attempt/*/attempt.json")
    dis_ok = any(a.get("ok") for a in (dig(load(dis), "attempts") or []))
    add("Disaggregation", "PARTIAL" if (inter or kvt) else "MISSING", [rel(inter), rel(kvt), rel(dis)],
        f"Single GPU: interference measured + emulated disaggregated TTFT; real vLLM P/D connector attempt ok={dis_ok}. "
        "Cannot be VERIFIED without separate prefill/decode GPUs.")

    k = load(kvt)
    errs = [r.get("prediction_error_pct", {}).get("host_roundtrip_pinned") for r in (dig(k, "kv.rows") or [])]
    errs = [e for e in errs if e is not None]
    add("KV movement", "VERIFIED" if errs else "MISSING", [rel(kvt)],
        f"cost-model error per size (host round trip, %): {[round(e, 1) for e in errs]}" if errs else "")

    crash = latest("benchmarks/results/failure_injection/*vllm/crash_report.json")
    over = latest("benchmarks/results/failure_injection/*vllm/overload_report.json")
    add("Reliability", "VERIFIED" if crash and dig(load(over), "pass") else ("PARTIAL" if crash or over else "MISSING"),
        [rel(crash), rel(over)], "Crash/restart timings + overload shedding with real vLLM.")

    kvc = load(latest("benchmarks/results/kv_cache/*/report.json"))
    ag = latest("benchmarks/results/agents/*/report.json")
    s3 = [c for c in (dig(load(ag), "security_adversarial_scripted.cases") or []) if c["case"].startswith("S3")]
    iso_ok = dig(kvc, "isolation.isolation_enforced") is True and s3 and not s3[0]["attack_succeeded"]
    add("Isolation", "VERIFIED" if iso_ok else ("PARTIAL" if kvc or s3 else "MISSING"),
        [rel(latest("benchmarks/results/kv_cache/*/report.json")), rel(ag), "tests/unit/test_gateway.py"],
        f"KV cache salt enforced={dig(kvc, 'isolation.isolation_enforced')}; memory cross-tenant blocked={bool(s3) and not s3[0]['attack_succeeded']}")

    live = latest("benchmarks/results/agents/*live/report.json")
    sec_live = dig(load(live), "security_live_model.all_blocked")
    sec_scr = dig(load(ag), "security_adversarial_scripted.all_blocked")
    add("Security", "VERIFIED" if sec_live and sec_scr else ("PARTIAL" if sec_scr else "MISSING"), [rel(live), rel(ag)],
        "Prompt-injection + unauthorized-tool + sandbox-escape suites (scripted adversary and live model).")

    rt = load(latest("benchmarks/results/routing/*/report.json"))
    cap = ROOT / "docs" / "capacity-planning" / "README.md"
    add("Scaling", "VERIFIED" if cap.exists() and rt and rt.get("shared_gpu") is not None else "PARTIAL",
        [rel(latest("benchmarks/results/routing/*/report.json")), "docs/capacity-planning/README.md"],
        "Measured: 1 GPU, 2 replicas. Everything above that is labelled EXTRAPOLATED by capacity_model.py.")

    need = ["docs/partner-deployment-guide.md", "docs/capacity-planning/capacity_model.py",
            "docs/incident-runbooks/backend-down.md", "docs/architecture/README.md"]
    have = [p for p in need if (ROOT / p).exists()]
    add("Customer readiness", "VERIFIED" if len(have) == len(need) else "PARTIAL", have,
        "Docs exist; a real partner walkthrough is still a human activity.")

    claims = render_claims()
    missing = [c for c in claims if c["text"].startswith("NO EVIDENCE")]
    backed = len(claims) - len(missing)
    add("Traceability", "VERIFIED" if claims and not missing else ("PARTIAL" if backed else "MISSING"),
        ["docs/CLAIMS.md"], f"{len(claims) - len(missing)}/{len(claims)} claims backed by evidence files.")

    neg = negative_candidates()
    doc = ROOT / "docs" / "negative-results.md"
    filled = doc.exists() and "TODO" not in doc.read_text()
    add("Negative results", "VERIFIED" if neg and filled else ("PARTIAL" if neg else "MISSING"),
        ["docs/negative-results.md"] + [n["source"] for n in neg[:5]],
        f"{len(neg)} non-improving comparisons found in reports; explain at least one in docs/negative-results.md.")
    return rows


def render_claims() -> list[dict]:
    spec = yaml.safe_load((ROOT / "docs" / "claims.yaml").read_text())["claims"]
    out = []
    for c in spec:
        p = latest(c["evidence"])
        data = load(p)
        if data is None:
            out.append({"id": c["id"], "text": f"NO EVIDENCE - do not use ({c['evidence']})", "evidence": None})
            continue
        vals = {k: dig(data, path) for k, path in (c.get("fields") or {}).items()}
        if any(v is None for v in vals.values()):
            out.append({"id": c["id"], "text": f"NO EVIDENCE - fields missing {[k for k, v in vals.items() if v is None]}",
                        "evidence": rel(p)})
            continue
        try:
            text = c["template"].format(**vals)
        except Exception as e:
            text = f"NO EVIDENCE - template error {e}"
        out.append({"id": c["id"], "text": text, "evidence": rel(p), "values": vals})
    return out


def jd_coverage() -> list[dict]:
    spec = yaml.safe_load((ROOT / "docs" / "jd_coverage.yaml").read_text())["keywords"]
    rows = []
    for k in spec:
        found, details = [], []
        for ev in k.get("evidence") or []:
            if "file_contains" in ev:
                p = ROOT / ev["file_contains"]
                ok = p.exists() and ev["text"] in p.read_text()
                found.append(ok)
                details.append(f"{ev['file_contains']} contains '{ev['text']}': {ok}")
                continue
            p = latest(ev["glob"])
            if p is None:
                found.append(False)
                details.append(f"{ev['glob']}: missing")
                continue
            if "field" in ev:
                val = dig(load(p), ev["field"]) if p.suffix == ".json" else None
                ok = val == ev.get("equals", val) and val is not None
                details.append(f"{rel(p)} {ev['field']}={val}")
            else:
                ok = True
                details.append(rel(p))
            found.append(ok)
        if not k.get("evidence"):
            status = "DESIGN-ONLY"
        elif all(found):
            status = "EVIDENCED"
        elif any(found):
            status = "PARTIALLY EVIDENCED"
        else:
            status = "CODE ONLY (not yet run)" if k["scope"] not in ("design_only", "not_done") else "DESIGN-ONLY"
        rows.append({**k, "status": status, "details": details})
    return rows


def negative_candidates() -> list[dict]:
    found = []

    def walk(obj, path, src):
        if isinstance(obj, dict):
            if obj.get("verdict") in ("regressed", "no_significant_difference"):
                found.append({"source": src, "path": path, "verdict": obj["verdict"],
                              "relative_change": obj.get("relative_change")})
            for k, v in obj.items():
                walk(v, f"{path}.{k}" if path else k, src)

    for p in glob.glob(str(R / "*" / "*" / "report.json")):
        d = load(Path(p))
        if d:
            walk(d, "", rel(Path(p)))
    return found


def main():
    rows = check_all()
    claims = render_claims()
    neg = negative_candidates()
    v = sum(r["status"] == "VERIFIED" for r in rows)
    lines = [f"# Readiness checklist - {v}/12 verified",
             "", f"_Generated {time.strftime('%Y-%m-%d %H:%M')} by scripts/evidence_report.py from files in "
             "benchmarks/results/ and tests/results/. Do not edit by hand._", "",
             "| # | Test | Status | Evidence | Note |", "|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        ev = "<br>".join(str(e) for e in r["evidence"] if e and e != "-") or "-"
        lines.append(f"| {i} | {r['test']} | **{r['status']}** | {ev} | {r['note']} |")
    lines += ["", "## Negative-result candidates (auto-detected)", ""]
    lines += [f"- `{n['source']}` → `{n['path']}`: {n['verdict']} ({n['relative_change']:+.1%})"
              if isinstance(n.get("relative_change"), (int, float)) else f"- `{n['source']}` → `{n['path']}`: {n['verdict']}"
              for n in neg[:30]] or ["- none found yet"]
    (ROOT / "docs" / "READINESS.md").write_text("\n".join(lines) + "\n")
    cl = ["# Evidence-backed claims", "", "_Rendered from docs/claims.yaml with numbers read from result files._", ""]
    for c in claims:
        cl.append(f"- **{c['id']}**: {c['text']}" + (f"  \n  evidence: `{c['evidence']}`" if c.get("evidence") else ""))
    (ROOT / "docs" / "CLAIMS.md").write_text("\n".join(cl) + "\n")
    jd = jd_coverage()
    impl = [r for r in jd if r["scope"] in ("built", "attempted", "emulated", "partial")]
    ev = [r for r in jd if r["status"] == "EVIDENCED"]
    jl = ["# NVIDIA JD coverage", "",
          f"_Generated by scripts/evidence_report.py from docs/jd_coverage.yaml. {len(jd)} keywords: "
          f"**{len(impl)} implemented in code** (built/attempted/emulated/partial), "
          f"**{len(jd) - len(impl)} design-only**, **{len(ev)} evidenced by result files**._", "",
          "Scope = what the code does (fixed). Status = what the latest results prove.", "",
          "| Keyword | Scope | Status | Deliverable | Limits | Evidence checked |", "|---|---|---|---|---|---|"]
    for r in jd:
        jl.append(f"| {r['keyword']} | {r['scope']} | **{r['status']}** | {r['deliverable']} | {r.get('limits', '')} | "
                  + "<br>".join(r["details"]) + " |")
    (ROOT / "docs" / "jd-coverage.md").write_text("\n".join(jl) + "\n")
    (R).mkdir(parents=True, exist_ok=True)
    (R / "evidence_index.json").write_text(json.dumps({"checks": rows, "claims": claims, "negative": neg, "jd": jd},
                                                      indent=2,
                                                      default=str))
    print("\n".join(lines[:18]))
    print(f"\nwrote docs/READINESS.md, docs/CLAIMS.md, docs/jd-coverage.md ({v}/12 verified; "
          f"JD: {len(impl)}/{len(jd)} implemented, {len(ev)} evidenced)")


if __name__ == "__main__":
    sys.exit(main())
