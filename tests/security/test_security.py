"""Security suite (CPU): adversarial scripted model vs governance, sandbox escapes."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "agents" / "eval"))

from agents.sandbox.sandbox import run_python  # noqa: E402
from run_eval import ADVERSARIAL, security  # noqa: E402
from agents.agent import ScriptedLLM  # noqa: E402
from agents.skills.registry import SkillRegistry  # noqa: E402
from gateway.config import load_config  # noqa: E402


@pytest.fixture(scope="module")
def report():
    cfg = load_config(ROOT / "configs/gateway.colab.yaml")
    tenants = {t.tenant_id: set(t.allowed_tools) for t in cfg.tenants.values()}
    return security(lambda c: ScriptedLLM(ADVERSARIAL[c]), SkillRegistry(), tenants, "pytest")


def test_every_attack_blocked(report):
    failed = [c["case"] for c in report["cases"] if c["attack_succeeded"]]
    assert not failed, failed


def test_scripted_model_actually_attempted_attacks(report):
    # guards against a vacuous pass where the adversary never tried anything
    assert all(c["model_attempted"] for c in report["cases"])


def test_injection_taints_context(report):
    s2 = next(c for c in report["cases"] if c["case"] == "S2_prompt_injection")
    assert s2["detail"]["tainted"] and not s2["detail"]["leaked_key_written"]


def test_sandbox_allows_benign_code():
    r = run_python("import math, json\nprint(json.dumps({'v': math.factorial(10)}))")
    assert r.ok and '3628800' in r.stdout


def test_sandbox_timeout():
    r = run_python("import time\ntime.sleep(10)", timeout_s=1)
    assert r.timed_out and not r.ok


def test_sandbox_env_scrubbed(monkeypatch):
    monkeypatch.setenv("SUPER_SECRET_TOKEN", "abc123")
    r = run_python("import os\nprint(sorted(os.environ))")
    assert "SUPER_SECRET_TOKEN" not in r.stdout


def test_memory_limit_enforced_without_rlimit_as(monkeypatch):
    """macOS ignores RLIMIT_AS; the RSS watchdog must still kill a memory bomb (regression)."""
    monkeypatch.setenv("ISX_SANDBOX_SKIP_RLIMIT_AS", "1")
    monkeypatch.setenv("ISX_SANDBOX_SKIP_RLIMIT_DATA", "1")
    r = run_python("x = b'x' * (4 * 1024**3)", timeout_s=10, mem_mb=256)
    assert not r.ok and "memory limit exceeded" in r.stderr


def test_other_limits_survive_unsupported_rlimit(monkeypatch):
    """An unsupported limit must not silently drop the file-size limit (the original macOS bug)."""
    monkeypatch.setenv("ISX_SANDBOX_SKIP_RLIMIT_AS", "1")
    r = run_python("open('big.bin', 'wb').write(b'x' * (8 * 1024 * 1024))")
    assert not r.ok
