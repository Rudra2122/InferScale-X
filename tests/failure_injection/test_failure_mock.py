"""Crash/recovery + overload experiment on CPU mock backends (same code path as the vLLM run)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_crash_and_overload_mock(tmp_path):
    p = subprocess.run([sys.executable, "-m", "tests.failure_injection.exp_failure", "--kind", "mock",
                        "--duration", "8", "--rate", "30", "--out", str(tmp_path)],
                       cwd=ROOT, capture_output=True, text=True, timeout=240)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    crash = json.loads((tmp_path / "crash_report.json").read_text())
    over = json.loads((tmp_path / "overload_report.json").read_text())
    assert crash["time_to_detect_s"] is not None and crash["time_to_detect_s"] < 5
    assert crash["time_to_readmit_from_restart_s"] is not None
    # requests that STARTED during the outage must all succeed via the healthy replica
    assert crash["per_window"]["outage"]["failed"] == 0
    assert crash["per_window"]["after_readmit"]["served_by"]["replica-b"] > 0
    assert over["pass"], over["freetier_statuses"]
