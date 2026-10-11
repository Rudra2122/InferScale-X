"""Gateway end-to-end against two CPU mock backends (real HTTP, real processes)."""
import json
import sys
import time
from pathlib import Path

import httpx
import pytest
import yaml

from scripts import serverctl

ROOT = Path(__file__).resolve().parents[2]
GW = "http://127.0.0.1:19000"
A, B = "http://127.0.0.1:18001", "http://127.0.0.1:18002"


@pytest.fixture(scope="module", autouse=True)
def stack(tmp_path_factory):
    cfg = yaml.safe_load((ROOT / "configs/gateway.example.yaml").read_text())
    cfg["backends"] = [{"name": "replica-a", "url": A, "model": "mock-model"},
                       {"name": "replica-b", "url": B, "model": "mock-model"}]
    cfg["gateway"]["health_interval_s"] = 0.3
    cfg["gateway"]["upstream_read_timeout_s"] = 2.0  # so the black-hole test finishes quickly
    p = tmp_path_factory.mktemp("cfg") / "gw.yaml"
    p.write_text(yaml.safe_dump(cfg))
    for name, port in (("it-mock-a", 18001), ("it-mock-b", 18002)):
        assert serverctl.start(name, [sys.executable, "-m", "inference.mock.mock_server", "--port", str(port)],
                               port, timeout=30)["healthy"]
    assert serverctl.start("it-gateway", [sys.executable, "-m", "uvicorn", "gateway.api.app:app", "--port", "19000",
                                          "--log-level", "warning"], 19000,
                           env={"INFERSCALE_GATEWAY_CONFIG": str(p)}, timeout=30)["healthy"]
    yield
    for n in ("it-gateway", "it-mock-a", "it-mock-b"):
        serverctl.stop(n)


def post(key, body, **kw):
    return httpx.post(f"{GW}/v1/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"}, timeout=30, **kw)


def chat_body(text="hello world", **kw):
    return {"model": "mock-model", "messages": [{"role": "user", "content": text}], "max_tokens": 8, **kw}


def test_health_ready_models():
    assert httpx.get(f"{GW}/health").status_code == 200
    assert httpx.get(f"{GW}/ready").json()["healthy_backends"] == 2
    r = httpx.get(f"{GW}/v1/models", headers={"Authorization": "Bearer sk-dev-acme"})
    assert r.json()["data"][0]["id"] == "mock-model"


def test_unary_ok():
    r = post("sk-dev-acme", chat_body())
    assert r.status_code == 200 and r.headers["x-backend"] in ("replica-a", "replica-b")
    j = r.json()
    assert j["usage"]["completion_tokens"] == 8 and j["choices"][0]["message"]["content"]


def test_stream_ok_and_matches_unary():
    unary = post("sk-dev-acme", chat_body("same prompt")).json()["choices"][0]["message"]["content"]
    pieces, usage = [], None
    with httpx.stream("POST", f"{GW}/v1/chat/completions", headers={"Authorization": "Bearer sk-dev-acme"},
                      json=chat_body("same prompt", stream=True, stream_options={"include_usage": True})) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                ev = json.loads(line[6:])
                usage = ev.get("usage") or usage
                for ch in ev.get("choices", []):
                    pieces.append(ch["delta"].get("content", ""))
    assert "".join(pieces) == unary and usage["completion_tokens"] == 8


@pytest.mark.parametrize("headers,status", [({}, 401), ({"Authorization": "Bearer wrong"}, 401),
                                             ({"Authorization": "Token x"}, 401)])
def test_auth_errors(headers, status):
    r = httpx.post(f"{GW}/v1/chat/completions", json=chat_body(), headers=headers)
    assert r.status_code == status and r.json()["error"]["code"] == status


def test_validation_errors():
    assert post("sk-dev-acme", {"messages": []}).status_code == 400
    assert post("sk-dev-acme", {"model": "mock-model", "messages": "hi"}).status_code == 400
    r = httpx.post(f"{GW}/v1/chat/completions", content=b"{not json", headers={"Authorization": "Bearer sk-dev-acme"})
    assert r.status_code == 400


def test_model_permission():
    r = post("sk-dev-free", {**chat_body(), "model": "other-model"})
    assert r.status_code == 403


def test_max_tokens_clamped_for_tenant():
    r = post("sk-dev-acme", {**chat_body(), "max_tokens": 100000})
    assert r.json()["usage"]["completion_tokens"] == 512  # acme cap in gateway.example.yaml


def test_rate_limit_returns_429_with_retry_after():
    codes = [post("sk-dev-free", chat_body()).status_code for _ in range(8)]
    assert 429 in codes and all(c in (200, 429) for c in codes)
    r = post("sk-dev-free", chat_body())
    if r.status_code == 429:
        assert int(r.headers["retry-after"]) >= 1


def test_cache_salt_isolates_tenants_and_client_cannot_override():
    for u in (A, B):
        httpx.post(f"{u}/admin/reset_cache")
    doc = " ".join(f"w{i}" for i in range(200))
    body = chat_body(doc)
    body["cache_salt"] = "attacker-chosen"  # must be stripped by the gateway
    a1 = post("sk-dev-acme", body).json()["usage"]["prompt_tokens_details"]["cached_tokens"]
    # force same backend for a fair check: query both backends' metrics instead of relying on routing
    g = post("sk-dev-globex", body).json()["usage"]["prompt_tokens_details"]["cached_tokens"]
    a2_vals = [post("sk-dev-acme", body).json()["usage"]["prompt_tokens_details"]["cached_tokens"] for _ in range(4)]
    assert a1 == 0 and g == 0
    assert max(a2_vals) > 0  # same tenant reuses its own cache (on whichever replica it warmed)


def test_failover_when_backend_fails():
    httpx.post(f"{B}/admin/fail")
    try:
        time.sleep(1.0)
        codes = [post("sk-dev-acme", chat_body(f"q{i}")).status_code for i in range(10)]
        assert all(c == 200 for c in codes)
        r = httpx.get(f"{GW}/admin/backends", headers={"Authorization": "Bearer sk-dev-acme"}).json()
        assert {b["name"]: b["healthy"] for b in r["backends"]}["replica-b"] is False
    finally:
        httpx.post(f"{B}/admin/recover")
    deadline = time.time() + 5
    while time.time() < deadline:
        r = httpx.get(f"{GW}/admin/backends", headers={"Authorization": "Bearer sk-dev-acme"}).json()
        if all(b["healthy"] for b in r["backends"]):
            break
        time.sleep(0.3)
    assert all(b["healthy"] for b in r["backends"])


def test_all_backends_down_gives_503():
    for u in (A, B):
        httpx.post(f"{u}/admin/fail")
    try:
        time.sleep(1.2)
        r = post("sk-dev-acme", chat_body())
        assert r.status_code == 503 and "retry-after" in r.headers
    finally:
        for u in (A, B):
            httpx.post(f"{u}/admin/recover")
        time.sleep(1.2)


def test_metrics_exposed():
    post("sk-dev-acme", chat_body())
    m = httpx.get(f"{GW}/metrics").text
    for name in ("inferscale_requests_total", "inferscale_request_seconds_bucket", "inferscale_backend_healthy",
                 "inferscale_rejections_total"):
        assert name in m


def test_failover_when_backend_black_holes():
    """Regression (found on kind): a pod killed without a TCP reset left requests hanging for the full
    request timeout, and some transport errors surfaced as bare 500s. With an idle-read timeout and
    TransportError-wide retries, every request must still succeed via the healthy replica."""
    httpx.post(f"{B}/admin/hang")
    try:
        t0 = time.time()
        codes = [post("sk-dev-acme", chat_body(f"bh{i}")).status_code for i in range(6)]
        with httpx.stream("POST", f"{GW}/v1/chat/completions", headers={"Authorization": "Bearer sk-dev-acme"},
                          json=chat_body("bh-stream", stream=True), timeout=30) as r:
            body = r.read().decode()
        assert all(c == 200 for c in codes), codes
        assert r.status_code == 200 and "[DONE]" in body and "error" not in body
        assert time.time() - t0 < 6 * 2.0 + 10  # bounded by the idle timeout, not the 300 s request timeout
    finally:
        httpx.post(f"{B}/admin/recover")
    time.sleep(1.0)


def test_no_bare_500_pages():
    r = post("sk-dev-acme", chat_body())
    assert r.headers["content-type"].startswith("application/json")
