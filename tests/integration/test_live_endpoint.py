"""Correctness tests against a REAL model server (run on Colab after starting vLLM + gateway).

INFERSCALE_LIVE_URL=http://127.0.0.1:9000 INFERSCALE_LIVE_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
    pytest tests/integration/test_live_endpoint.py --junitxml tests/results/junit-live.xml
Skipped automatically when INFERSCALE_LIVE_URL is not set.
"""
import json
import os

import httpx
import pytest

URL = os.environ.get("INFERSCALE_LIVE_URL")
MODEL = os.environ.get("INFERSCALE_LIVE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
KEY = os.environ.get("INFERSCALE_LIVE_KEY", "sk-dev-acme")
pytestmark = pytest.mark.skipif(not URL, reason="INFERSCALE_LIVE_URL not set")
H = {"Authorization": f"Bearer {KEY}"}


def body(prompt, **kw):
    return {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0,
            "max_tokens": 48, "seed": 0, **kw}


def unary(prompt, **kw):
    r = httpx.post(f"{URL}/v1/chat/completions", json=body(prompt, **kw), headers=H, timeout=120)
    assert r.status_code == 200, r.text
    return r.json()


def stream(prompt, **kw):
    text, usage, n = [], None, 0
    with httpx.stream("POST", f"{URL}/v1/chat/completions", headers=H, timeout=120,
                      json=body(prompt, stream=True, stream_options={"include_usage": True}, **kw)) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            assert "error" not in ev, ev
            usage = ev.get("usage") or usage
            for ch in ev.get("choices", []):
                text.append(ch.get("delta", {}).get("content") or "")
                n += 1
    return "".join(text), usage, n


def test_basic_answer_is_sane():
    out = unary("What is 17 + 25? Reply with only the number.")["choices"][0]["message"]["content"]
    assert "42" in out


def test_greedy_is_deterministic():
    outs = {unary("Name three primary colors.")["choices"][0]["message"]["content"] for _ in range(3)}
    assert len(outs) == 1


def test_stream_matches_unary_and_usage_consistent():
    p = "Write one sentence about GPUs."
    u = unary(p)
    s_text, s_usage, n_chunks = stream(p)
    assert s_text == u["choices"][0]["message"]["content"]
    assert s_usage["completion_tokens"] == u["usage"]["completion_tokens"]
    assert n_chunks >= 1


def test_max_tokens_honoured():
    j = unary("Count from 1 to 200 separated by spaces.", max_tokens=10)
    assert j["usage"]["completion_tokens"] <= 10 and j["choices"][0]["finish_reason"] == "length"


def test_stop_sequence():
    out = unary("List the letters A, B, C, D each on its own line.", stop=["C"])["choices"][0]["message"]["content"]
    assert "D" not in out


def test_completions_endpoint():
    r = httpx.post(f"{URL}/v1/completions", headers=H, timeout=60,
                   json={"model": MODEL, "prompt": "The capital of France is", "max_tokens": 5, "temperature": 0})
    assert r.status_code == 200 and "Paris" in r.json()["choices"][0]["text"]


def test_concurrent_requests_all_succeed():
    import concurrent.futures as cf

    with cf.ThreadPoolExecutor(16) as ex:
        res = list(ex.map(lambda i: unary(f"Say the number {i}.")["usage"]["completion_tokens"], range(32)))
    assert len(res) == 32 and all(r > 0 for r in res)


def test_error_format_for_bad_model():
    r = httpx.post(f"{URL}/v1/chat/completions", headers=H, json=body("hi", model="no-such-model"), timeout=60)
    assert r.status_code in (400, 403, 404) and "error" in r.json() or "detail" in r.json()
