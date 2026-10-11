import asyncio

import pytest

from gateway.api.app import tenant_cache_salt
from gateway.api.router import Router
from gateway.authentication.auth import AuthError, authenticate, authorize_model
from gateway.config import config_from_dict
from gateway.tenant_policies.policies import AdmissionController, TokenBucket, clamp_max_tokens

RAW = {
    "gateway": {"routing_policy": "round_robin", "global_max_inflight": 3, "unhealthy_after_failures": 2,
                "cache_salt_secret": "s"},
    "backends": [{"name": "a", "url": "http://a"}, {"name": "b", "url": "http://b"}, {"name": "c", "url": "http://c"}],
    "tenants": {
        "acme": {"api_key": "k-acme", "max_concurrent": 2, "requests_per_minute": 60, "burst": 2, "max_output_tokens": 100},
        "free": {"api_key": "k-free", "allowed_models": ["small"], "max_concurrent": 1},
    },
}


def cfg(**gw):
    raw = {**RAW, "gateway": {**RAW["gateway"], **gw}}
    return config_from_dict(raw)


def test_auth_paths():
    c = cfg()
    assert authenticate(c, "Bearer k-acme").tenant_id == "acme"
    for bad in (None, "", "k-acme", "Basic k-acme", "Bearer ", "Bearer nope"):
        with pytest.raises(AuthError) as e:
            authenticate(c, bad)
        assert e.value.status == 401


def test_keys_are_hashed_in_memory():
    c = cfg()
    assert all(len(t.api_key_sha256) == 64 and "k-" not in t.api_key_sha256 for t in c.tenants.values())


def test_model_authorization():
    c = cfg()
    authorize_model(c.tenants["acme"], "anything")
    authorize_model(c.tenants["free"], "small")
    with pytest.raises(AuthError) as e:
        authorize_model(c.tenants["free"], "big")
    assert e.value.status == 403


def test_token_bucket_with_fake_clock():
    t = [0.0]
    b = TokenBucket(rate_per_s=1.0, burst=2, clock=lambda: t[0])
    assert b.try_take()[0] and b.try_take()[0]
    ok, wait = b.try_take()
    assert not ok and 0.9 < wait <= 1.0
    t[0] += 1.0
    assert b.try_take()[0]


def test_admission_concurrency_and_global():
    c = cfg()
    adm = AdmissionController(c, clock=lambda: 0.0)
    adm.buckets["acme"] = TokenBucket(1000, 1000)  # isolate concurrency logic from the rate limiter
    acme = c.tenants["acme"]

    async def go():
        assert await adm.admit(acme) is None
        assert await adm.admit(acme) is None
        r = await adm.admit(acme)  # tenant concurrency 2
        assert r.status == 429 and r.reason == "tenant_concurrency"
        await adm.release(acme)
        free = c.tenants["free"]
        assert await adm.admit(free) is None  # in flight: acme 1 + free 1 = 2 (global cap 3)
        assert await adm.admit(acme) is None  # total 3
        r = await adm.admit(free)
        assert r.status == 503 and r.reason == "global_overload"

    asyncio.run(go())


def test_rate_limit_rejection():
    c = cfg()
    adm = AdmissionController(c, clock=lambda: 0.0)
    acme = c.tenants["acme"]

    async def go():
        for _ in range(2):
            assert await adm.admit(acme) is None
            await adm.release(acme)
        r = await adm.admit(acme)
        assert r.status == 429 and r.reason == "tenant_rate_limit" and r.retry_after_s > 0

    asyncio.run(go())


def test_clamp_max_tokens():
    t = cfg().tenants["acme"]
    assert clamp_max_tokens({"max_tokens": 5000}, t)["max_tokens"] == 100
    assert clamp_max_tokens({"max_tokens": 10}, t)["max_tokens"] == 10
    assert clamp_max_tokens({}, t)["max_tokens"] == 100
    assert clamp_max_tokens({"max_completion_tokens": 999}, t)["max_completion_tokens"] == 100


def test_round_robin_cycles():
    r = Router(cfg())
    assert [r.choose({}).name for _ in range(6)] == ["a", "b", "c", "a", "b", "c"]


def test_least_outstanding():
    r = Router(cfg(routing_policy="least_outstanding"))
    a, b, c = r.backends
    a.outstanding, b.outstanding, c.outstanding = 5, 1, 3
    assert r.choose({}).name == "b"


def test_prefix_affinity_is_sticky_and_spreads():
    r = Router(cfg(routing_policy="prefix_affinity", prefix_affinity_chars=64))
    body = lambda s: {"messages": [{"role": "user", "content": s}]}  # noqa: E731
    first = r.choose(body("shared system prompt " * 10 + "q1")).name
    assert all(r.choose(body("shared system prompt " * 10 + f"q{i}")).name == first for i in range(20))
    owners = {r.choose(body(f"doc {i} " * 20)).name for i in range(60)}
    assert len(owners) == 3  # different prefixes spread over replicas


def test_prefix_affinity_falls_back_when_overloaded():
    r = Router(cfg(routing_policy="prefix_affinity", prefix_affinity_max_imbalance=2))
    body = {"prompt": "x" * 100}
    pref = r.choose(body)
    pref.outstanding = 10
    assert r.choose(body).name != pref.name


def test_unhealthy_excluded_and_marked():
    r = Router(cfg(routing_policy="round_robin"))
    a = r.backends[0]
    r.on_start(a), r.on_finish(a, ok=False)
    assert a.healthy
    r.on_start(a), r.on_finish(a, ok=False)
    assert not a.healthy
    assert "a" not in {r.choose({}).name for _ in range(10)}


def test_cache_salt_per_tenant():
    c = cfg()
    s1, s2 = tenant_cache_salt(c, "acme"), tenant_cache_salt(c, "free")
    assert s1 != s2 and s1 == tenant_cache_salt(c, "acme") and len(s1) == 24


def test_env_override(monkeypatch):
    monkeypatch.setenv("INFERSCALE_BACKENDS", "x=http://x:1,y=http://y:2")
    monkeypatch.setenv("INFERSCALE_ROUTING_POLICY", "prefix_affinity")
    c = cfg()
    assert [b.name for b in c.backends] == ["x", "y"] and c.routing_policy == "prefix_affinity"
