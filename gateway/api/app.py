"""InferScale-X gateway: OpenAI-compatible front door for vLLM / SGLang / mock backends.

Run:
    INFERSCALE_GATEWAY_CONFIG=configs/gateway.colab.yaml \
        uvicorn gateway.api.app:app --host 0.0.0.0 --port 9000
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from gateway.api import observability as obs
from gateway.api.router import NoHealthyBackend, Router
from gateway.authentication.auth import AuthError, authenticate, authorize_model
from gateway.config import GatewayConfig, TenantConfig, load_config
from gateway.tenant_policies.policies import AdmissionController, clamp_max_tokens

log = obs.get_logger("inferscale.gateway")


def error(status: int, message: str, etype: str = "invalid_request_error", headers: dict | None = None):
    return JSONResponse({"error": {"message": message, "type": etype, "code": status}}, status_code=status,
                        headers=headers)


def tenant_cache_salt(cfg: GatewayConfig, tenant_id: str) -> str:
    return hmac.new(cfg.cache_salt_secret.encode(), tenant_id.encode(), hashlib.sha256).hexdigest()[:24]


class GatewayState:
    def __init__(self, cfg: GatewayConfig):
        self.cfg = cfg
        self.router = Router(cfg)
        self.admission = AdmissionController(cfg)
        # read= is the max gap between bytes, not total time: a backend killed without a TCP reset
        # (force-deleted pod, dead node) is detected in upstream_read_timeout_s instead of request_timeout_s.
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(cfg.request_timeout_s, connect=5.0,
                                                              read=cfg.upstream_read_timeout_s),
                                        limits=httpx.Limits(max_connections=1024, max_keepalive_connections=256))


def create_app(cfg: GatewayConfig | None = None, start_health: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state = GatewayState(cfg or load_config())
        app.state.gw = state
        obs.init_tracing()
        if start_health:
            state.router.start_health_checks()
        log_event("gateway_start", policy=state.cfg.routing_policy,
                  backends=[b.url for b in state.cfg.backends])
        yield
        await state.router.stop()
        await state.client.aclose()

    app = FastAPI(title="InferScale-X Gateway", version="2.0.0", lifespan=lifespan)

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception):  # never leak a bare 500 page
        log_event("unhandled_error", path=request.url.path, err=repr(exc)[:300])
        return error(502, f"gateway error: {type(exc).__name__}", "backend_error")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request):
        gw: GatewayState = request.app.state.gw
        n = len(gw.router.healthy())
        return JSONResponse({"ready": n > 0, "healthy_backends": n}, status_code=200 if n else 503)

    @app.get("/metrics")
    async def metrics(request: Request):
        gw: GatewayState = request.app.state.gw
        for b in gw.router.backends:
            obs.BACKEND_HEALTHY.labels(b.name).set(1 if b.healthy else 0)
            obs.BACKEND_OUTSTANDING.labels(b.name).set(b.outstanding)
        return Response(generate_latest(obs.REGISTRY), media_type=CONTENT_TYPE_LATEST)

    @app.get("/admin/backends")
    async def backends(request: Request):
        gw: GatewayState = request.app.state.gw
        tenant = _auth_or_none(gw, request)
        if isinstance(tenant, JSONResponse):
            return tenant
        return {"policy": gw.cfg.routing_policy, "backends": gw.router.snapshot()}

    @app.get("/v1/models")
    async def models(request: Request):
        gw: GatewayState = request.app.state.gw
        tenant = _auth_or_none(gw, request)
        if isinstance(tenant, JSONResponse):
            return tenant
        names = {b.model for b in gw.cfg.backends if b.model}
        for b in gw.router.healthy():  # backends configured without a model name: ask them
            if not b.cfg.model:
                try:
                    r = await gw.client.get(f"{b.cfg.url}/v1/models", timeout=3.0)
                    names |= {m["id"] for m in r.json().get("data", []) if m.get("id")}
                except Exception:
                    pass
        names = sorted(names)
        names = [m for m in names if tenant.model_allowed(m)]
        return {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "inferscale"} for m in names]}

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await _proxy(request, "/v1/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await _proxy(request, "/v1/completions")

    return app


def log_event(msg: str, **fields):
    obs.log_event(log, msg, **fields)


def _auth_or_none(gw: GatewayState, request: Request):
    try:
        return authenticate(gw.cfg, request.headers.get("authorization"))
    except AuthError as e:
        return error(e.status, e.message, "authentication_error" if e.status == 401 else "permission_error")


async def _proxy(request: Request, path: str):
    gw: GatewayState = request.app.state.gw
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex
    t0 = time.perf_counter()

    tenant = _auth_or_none(gw, request)
    if isinstance(tenant, JSONResponse):
        return tenant

    try:
        body = await request.json()
    except Exception:
        return error(400, "body must be valid JSON")
    if not isinstance(body, dict) or not body.get("model"):
        return error(400, "'model' is required")
    if path.endswith("chat/completions") and not isinstance(body.get("messages"), list):
        return error(400, "'messages' must be a list")
    if path.endswith("/completions") and not path.endswith("chat/completions") and "prompt" not in body:
        return error(400, "'prompt' is required")
    try:
        authorize_model(tenant, body["model"])
    except AuthError as e:
        return error(e.status, e.message, "permission_error")

    body = clamp_max_tokens(body, tenant)
    body.pop("cache_salt", None)  # clients may never choose another tenant's salt
    if gw.cfg.inject_cache_salt:
        body["cache_salt"] = tenant_cache_salt(gw.cfg, tenant.tenant_id)
    stream = bool(body.get("stream"))

    rej = await gw.admission.admit(tenant)
    if rej:
        obs.REJECTS.labels(tenant.tenant_id, rej.reason).inc()
        log_event("rejected", request_id=rid, tenant=tenant.tenant_id, reason=rej.reason)
        return error(rej.status, f"request rejected: {rej.reason}", "rate_limit_error" if rej.status == 429
                     else "overloaded_error", headers={"Retry-After": str(max(1, int(rej.retry_after_s + 0.999)))})
    obs.INFLIGHT.inc()

    released = False

    async def release():
        nonlocal released
        if not released:
            released = True
            obs.INFLIGHT.dec()
            await gw.admission.release(tenant)

    try:
        if stream:
            return await _proxy_stream(gw, tenant, body, path, rid, t0, release)
        return await _proxy_unary(gw, tenant, body, path, rid, t0)
    finally:
        if not stream:
            await release()


async def _open_backend(gw: GatewayState, body: dict, path: str, rid: str, stream: bool):
    """Pick a backend and send. Retries on connect errors / 5xx before any byte reached the client."""
    tried: set[str] = set()
    attempts = 1 + (gw.cfg.max_retries if gw.cfg.retry_on_backend_error else 0)
    last_err = "no attempt"
    for attempt in range(attempts):
        try:
            b = gw.router.choose(body, exclude=tried)
        except NoHealthyBackend:
            break
        tried.add(b.name)
        gw.router.on_start(b)
        try:
            req = gw.client.build_request("POST", f"{b.cfg.url}{path}", json=body, headers={"x-request-id": rid})
            resp = await gw.client.send(req, stream=stream)
            if resp.status_code >= 500:
                last_err = f"backend {b.name} returned {resp.status_code}"
                await resp.aclose()
                gw.router.on_finish(b, ok=False)
                if attempt + 1 < attempts:
                    obs.RETRIES.labels(b.name).inc()
                continue
            return b, resp
        except httpx.TransportError as e:  # connect/read/write errors and all timeouts -> try another replica
            last_err = f"backend {b.name} unreachable: {type(e).__name__}"
            gw.router.on_finish(b, ok=False)
            if attempt + 1 < attempts:
                obs.RETRIES.labels(b.name).inc()
    raise NoHealthyBackend(last_err)


async def _proxy_unary(gw: GatewayState, tenant: TenantConfig, body, path, rid, t0):
    try:
        b, resp = await _open_backend(gw, body, path, rid, stream=False)
    except NoHealthyBackend as e:
        obs.REQUESTS.labels(tenant.tenant_id, "none", "503").inc()
        log_event("no_backend", request_id=rid, tenant=tenant.tenant_id, err=str(e))
        return error(503, f"no healthy backend: {e}", "overloaded_error", headers={"Retry-After": "1"})
    ok = resp.status_code < 500
    gw.router.on_finish(b, ok=ok)
    dt = time.perf_counter() - t0
    obs.E2E.labels(tenant.tenant_id, "false").observe(dt)
    obs.REQUESTS.labels(tenant.tenant_id, b.name, str(resp.status_code)).inc()
    try:
        payload = resp.json()
        usage = payload.get("usage") or {}
        if usage.get("completion_tokens"):
            obs.OUT_TOKENS.labels(tenant.tenant_id).inc(usage["completion_tokens"])
    except Exception:
        payload = {"error": {"message": resp.text[:500], "type": "backend_error", "code": resp.status_code}}
    log_event("request", request_id=rid, tenant=tenant.tenant_id, backend=b.name, status=resp.status_code,
              stream=False, e2e_s=round(dt, 4), model=body.get("model"))
    return JSONResponse(payload, status_code=resp.status_code, headers={"x-request-id": rid, "x-backend": b.name})


async def _proxy_stream(gw: GatewayState, tenant: TenantConfig, body, path, rid, t0, release):
    try:
        b, resp = await _open_backend(gw, body, path, rid, stream=True)
    except NoHealthyBackend as e:
        await release()
        obs.REQUESTS.labels(tenant.tenant_id, "none", "503").inc()
        log_event("no_backend", request_id=rid, tenant=tenant.tenant_id, err=str(e))
        return error(503, f"no healthy backend: {e}", "overloaded_error", headers={"Retry-After": "1"})

    if resp.status_code >= 400:  # 4xx from backend: pass through as JSON
        content = await resp.aread()
        await resp.aclose()
        gw.router.on_finish(b, ok=True)
        await release()
        obs.REQUESTS.labels(tenant.tenant_id, b.name, str(resp.status_code)).inc()
        return Response(content, status_code=resp.status_code, media_type="application/json")

    async def gen():
        first = True
        ok = True
        completion_tokens = 0
        buf = b""
        try:
            async for chunk in resp.aiter_raw():
                if first:
                    first = False
                    obs.TTFT.labels(tenant.tenant_id).observe(time.perf_counter() - t0)
                buf += chunk
                while b"\n\n" in buf:  # sniff usage without altering the stream
                    event, buf = buf.split(b"\n\n", 1)
                    if event.startswith(b"data: {") and b'"usage"' in event:
                        try:
                            u = json.loads(event[6:]).get("usage") or {}
                            completion_tokens = u.get("completion_tokens") or completion_tokens
                        except Exception:
                            pass
                yield chunk
        except Exception as e:  # backend died mid-stream: tell the client explicitly
            ok = False
            log_event("stream_broken", request_id=rid, tenant=tenant.tenant_id, backend=b.name, err=repr(e))
            err = {"error": {"message": "backend stream interrupted", "type": "backend_error", "code": 502}}
            yield f"data: {json.dumps(err)}\n\ndata: [DONE]\n\n".encode()
        finally:
            await resp.aclose()
            gw.router.on_finish(b, ok=ok)
            dt = time.perf_counter() - t0
            obs.E2E.labels(tenant.tenant_id, "true").observe(dt)
            obs.REQUESTS.labels(tenant.tenant_id, b.name, "200" if ok else "502_midstream").inc()
            if completion_tokens:
                obs.OUT_TOKENS.labels(tenant.tenant_id).inc(completion_tokens)
            log_event("request", request_id=rid, tenant=tenant.tenant_id, backend=b.name, status=200 if ok else 502,
                      stream=True, e2e_s=round(dt, 4), model=body.get("model"))
            await release()

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"x-request-id": rid, "x-backend": b.name, "cache-control": "no-cache"})


app = create_app()
