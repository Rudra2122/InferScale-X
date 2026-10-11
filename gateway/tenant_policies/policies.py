"""Per-tenant rate limiting and admission control.

- TokenBucket: requests-per-minute with burst.
- AdmissionController: per-tenant concurrency cap + global in-flight cap.
  Rejections are explicit (429 / 503 with Retry-After) instead of unbounded
  queueing, so overload behaviour is measurable in failure-injection tests.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from gateway.config import GatewayConfig, TenantConfig


class TokenBucket:
    def __init__(self, rate_per_s: float, burst: int, clock=time.monotonic):
        self.rate = rate_per_s
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.clock = clock
        self.last = clock()

    def try_take(self, n: float = 1.0) -> tuple[bool, float]:
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
        self.last = now
        if self.tokens >= n:
            self.tokens -= n
            return True, 0.0
        deficit = n - self.tokens
        return False, deficit / self.rate if self.rate > 0 else 60.0


@dataclass
class Rejection:
    status: int
    reason: str
    retry_after_s: float


class AdmissionController:
    def __init__(self, cfg: GatewayConfig, clock=time.monotonic):
        self.cfg = cfg
        self.buckets = {
            t.tenant_id: TokenBucket(t.requests_per_minute / 60.0, t.burst, clock) for t in cfg.tenants.values()
        }
        self.inflight_by_tenant: dict[str, int] = {t: 0 for t in cfg.tenants}
        self.inflight_total = 0
        self._lock = asyncio.Lock()

    async def admit(self, tenant: TenantConfig) -> Rejection | None:
        async with self._lock:
            if self.inflight_total >= self.cfg.global_max_inflight:
                return Rejection(503, "global_overload", 1.0)
            if self.inflight_by_tenant.get(tenant.tenant_id, 0) >= tenant.max_concurrent:
                return Rejection(429, "tenant_concurrency", 0.5)
            ok, wait = self.buckets[tenant.tenant_id].try_take()
            if not ok:
                return Rejection(429, "tenant_rate_limit", round(wait, 3))
            self.inflight_by_tenant[tenant.tenant_id] = self.inflight_by_tenant.get(tenant.tenant_id, 0) + 1
            self.inflight_total += 1
            return None

    async def release(self, tenant: TenantConfig) -> None:
        async with self._lock:
            self.inflight_by_tenant[tenant.tenant_id] = max(0, self.inflight_by_tenant.get(tenant.tenant_id, 0) - 1)
            self.inflight_total = max(0, self.inflight_total - 1)


def clamp_max_tokens(body: dict, tenant: TenantConfig) -> dict:
    """Enforce the tenant's output-token cap regardless of what the client asked for."""
    for key in ("max_tokens", "max_completion_tokens"):
        if key in body and body[key] is not None:
            body[key] = min(int(body[key]), tenant.max_output_tokens)
    if "max_tokens" not in body and "max_completion_tokens" not in body:
        body["max_tokens"] = tenant.max_output_tokens
    return body
