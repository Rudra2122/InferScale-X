"""Backend routing and health tracking.

Policies
--------
round_robin        : baseline; ignores load and prompt content.
least_outstanding  : picks the healthy backend with the fewest in-flight requests.
prefix_affinity    : KV-cache-aware approximation. Rendezvous-hashes the prompt
                     prefix so requests sharing a prefix land on the same replica
                     (where its prefix cache is warm), but falls back to
                     least_outstanding when that replica is more than
                     `max_imbalance` requests busier than the least-loaded one.

This is the single-GPU stand-in for Dynamo's KV-aware router: same idea, far
simpler implementation. Experiments compare all three against round_robin.
"""
from __future__ import annotations

import asyncio
import hashlib
import itertools
import time
from dataclasses import dataclass, field

import httpx

from gateway.config import BackendConfig, GatewayConfig

POLICIES = ("round_robin", "least_outstanding", "prefix_affinity")


@dataclass
class BackendState:
    cfg: BackendConfig
    healthy: bool = True
    outstanding: int = 0
    consecutive_failures: int = 0
    total_requests: int = 0
    total_errors: int = 0
    last_change: float = field(default_factory=time.time)

    @property
    def name(self) -> str:
        return self.cfg.name


class NoHealthyBackend(Exception):
    pass


def _score(key: str, backend_name: str) -> int:
    return int.from_bytes(hashlib.blake2b(f"{key}|{backend_name}".encode(), digest_size=8).digest(), "big")


def prefix_key_from_body(body: dict, n_chars: int) -> str:
    if "messages" in body:
        text = "".join(
            m.get("content") if isinstance(m.get("content"), str) else str(m.get("content"))
            for m in body.get("messages") or []
        )
    else:
        p = body.get("prompt", "")
        text = p if isinstance(p, str) else str(p)
    return text[:n_chars]


class Router:
    def __init__(self, cfg: GatewayConfig):
        if cfg.routing_policy not in POLICIES:
            raise ValueError(f"unknown routing policy {cfg.routing_policy}; choose from {POLICIES}")
        self.cfg = cfg
        self.backends = [BackendState(b) for b in cfg.backends]
        self._rr = itertools.count()
        self._health_task: asyncio.Task | None = None

    # ---------------- selection ----------------
    def healthy(self, exclude: set[str] | None = None) -> list[BackendState]:
        exclude = exclude or set()
        return [b for b in self.backends if b.healthy and b.name not in exclude]

    def choose(self, body: dict, exclude: set[str] | None = None) -> BackendState:
        cands = self.healthy(exclude)
        if not cands:
            raise NoHealthyBackend("no healthy backends")
        policy = self.cfg.routing_policy
        if policy == "round_robin":
            return cands[next(self._rr) % len(cands)]
        least = min(cands, key=lambda b: (b.outstanding, b.total_requests))
        if policy == "least_outstanding":
            return least
        key = prefix_key_from_body(body, self.cfg.prefix_affinity_chars)
        preferred = max(cands, key=lambda b: _score(key, b.name))
        if preferred.outstanding - least.outstanding > self.cfg.prefix_affinity_max_imbalance:
            return least
        return preferred

    # ---------------- bookkeeping ----------------
    def on_start(self, b: BackendState) -> None:
        b.outstanding += 1
        b.total_requests += 1

    def on_finish(self, b: BackendState, ok: bool) -> None:
        b.outstanding = max(0, b.outstanding - 1)
        if ok:
            b.consecutive_failures = 0
        else:
            b.total_errors += 1
            b.consecutive_failures += 1
            if b.consecutive_failures >= self.cfg.unhealthy_after_failures and b.healthy:
                b.healthy = False
                b.last_change = time.time()

    # ---------------- health checking ----------------
    async def probe_once(self, client: httpx.AsyncClient) -> None:
        async def probe(b: BackendState):
            try:
                r = await client.get(f"{b.cfg.url}/health", timeout=2.0)
                ok = r.status_code == 200
            except Exception:
                ok = False
            if ok and not b.healthy:
                b.healthy, b.consecutive_failures, b.last_change = True, 0, time.time()
            elif not ok and b.healthy:
                b.consecutive_failures += 1
                if b.consecutive_failures >= self.cfg.unhealthy_after_failures:
                    b.healthy, b.last_change = False, time.time()

        await asyncio.gather(*(probe(b) for b in self.backends))

    async def _health_loop(self) -> None:
        async with httpx.AsyncClient() as client:
            while True:
                await self.probe_once(client)
                await asyncio.sleep(self.cfg.health_interval_s)

    def start_health_checks(self) -> None:
        if self._health_task is None:
            self._health_task = asyncio.create_task(self._health_loop())

    async def stop(self) -> None:
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except (asyncio.CancelledError, Exception):
                pass
            self._health_task = None

    def snapshot(self) -> list[dict]:
        return [
            dict(name=b.name, url=b.cfg.url, healthy=b.healthy, outstanding=b.outstanding,
                 total_requests=b.total_requests, total_errors=b.total_errors)
            for b in self.backends
        ]
