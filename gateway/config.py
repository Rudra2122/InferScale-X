"""Gateway configuration loading.

Config is a single YAML file (see configs/gateway.example.yaml). Environment
variable INFERSCALE_GATEWAY_CONFIG points at it. Nothing here talks to a GPU.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class TenantConfig:
    tenant_id: str
    api_key_sha256: str
    allowed_models: list[str] = field(default_factory=lambda: ["*"])
    requests_per_minute: float = 600.0
    burst: int = 20
    max_concurrent: int = 16
    max_output_tokens: int = 1024
    priority: int = 1  # higher = more important; used only for logging/fairness reports
    allowed_tools: list[str] = field(default_factory=list)

    def model_allowed(self, model: str) -> bool:
        return "*" in self.allowed_models or model in self.allowed_models


@dataclass
class BackendConfig:
    name: str
    url: str  # base url, e.g. http://127.0.0.1:8001
    model: str | None = None  # served model name; None = accept any


@dataclass
class GatewayConfig:
    tenants: dict[str, TenantConfig]
    backends: list[BackendConfig]
    routing_policy: str = "least_outstanding"  # round_robin | least_outstanding | prefix_affinity
    prefix_affinity_chars: int = 512
    prefix_affinity_max_imbalance: int = 8
    global_max_inflight: int = 256
    request_timeout_s: float = 300.0
    upstream_read_timeout_s: float = 60.0  # max silence between bytes from a backend before failing over
    health_interval_s: float = 2.0
    unhealthy_after_failures: int = 2
    retry_on_backend_error: bool = True
    max_retries: int = 1
    inject_cache_salt: bool = False  # per-tenant vLLM prefix-cache isolation
    cache_salt_secret: str = "change-me"
    log_prompts: bool = False

    @property
    def key_index(self) -> dict[str, TenantConfig]:
        return {t.api_key_sha256: t for t in self.tenants.values()}


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _tenant_from_dict(tid: str, d: dict[str, Any]) -> TenantConfig:
    d = dict(d)
    if "api_key" in d:  # convenience for local dev; hashed immediately
        d["api_key_sha256"] = sha256_hex(d.pop("api_key"))
    return TenantConfig(tenant_id=tid, **d)


def load_config(path: str | os.PathLike | None = None) -> GatewayConfig:
    path = path or os.environ.get("INFERSCALE_GATEWAY_CONFIG", "configs/gateway.example.yaml")
    raw = yaml.safe_load(Path(path).read_text())
    return config_from_dict(raw)


def config_from_dict(raw: dict[str, Any]) -> GatewayConfig:
    tenants = {tid: _tenant_from_dict(tid, td) for tid, td in (raw.get("tenants") or {}).items()}
    backends = [BackendConfig(**b) for b in (raw.get("backends") or [])]
    # Allow env override of backends: INFERSCALE_BACKENDS="a=http://h:1,b=http://h:2"
    env_b = os.environ.get("INFERSCALE_BACKENDS")
    if env_b:
        backends = []
        for item in env_b.split(","):
            name, url = item.split("=", 1)
            backends.append(BackendConfig(name=name.strip(), url=url.strip()))
    opts = {k: v for k, v in (raw.get("gateway") or {}).items()}
    if os.environ.get("INFERSCALE_ROUTING_POLICY"):
        opts["routing_policy"] = os.environ["INFERSCALE_ROUTING_POLICY"]
    if os.environ.get("INFERSCALE_INJECT_CACHE_SALT"):
        opts["inject_cache_salt"] = os.environ["INFERSCALE_INJECT_CACHE_SALT"].lower() in ("1", "true", "yes")
    if not backends:
        raise ValueError("No backends configured")
    if not tenants:
        raise ValueError("No tenants configured")
    return GatewayConfig(tenants=tenants, backends=backends, **opts)
