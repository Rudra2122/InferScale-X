"""API-key authentication.

Keys are never stored in plaintext: the config holds sha256(key). Lookup is a
dict hit on the hash followed by a constant-time comparison.
"""
from __future__ import annotations

import hmac

from gateway.config import GatewayConfig, TenantConfig, sha256_hex


class AuthError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def extract_bearer(header_value: str | None) -> str:
    if not header_value:
        raise AuthError(401, "missing Authorization header")
    parts = header_value.strip().split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise AuthError(401, "malformed Authorization header; expected 'Bearer <key>'")
    return parts[1].strip()


def authenticate(cfg: GatewayConfig, header_value: str | None) -> TenantConfig:
    key = extract_bearer(header_value)
    digest = sha256_hex(key)
    tenant = cfg.key_index.get(digest)
    if tenant is None or not hmac.compare_digest(tenant.api_key_sha256, digest):
        raise AuthError(401, "invalid API key")
    return tenant


def authorize_model(tenant: TenantConfig, model: str) -> None:
    if not tenant.model_allowed(model):
        raise AuthError(403, f"tenant '{tenant.tenant_id}' is not allowed to use model '{model}'")
