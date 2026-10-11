"""Persistent per-tenant agent memory (SQLite).

Isolation is structural: every query is parameterised by tenant_id taken from the
authenticated context, never from model output. There is no API that accepts a
tenant id from the model.
"""
from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

KEY_RE = re.compile(r"^[a-zA-Z0-9_.-]{1,64}$")
MAX_VALUE = 4096


class MemoryStore:
    def __init__(self, path: str | Path = ":memory:"):
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS memory (
            tenant_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
            updated REAL NOT NULL, version INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (tenant_id, key))""")
        self.db.commit()

    def write(self, tenant_id: str, key: str, value: str) -> dict:
        if not KEY_RE.match(key):
            raise ValueError("invalid memory key")
        if len(value) > MAX_VALUE:
            raise ValueError("memory value too large")
        cur = self.db.execute("SELECT version FROM memory WHERE tenant_id=? AND key=?", (tenant_id, key)).fetchone()
        ver = (cur[0] + 1) if cur else 1
        self.db.execute("INSERT OR REPLACE INTO memory VALUES (?,?,?,?,?)", (tenant_id, key, value, time.time(), ver))
        self.db.commit()
        return {"key": key, "version": ver}

    def read(self, tenant_id: str, key: str) -> str | None:
        if not KEY_RE.match(key):
            raise ValueError("invalid memory key")
        row = self.db.execute("SELECT value FROM memory WHERE tenant_id=? AND key=?", (tenant_id, key)).fetchone()
        return row[0] if row else None

    def keys(self, tenant_id: str) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT key FROM memory WHERE tenant_id=? ORDER BY key", (tenant_id,))]
