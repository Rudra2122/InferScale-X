"""Versioned skill registry.

A skill = name + semver + system prompt + declared tool allowlist + limits. Manifests
live in agents/skills/manifests/<name>/<version>.yaml. The registry computes a
content hash per version so audit logs record exactly which skill text ran, and
callers can pin a version or take the latest. Editing a published version in place
is detected (hash mismatch against the lockfile) and refused.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

HERE = Path(__file__).parent


@dataclass(frozen=True)
class Skill:
    name: str
    version: str
    description: str
    system_prompt: str
    tools: tuple[str, ...]
    max_tool_calls: int
    content_hash: str


def _vkey(v: str) -> tuple:
    return tuple(int(x) for x in v.split("."))


def _hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


class SkillRegistry:
    def __init__(self, root: Path = HERE / "manifests", lockfile: Path | None = HERE / "skills.lock.json"):
        self.root = Path(root)
        self.lockfile = lockfile
        self.skills: dict[str, dict[str, Skill]] = {}
        lock = json.loads(lockfile.read_text()) if lockfile and lockfile.exists() else {}
        for f in sorted(self.root.glob("*/*.yaml")):
            d = yaml.safe_load(f.read_text())
            h = _hash(d)
            key = f"{d['name']}@{d['version']}"
            if key in lock and lock[key] != h:
                raise ValueError(f"skill {key} was modified after publication (hash {h} != locked {lock[key]}); "
                                 "bump the version instead")
            s = Skill(d["name"], str(d["version"]), d.get("description", ""), d["system_prompt"],
                      tuple(d.get("tools", [])), int(d.get("max_tool_calls", 6)), h)
            self.skills.setdefault(s.name, {})[s.version] = s

    def get(self, name: str, version: str | None = None) -> Skill:
        if name not in self.skills:
            raise KeyError(f"unknown skill {name}")
        vs = self.skills[name]
        if version is None:
            return vs[max(vs, key=_vkey)]
        if version not in vs:
            raise KeyError(f"unknown version {name}@{version}; available {sorted(vs, key=_vkey)}")
        return vs[version]

    def write_lock(self) -> dict:
        lock = {f"{s.name}@{s.version}": s.content_hash for vs in self.skills.values() for s in vs.values()}
        if self.lockfile:
            self.lockfile.write_text(json.dumps(lock, indent=2, sort_keys=True))
        return lock
