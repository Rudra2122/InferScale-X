"""Dependency-free BM25 retrieval with tenant-scoped corpora.

Each tenant sees: the shared corpus + its own private documents. Private chunks are
stored under the tenant namespace and are never scored for other tenants.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

_TOK = re.compile(r"[a-z0-9]+")
STOP = set("the a an and or of to in on for is are was were be by with as at from that this it its".split())


def tokenize(text: str) -> list[str]:
    return [t for t in _TOK.findall(text.lower()) if t not in STOP]


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    namespace: str  # "shared" or tenant id
    trusted: bool = True  # False for user-uploaded / web content -> treated as untrusted by governance


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.chunks: list[Chunk] = []
        self.tfs: list[Counter] = []
        self.df: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))  # namespace -> term -> df
        self.lens: list[int] = []

    def add(self, chunk: Chunk) -> None:
        toks = tokenize(chunk.text)
        tf = Counter(toks)
        self.chunks.append(chunk)
        self.tfs.append(tf)
        self.lens.append(len(toks))
        for t in tf:
            self.df[chunk.namespace][t] += 1

    def add_markdown_dir(self, path: str | Path, namespace: str = "shared", trusted: bool = True,
                         chunk_chars: int = 800) -> int:
        n = 0
        for p in sorted(Path(path).glob("*.md")):
            text = p.read_text()
            parts = [s.strip() for s in re.split(r"\n(?=#)|\n\n", text) if s.strip()]
            buf = ""
            for part in parts:
                if len(buf) + len(part) > chunk_chars and buf:
                    self.add(Chunk(f"{p.stem}#{n}", buf, p.name, namespace, trusted))
                    n += 1
                    buf = ""
                buf += part + "\n"
            if buf:
                self.add(Chunk(f"{p.stem}#{n}", buf, p.name, namespace, trusted))
                n += 1
        return n

    def search(self, query: str, tenant: str, k: int = 3) -> list[tuple[float, Chunk]]:
        q = tokenize(query)
        visible = [i for i, c in enumerate(self.chunks) if c.namespace in ("shared", tenant)]
        if not visible or not q:
            return []
        n = len(visible)
        avg = sum(self.lens[i] for i in visible) / n
        df = Counter()
        for ns in ("shared", tenant):
            for t in q:
                df[t] += self.df.get(ns, {}).get(t, 0)
        scored = []
        for i in visible:
            tf, L, s = self.tfs[i], self.lens[i], 0.0
            for t in q:
                if t not in tf:
                    continue
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * L / avg))
            if s > 0:
                scored.append((s, self.chunks[i]))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]
