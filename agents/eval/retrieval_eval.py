"""Retrieval quality evaluation, separate from end-to-end agent success.

Corpus: the project's own engineering docs + the Orbit agent corpus (fixed file list, so
results are comparable across runs). Labels: agents/eval/retrieval_qrels.yaml maps each
query to the document(s) that answer it (document-level relevance, written by hand from the
docs' content — not tuned to any retriever).

Retrievers
  bm25    agents/retrieval/bm25.py (always available)
  dense   sentence-transformers all-MiniLM-L6-v2, cosine (if installed; pip install sentence-transformers)
  hybrid  reciprocal-rank fusion of bm25 + dense (k=60)
Metrics: hit@1, recall@3, recall@5, MRR@10, nDCG@5, latency per query.

python agents/eval/retrieval_eval.py [--no-dense]
"""
from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from agents.retrieval.bm25 import BM25Index, Chunk  # noqa: E402

QRELS = Path(__file__).parent / "retrieval_qrels.yaml"


def corpus_files() -> list[Path]:
    spec = yaml.safe_load(QRELS.read_text())
    return [ROOT / p for p in spec["corpus"]]


def chunk_file(p: Path, chunk_chars: int = 700) -> list[Chunk]:
    rel = str(p.relative_to(ROOT))
    text = p.read_text()
    parts = [s.strip() for s in re.split(r"\n(?=#{1,3} )", text) if s.strip()]
    chunks, buf, i = [], "", 0
    for part in parts:
        while len(part) > chunk_chars * 2:  # split very long sections
            chunks.append(Chunk(f"{rel}#{i}", part[:chunk_chars], rel, "shared"))
            part, i = part[chunk_chars:], i + 1
        if buf and len(buf) + len(part) > chunk_chars:
            chunks.append(Chunk(f"{rel}#{i}", buf, rel, "shared"))
            buf, i = "", i + 1
        buf += part + "\n"
    if buf:
        chunks.append(Chunk(f"{rel}#{i}", buf, rel, "shared"))
    return chunks


def doc_ranking(chunk_ranking: list[str]) -> list[str]:
    seen, out = set(), []
    for src in chunk_ranking:
        if src not in seen:
            seen.add(src)
            out.append(src)
    return out


def metrics(ranked_docs: list[str], relevant: set[str]) -> dict:
    def recall(k):
        return len(set(ranked_docs[:k]) & relevant) / len(relevant)

    rr = 0.0
    for i, d in enumerate(ranked_docs[:10]):
        if d in relevant:
            rr = 1 / (i + 1)
            break
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked_docs[:5]) if d in relevant)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(5, len(relevant))))
    return {"hit@1": float(bool(ranked_docs[:1]) and ranked_docs[0] in relevant), "recall@3": recall(3),
            "recall@5": recall(5), "mrr@10": rr, "ndcg@5": dcg / idcg if idcg else 0.0}


class Bm25Retriever:
    name = "bm25"

    def __init__(self, chunks):
        self.idx = BM25Index()
        for c in chunks:
            self.idx.add(c)

    def rank(self, q: str) -> list[str]:
        return [c.source for _, c in self.idx.search(q, tenant="shared", k=40)]


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks, model="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)
        self.chunks = chunks
        self.emb = self.model.encode([c.text for c in chunks], normalize_embeddings=True, convert_to_numpy=True)

    def rank(self, q: str) -> list[str]:
        qv = self.model.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
        order = (self.emb @ qv).argsort()[::-1][:40]
        return [self.chunks[i].source for i in order]


class HybridRetriever:
    name = "hybrid_rrf"

    def __init__(self, a, b, k: int = 60):
        self.a, self.b, self.k = a, b, k

    def rank(self, q: str) -> list[str]:
        score: dict[str, float] = {}
        for r in (doc_ranking(self.a.rank(q)), doc_ranking(self.b.rank(q))):
            for i, d in enumerate(r):
                score[d] = score.get(d, 0.0) + 1 / (self.k + i + 1)
        return sorted(score, key=lambda d: -score[d])


def evaluate(retriever, queries) -> dict:
    rows = []
    for q in queries:
        t = time.perf_counter()
        ranked = doc_ranking(retriever.rank(q["query"]))
        dt = time.perf_counter() - t
        m = metrics(ranked, set(q["relevant"]))
        rows.append({"id": q["id"], "query": q["query"], "relevant": q["relevant"], "top3": ranked[:3],
                     "latency_ms": dt * 1e3, **m})
    agg = {k: statistics.mean(r[k] for r in rows) for k in ("hit@1", "recall@3", "recall@5", "mrr@10", "ndcg@5")}
    agg["latency_ms_p50"] = statistics.median(r["latency_ms"] for r in rows)
    return {"aggregate": agg, "per_query": rows, "failures": [r for r in rows if r["recall@5"] == 0]}


def run(no_dense: bool = False) -> dict:
    spec = yaml.safe_load(QRELS.read_text())
    chunks = [c for p in corpus_files() for c in chunk_file(p)]
    bm = Bm25Retriever(chunks)
    report = {"experiment": "retrieval_eval", "n_queries": len(spec["queries"]), "n_docs": len(spec["corpus"]),
              "n_chunks": len(chunks), "relevance": "document-level, hand-labelled", "retrievers": {}}
    report["retrievers"]["bm25"] = evaluate(bm, spec["queries"])
    if not no_dense:
        try:
            dense = DenseRetriever(chunks)
            report["retrievers"]["dense"] = evaluate(dense, spec["queries"])
            report["retrievers"]["hybrid_rrf"] = evaluate(HybridRetriever(bm, dense), spec["queries"])
        except Exception as e:
            report["dense_skipped"] = f"{type(e).__name__}: {e}"[:300]
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from benchmarks.runners.common import result_dir, save_json

    rep = run(a.no_dense)
    out = Path(a.out) if a.out else result_dir("retrieval")
    save_json(out / "report.json", rep)
    for name, r in rep["retrievers"].items():
        g = r["aggregate"]
        print(f"[retrieval] {name:10s} hit@1={g['hit@1']:.2f} recall@3={g['recall@3']:.2f} recall@5={g['recall@5']:.2f} "
              f"mrr={g['mrr@10']:.2f} ndcg@5={g['ndcg@5']:.2f} misses={len(r['failures'])}")
    if "dense_skipped" in rep:
        print(f"[retrieval] dense skipped: {rep['dense_skipped'][:120]}")
    print(f"[retrieval] wrote {out}/report.json")


if __name__ == "__main__":
    main()
