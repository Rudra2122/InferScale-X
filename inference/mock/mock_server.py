"""Deterministic OpenAI-compatible mock backend (CPU only).

Used by CI, unit/integration/failure tests, and laptop demos. It is NOT a model:
it fakes latency with a simple cost model so gateway behaviour (routing,
admission, retries, streaming, cache isolation) can be tested without a GPU.

Latency model (all configurable by env):
    TTFT = base + uncached_prompt_tokens * prefill_per_token
    each output token = tpot
A block-level prefix cache (16 "tokens"/block, LRU) honours `cache_salt`
exactly the way vLLM does: salt is mixed into the first block hash.

Run:  python -m inference.mock.mock_server --port 8001
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
import uuid
from collections import OrderedDict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

BLOCK = 16


class MockState:
    def __init__(self):
        self.model = os.environ.get("MOCK_MODEL", "mock-model")
        self.base_ms = float(os.environ.get("MOCK_TTFT_BASE_MS", "5"))
        self.prefill_us = float(os.environ.get("MOCK_PREFILL_US_PER_TOKEN", "50"))
        self.tpot_ms = float(os.environ.get("MOCK_TPOT_MS", "2"))
        self.cache_blocks = int(os.environ.get("MOCK_CACHE_BLOCKS", "4096"))
        self.cache: OrderedDict[str, None] = OrderedDict()
        self.failing = False
        self.hanging = False
        self.hits = 0
        self.queries = 0
        self.requests = 0
        self.inflight = 0


S = MockState()
app = FastAPI(title="InferScale-X mock backend")


def tokenize(text: str) -> list[str]:
    return text.split()


def block_hashes(tokens: list[str], salt: str | None) -> list[str]:
    out, prev = [], (salt or "")
    for i in range(0, len(tokens) - len(tokens) % BLOCK, BLOCK):
        prev = hashlib.sha1((prev + "|" + " ".join(tokens[i:i + BLOCK])).encode()).hexdigest()
        out.append(prev)
    return out


def cached_prefix_tokens(tokens: list[str], salt: str | None) -> int:
    hashes = block_hashes(tokens, salt)
    hit = 0
    for h in hashes:
        if h in S.cache:
            S.cache.move_to_end(h)
            hit += 1
        else:
            break
    for h in hashes[hit:]:
        S.cache[h] = None
        if len(S.cache) > S.cache_blocks:
            S.cache.popitem(last=False)
    S.queries += len(tokens)
    S.hits += hit * BLOCK
    return hit * BLOCK


def prompt_text(body: dict) -> str:
    if "messages" in body:
        return "\n".join(str(m.get("content", "")) for m in body["messages"])
    p = body.get("prompt", "")
    return p if isinstance(p, str) else " ".join(map(str, p))


def make_output(prompt: str, n: int) -> list[str]:
    seed = hashlib.md5(prompt.encode()).hexdigest()
    words = ["alpha", "beta", "gamma", "delta", "omega", "sigma", "tau", "kappa"]
    return [words[int(seed[i % 32], 16) % len(words)] for i in range(n)]


@app.get("/health")
async def health():
    if S.failing:
        return JSONResponse({"status": "failing"}, status_code=503)
    return {"status": "ok"}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": S.model, "object": "model"}]}


@app.get("/metrics")
async def metrics():
    lines = [
        f"vllm:prefix_cache_hits_total {S.hits}",
        f"vllm:prefix_cache_queries_total {S.queries}",
        f"mock_requests_total {S.requests}",
        f"mock_inflight {S.inflight}",
    ]
    return PlainTextResponse("\n".join(lines) + "\n")


@app.post("/admin/fail")
async def fail():
    S.failing = True
    return {"failing": True}


@app.post("/admin/hang")
async def hang():
    S.hanging = True
    return {"hanging": True}


@app.post("/admin/recover")
async def recover():
    S.failing = False
    S.hanging = False
    return {"failing": False}


@app.post("/admin/reset_cache")
async def reset_cache():
    S.cache.clear()
    S.hits = S.queries = 0
    return {"ok": True}


async def handle(request: Request, chat: bool):
    if S.hanging:  # black hole: hold the connection open, send nothing
        await asyncio.sleep(3600)
    if S.failing:
        return JSONResponse({"error": {"message": "injected failure", "code": 500}}, status_code=500)
    body = await request.json()
    S.requests += 1
    text = prompt_text(body)
    toks = tokenize(text)
    n_out = int(body.get("max_tokens") or body.get("max_completion_tokens") or 16)
    if body.get("ignore_eos") is False:
        n_out = min(n_out, 16)
    hit = cached_prefix_tokens(toks, body.get("cache_salt"))
    ttft = (S.base_ms + (len(toks) - hit) * S.prefill_us / 1000.0) / 1000.0
    out = make_output(text, n_out)
    rid = f"cmpl-{uuid.uuid4().hex[:12]}"
    usage = {"prompt_tokens": len(toks), "completion_tokens": n_out, "total_tokens": len(toks) + n_out,
             "prompt_tokens_details": {"cached_tokens": hit}}
    created = int(time.time())
    S.inflight += 1

    if not body.get("stream"):
        try:
            await asyncio.sleep(ttft + n_out * S.tpot_ms / 1000.0)
        finally:
            S.inflight -= 1
        txt = " ".join(out)
        choice = ({"index": 0, "message": {"role": "assistant", "content": txt}, "finish_reason": "length"} if chat
                  else {"index": 0, "text": txt, "finish_reason": "length"})
        return {"id": rid, "object": "chat.completion" if chat else "text_completion", "created": created,
                "model": S.model, "choices": [choice], "usage": usage}

    include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

    async def gen():
        try:
            await asyncio.sleep(ttft)
            for i, w in enumerate(out):
                if i:
                    await asyncio.sleep(S.tpot_ms / 1000.0)
                if S.failing:  # simulate a crash mid-stream
                    raise RuntimeError("injected mid-stream failure")
                piece = (" " if i else "") + w
                ch = ({"index": 0, "delta": {"content": piece}, "finish_reason": None} if chat
                      else {"index": 0, "text": piece, "finish_reason": None})
                yield f"data: {json.dumps({'id': rid, 'object': 'chat.completion.chunk', 'model': S.model, 'choices': [ch]})}\n\n"
            if include_usage:
                yield f"data: {json.dumps({'id': rid, 'choices': [], 'usage': usage})}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            S.inflight -= 1

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await handle(request, chat=True)


@app.post("/v1/completions")
async def completions(request: Request):
    return await handle(request, chat=False)


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    a = ap.parse_args()
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
