"""Tool implementations. Tenant identity is injected by the runtime, never taken from args."""
from __future__ import annotations

import ast
import operator as op
from dataclasses import dataclass

from agents.memory.store import MemoryStore
from agents.retrieval.bm25 import BM25Index
from agents.sandbox.sandbox import run_python

_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.FloorDiv: op.floordiv,
        ast.Mod: op.mod, ast.Pow: op.pow, ast.USub: op.neg, ast.UAdd: op.pos}

TOOL_SPECS = {
    "calculator": "calculator(expression: str) -> number. Arithmetic only, e.g. {\"expression\": \"12*(3+4)\"}",
    "search_docs": "search_docs(query: str) -> top passages from the documentation",
    "memory_read": "memory_read(key: str) -> stored value or null",
    "memory_write": "memory_write(key: str, value: str) -> stores a value for this user",
    "python_exec": "python_exec(code: str) -> stdout of a sandboxed Python program (no network, no files outside tmp)",
}


def safe_calc(expr: str) -> float:
    if len(expr) > 200:
        raise ValueError("expression too long")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            if isinstance(n.op, ast.Pow) and abs(ev(n.right)) > 64:
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError(f"unsupported expression element: {type(n).__name__}")

    v = ev(ast.parse(expr.replace("^", "**"), mode="eval"))
    return int(v) if isinstance(v, float) and v.is_integer() else v


@dataclass
class ToolResult:
    ok: bool
    content: str
    untrusted: bool = False  # content came from a source the tenant does not control
    source: str = ""


class Toolbox:
    def __init__(self, index: BM25Index, memory: MemoryStore):
        self.index, self.memory = index, memory

    def run(self, tenant_id: str, tool: str, args: dict) -> ToolResult:
        try:
            if tool == "calculator":
                return ToolResult(True, str(safe_calc(str(args.get("expression", "")))))
            if tool == "search_docs":
                hits = self.index.search(str(args.get("query", "")), tenant=tenant_id, k=3)
                if not hits:
                    return ToolResult(True, "no results")
                untrusted = any(not c.trusted for _, c in hits)
                text = "\n---\n".join(f"[{c.source}] {c.text.strip()}" for _, c in hits)
                srcs = ",".join(c.source for _, c in hits if not c.trusted)
                return ToolResult(True, text, untrusted=untrusted, source=srcs)
            if tool == "memory_read":
                v = self.memory.read(tenant_id, str(args.get("key", "")))
                return ToolResult(True, "null" if v is None else v)
            if tool == "memory_write":
                r = self.memory.write(tenant_id, str(args.get("key", "")), str(args.get("value", "")))
                return ToolResult(True, f"stored {r['key']} (v{r['version']})")
            if tool == "python_exec":
                r = run_python(str(args.get("code", "")))
                return ToolResult(r.ok, (r.stdout or "") + (f"\n[stderr]\n{r.stderr[-1000:]}" if not r.ok else ""))
            return ToolResult(False, f"unknown tool {tool}")
        except Exception as e:
            return ToolResult(False, f"tool error: {type(e).__name__}: {e}")
