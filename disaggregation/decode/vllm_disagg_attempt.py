"""EXPERIMENTAL: real vLLM prefill/decode disaggregation on ONE GPU via a KV connector.

Starts a prefill instance (kv_producer) and a decode instance (kv_consumer) on the same
A100 (gpu_memory_utilization 0.4 each) with vLLM's NixlConnector (falls back to
trying P2pNcclConnector), and a tiny proxy that sends each request to the prefill
instance with max_tokens=1 and then to the decode instance.

KV-connector flags change between vLLM releases. This script records exactly what
was tried and whether it worked in results/disagg_attempt/<stamp>/attempt.json.
If it fails, that IS the result: document it in docs/compatibility-matrix.md.

python disaggregation/decode/vllm_disagg_attempt.py
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx  # noqa: E402

from benchmarks.runners.common import MODEL, result_dir, save_json  # noqa: E402
from scripts import serverctl  # noqa: E402

CONNECTORS = [
    ("NixlConnector", {"kv_connector": "NixlConnector", "kv_role": "kv_both"},
     {"kv_connector": "NixlConnector", "kv_role": "kv_both"}),
    ("P2pNcclConnector", {"kv_connector": "P2pNcclConnector", "kv_role": "kv_producer", "kv_rank": 0,
                          "kv_parallel_size": 2},
     {"kv_connector": "P2pNcclConnector", "kv_role": "kv_consumer", "kv_rank": 1, "kv_parallel_size": 2}),
]


def disagg_request(prompt: str, max_tokens: int = 64) -> dict:
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0}
    t0 = time.perf_counter()
    # 1) prefill on producer; NixlConnector needs kv_transfer_params handshake
    pre = httpx.post("http://127.0.0.1:8101/v1/chat/completions", timeout=120,
                     json={**body, "max_tokens": 1,
                           "kv_transfer_params": {"do_remote_decode": True}}).json()
    t_pre = time.perf_counter() - t0
    params = pre.get("kv_transfer_params")
    dec_body = {**body, "max_tokens": max_tokens}
    if params:
        dec_body["kv_transfer_params"] = params
    dec = httpx.post("http://127.0.0.1:8102/v1/chat/completions", json=dec_body, timeout=120).json()
    return {"prefill_s": t_pre, "total_s": time.perf_counter() - t0,
            "decode_usage": dec.get("usage"), "text": (dec.get("choices") or [{}])[0].get("message", {}).get("content"),
            "kv_transfer_params_returned": params is not None}


def main():
    out = result_dir("disagg_attempt")
    attempts = []
    for name, pcfg, dcfg in CONNECTORS:
        rec = {"connector": name, "ok": False}
        try:
            serverctl.stopall()
            p = serverctl.start("prefill", serverctl.vllm_cmd(MODEL, 8101, 0.4, 4096,
                                f"--kv-transfer-config '{json.dumps(pcfg)}'"), 8101, timeout=900,
                                env={"VLLM_NIXL_SIDE_CHANNEL_PORT": "5600", "UCX_TLS": "all"})
            d = serverctl.start("decode", serverctl.vllm_cmd(MODEL, 8102, 0.4, 4096,
                                f"--kv-transfer-config '{json.dumps(dcfg)}'"), 8102, timeout=900,
                                env={"VLLM_NIXL_SIDE_CHANNEL_PORT": "5601", "UCX_TLS": "all"})
            rec["prefill_healthy"], rec["decode_healthy"] = p.get("healthy"), d.get("healthy")
            if not (p.get("healthy") and d.get("healthy")):
                rec["prefill_log_tail"] = serverctl.tail("prefill", 30)
                rec["decode_log_tail"] = serverctl.tail("decode", 30)
            else:
                rec["sample"] = disagg_request("Explain KV cache transfer in one sentence. " * 50)
                rec["ok"] = bool(rec["sample"].get("text"))
                rec["decode_log_mentions_kv"] = [ln for ln in serverctl.tail("decode", 200).splitlines()
                                                 if "kv" in ln.lower() or "nixl" in ln.lower()][-10:]
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["traceback"] = traceback.format_exc()[-2000:]
        finally:
            serverctl.stopall()
        attempts.append(rec)
        print(f"[disagg] {name}: ok={rec['ok']}")
        if rec["ok"]:
            break
    save_json(out / "attempt.json", {"label": "EXPERIMENTAL single-GPU disaggregation attempt", "attempts": attempts})
    print(f"[disagg] wrote {out}/attempt.json")


if __name__ == "__main__":
    main()
