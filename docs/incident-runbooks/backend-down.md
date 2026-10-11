# Runbook: inference backend down

**Alert:** `BackendDown` (`inferscale_backend_healthy == 0` for 30 s)

## Expected behaviour (from the failure-injection experiment)
- Gateway ejects the replica after `unhealthy_after_failures` failed probes/requests (detection time measured in `crash_report.json → time_to_detect_s`).
- Requests that start after detection go to healthy replicas; requests that hit the dead replica before the first byte are retried once.
- **Streams already in progress on the dead replica fail** (SSE error event). This is expected, not a bug.
- A backend that dies **without a TCP reset** (force-deleted pod, dead node, network partition) is detected by the
  idle-read timeout `upstream_read_timeout_s` (default 60 s; 180 s in the Colab config), not by connection errors.
  Found on kind: before this setting existed such requests hung for the full 300 s request timeout, and some
  transport errors surfaced as bare HTTP 500s (now: any `httpx.TransportError` before the first byte is retried).

## Triage
1. `curl -H "Authorization: Bearer $KEY" $GW/admin/backends` — which replica, since when.
2. Engine log: `scripts/serverctl.py tail <name>` (Colab) / `kubectl logs` (K8s). Common causes: CUDA OOM at startup (gpu-memory-utilization too high), Xid errors (hardware), model download failure.
3. `nvidia-smi` — GPU present? ECC/Xid errors? Other processes holding memory?

## Mitigate
- Restart the replica (`serverctl.py vllm ...` / delete pod). Expected time-to-healthy ≈ `backend_restart_s` from the experiment (dominated by model load + CUDA graph capture).
- If capacity is insufficient on the remaining replicas, the gateway will start returning 503 + Retry-After rather than queueing unboundedly — lower traffic or scale out.

## Follow-up
Record actual detection/restart times next to the experiment's; a large difference means the health settings or startup path changed.
