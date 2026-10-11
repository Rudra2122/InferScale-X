# ADR-003: When to disaggregate prefill and decode

Status: proposed — decided by measurement, not by default

## Model
Disaggregation removes prefill→decode interference but adds a KV transfer per request.

    saving per request ≈ ΔTPOT_interference × output_tokens          (exp_interference.py, measured)
    cost per request   ≈ α + KV_bytes(prompt_len) / fabric_bandwidth (kv_transfer_bench.py: α, β measured on
                                                                      PCIe; NVLink / IB values EXTRAPOLATED)

Disaggregate when saving > cost for the traffic mix, *and* the operational cost of two pools
(capacity planning per pool, failure domains, transfer path health) is justified.

## Evidence to fill in after the run
- ΔTPOT at default chunk budget: `interference/*/report.json → delta_tpot_p50_s`
- KV bytes/token for the target model (exact): `kv_transfer → kv.rows[*].kv_bytes_measured / seq_len`
- Break-even output length per fabric: `kv_transfer → break_even.by_seq_len[*].min_output_tokens_to_break_even`

## Alternatives
- Tune chunked prefill (`--max-num-batched-tokens`) on colocated replicas: cheaper; the
  interference experiment measures how much of the gap it closes. If it closes most of it,
  the right recommendation is **not** to disaggregate — record that as a negative result.
