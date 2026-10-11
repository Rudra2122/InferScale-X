# Engine comparison (same A100, same workloads, separate runtimes)

| engine | workload | tok/s (mean ± CI) | TTFT p50 ms | TPOT p50 ms | error rate |
|---|---|---|---|---|---|
| vllm | chat_short | 5087 ± 172 | 114.7 | 5.12 | 0.000 |
| vllm | prefix_heavy | 2683 ± 166 | 165.1 | 8.69 | 0.000 |
| sglang | chat_short | 4859 ± 277 | 163.1 | 4.90 | 0.000 |
| sglang | prefix_heavy | 2473 ± 212 | 290.1 | 7.30 | 0.000 |
| trtllm | chat_short | 1715 ± 30 | 117.9 | 17.25 | 0.000 |
| trtllm | prefix_heavy | 1234 ± 153 | 150.7 | 23.47 | 0.000 |
