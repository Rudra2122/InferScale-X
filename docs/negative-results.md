# Negative results

Optimisations that did not help, and experiment designs that were wrong, from the Colab A100 run of
2026-10-10 (vLLM 0.31.0, torch 2.13.0, Qwen2.5 models). Every number below is quoted from the console log of
that run; the linked report files hold the full distributions, CIs and verdicts. Re-check them after re-running.

## 1. KV-cache-aware (prefix-affinity) routing did not beat round-robin
- **Evidence:** `benchmarks/results/routing/*/report.json` → `runs.prefix_heavy.vs_round_robin.prefix_affinity`
- **Result:** prefix_heavy p50 TTFT per rep: round-robin 106 / 93 / 145 ms vs prefix-affinity 138 / 214 / 170 ms
  (≈ +50% worse on the mean); throughput unchanged (~1.66k vs ~1.67k tok/s).
- **Why:** the workload has only 8 shared 2,000-token prefixes (~16k tokens of KV). Each replica's prefix cache
  holds all of them, so round-robin already gets near-full cache hits on both replicas; affinity adds nothing
  but concentrates popular prefixes on one replica (load imbalance → queueing). Both replicas also share one
  GPU, so "spreading" load doesn't add compute.
- **What would change it:** more distinct prefixes than one replica's KV can hold (cache-capacity-bound), or
  replicas on separate GPUs. That is the regime KV-aware routers (e.g. Dynamo's) target.

## 2. Triton FlashAttention-style kernel loses to PyTorch SDPA, badly at head_dim 128
- **Evidence:** `benchmarks/results/triton_attention/*/report.json` → `performance`
- **Result:** correct on 24/24 cases, but 64–91% of SDPA throughput at head_dim 64 (e.g. 135.8 vs 159.8 TFLOP/s,
  B4 H16 N2048 non-causal) and only 35–44% at head_dim 128, N=8192 (64.0 vs 182.7 TFLOP/s causal).
- **Why:** fixed tile sizes (BLOCK_M=64, BLOCK_N=64, 2 stages for D=128) and no autotuning; SDPA dispatches to a
  hand-tuned FlashAttention-2 kernel. D=128 tiles need more shared memory/registers per program, so occupancy
  drops sharply with these choices.
- **What would change it:** `@triton.autotune` over BLOCK_M/BLOCK_N/num_warps/num_stages, larger BLOCK_N for D=128,
  pipelining K/V loads. Expectation: close much of the gap at D=64, partial at D=128.

## 3. Dense retrieval underperformed BM25 on this benchmark
- **Evidence:** `benchmarks/results/retrieval/*/report.json`
- **Result:** MRR@10 BM25 0.94, dense (all-MiniLM-L6-v2) 0.83, hybrid RRF 0.92.
- **Why:** the corpus is technical docs and the queries reuse their exact terms (flag names, metric names), which
  favours lexical matching; a small general-purpose embedding model adds little. Hybrid did not beat BM25 either.
- **Caveat:** 32 hand-written queries, same author as the docs: an easy, lexically biased benchmark.

## 4. The agent was no more accurate than single-shot RAG
- **Evidence:** `benchmarks/results/agents/*live/report.json` → `functional`
- **Result:** task success 83% agent vs 83% RAG (Qwen2.5-7B), with the agent using more tokens and latency.
- **Why:** most tasks are answerable from one retrieved passage; tool use only matters for the arithmetic tasks.
  The agent's value here is governance (all live and scripted attacks blocked), not accuracy.

## 5. Experiment-design bugs found by the run (fixed in v2 of the scripts)
- **Interference v1 measured nothing** (TPOT 4.16 ms alone vs 4.16 ms "noisy" at every chunk budget): the 48
  long-prefill noise requests finished in roughly the first 20% of the probe, so most probe tokens never
  overlapped a prefill. v2 keeps noise running until the probe ends and adds ITL p99.
- **KV-pressure v1 created no pressure** (TTFT/throughput identical at gpu_memory_utilization 0.85 / 0.5 / 0.3):
  even 0.3 × 40 GB leaves ~9 GB of KV for a 1.5B model, far more than the 64k-token working set. v2 caps the KV
  directly with `--num-gpu-blocks-override` (8000 / 3000 / 1200 blocks vs a 4000-block working set).
- **v2 results (both fixes worked):**
  - Interference (`interference/20261010-102752`), with 184–232 noise requests per rep:
    - Decode TPOT p50 went from 4.16 ms alone to 7.5 / 8.1 / 8.3 ms at chunk budgets 512 / 2048 / 8192.
    - ITL p99 went from 4.7 ms to 22.6 / 44.8 / 110 ms.
    - A bigger chunk budget means longer prefill chunks per step, so worse decode-tail stalls. This is the
      cost that prefill/decode disaggregation removes.
  - Pressure (`kv_pressure/20261010-103719`), at 8000 / 3000 / 1200 blocks:
    - cached-token fraction 0.57 / 0.31 / 0.14
    - TTFT p50 172 / 360 / 1552 ms
    - throughput 2320 / 1642 / 1018 tok/s
- **Speculative decoding v1 crashed:** Qwen2.5-7B (152,064-token vocab) and Qwen2.5-0.5B (151,936) are treated as
  different tokenizers by transformers. v2 uses a 3B target with the same vocab as the 0.5B draft.

## 6. HF speculative decoding: good acceptance, no speedup
- **Evidence:** `benchmarks/results/speculative_decoding/20261010-104217/report.json` → `hf`, rows in `hf_rows.jsonl`
- **Result:** Qwen2.5-3B target + 0.5B draft, greedy, batch 1, 128 new tokens: 2.6–4.6 tokens accepted per target
  forward, yet tokens/s was equal or lower (e.g. 25.8 standard vs 24.3 speculative; worst prompt 14.8).
- **Why:** HF eager `generate` at batch 1 runs at ~26 tok/s, far below what A100 memory bandwidth allows for a 3B
  model. It is bound by Python and kernel-launch overhead per layer, not by weight reads. The 0.5B draft has
  24 layers versus the target's 36, so a draft step costs roughly two-thirds of a target step. Four draft steps plus
  one verify step cost about as much as the ~3 target steps they save. Speculation only pays when the target
  step is memory-bound and much more expensive than the draft step.
- **Correctness note:** some outputs were not identical to plain greedy decoding. That is bf16 numerics: the target
  verifies k tokens in one batched forward instead of one at a time, so logits differ slightly and can flip
  near-tie argmaxes. It is not a bug in the method. The rate is in `identical_output_rate`.
- **What would change it:** a serving engine with CUDA graphs (the vLLM part of this experiment), a larger target
  (7B+ with a matching-vocab draft), or a shallower draft (n-gram/EAGLE heads).

## 7. Plain DPO destroyed arithmetic accuracy
- **Evidence:** `training/runs/*-dpo/log_history.json`; `training/runs/eval_report.json` from the 2026-10-10 run
- **Result:** Qwen2.5-0.5B with LoRA, 200 steps, beta 0.1:
  - accuracy 0.345 (base) → **0.015** (DPO from base), format compliance 0.54 → 0.71
  - SFT→DPO also dropped below base
  - preference accuracy hit 1.0 while `rewards/chosen` fell to about −6, meaning log p(chosen) dropped by
    about 60 nats against the reference model
- **Why:** half the rejected answers repeat the chosen steps and change only the final number. DPO only
  needs the *margin* to grow, and the cheapest way is to push both responses down
  ("likelihood displacement"), which damages the shared arithmetic tokens.
- **Fix:** add an SFT (NLL) term on chosen (`loss_type=["sigmoid","sft"]`, RPO style). This is now the
  default; `--sft-weight 0` reproduces plain DPO.
- **Result of the fix** (`eval_report*.json`, n=200, one seed):

  | Run | Plain DPO accuracy | DPO + SFT term accuracy |
  |---|---|---|
  | DPO from base | 0.015 | **0.980** |
  | DPO from SFT (SFT alone scored 0.995) | 0.325, format 0.000 | **0.970** |

- **The fix works, but DPO still adds nothing here.** Plain SFT (0.995) is the best model. On this task the
  rejected answers carry no information that the SFT demonstrations don't already contain. The differences
  between 0.995, 0.98 and 0.97 are within about 1–2 binomial standard errors (n=200, one seed), so they are
  not significant.

## 8. GRPO with a format-only reward was reward-hacked (expected; this is the control for RLVR)
- **Evidence:** `training/runs/eval_report.json`; `training/runs/*-grpo/log_history.json`
- **Result:** format compliance 1.000, accuracy **0.005**, mean output length **4 tokens**. The policy
  learned to print a bare `Answer: <n>` with no working. Completion length fell from about 13 to about 5
  tokens within 10 steps.
- **Same hyperparameters, verifiable reward (RLVR):** accuracy 0.345 → **0.955**, format 1.000, about 53 tokens.
- **Takeaway:** the reward function, not the RL algorithm, decided the outcome. A proxy reward that doesn't
  check correctness is optimised exactly as written.

## 9. Evaluation selector bug (caught before publishing)
In the first run, the evaluation cell chose runs with `glob('*-sft')`, which also matched `<ts>-dpo-from-sft`.
So the "SFT" row was really SFT→DPO, and the chained evaluation merged the DPO adapter into itself
(accuracy 0.000, 144 tokens). Fixed by matching method names exactly. The affected rows were re-evaluated.

## 10. TensorRT-LLM, untuned, was about 3× slower than vLLM on decode
- **Evidence:** `benchmarks/results/engines/*/report.json`, `engines/COMPARISON.md`
- **Setup:** same A100, Qwen2.5-1.5B, same workloads, each engine in its own runtime.

  | Engine | chat_short tok/s | TTFT p50 | TPOT p50 |
  |---|---|---|---|
  | vLLM 0.31 | 5,087 | 115 ms | 5.1 ms |
  | SGLang 0.5.21 | 4,859 | 163 ms | 4.9 ms |
  | TensorRT-LLM 1.2.1 | 1,715 | 118 ms | **17.3 ms** |
  | Dynamo frontend → vLLM worker | 4,989 | 122 ms | 5.2 ms |

- **Why TensorRT-LLM lost:** this is `trtllm-serve` with defaults (PyTorch backend, no tuned engine build,
  CUDA-graph and batch settings left alone) and with `ignore_eos` off. TTFT matched vLLM, so the gap is entirely
  per-token decode cost. That points at configuration, not at a TensorRT-LLM ceiling, so this is **not** evidence
  that TensorRT-LLM is slower in general. Getting it to run at all needed a Python 3.12 venv (no 3.13 wheels) and
  `pyarrow==19.0.1` to fit its `numpy<2` pin.
- **Dynamo:** the routing frontend over one vLLM worker cost about 2% throughput and about +7 ms TTFT against
  direct vLLM. That is the price of the control plane on a single node (2 repeats only).
- **What would change it:** enable CUDA graphs and tune `max_batch_size`/`max_num_tokens` in an `extra_llm_api_options`
  YAML, or build a TensorRT engine ahead of time.

## Quantisation note (mixed, not purely negative)
At concurrency 1 every quantised variant decoded faster than BF16 (AWQ-INT4 332 vs 225 tok/s; GPTQ-INT8 295;
FP8 weight-only 291, even without FP8 tensor cores on A100, because single-stream decode is memory-bound).
At concurrency 32 the gain mostly disappeared (AWQ 4.58k vs 4.34k tok/s; GPTQ/FP8 ≈ BF16) and p50 TTFT got
worse (~215 ms vs ~150 ms) because prefill is compute-bound and pays dequantisation cost.
