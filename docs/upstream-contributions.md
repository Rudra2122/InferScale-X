# Upstream contributions

Goal: real, merged contributions to the projects this platform depends on. Nothing below is a
known open bug — these are **places where running this project is likely to surface one**, plus
how to find genuine open issues. Update the tracker as you go; `docs/jd-coverage.md` only counts
Open Source as evidenced once a row says `merged`.

## How to find real issues (run on your Mac)
```bash
gh issue list -R vllm-project/vllm --label "good first issue" --state open --limit 30
gh issue list -R vllm-project/vllm --search "prefix caching cache_salt in:title,body" --state open
gh issue list -R vllm-project/vllm --search "kv offloading OR OffloadingConnector" --state open
gh issue list -R sgl-project/sglang --label "good first issue" --state open
gh issue list -R ai-dynamo/dynamo --label "good first issue" --state open
gh issue list -R ai-dynamo/nixl --state open --limit 30
gh issue list -R LMCache/LMCache --label "good first issue" --state open
gh issue list -R huggingface/trl --label "good first issue" --state open
gh issue list -R triton-lang/triton --search "interpreter in:title" --state open
```

## Where this project will hit upstream edges (verify before filing anything)
| Area | What to watch for while running | Typical contribution |
|---|---|---|
| vLLM CLI flags | a flag used here was renamed/removed (`--enable-prompt-tokens-details`, prefix-cache toggles, `--num-gpu-blocks-override`, `--kv-offloading-*`) | doc fix / deprecation message |
| vLLM `cache_salt` | behaviour or docs for multi-tenant prefix-cache isolation unclear | docs example showing gateway-side salting |
| vLLM metrics | metric names differ from docs (`vllm:prefix_cache_*`) | docs / metric-name fix |
| vLLM KV connectors on 1 GPU | NixlConnector / OffloadingConnector fail single-GPU with unclear errors | clearer error, docs on single-node setup |
| LMCache | env-var config mismatches with docs | docs fix |
| Dynamo without Docker | etcd/NATS bring-up friction (`try_dynamo.sh`) | "run without docker compose" docs |
| NIXL Python API | loopback example missing or out of date | example script |
| TRL GRPO | reward-function signature confusion for conversational datasets | docs/example |
| Triton interpreter | kernels that pass on GPU but fail in `TRITON_INTERPRET=1` (or vice versa) | minimal repro issue |
| SGLang | OpenAI-compat differences vs vLLM (`stream_options`, `ignore_eos`) found by `loadgen` | compat fix or docs |

Rules: reproduce on the latest release first, search existing issues, attach a minimal repro (not
this whole repo), link the PR back here.

## Tracker
| # | Project | Type | Link | Status |
|---|---|---|---|---|
| 1 | | | | todo |
