# ADR-001: Scope the measured system to one A100; label everything else

Status: accepted

## Context
The original plan included Dynamo multi-node disaggregation, NIXL over RDMA, Kubernetes, Grove
and "10,000-GPU validation". The only hardware available is a single Colab A100 (no Docker,
no Kubernetes, no second GPU, sessions that disconnect).

## Decision
- Build and measure everything that is physically possible on one GPU: gateway, routing
  across co-located replicas, failure injection, prefix caching and isolation, KV movement
  over the paths that exist (D2D, host-staged), kernels, quantisation, speculative decoding,
  agents, model customisation.
- Ship cluster pieces (K8s, Helm, Grove, multi-node Dynamo) as reference designs validated
  only syntactically, and label them `reference`.
- Run "attempts" for Dynamo and vLLM KV connectors on one GPU and record success or failure.
- Every report states whether a number is measured, emulated, simulated or extrapolated.

## Consequences
+ Every claim is defensible in an interview because it points to a file.
− Disaggregation can only be PARTIAL in the readiness checklist. That is the correct status.
