# Grove evaluation (NOT executed)

NVIDIA Grove is a Kubernetes API for gang-scheduling and scaling multi-component
inference workloads (e.g. prefill + decode worker groups as one unit). It requires
a Kubernetes cluster; Colab has none, so **this project contains no Grove results**.

What a real evaluation should cover (checklist for a cluster run):

1. Install Grove's operator and CRDs on a cluster with >= 2 GPU nodes, pinned version recorded in `docs/compatibility-matrix.md`.
2. Express the disaggregated topology (1 prefill group, N decode groups) as a Grove resource; compare with two independent Deployments.
3. Measure: time to schedule the full group, behaviour when one GPU node is drained (does the group reschedule atomically?), scale-up latency of the decode group under load.
4. Record negative results (e.g. partial scheduling, version incompatibility with the Dynamo release in use).

Until those runs exist, résumé claims must not mention Grove beyond "evaluated design / reviewed".
