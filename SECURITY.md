# Security policy

This is a reference platform, not a hardened product. Known gaps are listed in `docs/security-threat-model.md`
(notably: the tool sandbox is not a VM/gVisor boundary).

The dev API keys in `configs/*.example.yaml` and `configs/gateway.colab.yaml` are **public test keys** — never
expose a gateway using them. `scripts/colab_public_endpoint.sh` generates a fresh key per session for that reason.

Report vulnerabilities privately via GitHub's "Report a vulnerability" (Security Advisories) on this repository
rather than a public issue. Include the commit, a reproduction, and impact.
