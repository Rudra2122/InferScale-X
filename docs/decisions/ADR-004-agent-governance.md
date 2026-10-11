# ADR-004: Enforce agent permissions outside the model

Status: accepted

## Decision
The model proposes; the policy engine disposes. A tool runs only if it is allowed by the
tenant config AND declared by the pinned skill version AND within the run's budget AND not
high-risk in a context tainted by untrusted content. Tenant identity is bound by the runtime
and never accepted from model arguments. Every decision goes to a hash-chained audit log.

## Why
Prompt-injection detection is heuristic and will miss attacks; authorisation must not depend
on it. The security suite therefore runs an adversarial *scripted* model that always attempts
the attack — enforcement is tested independently of how cooperative the real model is.

## Consequences
− Taint tracking is coarse (whole run). False positives disable memory writes after reading
  user-submitted docs; acceptable for a reference design, documented in the threat model.
