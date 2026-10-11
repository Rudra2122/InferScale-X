"""Build SFT / DPO / RL datasets with lineage, dedup and train/eval leakage checks.

All data is generated programmatically from benchmarks/analysis/arith_eval.py, so every
label is verifiable and every example traces back to a seed.

Outputs (training/data/):
  sft.jsonl   {"messages": [...system, user, assistant]}
  dpo.jsonl   {"prompt": [...], "chosen": [assistant], "rejected": [assistant]}
  rl.jsonl    {"prompt": [...], "answer": int}
  eval.jsonl  held-out problems (disjoint seed range)
  lineage.json  generator version, seeds, counts, sha256 of each file, leakage check result

python training/data-curation/build_datasets.py --n-sft 2000 --n-dpo 1000 --n-rl 1000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.analysis.arith_eval import SYSTEM, eval_set, make_problem, solution_text  # noqa: E402

GENERATOR_VERSION = "arith-v1"
OUT = Path(__file__).resolve().parents[1] / "data"


def norm(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


def qhash(q: str) -> str:
    return hashlib.sha256(norm(q).encode()).hexdigest()[:16]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dedup(problems: list[dict], seen: set[str]) -> list[dict]:
    out = []
    for p in problems:
        h = qhash(p["question"])
        if h in seen:
            continue
        seen.add(h)
        out.append(p)
    return out


def wrong_answer(p: dict, rng: random.Random) -> str:
    delta = rng.choice([-10, -3, -2, -1, 1, 2, 3, 7, 10])
    bad = p["answer"] + delta
    style = rng.random()
    if style < 0.5:  # plausible-looking reasoning with an arithmetic slip
        steps = list(p.get("steps") or [])
        if steps:
            steps[-1] = steps[-1].rsplit("=", 1)[0] + f"= {bad}"
        return "\n".join(steps + [f"Answer: {bad}"])
    return f"The answer is probably around {bad}."  # also violates the required format


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sft", type=int, default=2000)
    ap.add_argument("--n-dpo", type=int, default=1000)
    ap.add_argument("--n-rl", type=int, default=1000)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(1234)

    ev = eval_set(a.n_eval)
    eval_hashes = {qhash(p["question"]) for p in ev}
    seen = set(eval_hashes)  # anything identical to an eval question is dropped from training
    # disjoint seed ranges per split; oversample to survive dedup
    sft = dedup([make_problem(s) for s in range(0, a.n_sft * 3)], seen)[:a.n_sft]
    dpo = dedup([make_problem(s) for s in range(200_000, 200_000 + a.n_dpo * 3)], seen)[:a.n_dpo]
    rl = dedup([make_problem(s) for s in range(400_000, 400_000 + a.n_rl * 3)], seen)[:a.n_rl]

    def sysuser(p):
        return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p["question"]}]

    files = {
        "sft.jsonl": [{"messages": sysuser(p) + [{"role": "assistant", "content": solution_text(p)}], "id": p["id"]}
                      for p in sft],
        "dpo.jsonl": [{"prompt": sysuser(p), "chosen": [{"role": "assistant", "content": solution_text(p)}],
                       "rejected": [{"role": "assistant", "content": wrong_answer(p, rng)}], "id": p["id"]}
                      for p in dpo],
        "rl.jsonl": [{"prompt": sysuser(p), "answer": p["answer"], "id": p["id"]} for p in rl],
        "eval.jsonl": [{"prompt": sysuser(p), "answer": p["answer"], "id": p["id"], "question": p["question"]}
                       for p in ev],
    }
    for name, rows in files.items():
        with open(out / name, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    train_hashes = {qhash(p["question"]) for p in sft + dpo + rl}
    lineage = {
        "generator": GENERATOR_VERSION,
        "seed_ranges": {"sft": [0, a.n_sft * 3], "dpo": [200_000, 200_000 + a.n_dpo * 3],
                        "rl": [400_000, 400_000 + a.n_rl * 3], "eval": [10**6, 10**6 + a.n_eval]},
        "counts": {k: len(v) for k, v in files.items()},
        "sha256": {k: sha(out / k) for k in files},
        "leakage_check": {"train_eval_overlap": len(train_hashes & eval_hashes), "pass": not (train_hashes & eval_hashes)},
        "note": "Arithmetic templates repeat structure; exact-question overlap is zero but templates are shared. "
                "Report results as in-distribution generalisation, not general reasoning.",
    }
    (out / "lineage.json").write_text(json.dumps(lineage, indent=2))
    print(json.dumps({k: lineage[k] for k in ("counts", "leakage_check")}, indent=2))


if __name__ == "__main__":
    main()
