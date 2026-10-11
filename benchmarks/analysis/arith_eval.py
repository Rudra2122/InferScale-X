"""Verifiable arithmetic word problems: shared quality eval + RLVR reward.

Deterministic, seeded, no downloads. Used by:
  - quantization quality check (BF16 vs AWQ/GPTQ through the OpenAI API)
  - speculative decoding correctness (outputs must match greedy target)
  - SFT/DPO/GRPO held-out evaluation and the RLVR reward function

Splits are disjoint by seed range: train seeds [0, 100000), eval seeds [10**6, ...).
"""
from __future__ import annotations

import random
import re

NAMES = ["Ava", "Ben", "Chen", "Dara", "Eli", "Fatima", "Gus", "Hana", "Ivan", "Jo"]
ITEMS = ["apples", "books", "coins", "marbles", "stamps", "tickets", "cards", "pens"]
SYSTEM = "Solve the problem. Think briefly, then give the final answer on the last line as 'Answer: <number>'."


def make_problem(seed: int) -> dict:
    r = random.Random(seed)
    name, item = r.choice(NAMES), r.choice(ITEMS)
    kind = r.randint(0, 3)
    a, b, c = r.randint(3, 60), r.randint(2, 30), r.randint(2, 9)
    if kind == 0:
        q = f"{name} has {a} {item}, buys {b} more, then gives away {c}. How many {item} does {name} have now?"
        ans = a + b - c
        steps = [f"{a} + {b} = {a + b}", f"{a + b} - {c} = {ans}"]
    elif kind == 1:
        q = f"{name} has {c} boxes with {a} {item} each and loses {b} {item}. How many {item} remain?"
        ans = c * a - b
        steps = [f"{c} x {a} = {c * a}", f"{c * a} - {b} = {ans}"]
    elif kind == 2:
        total = a * c
        q = f"{name} shares {total} {item} equally among {c} friends, then each friend gets {b} more. How many {item} does each friend have?"
        ans = a + b
        steps = [f"{total} / {c} = {a}", f"{a} + {b} = {ans}"]
    else:
        q = f"{name} has {a} {item}. A friend has {b} more than {name}, and a third person has {c} times the friend's amount. How many {item} does the third person have?"
        ans = (a + b) * c
        steps = [f"{a} + {b} = {a + b}", f"{a + b} x {c} = {ans}"]
    return {"id": f"arith-{seed}", "question": q, "answer": ans, "seed": seed, "steps": steps}


def train_set(n: int, offset: int = 0) -> list[dict]:
    return [make_problem(s) for s in range(offset, offset + n)]


def eval_set(n: int = 200) -> list[dict]:
    return [make_problem(10**6 + s) for s in range(n)]


_ANS = re.compile(r"answer\s*[:=]\s*\$?(-?\d[\d,]*)", re.IGNORECASE)
_NUM = re.compile(r"-?\d[\d,]*")


def extract_answer(text: str) -> int | None:
    if not text:
        return None
    m = _ANS.findall(text)
    if m:
        s = m[-1]
    else:
        nums = _NUM.findall(text)
        if not nums:
            return None
        s = nums[-1]
    try:
        return int(s.replace(",", ""))
    except ValueError:
        return None


def reward(completion: str, answer: int) -> float:
    """RLVR reward: 1.0 exact match with explicit 'Answer:' line, 0.5 if only last number matches, else 0."""
    if _ANS.search(completion or "") and extract_answer(completion) == answer:
        return 1.0
    nums = _NUM.findall(completion or "")
    if nums and nums[-1].replace(",", "").lstrip("-").isdigit() and int(nums[-1].replace(",", "")) == answer:
        return 0.5
    return 0.0


def score(predictions: list[str], problems: list[dict]) -> dict:
    correct = sum(1 for p, q in zip(predictions, problems) if extract_answer(p) == q["answer"])
    formatted = sum(1 for p in predictions if _ANS.search(p or ""))
    n = len(problems)
    return {"n": n, "accuracy": correct / n if n else None, "format_rate": formatted / n if n else None}


def messages(problem: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": problem["question"]}]


def solution_text(problem: dict) -> str:
    """Reference chain-of-thought used to build SFT targets (programmatic, verifiable)."""
    steps = problem.get("steps") or []
    return "\n".join(steps + [f"Answer: {problem['answer']}"])
