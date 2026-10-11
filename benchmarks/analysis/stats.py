"""Small, dependency-light statistics helpers used by every experiment."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

# two-sided 95% t critical values for small n (df = n-1); falls back to 1.96
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
        10: 2.228, 15: 2.131, 20: 2.086, 30: 2.042}


def t95(df: int) -> float:
    if df <= 0:
        return float("nan")
    keys = sorted(k for k in _T95 if k <= df)
    return _T95[keys[-1]] if df <= 30 else 1.96


def pct(values: Sequence[float], q: float) -> float | None:
    v = [x for x in values if x is not None and not math.isnan(x)]
    return float(np.percentile(v, q)) if v else None


def dist(values: Iterable[float]) -> dict:
    v = [float(x) for x in values if x is not None and not math.isnan(float(x))]
    if not v:
        return {"n": 0}
    a = np.asarray(v)
    return {"n": len(v), "mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(a.min()), "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)), "max": float(a.max())}


def mean_ci(values: Sequence[float]) -> dict:
    """Mean with 95% CI across repeated runs (t-distribution)."""
    v = [float(x) for x in values if x is not None]
    n = len(v)
    if n == 0:
        return {"n": 0}
    m = float(np.mean(v))
    if n == 1:
        return {"n": 1, "mean": m, "std": 0.0, "ci95": [m, m], "note": "single run - no variance estimate"}
    s = float(np.std(v, ddof=1))
    h = t95(n - 1) * s / math.sqrt(n)
    return {"n": n, "mean": m, "std": s, "cv": s / m if m else None, "ci95": [m - h, m + h]}


def jain_fairness(xs: Sequence[float]) -> float | None:
    xs = [float(x) for x in xs]
    if not xs or sum(xs) == 0:
        return None
    return (sum(xs) ** 2) / (len(xs) * sum(x * x for x in xs))


def compare(baseline: Sequence[float], candidate: Sequence[float], lower_is_better: bool = True) -> dict:
    """Relative change of candidate vs baseline with a conservative CI-overlap verdict."""
    b, c = mean_ci(baseline), mean_ci(candidate)
    if not b.get("n") or not c.get("n"):
        return {"verdict": "insufficient_data"}
    delta = (c["mean"] - b["mean"]) / b["mean"] if b["mean"] else float("nan")
    verdict = "inconclusive"
    if b["n"] > 1 and c["n"] > 1:
        overlap = not (c["ci95"][1] < b["ci95"][0] or c["ci95"][0] > b["ci95"][1])
        if not overlap:
            better = c["mean"] < b["mean"] if lower_is_better else c["mean"] > b["mean"]
            verdict = "improved" if better else "regressed"
        else:
            verdict = "no_significant_difference"
    else:
        verdict = "single_run_not_conclusive"
    return {"baseline": b, "candidate": c, "relative_change": delta, "verdict": verdict}
