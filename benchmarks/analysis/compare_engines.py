"""Compare engine runs (each produced in its own Colab runtime by inference/exp_engines.py).

python benchmarks/analysis/compare_engines.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    reports = {}
    for p in sorted(glob.glob(str(ROOT / "benchmarks/results/engines/*/report.json"))):
        r = json.loads(Path(p).read_text())
        reports[r["engine"]] = r  # latest per engine wins (sorted by timestamped dir)
    if not reports:
        print("no engine reports found")
        return
    lines = ["| engine | workload | tok/s (mean ± CI) | TTFT p50 ms | TPOT p50 ms | error rate |", "|---|---|---|---|---|---|"]
    for eng, r in reports.items():
        if r.get("error"):
            lines.append(f"| {eng} | - | FAILED TO START | | | |")
            continue
        for wl, agg in r["workloads"].items():
            t = agg["output_tok_per_s"]
            ci = (t["ci95"][1] - t["ci95"][0]) / 2 if "ci95" in t else 0
            lines.append(f"| {eng} | {wl} | {t['mean']:.0f} ± {ci:.0f} | {agg['ttft_p50_s']['mean']*1e3:.1f} | "
                         f"{agg['tpot_p50_s']['mean']*1e3:.2f} | {agg['error_rate']['mean']:.3f} |")
    md = "\n".join(lines)
    out = ROOT / "benchmarks/results/engines/COMPARISON.md"
    out.write_text("# Engine comparison (same A100, same workloads, separate runtimes)\n\n" + md + "\n")
    print(md)


if __name__ == "__main__":
    main()
