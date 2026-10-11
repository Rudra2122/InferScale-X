"""Shared training utilities: data loading, LoRA config, run metadata."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "training" / "data"
RUNS = ROOT / "training" / "runs"
BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def load_jsonl(name: str) -> list[dict]:
    with open(DATA / name) as f:
        return [json.loads(line) for line in f]


def hf_dataset(name: str, n: int | None = None):
    from datasets import Dataset

    rows = load_jsonl(name)
    if n:
        rows = rows[:n]
    for r in rows:
        r.pop("id", None)
    return Dataset.from_list(rows)


def lora_config(r: int = 16):
    from peft import LoraConfig

    return LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])


def run_dir(method: str, tag: str = "") -> Path:
    d = RUNS / f"{time.strftime('%Y%m%d-%H%M%S')}-{method}{('-' + tag) if tag else ''}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_meta(d: Path, method: str, config: dict, train_result=None) -> None:
    from benchmarks.runners.env_capture import capture

    lineage = json.loads((DATA / "lineage.json").read_text()) if (DATA / "lineage.json").exists() else None
    meta = {"method": method, "base_model": config.get("base_model", BASE_MODEL), "config": config,
            "data_lineage": lineage, "env": capture()}
    if train_result is not None:
        meta["train_metrics"] = getattr(train_result, "metrics", None)
    (d / "run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
