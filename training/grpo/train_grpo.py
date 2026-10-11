"""GRPO with LoRA (TRL). Two reward regimes so the comparison is meaningful:

  --reward heuristic : rewards the *format* (an 'Answer: <number>' line, brevity) but
                       NOT correctness - a proxy/learned-reward stand-in that is
                       known to be hackable.
  --reward verifiable: RLVR - reward = exact correctness checked by a program
                       (benchmarks/analysis/arith_eval.reward). training/rlvr/ calls this.

python training/grpo/train_grpo.py --reward heuristic --max-steps 150
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.analysis.arith_eval import reward as verifiable_reward  # noqa: E402
from training.common import BASE_MODEL, hf_dataset, lora_config, run_dir, write_meta  # noqa: E402


def _text(c) -> str:
    if isinstance(c, list):  # conversational: [{"role": "assistant", "content": ...}]
        return c[-1].get("content", "") if c else ""
    return str(c)


def heuristic_reward(completions, **kwargs) -> list[float]:
    out = []
    for c in completions:
        t = _text(c)
        r = 0.0
        if re.search(r"answer\s*:\s*-?\d+", t, re.IGNORECASE):
            r += 0.7
        r += 0.3 * max(0.0, 1 - len(t) / 400)
        out.append(r)
    return out


def correctness_reward(completions, answer, **kwargs) -> list[float]:
    return [verifiable_reward(_text(c), int(a)) for c, a in zip(completions, answer)]


def train(reward: str, method_name: str, a) -> Path:
    from trl import GRPOConfig, GRPOTrainer

    d = run_dir(method_name)
    funcs = [correctness_reward] if reward == "verifiable" else [heuristic_reward]
    cfg = GRPOConfig(output_dir=str(d / "ckpt"), max_steps=a.max_steps, learning_rate=a.lr,
                     per_device_train_batch_size=a.batch, num_generations=a.num_generations,
                     max_completion_length=a.max_completion, logging_steps=5, save_strategy="no", bf16=True,
                     report_to=[], seed=a.seed, beta=a.beta, temperature=0.9)
    trainer = GRPOTrainer(model=a.model, reward_funcs=funcs, args=cfg, train_dataset=hf_dataset("rl.jsonl"),
                          peft_config=lora_config())
    res = trainer.train()
    trainer.save_model(str(d / "adapter"))
    trainer.processing_class.save_pretrained(str(d / "adapter"))
    write_meta(d, method_name, {**vars(a), "reward": reward, "base_model": a.model}, res)
    (d / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    print(f"[{method_name}] adapter at {d / 'adapter'}")
    return d


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=16, help="must be divisible by --num-generations")
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--max-completion", type=int, default=128)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--seed", type=int, default=42)
    return ap


if __name__ == "__main__":
    p = parser()
    p.add_argument("--reward", choices=["heuristic", "verifiable"], default="heuristic")
    a = p.parse_args()
    train(a.reward, "grpo" if a.reward == "heuristic" else "rlvr", a)
