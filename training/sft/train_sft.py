"""SFT with LoRA (TRL) on verifiable arithmetic data.

python training/sft/train_sft.py --max-steps 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.common import BASE_MODEL, hf_dataset, lora_config, run_dir, write_meta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    from trl import SFTConfig, SFTTrainer

    d = run_dir("sft")
    cfg = SFTConfig(output_dir=str(d / "ckpt"), max_steps=a.max_steps, learning_rate=a.lr,
                    per_device_train_batch_size=a.batch, gradient_accumulation_steps=1, logging_steps=10,
                    save_strategy="no", bf16=True, report_to=[], seed=a.seed, lr_scheduler_type="cosine",
                    warmup_steps=max(1, int(0.05 * a.max_steps)))  # transformers 5 removed warmup_ratio
    trainer = SFTTrainer(model=a.model, args=cfg, train_dataset=hf_dataset("sft.jsonl", a.n), peft_config=lora_config())
    res = trainer.train()
    trainer.save_model(str(d / "adapter"))
    trainer.processing_class.save_pretrained(str(d / "adapter"))
    write_meta(d, "sft", {**vars(a), "base_model": a.model}, res)
    (d / "log_history.json").write_text(__import__("json").dumps(trainer.state.log_history, indent=2))
    print(f"[sft] adapter at {d / 'adapter'}")


if __name__ == "__main__":
    main()
