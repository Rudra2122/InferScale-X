"""DPO with LoRA (TRL). Preference pairs: correct+formatted (chosen) vs wrong or unformatted (rejected).

python training/dpo/train_dpo.py --max-steps 200
python training/dpo/train_dpo.py --init-adapter training/runs/<sft-run>/adapter   # SFT -> DPO chain
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.common import BASE_MODEL, hf_dataset, lora_config, run_dir, write_meta  # noqa: E402


def load_policy(model_name: str, init_adapter: str | None):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    if init_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, init_adapter).merge_and_unload()  # start DPO from SFT weights
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--init-adapter", default=None)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sft-weight", type=float, default=1.0,
                    help="weight of an NLL term on 'chosen' (RPO). 0 = plain DPO, which collapsed in run 1")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    tag = "-".join(t for t in ("from-sft" if a.init_adapter else "", "rpo" if a.sft_weight > 0 else "") if t)
    d = run_dir("dpo", tag)
    tok = AutoTokenizer.from_pretrained(a.model)
    cfg = DPOConfig(output_dir=str(d / "ckpt"), max_steps=a.max_steps, learning_rate=a.lr, beta=a.beta,
                    per_device_train_batch_size=a.batch, logging_steps=10, save_strategy="no", bf16=True,
                    report_to=[], seed=a.seed,
                    # plain DPO pushed log p(chosen) down too ("likelihood displacement": chosen and rejected share
                    # every token but the final number) and broke arithmetic; an SFT term on chosen anchors it
                    **({"loss_type": ["sigmoid", "sft"], "loss_weights": [1.0, a.sft_weight]} if a.sft_weight > 0 else {}))
    trainer = DPOTrainer(model=load_policy(a.model, a.init_adapter), args=cfg,
                         train_dataset=hf_dataset("dpo.jsonl"), processing_class=tok, peft_config=lora_config())
    res = trainer.train()
    trainer.save_model(str(d / "adapter"))
    tok.save_pretrained(str(d / "adapter"))
    write_meta(d, "dpo", {**vars(a), "base_model": a.model}, res)
    (d / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
    print(f"[dpo] adapter at {d / 'adapter'}  (if --init-adapter was used, evaluate with --merge-first {a.init_adapter})")


if __name__ == "__main__":
    main()
