"""RLVR: GRPO where the reward is a program that verifies correctness (no reward model).

Same hyper-parameters as training/grpo/train_grpo.py --reward heuristic, so the only
difference between the two runs is the reward signal.

python training/rlvr/train_rlvr.py --max-steps 150
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.grpo.train_grpo import parser, train  # noqa: E402

if __name__ == "__main__":
    a = parser().parse_args()
    train("verifiable", "rlvr", a)
