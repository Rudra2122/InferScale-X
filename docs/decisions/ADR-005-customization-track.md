# ADR-005: Keep SFT/DPO/GRPO/RLVR out of the serving path

Status: accepted

Model customisation is a separate research track (own runtime, own dependencies, LoRA on a
0.5B model) that produces adapters and an evaluation report. The serving platform does not
depend on it. The question it answers for the platform is: *does customisation change
quality, latency or cost?* — hence `eval_models.py` reports mean output tokens per method
(RL often lengthens outputs, which raises serving cost) alongside accuracy.

GRPO (heuristic, hackable reward) and RLVR (programmatically verified reward) share identical
hyper-parameters so the reward signal is the only variable.
