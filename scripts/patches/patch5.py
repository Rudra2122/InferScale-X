import os
os.chdir(os.environ.get("ISX_ROOT", "/content/InferScale-X"))

def rep(path, old, new, marker):
    s = open(path).read()
    if marker in s:
        print(f"skip (already applied): {path}"); return
    assert old in s, f"pattern not found in {path}: {old[:70]!r}"
    open(path, "w").write(s.replace(old, new, 1)); print(f"patched: {path}")

P = "training/dpo/train_dpo.py"
rep(P, '''    ap.add_argument("--seed", type=int, default=42)
''', '''    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sft-weight", type=float, default=1.0,
                    help="weight of an NLL term on 'chosen' (RPO). 0 = plain DPO, which collapsed in run 1")
''', "--sft-weight")
rep(P, '''    d = run_dir("dpo", "from-sft" if a.init_adapter else "")''',
'''    tag = "-".join(t for t in ("from-sft" if a.init_adapter else "", "rpo" if a.sft_weight > 0 else "") if t)
    d = run_dir("dpo", tag)''', 'tag = "-".join')
rep(P, '''                    report_to=[], seed=a.seed)''',
'''                    report_to=[], seed=a.seed,
                    # plain DPO pushed log p(chosen) down too ("likelihood displacement": chosen and rejected share
                    # every token but the final number) and broke arithmetic; an SFT term on chosen anchors it
                    **({"loss_type": ["sigmoid", "sft"], "loss_weights": [1.0, a.sft_weight]} if a.sft_weight > 0 else {}))''',
"likelihood displacement")
print("\nPATCH5 DONE")
