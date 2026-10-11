import os
os.chdir(os.environ.get("ISX_ROOT", "."))

def rep(path, old, new, marker):
    s = open(path).read()
    if marker in s:
        print(f"skip (already applied): {path}"); return
    assert old in s, f"pattern not found in {path}: {old[:70]!r}"
    open(path, "w").write(s.replace(old, new, 1)); print(f"patched: {path}")

# 1) serverctl: stop waiting as soon as the server log shows a fatal startup error
P = "scripts/serverctl.py"
rep(P, '''    ok = wait_healthy(f"http://127.0.0.1:{port}", timeout=timeout, proc=p, path=health_path)''',
'''    ok = wait_healthy(f"http://127.0.0.1:{port}", timeout=timeout, proc=p, path=health_path,
                      log_path=LOGS / f"{name}.log")''', "log_path=LOGS")
rep(P, '''def wait_healthy(url: str, timeout: float = 600, proc: subprocess.Popen | None = None, path: str = "/health") -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False''',
'''FATAL_MARKERS = ("Engine core initialization failed", "pydantic_core._pydantic_core.ValidationError",
                 "CUDA out of memory", "No available memory for the cache blocks", "Unsupported connector type")


def wait_healthy(url: str, timeout: float = 600, proc: subprocess.Popen | None = None, path: str = "/health",
                 log_path: Path | None = None) -> bool:
    """Poll /health. Fail fast if the process exits OR its log shows a fatal startup error (vLLM's API server
    can stay alive for the whole timeout after its engine core has already died)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        if log_path is not None and log_path.exists():
            try:
                tail = log_path.read_text(errors="replace")[-20000:]
                if any(m in tail for m in FATAL_MARKERS):
                    return False
            except OSError:
                pass''', "FATAL_MARKERS")

# 2) specdec: really free the HF models before starting vLLM
P = "optimization/speculative-decoding/exp_specdec.py"
rep(P, '''    del tgt, drf
''', '''    tgt.forward = orig_forward  # drop the counting wrapper: its closure kept the model alive
    del tgt, drf, orig_forward, counting_forward
    import gc

    gc.collect()
''', "drop the counting wrapper")
rep(P, '''    for name, extra in configs.items():
        serverctl.stopall()''', '''    import torch

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        res["gpu_free_before_vllm_gib"] = round(free / 2**30, 2)
        if free / total < 0.85:
            print(f"[spec] WARNING only {free / 2**30:.1f} GiB free before vLLM; run with --skip-hf in a separate process")
    for name, extra in configs.items():
        serverctl.stopall()''', "gpu_free_before_vllm_gib")
print("\nPATCH4 DONE")
