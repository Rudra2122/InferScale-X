import os
os.chdir(os.environ.get("ISX_ROOT", "/content/InferScale-X"))

def rep(path, old, new, marker):
    s = open(path).read()
    if marker in s:
        print(f"skip (already applied): {path}"); return
    assert old in s, f"pattern not found in {path}: {old[:70]!r}"
    open(path, "w").write(s.replace(old, new, 1)); print(f"patched: {path}")

P = "inference/exp_engines.py"
rep(P, '''        info = launch(a.engine, a.model, a.gpu_mem)
''', '''        try:
            info = launch(a.engine, a.model, a.gpu_mem)
        except OSError as e:  # e.g. trtllm-serve not installed: record it instead of crashing with no report
            info = {"healthy": False, "cmd": None}
            report["launch_exception"] = f"{type(e).__name__}: {e}"
''', "launch_exception")
print("\nPATCH6 DONE")
