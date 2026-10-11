"""Sandboxed Python execution for the `python_exec` tool.

Layers (defence in depth, NOT a hard security boundary):
  1. separate interpreter process (`python -I`), scrubbed environment, fresh temp cwd
  2. rlimits: CPU seconds, address space, output file size, open files
  3. a PEP 578 audit hook installed before user code runs that blocks: network
     (socket.connect/bind), subprocess/exec/system, ctypes, writes outside the
     temp dir, and imports of a deny-listed set of modules
  4. wall-clock timeout and output truncation

Production deployments should run this inside gVisor / Firecracker / nsjail with
no network namespace. docs/security-threat-model.md records this gap explicitly.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass

PRELUDE = textwrap.dedent(r'''
import sys, os
_ROOT = os.path.realpath(os.getcwd())
_DENY_MODULES = {"socket", "ssl", "urllib.request", "http.client", "ctypes", "subprocess", "multiprocessing",
                 "asyncio.subprocess", "ftplib", "smtplib", "telnetlib", "pty", "_posixsubprocess"}
_BLOCK_EVENTS = {"socket.connect", "socket.bind", "socket.getaddrinfo", "subprocess.Popen", "os.system",
                 "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty", "ctypes.dlopen",
                 "os.kill", "os.chmod", "os.chown", "os.remove", "os.rename", "os.rmdir", "shutil.rmtree"}
def _hook(event, args):
    if event in _BLOCK_EVENTS:
        raise PermissionError(f"sandbox: {event} is blocked")
    if event == "import" and args and args[0] in _DENY_MODULES:
        raise PermissionError(f"sandbox: import of {args[0]} is blocked")
    if event == "open" and args:
        path, mode = args[0], (args[1] if len(args) > 1 else "r")
        if isinstance(path, (str, bytes)) and isinstance(mode, str) and any(c in mode for c in "wax+"):
            p = os.path.realpath(path if isinstance(path, str) else path.decode())
            if not p.startswith(_ROOT):
                raise PermissionError("sandbox: writes outside the sandbox dir are blocked")
sys.addaudithook(_hook)
del _hook
''')


@dataclass
class SandboxResult:
    ok: bool
    stdout: str
    stderr: str
    exit_code: int | None
    timed_out: bool

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _limits(cpu_s: int, mem_mb: int):
    """Applied in the child before exec. Each limit is set independently: macOS rejects
    RLIMIT_AS, and that must not silently drop the others (this was a real bug on macOS)."""
    def f():
        try:
            import resource
        except ImportError:
            resource = None
        if resource is not None:
            mem = mem_mb * 2**20
            for name, val in (("RLIMIT_CPU", cpu_s), ("RLIMIT_AS", mem), ("RLIMIT_DATA", mem),
                              ("RLIMIT_FSIZE", 2**20), ("RLIMIT_NOFILE", 32)):
                if os.environ.get("ISX_SANDBOX_SKIP_" + name):  # test hook: simulate a platform without it
                    continue
                lim = getattr(resource, name, None)
                if lim is None:
                    continue
                try:
                    resource.setrlimit(lim, (val, val))
                except (ValueError, OSError):
                    pass  # unsupported on this OS; the RSS watchdog below still enforces memory
        os.setsid()
    return f


def _rss_mb(pid: int) -> float | None:
    """Resident memory of a process in MB, portable across Linux and macOS."""
    try:
        with open(f"/proc/{pid}/statm") as fh:  # Linux fast path
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except (OSError, ValueError, IndexError):
        pass
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=1).stdout
        return int(out.strip()) / 1024 if out.strip() else None
    except Exception:
        return None


def run_python(code: str, timeout_s: float = 5.0, cpu_s: int = 3, mem_mb: int = 512,
               max_output: int = 8000, poll_s: float = 0.02) -> SandboxResult:
    """Run untrusted code. Memory is enforced twice: RLIMIT_AS where the OS supports it, and a
    parent-side RSS watchdog that SIGKILLs the process group above `mem_mb` (works on macOS)."""
    if len(code) > 20_000:
        return SandboxResult(False, "", "code too long", None, False)
    import signal
    import time

    with tempfile.TemporaryDirectory(prefix="isx-sbx-") as d:
        script = os.path.join(d, "main.py")
        with open(script, "w") as f:
            f.write(PRELUDE + "\n" + "exec(compile(" + json.dumps(code) + ", '<agent>', 'exec'), {'__name__': '__main__'})\n")
        env = {"PATH": "/usr/bin:/bin", "HOME": d, "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8",
               "no_proxy": "*", "NO_PROXY": "*"}
        out_f, err_f = tempfile.TemporaryFile("w+"), tempfile.TemporaryFile("w+")
        p = subprocess.Popen([sys.executable, "-I", script], cwd=d, env=env, stdout=out_f, stderr=err_f, text=True,
                             preexec_fn=_limits(cpu_s, mem_mb))
        deadline = time.monotonic() + timeout_s
        killed = None
        while p.poll() is None:
            if time.monotonic() > deadline:
                killed = "timeout"
            else:
                rss = _rss_mb(p.pid)
                if rss is not None and rss > mem_mb:
                    killed = f"memory limit exceeded ({rss:.0f} MB > {mem_mb} MB)"
            if killed:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()
                break
            time.sleep(poll_s)
        out_f.seek(0)
        err_f.seek(0)
        stdout, stderr = out_f.read()[:max_output], err_f.read()[-max_output:]
        out_f.close()
        err_f.close()
        if killed == "timeout":
            return SandboxResult(False, stdout, "timeout", None, True)
        if killed:
            return SandboxResult(False, stdout, (stderr + "\n" + killed).strip(), p.returncode, False)
        return SandboxResult(p.returncode == 0, stdout, stderr, p.returncode, False)
