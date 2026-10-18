"""One subprocess at a time, stdout streamed as events (web counterpart of ProcessRunner).

Same busy/cancel semantics as rl_app.ProcessRunner: `_starting` counts as busy, a second
start is rejected, a stop requested before Popen returns is honoured, and shared state is
only cleared when it still refers to this process. Lines go into a queue; `drain()` is
called by the pump that forwards batches to the page.
"""
from __future__ import annotations

import os
import queue
import subprocess
import threading


class Runner:
    def __init__(self, cwd):
        self.cwd = str(cwd)
        self.proc = None
        self.q = queue.Queue()
        self._starting = False
        self._cancel = False
        self._lock = threading.Lock()
        self.page = None

    def is_running(self):
        with self._lock:
            return self._starting or self.proc is not None

    def start(self, cmd, page):
        """One command; queue gets ("line", page, text)* then ("done", page, {rc, cancelled})."""
        return self.start_sequence([(None, cmd)], page)

    def start_sequence(self, stages, page):
        """Run [(stage_name, cmd), ...] one after another, gated on exit code (the CTk
        pipeline rule): a failing or cancelled stage stops the rest. Named stages add
        ("stage", page, {index, total, name, command}) and ("stage_done", page, {...})."""
        with self._lock:
            if self._starting or self.proc is not None:
                return False
            self._starting = True
            self._cancel = False
            self.page = page
        named = [n for n, _ in stages if n]

        def worker():
            rc, cancelled, failed = 0, False, None
            for i, (name, cmd) in enumerate(stages):
                with self._lock:
                    if self._cancel:
                        cancelled = True
                        break
                if name:
                    self.q.put(("stage", page, {"index": i, "total": len(named), "name": name, "command": " ".join(map(str, cmd[1:]))}))
                rc, cancelled = self._run_one(cmd, page)
                if cancelled or rc != 0:
                    failed = name
                    break
                if name:
                    self.q.put(("stage_done", page, {"index": i, "total": len(named), "name": name}))
            with self._lock:
                self._starting = False
                self._cancel = False
            done = {"rc": rc, "cancelled": cancelled}
            if named:
                done["failed_stage"] = failed
            self.q.put(("done", page, done))

        threading.Thread(target=worker, daemon=True).start()
        return True

    def _run_one(self, cmd, page):
        proc, rc = None, 1
        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace", cwd=self.cwd, bufsize=1, env=env,
                                    creationflags=flags)
            with self._lock:
                self.proc = proc              # _starting stays True until the whole sequence ends
                cancel_now = self._cancel
            if cancel_now and proc.poll() is None:
                proc.terminate()
            for line in iter(proc.stdout.readline, ""):
                line = line.rstrip()
                if line:
                    self.q.put(("line", page, line))
            proc.wait()
            rc = proc.returncode
        except Exception as exc:
            self.q.put(("line", page, f"Process start/run failed: {exc}"))
        finally:
            with self._lock:
                cancelled = self._cancel
                if self.proc is proc:
                    self.proc = None
            if proc is not None and getattr(proc, "pid", None):
                try:
                    from artifact_paths import mark_process_runs
                    mark_process_runs(proc.pid, "cancelled" if cancelled else "failed")
                except Exception:
                    pass
        return rc, cancelled

    def stop(self):
        with self._lock:
            self._cancel = True
            proc = self.proc
        if proc:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    if proc.poll() is None:
                        proc.kill()
            except Exception:
                pass

    def drain(self, limit=500):
        out = []
        try:
            while len(out) < limit:
                out.append(self.q.get_nowait())
        except queue.Empty:
            pass
        return out
