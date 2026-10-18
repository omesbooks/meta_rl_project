"""RL Trading Studio.exe - the double-click launcher (built by tools/build/rl_studio.spec).

Why a launcher and not a frozen app: every page drives the CLI scripts (rl_train.py,
backtest_live.py, ...) with the project's own interpreter, so the exe only has to find the
project folder and start rl_web.py with .venv\\Scripts\\pythonw.exe (no console window).

- the exe lives in the project folder (or any folder below it)
- no .venv yet -> runs run_rl_app.bat in a console window, which creates it and installs
  requirements.txt, then opens the app
- arguments are passed through (--classic, --debug); with --smoke the launcher waits and
  returns the self-test's exit code (used by the build check)
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def project_root():
    here = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
    for d in (here, *here.parents):
        if (d / "rl_web.py").is_file() and (d / "webgui").is_dir():
            return d
    return None


def message(text):
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, "RL Trading Studio", 0x40)
    except Exception:
        print(text)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    root = project_root()
    if root is None:
        message("Put 'RL Trading Studio.exe' in the project folder (next to rl_web.py) and start it again.")
        return 1
    smoke = "--smoke" in args
    pyw = root / ".venv" / "Scripts" / ("python.exe" if smoke else "pythonw.exe")
    if pyw.is_file():
        proc = subprocess.Popen([str(pyw), str(root / "rl_web.py"), *args], cwd=str(root))
    else:
        flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        proc = subprocess.Popen(["cmd.exe", "/c", str(root / "run_rl_app.bat"), *args], cwd=str(root), creationflags=flags)
    return proc.wait() if smoke else 0


if __name__ == "__main__":
    sys.exit(main())
