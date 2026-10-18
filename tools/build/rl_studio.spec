# PyInstaller spec for "RL Trading Studio.exe" (the launcher in rl_studio_launcher.py).
#
#   .venv\Scripts\python.exe -m pip install -r requirements-build.txt
#   .venv\Scripts\python.exe -m PyInstaller tools\build\rl_studio.spec --distpath dist --workpath build\pyinstaller --noconfirm
#   copy "dist\RL Trading Studio.exe" .        (the exe must sit in the project folder)
#
# One small windowed exe (~10 MB): it starts rl_web.py with the project's .venv, so the ML
# stack is NOT frozen into it (see the launcher's docstring for why).
from pathlib import Path

HERE = Path(SPECPATH)
ROOT = HERE.parents[1]
ICON = ROOT / "assets" / "rl_studio.ico"

a = Analysis([str(HERE / "rl_studio_launcher.py")], pathex=[], binaries=[], datas=[], hiddenimports=[],
             hookspath=[], runtime_hooks=[], excludes=["tkinter", "numpy", "pandas", "torch"], noarchive=False)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="RL Trading Studio", debug=False, bootloader_ignore_signals=False,
          strip=False, upx=False, runtime_tmpdir=None, console=False, disable_windowed_traceback=False,
          icon=str(ICON) if ICON.exists() else None)
