"""RL Trading Studio — web GUI (pywebview). Native app window, no browser.

    run_rl_app.bat                 (default launcher; sets up .venv on first run)
    run_rl_app.bat --classic       (the CustomTkinter app, rl_app.py)
    .venv/Scripts/python.exe rl_web.py [--debug] [--smoke] [--classic]

--debug   opens the WebView2 dev tools
--smoke   self-test: hidden window, every page opened once, exit code 0 = OK (used by the
          fresh-machine check; prints one SMOKE line)
--classic hands over to rl_app.py

All pages run here (docs/web_gui_migration_plan_2026-10-18.md); backend: webgui/api.py.
When the Edge WebView2 runtime is missing the app prints the download link and falls back
to the classic app instead of failing.
"""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
if __name__ == "__main__" and sys.stdout is not None and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PAGE = ROOT / "webgui" / "ui" / "index.html"
APP_ID = "MetaFXClub.RLTradingStudio"
WEBVIEW2_URL = "https://developer.microsoft.com/microsoft-edge/webview2/"
PAGES = ["tools", "pipeline", "train", "backtest", "walkfwd", "finetune", "analyze", "regime", "models", "settings"]


def webview2_installed():
    """Edge WebView2 runtime present? (EdgeUpdate client key, machine or user install). Non-Windows: True."""
    if sys.platform != "win32":
        return True
    import winreg
    guid = r"{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
    keys = [(winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{guid}"),
            (winreg.HKEY_LOCAL_MACHINE, rf"SOFTWARE\Microsoft\EdgeUpdate\Clients\{guid}"),
            (winreg.HKEY_CURRENT_USER, rf"Software\Microsoft\EdgeUpdate\Clients\{guid}")]
    for hive, key in keys:
        try:
            with winreg.OpenKey(hive, key) as k:
                version, _ = winreg.QueryValueEx(k, "pv")
                if version and version != "0.0.0.0":
                    return True
        except OSError:
            continue
    return False


def run_classic(reason=""):
    if reason:
        if sys.stdout is None:                      # pythonw / the exe launcher: no console to print to
            try:
                import ctypes
                ctypes.windll.user32.MessageBoxW(None, reason, "RL Trading Studio", 0x30)
            except Exception:
                pass
        else:
            print(reason, flush=True)
    return subprocess.call([sys.executable, str(ROOT / "rl_app.py")], cwd=str(ROOT))


def window_icon_file():
    """.ico for the title bar / taskbar: branding window_icon, else the sidebar logo image
    (converted once to an .ico next to it in assets/.cache). None when neither exists."""
    try:
        import app_settings
        b = app_settings.effective_branding()["branding"]
    except Exception:
        return None
    for rel in (b.get("window_icon"), b.get("logo_image")):
        if not rel:
            continue
        path = Path(rel) if Path(rel).is_absolute() else ROOT / rel
        if not path.is_file():
            continue
        if path.suffix.lower() == ".ico":
            return path
        try:
            from PIL import Image
            cache = ROOT / "assets" / ".cache"
            cache.mkdir(parents=True, exist_ok=True)
            ico = cache / (path.stem + ".ico")
            if not ico.exists() or ico.stat().st_mtime < path.stat().st_mtime:
                img = Image.open(path).convert("RGBA")
                side = max(img.size)
                square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
                square.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
                square.save(ico, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
            return ico
        except Exception:
            continue
    return None


def set_window_icon(title, ico):
    """WM_SETICON on the pywebview window (pywebview's own icon= is GTK/Qt only)."""
    if sys.platform != "win32" or not ico:
        return
    import ctypes
    user32 = ctypes.windll.user32
    user32.LoadImageW.restype = ctypes.c_void_p
    user32.FindWindowW.restype = ctypes.c_void_p
    user32.SendMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p]
    hwnd = user32.FindWindowW(None, title)
    if not hwnd:
        return
    for which, size in ((0, 16), (1, 32)):            # ICON_SMALL, ICON_BIG
        h = user32.LoadImageW(None, str(ico), 1, size, size, 0x0010)   # IMAGE_ICON, LR_LOADFROMFILE
        if h:
            user32.SendMessageW(hwnd, 0x0080, ctypes.c_void_p(which), ctypes.c_void_p(h))   # WM_SETICON


def smoke_test(window, api):
    """--smoke: wait for the page, open every sidebar page, report and exit."""
    import time
    js = window.evaluate_js
    ok, detail = False, "page never became ready"
    try:
        for _ in range(300):
            if js("!!window.__rlweb_ready"):
                break
            time.sleep(0.1)
        if js("!!window.__rlweb_ready"):
            opened = []
            for p in PAGES:
                js(f"document.querySelector('#side a[data-p={p}]').click()")
                time.sleep(0.6)
                if js(f"document.getElementById('p-{p}').classList.contains('on')"):
                    opened.append(p)
            classic = js("document.getElementById('p-classic').classList.contains('on')")
            ok = len(opened) == len(PAGES) and not classic
            detail = f"{len(opened)}/{len(PAGES)} pages opened" + ("" if ok else f" (missing: {sorted(set(PAGES) - set(opened))})")
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}"
    print(f"SMOKE {'OK' if ok else 'FAIL'} - {detail}", flush=True)
    window.smoke_ok = ok
    api.shutdown()
    window.destroy()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--classic" in argv:
        return run_classic()
    smoke = "--smoke" in argv
    if not webview2_installed():
        msg = ("[web] The Microsoft Edge WebView2 runtime is not installed, so the web GUI cannot open.\n"
               f"      Install it from {WEBVIEW2_URL} (Evergreen runtime, free), then start again.")
        if smoke:
            print(msg + "\nSMOKE FAIL - WebView2 runtime missing", flush=True)
            return 2
        return run_classic(msg + "\n      Opening the classic app instead...")

    import webview
    from webgui.api import Api, JsBridge, to_json
    import app_settings

    if sys.platform == "win32":
        try:                                        # own taskbar group + icon instead of python.exe's
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
        except Exception:
            pass

    def emit(events):
        if webview.windows:
            webview.windows[0].evaluate_js(f"window.rlweb && window.rlweb.onEvents({to_json(events)})")

    title = app_settings.effective_branding()["branding"].get("window_title") or "RL Trading Studio"
    ico = window_icon_file()

    def set_title(new_title):
        if webview.windows:
            webview.windows[0].set_title(new_title)
            set_window_icon(new_title, ico)

    api = Api(ROOT, emit=emit, set_title=set_title)
    window = webview.create_window(title, PAGE.as_uri(), js_api=JsBridge(api), width=1500, height=980,
                                   min_size=(1180, 720), background_color="#dfe8f6", text_select=True, hidden=smoke)
    window.smoke_ok = False

    def on_closing():
        if api.busy() and not smoke:
            ok = window.create_confirmation_dialog("A task is running",
                                                   "Training or another task is still in progress.\nStop it and close the app?")
            if not ok:
                return False
        api.shutdown()
        return True

    window.events.closing += on_closing
    window.events.shown += lambda: set_window_icon(title, ico)

    def started():
        api.start_pump()
        if smoke:
            smoke_test(window, api)

    try:
        webview.start(started, debug="--debug" in argv)
    except Exception as exc:                        # e.g. a broken WebView2 install
        msg = f"[web] The web GUI failed to start: {type(exc).__name__}: {exc}"
        if smoke:
            print(msg + "\nSMOKE FAIL", flush=True)
            return 1
        return run_classic(msg + "\n      Opening the classic app instead...")
    if smoke:
        return 0 if window.smoke_ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
