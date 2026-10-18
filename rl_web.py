"""RL Trading Studio — web GUI (pywebview). Native app window, no browser.

    .venv/Scripts/python.exe rl_web.py          (or run_rl_web.bat)

Migration in progress (docs/web_gui_migration_plan_2026-10-18.md): the Train, Models and
Settings pages live here; the other pages still open the classic app (rl_app.py). Backend: webgui/api.py.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
if __name__ == "__main__" and sys.stdout is not None and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import webview  # noqa: E402

from webgui.api import Api, JsBridge, to_json  # noqa: E402

PAGE = ROOT / "webgui" / "ui" / "index.html"


def main():
    def emit(events):
        if webview.windows:
            webview.windows[0].evaluate_js(f"window.rlweb && window.rlweb.onEvents({to_json(events)})")

    def set_title(title):
        if webview.windows:
            webview.windows[0].set_title(title)

    import app_settings
    title = app_settings.effective_branding()["branding"].get("window_title") or "RL Trading Studio"
    api = Api(ROOT, emit=emit, set_title=set_title)
    window = webview.create_window(title, PAGE.as_uri(), js_api=JsBridge(api), width=1500, height=980,
                                   min_size=(1180, 720), background_color="#dfe8f6", text_select=True)

    def on_closing():
        if api.busy():
            ok = window.create_confirmation_dialog("Training is running",
                                                   "A training run is still in progress.\nStop it and close the app?")
            if not ok:
                return False
        api.shutdown()
        return True

    window.events.closing += on_closing
    webview.start(api.start_pump, debug="--debug" in sys.argv)


if __name__ == "__main__":
    main()
