"""Live Visual window for a training run (pywebview / Edge WebView2).

Serves tools/viz (viewer + vendored Three.js) and the run's viz/ folder on a
private 127.0.0.1 port, then opens tools/viz/network3d.html?live=1 in a native
window. The page polls viz/index.json and follows each new snapshot written
by `rl_train.py --viz` (one per evaluation), with a timeline to scrub back.

Usage (repo root):
    python tools/viz/live_viewer.py --run_dir artifacts/models/<model>/runs/<id>   (or the viz/ folder)
    python tools/viz/live_viewer.py --model <model>          # current run of a model
    python tools/viz/live_viewer.py --model <model> --browser   # default browser instead of a window
    python tools/viz/live_viewer.py --html out/<model>_network3d.html  # a capture_snapshot.py page
    python tools/viz/live_viewer.py --run_dir <dir> --serve_only --port 8766    # testing
"""
import argparse
import http.server
import json
import re
import io
import os
import socketserver
import sys
import threading
import urllib.parse
import webbrowser
from pathlib import Path

if __name__ == "__main__" and sys.stdout is not None and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")   # not on import (pytest capture)
VIZ_DIR = Path(__file__).resolve().parent
ROOT = VIZ_DIR.parents[1]


def resolve_viz_dir(run_dir=None, model=None):
    if model:
        sys.path.insert(0, str(ROOT))
        from artifact_paths import model_dir
        run_dir = model_dir(model)
    p = Path(run_dir).resolve()
    # accept the run folder or its viz/ folder. Before the first evaluation only
    # rollout_*.json / monitor.json exist (no index.json), so never descend by that test:
    # a viz dir passed in stayed "waiting for training data" forever (seen 2026-10-18).
    if p.name == "viz":
        return p
    if (p / "viz").is_dir():
        return p / "viz"
    return p


def find_peers(viz_data, limit=12):
    """Sibling runs for the overlay: models whose name is this model's name with a different
    `_s<seed>` suffix (obs_ablation naming), each at its newest run that has a viz folder.
    Returns [(label, viz_dir)] with this run first."""
    p = Path(viz_data).resolve()
    try:
        run_dir = p.parent if p.name == "viz" else p
        model_home = run_dir.parent.parent if run_dir.parent.name == "runs" else None
    except Exception:
        model_home = None
    if model_home is None or not model_home.is_dir():
        return [(p.parent.parent.name if p.name == "viz" else p.name, p)]
    name = model_home.name
    stem = re.sub(r"_s\d+$", "", name)
    out = [(name, p)]
    for home in sorted(model_home.parent.glob(stem + "_s*")):
        if home == model_home or not re.fullmatch(re.escape(stem) + r"_s\d+", home.name):
            continue
        runs = sorted((r for r in (home / "runs").glob("*/viz") if (r / "index.json").exists() or (r / "monitor.json").exists()),
                      key=lambda r: r.stat().st_mtime)
        if runs:
            out.append((home.name, runs[-1].resolve()))
        if len(out) >= limit:
            break
    return out


def make_handler(viz_data):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?", 1)[0] == "/peers.json":
                peers = find_peers(viz_data)
                body = json.dumps({"peers": [{"k": i, "name": n} for i, (n, _) in enumerate(peers)]}).encode("utf-8")
                self.send_response(200); self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
                return
            super().do_GET()

        def translate_path(self, path):
            path = urllib.parse.unquote(path.split("?", 1)[0].split("#", 1)[0])
            if path.startswith("/run/"):
                base, rel = viz_data, path[len("/run/"):]
            elif path.startswith("/peer/"):
                parts = path[len("/peer/"):].split("/", 1)
                peers = find_peers(viz_data)
                try:
                    base = peers[int(parts[0])][1]
                except (ValueError, IndexError):
                    return str(VIZ_DIR / "__forbidden__")
                rel = parts[1] if len(parts) > 1 else ""
            else:
                base, rel = VIZ_DIR, path.lstrip("/")
            target = (base / rel).resolve()
            if target != base.resolve() and base.resolve() not in target.parents:
                return str(base / "__forbidden__")          # no path traversal out of the two roots
            return str(target)

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def log_message(self, *args):
            pass

    Handler.extensions_map = dict(http.server.SimpleHTTPRequestHandler.extensions_map,
                                  **{".js": "text/javascript", ".json": "application/json"})
    return Handler


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser(description="Open the live 3D training visual")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run_dir", help="run folder (artifacts/models/<m>/runs/<id>) or its viz/ folder")
    src.add_argument("--model", help="model name: open its current run")
    src.add_argument("--html", help="static page under tools/viz (e.g. out/<model>_network3d.html)")
    ap.add_argument("--browser", action="store_true", help="open in the default browser instead of a window")
    ap.add_argument("--serve_only", action="store_true", help="only serve (for testing)")
    ap.add_argument("--port", type=int, default=0, help="fixed port (default: random free port)")
    ap.add_argument("--mode", choices=["lite", "cinematic", "dashboard", "monitor"], default=None,
                    help="display mode (default: lite for live runs, cinematic for --html pages)")
    ap.add_argument("--fps", type=int, default=None, help="frame cap override (default: the mode's own)")
    ap.add_argument("--query", default="", help="extra viewer options, e.g. 'bloom=0&k=4&pulses=0&dpr=1'")
    args = ap.parse_args()

    if args.html:
        page = (VIZ_DIR / args.html).resolve()
        if VIZ_DIR not in page.parents or not page.is_file():
            sys.exit(f"--html must be an existing file under {VIZ_DIR}: {args.html}")
        viz_data = VIZ_DIR / "out"
        path = page.relative_to(VIZ_DIR).as_posix()
        name, mode = page.stem.replace("_network3d", ""), "snapshot"
    else:
        viz_data = resolve_viz_dir(args.run_dir, args.model)
        path = "network3d.html?live=1"
        name = viz_data.parent.parent.parent.name if viz_data.parent.parent.name == "runs" else viz_data.parent.name
        mode = "live"
    viz_data.mkdir(parents=True, exist_ok=True)       # the window can open before the first snapshot
    server = Server(("127.0.0.1", args.port), make_handler(viz_data))
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}/{path}"
    extra = [f"mode={args.mode}" if args.mode else "", f"fps={args.fps}" if args.fps else ""]
    for q in extra:
        if q:
            url += ("&" if "?" in url else "?") + q
    if args.query:
        url += ("&" if "?" in url else "?") + args.query.lstrip("?&")
    print(f"[viz] serving {viz_data} at {url}", flush=True)

    try:
        if args.serve_only:
            threading.Event().wait()
        if not args.browser:
            try:
                import webview
            except ImportError:
                print("[viz] pywebview not installed — opening the default browser instead", flush=True)
                args.browser = True
        if args.browser:
            webbrowser.open(url)
            threading.Event().wait()
        else:
            win = webview.create_window(f"Neural Trader · {name} · {mode}", url, width=1600, height=920,
                                        background_color="#02030A", min_size=(900, 560))

            def host_hidden(flag):
                # WebView2 keeps animating when its window is minimised — tell the page
                try:
                    win.evaluate_js(f"window.__viz && window.__viz.setHidden({'true' if flag else 'false'})")
                except Exception:
                    pass
            try:
                win.events.minimized += lambda *_: host_hidden(True)
                win.events.restored += lambda *_: host_hidden(False)
                win.events.maximized += lambda *_: host_hidden(False)
            except Exception:
                pass
            webview.start()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
