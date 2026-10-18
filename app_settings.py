"""Settings storage shared by rl_app.py and the web GUI.

- branding_config.json : {"branding": {...text fields...}, "colors": {KEY: "#hex"}} (gitignored)
- api_keys.json        : {"gemini": "..."} (gitignored; never commit, never log)
- web_settings.json    : web-GUI-only preferences, e.g. {"theme": "clear"} (gitignored)

COLOR_PRESETS moved here verbatim from rl_app.py.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
BRANDING_CONFIG_FILE = ROOT / "branding_config.json"
API_KEYS_FILE = ROOT / "api_keys.json"
WEB_SETTINGS_FILE = ROOT / "web_settings.json"

COLOR_PRESETS = {
    "Default (Blue)": {
        "ACCENT":   "#2f81f7",
        "GREEN":    "#3fb950",
        "RED":      "#f85149",
        "YELLOW":   "#d29922",
        "PURPLE":   "#a371f7",
        "DIM":      "#8b949e",
        "BG_CARD":  "#1c2128",
        "BG_INPUT": "#22272e",
        "SIDEBAR_BG": "#161b22",
    },
    "Warm (Orange/Red)": {
        "ACCENT":   "#fb923c",
        "GREEN":    "#84cc16",
        "RED":      "#dc2626",
        "YELLOW":   "#facc15",
        "PURPLE":   "#ec4899",
        "DIM":      "#a8a29e",
        "BG_CARD":  "#1c1917",
        "BG_INPUT": "#292524",
        "SIDEBAR_BG": "#171513",
    },
    "Ocean (Cyan/Teal)": {
        "ACCENT":   "#06b6d4",
        "GREEN":    "#10b981",
        "RED":      "#ef4444",
        "YELLOW":   "#eab308",
        "PURPLE":   "#8b5cf6",
        "DIM":      "#94a3b8",
        "BG_CARD":  "#0f172a",
        "BG_INPUT": "#1e293b",
        "SIDEBAR_BG": "#0a1428",
    },
    "Pure Black (Minimal)": {
        "ACCENT":   "#ffffff",
        "GREEN":    "#22c55e",
        "RED":      "#dc2626",
        "YELLOW":   "#eab308",
        "PURPLE":   "#737373",
        "DIM":      "#525252",
        "BG_CARD":  "#000000",
        "BG_INPUT": "#171717",
        "SIDEBAR_BG": "#000000",
    },
    "Metafxclub (Brand)": {
        "ACCENT":   "#00d4ff",
        "GREEN":    "#3fb950",
        "RED":      "#f85149",
        "YELLOW":   "#fbbf24",
        "PURPLE":   "#c084fc",
        "DIM":      "#94a3b8",
        "BG_CARD":  "#0a1628",
        "BG_INPUT": "#0f1f37",
        "SIDEBAR_BG": "#050d18",
    },
}


# Text fields on the Settings page: (key, label, hint) — same order as the CTk page.
BRANDING_TEXT_FIELDS = [
    ("window_title", "Window Title", "ชื่อหน้าต่าง app"),
    ("logo_text", "Logo Text", "ชื่อใน sidebar"),
    ("logo_emoji", "Logo Emoji (ถ้าไม่มี image)", "เช่น 🤖 ⚡ 🚀"),
    ("subtitle", "Subtitle", "ข้อความใต้ logo"),
    ("logo_image", "Logo Image Path", "เช่น assets/logo.png"),
    ("window_icon", "Window Icon Path", "เช่น assets/icon.ico"),
    ("theme_label", "Theme Toggle Label", "ข้อความปุ่มล่าง sidebar"),
]
# Defaults of those text fields (rl_app.BRANDING; a test keeps them in sync).
BRANDING_TEXT_DEFAULTS = {
    "window_title": "Metafxclub RL Studio",
    "logo_text": "Metafxclub RL",
    "logo_emoji": "🤖",
    "subtitle": "v1.0 · Trading AI",
    "logo_image": "assets/metafx.png",
    "window_icon": "",
    "theme_label": "Dark Mode",
}
COLOR_FIELDS = [
    ("ACCENT", "Primary", "สี brand หลัก, ปุ่ม"),
    ("GREEN", "Success", "metric ดี, win trade"),
    ("RED", "Error", "metric แย่, loss"),
    ("YELLOW", "Warning", "metric เตือน"),
    ("PURPLE", "Special", "Export/Chart buttons"),
    ("DIM", "Secondary", "text รอง, hint"),
    ("BG_CARD", "Card BG", "พื้นหลังการ์ด"),
    ("BG_INPUT", "Input BG", "พื้นหลัง input"),
    ("SIDEBAR_BG", "Sidebar BG", "พื้นหลัง sidebar"),
]
DEFAULT_COLORS = dict(COLOR_PRESETS["Default (Blue)"])

WEB_THEMES = {"clear": "Clear Liquid (light glass)", "deep": "Deep Glass (dark)"}
WEB_DEFAULTS = {"theme": "clear", "brand_accent": False}

_HEX = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def is_hex_color(value) -> bool:
    return isinstance(value, str) and bool(_HEX.match(value.strip()))


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_api_keys(path: Path = None) -> dict:
    return _read_json(path or API_KEYS_FILE)


def save_api_keys(keys: dict, path: Path = None) -> None:
    try:
        (path or API_KEYS_FILE).write_text(json.dumps(keys, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[api_keys] save failed: {e}")


def load_branding_config(path: Path = None) -> dict:
    return _read_json(path or BRANDING_CONFIG_FILE)


def save_branding_config(branding_overrides: dict, colors: dict, path: Path = None) -> None:
    cfg = {"branding": branding_overrides, "colors": colors}
    (path or BRANDING_CONFIG_FILE).write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def effective_branding(path: Path = None) -> dict:
    """{"branding": text fields with saved overrides, "colors": colours with saved overrides}."""
    cfg = load_branding_config(path)
    branding = dict(BRANDING_TEXT_DEFAULTS)
    branding.update({k: str(v) for k, v in (cfg.get("branding") or {}).items() if k in BRANDING_TEXT_DEFAULTS})
    colors = dict(DEFAULT_COLORS)
    colors.update({k: v for k, v in (cfg.get("colors") or {}).items() if k in DEFAULT_COLORS and is_hex_color(v)})
    return {"branding": branding, "colors": colors, "saved": bool(cfg)}


def load_web_settings(path: Path = None) -> dict:
    data = dict(WEB_DEFAULTS)
    raw = _read_json(path or WEB_SETTINGS_FILE)
    if raw.get("theme") in WEB_THEMES:
        data["theme"] = raw["theme"]
    if isinstance(raw.get("brand_accent"), bool):
        data["brand_accent"] = raw["brand_accent"]
    return data


def save_web_settings(values: dict, path: Path = None) -> dict:
    data = load_web_settings(path)
    if values.get("theme") in WEB_THEMES:
        data["theme"] = values["theme"]
    if "brand_accent" in values:
        data["brand_accent"] = bool(values["brand_accent"])
    (path or WEB_SETTINGS_FILE).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data
