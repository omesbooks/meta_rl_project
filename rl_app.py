"""
RL Trading Studio — Modern GUI for RL Trading Pipeline
========================================================
GUI app ที่รวมทุกอย่าง: Train, Backtest, Walk-Forward, Fine-tune

ติดตั้ง:
    pip install customtkinter stable-baselines3 pandas numpy matplotlib

รัน:
    python rl_app.py
"""
import os, sys, io, subprocess, threading, queue, time, json, glob, re
from pathlib import Path
from datetime import datetime

import customtkinter as ctk
from tkinter import filedialog, messagebox, ttk, colorchooser
import tkinter as tk

# Force UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ============================================================
# THEME / GLOBAL CONFIG
# ============================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Colors
COLOR_ACCENT  = "#2f81f7"
COLOR_GREEN   = "#3fb950"
COLOR_RED     = "#f85149"
COLOR_YELLOW  = "#d29922"
COLOR_PURPLE  = "#a371f7"
COLOR_DIM     = "#8b949e"
COLOR_BG_CARD = "#1c2128"
COLOR_BG_INPUT = "#22272e"
COLOR_BG_APP = "#0d1117"
COLOR_BG_PANEL = "#161b22"
COLOR_BG_TERMINAL = "#070b10"
COLOR_BORDER = "#30363d"
COLOR_HOVER = "#2d333b"
COLOR_SELECTED = "#1f3a5f"
COLOR_TEXT = "#f0f6fc"

WORK_DIR = Path(__file__).parent.resolve()

# ============================================================
# 🎨 BRANDING — แก้ได้ตามต้องการ
# ============================================================
BRANDING = {
    # Window title
    "window_title":   "Metafxclub RL Studio",

    # Window icon (taskbar/title bar) — PNG/ICO path, empty = system default
    "window_icon":    "",                  # e.g. "assets/icon.ico"

    # Sidebar logo IMAGE (PNG/JPG/GIF) — recommended for branding
    # Empty string = ใช้ emoji แทน (ดู logo_emoji)
    # Path: relative to project root OR absolute path
    "logo_image":      "assets/metafx.png",                  # e.g. "assets/logo.png"
    "logo_image_size": (32, 32),            # (width, height) in pixels

    # Sidebar logo TEXT (ใช้คู่กับ image, หรือ emoji ถ้าไม่มี image)
    "logo_emoji":     "🤖",                 # emoji prefix (ใช้ถ้าไม่มี image)
    "logo_text":      "Metafxclub RL",      # ชื่อใน sidebar
    "logo_color":     COLOR_ACCENT,         # สีตัวอักษร logo
    "logo_size":      20,                   # ขนาดฟอนต์ logo

    # Subtitle (under logo)
    "subtitle":       "v1.0 · Trading AI",
    "subtitle_color": COLOR_DIM,
    "subtitle_size":  11,

    # Sidebar background color
    "sidebar_bg":     "#161b22",

    # Theme toggle label (bottom of sidebar)
    "theme_label":    "Dark Mode",
}

# ============================================================
# 📋 NAVIGATION — แก้ icon / label ของแต่ละหน้าได้
# ============================================================
NAV_ITEMS = [
    # (key,        icon,    sidebar label,           top-bar title)
    ("tools",     "🛠️",    "Data Tools",            "🛠️ Data Tools"),
    ("pipeline",  "▶",      "Pipeline",              "▶ Full Pipeline"),
    ("train",     "🎯",    "Train",                 "🎯 Train New Model"),
    ("backtest",  "📊",    "Backtest",              "📊 Backtest Model"),
    ("walkfwd",   "🔬",    "Walk-Forward",          "🔬 Walk-Forward Validation"),
    ("finetune",  "🔄",    "Fine-tune",             "🔄 Fine-tune Existing Model"),
    ("analyze",   "🔍",    "Analyze",               "🔍 Confidence Analysis"),
    ("models",    "🗂️",    "Models",                "🗂️ Model Library"),
    ("settings",  "⚙️",    "Settings",              "⚙️ Settings"),
]

# ============================================================
# 🎨 COLOR PRESETS — เลือกธีมสีสำเร็จได้ใน Settings page
# ============================================================
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

# ============================================================
# 💾 Branding persistence — save/load to branding_config.json
# ============================================================
BRANDING_CONFIG_FILE = WORK_DIR / "branding_config.json"

def _load_branding_config():
    """Load saved branding from JSON file (if exists) and override BRANDING + colors"""
    global BRANDING
    global COLOR_ACCENT, COLOR_GREEN, COLOR_RED, COLOR_YELLOW
    global COLOR_PURPLE, COLOR_DIM, COLOR_BG_CARD, COLOR_BG_INPUT
    if not BRANDING_CONFIG_FILE.exists():
        return
    try:
        import json
        cfg = json.loads(BRANDING_CONFIG_FILE.read_text(encoding="utf-8"))
        if "branding" in cfg:
            BRANDING.update(cfg["branding"])
        if "colors" in cfg:
            c = cfg["colors"]
            COLOR_ACCENT  = c.get("ACCENT",   COLOR_ACCENT)
            COLOR_GREEN   = c.get("GREEN",    COLOR_GREEN)
            COLOR_RED     = c.get("RED",      COLOR_RED)
            COLOR_YELLOW  = c.get("YELLOW",   COLOR_YELLOW)
            COLOR_PURPLE  = c.get("PURPLE",   COLOR_PURPLE)
            COLOR_DIM     = c.get("DIM",      COLOR_DIM)
            COLOR_BG_CARD = c.get("BG_CARD",  COLOR_BG_CARD)
            COLOR_BG_INPUT = c.get("BG_INPUT", COLOR_BG_INPUT)
            # Update sidebar_bg in BRANDING too if present
            if "SIDEBAR_BG" in c:
                BRANDING["sidebar_bg"] = c["SIDEBAR_BG"]
            # Update logo_color if it was COLOR_ACCENT (most common case)
            BRANDING["logo_color"] = c.get("ACCENT", BRANDING.get("logo_color"))
            BRANDING["subtitle_color"] = c.get("DIM", BRANDING.get("subtitle_color"))
        print(f"[branding] Loaded from {BRANDING_CONFIG_FILE.name}")
    except Exception as e:
        print(f"[branding] Failed to load config: {e}")

def _save_branding_config(branding_overrides: dict, colors: dict):
    """Save current branding + colors to JSON file"""
    import json
    cfg = {
        "branding": branding_overrides,
        "colors":   colors,
    }
    BRANDING_CONFIG_FILE.write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

# Apply saved config on import (before classes/widgets created)
_load_branding_config()


# ============================================================
# Metric Health Thresholds + Recommendations
# ============================================================
METRIC_INFO = {
    'approx_kl': {
        'good': (0.0, 0.05),
        'warn': (0.05, 0.15),
        'bad':  (0.15, float('inf')),
        'icon': '📐',
        'label': 'KL Divergence',
        'desc': 'Policy stability',
        'fix_high': lambda v: {
            'title': f'approx_kl = {v:.3f}',
            'severity': 'high' if v > 0.5 else 'med',
            'problem': 'Policy unstable — เปลี่ยนเร็วเกิน',
            'actions': [
                ('↓', 'learning_rate', '3e-4 → 1e-4', 'main'),
                ('↓', 'clip_range', '0.2 → 0.1', None),
                ('↓', 'window_size', 'current → 10', None),
            ],
        },
        'fix_low': lambda v: None,
    },
    'clip_fraction': {
        'good': (0.0, 0.2),
        'warn': (0.2, 0.3),
        'bad':  (0.3, 1.0),
        'icon': '✂️',
        'label': 'Clip Fraction',
        'desc': 'How often clipping triggers',
        'fix_high': lambda v: {
            'title': f'clip_fraction = {v:.3f}',
            'severity': 'high' if v > 0.5 else 'med',
            'problem': 'Policy พยายามกระโดด → โดน clip บ่อย',
            'actions': [
                ('↓', 'learning_rate', '3e-4 → 1e-4', 'main'),
                ('↑', 'n_steps', '2048 → 4096', None),
                ('⚠️', 'อย่าลด clip_range', 'จะแย่ลง!', 'warn'),
            ],
        },
        'fix_low': lambda v: None,
    },
    'explained_variance': {
        'good': (0.5, 1.0),
        'warn': (0.2, 0.5),
        'bad':  (-float('inf'), 0.2),
        'icon': '🎯',
        'label': 'Explained Variance',
        'desc': 'Critic accuracy',
        'fix_high': lambda v: None,
        'fix_low': lambda v: {
            'title': f'explained_variance = {v:.3f}',
            'severity': 'high' if v < 0 else 'med',
            'problem': 'Critic ทำนาย value แย่',
            'actions': [
                ('↑', 'Critic NN size', 'vf=[256,128,64]', 'main'),
                ('↑', 'gae_lambda', '0.95 → 0.97', None),
                ('↑', 'Training steps', '200k → 500k', None),
            ],
        },
    },
    'entropy_loss': {
        'good': (-2.0, -0.3),
        'warn': (-0.3, -0.1),
        'bad':  (-0.1, float('inf')),
        'icon': '🔍',
        'label': 'Entropy',
        'desc': 'Exploration level',
        'fix_high': lambda v: {
            'title': f'entropy_loss = {v:.3f}',
            'severity': 'med',
            'problem': 'Exploration หาย → policy collapse เร็ว',
            'actions': [
                ('↑', 'ent_coef', '0.01 → 0.05', 'main'),
                ('↓', 'learning_rate', '3e-4 → 1e-4', None),
                ('🔄', 'Restart', 'with different seed', None),
            ],
        },
        'fix_low': lambda v: {
            'title': f'entropy_loss = {v:.3f}',
            'severity': 'low',
            'problem': 'Exploration เยอะเกิน → ไม่ converge',
            'actions': [
                ('↓', 'ent_coef', '0.01 → 0.005', 'main'),
            ],
        },
    },
    'value_loss': {
        'good': (0.0, 0.01),
        'warn': (0.01, 0.1),
        'bad':  (0.1, float('inf')),
        'icon': '📉',
        'label': 'Value Loss',
        'desc': 'Critic training loss',
        'fix_high': lambda v: {
            'title': f'value_loss = {v:.4f}',
            'severity': 'high' if v > 1 else 'med',
            'problem': 'Critic เรียนไม่ทัน',
            'actions': [
                ('⚙️', 'Reward normalization', 'clip [-1, 1]', 'main'),
                ('↑', 'Critic NN size', 'vf=[256, 128]', None),
            ],
        },
        'fix_low': lambda v: None,
    },
    'ep_rew_mean': {
        'good': (0.5, float('inf')),
        'warn': (-0.1, 0.5),
        'bad':  (-float('inf'), -0.1),
        'icon': '💰',
        'label': 'Reward',
        'desc': 'Average episode reward',
        'fix_high': lambda v: None,
        'fix_low': lambda v: {
            'title': f'ep_rew_mean = {v:.2f}',
            'severity': 'high' if v < -0.5 else 'med',
            'problem': 'Agent กำลังขาดทุน',
            'actions': [
                ('⚙️', 'Reward function', 'ตรวจ logic', 'main'),
                ('↑', 'ent_coef', '0.01 → 0.05', None),
                ('⚙️', 'Reward shaping', 'penalty/bonus', None),
            ],
        },
    },
}


def classify_metric(name: str, value: float) -> str:
    """Returns 'good' | 'warn' | 'bad' | None"""
    info = METRIC_INFO.get(name)
    if not info:
        return None
    g_lo, g_hi = info['good']
    w_lo, w_hi = info['warn']
    if g_lo <= value <= g_hi:
        return 'good'
    elif w_lo <= value <= w_hi:
        return 'warn'
    else:
        return 'bad'


def get_recommendation(name: str, value: float):
    """Returns recommendation dict {title, severity, problem, actions} or None"""
    info = METRIC_INFO.get(name)
    if not info:
        return None
    status = classify_metric(name, value)
    if status == 'good':
        return None
    g_lo, g_hi = info['good']
    rec = None
    if value > g_hi and info.get('fix_high'):
        rec = info['fix_high'](value)
    elif value < g_lo and info.get('fix_low'):
        rec = info['fix_low'](value)
    if rec:
        rec['metric'] = name
        rec['icon'] = info.get('icon', '⚠️')
        rec['label'] = info.get('label', name)
    return rec


# ============================================================
# Helper: Subprocess runner with live output
# ============================================================
class ProcessRunner:
    """Run subprocess + stream output to callback"""

    def __init__(self):
        self.proc = None
        self.thread = None
        self.q = queue.Queue()

    def start(self, cmd, on_line=None, on_done=None):
        if self.proc is not None:
            return False

        def worker():
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace',
                cwd=str(WORK_DIR), bufsize=1, env=env,
            )
            for line in iter(self.proc.stdout.readline, ''):
                line = line.rstrip()
                if line:
                    self.q.put(('line', line))
            self.proc.wait()
            rc = self.proc.returncode
            self.proc = None
            self.q.put(('done', rc))

        self.thread = threading.Thread(target=worker, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        if self.proc:
            try:
                self.proc.terminate()
                time.sleep(0.5)
                if self.proc and self.proc.poll() is None:
                    self.proc.kill()
            except Exception:
                pass

    def is_running(self):
        return self.proc is not None


# ============================================================
# Reusable Components
# ============================================================
class Card(ctk.CTkFrame):
    """Styled card container"""
    def __init__(self, master, title=None, **kwargs):
        super().__init__(master, fg_color=COLOR_BG_CARD, corner_radius=8,
                          border_width=1, border_color=COLOR_BORDER, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        if title:
            self.title_label = ctk.CTkLabel(self, text=title,
                font=ctk.CTkFont(size=15, weight="bold"),
                text_color=COLOR_TEXT)
            self.title_label.grid(row=0, column=0, sticky="w", padx=18, pady=(16, 6))


class StatCard(ctk.CTkFrame):
    """Metric stat card"""
    def __init__(self, master, label, value, change="", color=None, **kwargs):
        super().__init__(master, fg_color=COLOR_BG_INPUT, corner_radius=8,
                          border_width=1, border_color=COLOR_BORDER, **kwargs)
        ctk.CTkLabel(self, text=label.upper(), text_color=COLOR_DIM,
                      font=ctk.CTkFont(size=10, weight="bold")
                      ).pack(anchor="w", padx=14, pady=(10, 0))
        ctk.CTkLabel(self, text=value, text_color=color or "white",
                      font=ctk.CTkFont(size=22, weight="bold", family="Consolas")
                      ).pack(anchor="w", padx=14, pady=(2, 0))
        if change:
            ctk.CTkLabel(self, text=change, text_color=COLOR_DIM,
                          font=ctk.CTkFont(size=11)
                          ).pack(anchor="w", padx=14, pady=(0, 10))
        else:
            ctk.CTkLabel(self, text="").pack(padx=14, pady=(0, 6))


class ScrollableOptionMenu(ctk.CTkFrame):
    """Option menu replacement for long file/model lists with mouse-wheel scroll."""
    _open_widget = None

    def __init__(self, master, values=None, command=None, width=180, height=34,
                 max_items=14, max_render=80, fg_color=None, button_color=None,
                 button_hover_color=None, text_color=None, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._values = list(values or [])
        self._command = command
        self._width = width
        self._height = height
        self._max_items = max_items
        self._max_render = max_render
        self._fg = fg_color or COLOR_BG_INPUT
        self._button = button_color or self._fg
        self._hover = button_hover_color or COLOR_HOVER
        self._text_color = text_color or COLOR_TEXT
        self._popup = None
        self._popup_scroll = None
        self._search_var = None
        self._popup_value_count = 0
        self._value = self._values[0] if self._values else ""

        self.grid_columnconfigure(0, weight=1)
        self._display = ctk.CTkButton(
            self, text=self._value, anchor="w",
            fg_color=self._fg, hover_color=self._hover,
            text_color=self._text_color, corner_radius=8,
            height=height, command=self._toggle)
        self._display.grid(row=0, column=0, sticky="ew")

        self._arrow = ctk.CTkButton(
            self, text="⌄", width=36, height=height,
            fg_color=self._button, hover_color=self._hover,
            text_color=self._text_color, corner_radius=8,
            command=self._toggle)
        self._arrow.grid(row=0, column=1, sticky="e")

        if width:
            self.configure(width=width)

    def get(self):
        return self._value

    def set(self, value):
        self._value = str(value)
        self._display.configure(text=self._value)

    def configure(self, **kwargs):
        if "values" in kwargs:
            self._values = list(kwargs.pop("values") or [])
            if self._value in ("", "(none)") and self._values:
                self.set(self._values[0])
        if "command" in kwargs:
            self._command = kwargs.pop("command")
        if "fg_color" in kwargs:
            self._fg = kwargs.pop("fg_color")
            self._display.configure(fg_color=self._fg)
        if "button_color" in kwargs:
            self._button = kwargs.pop("button_color")
            self._arrow.configure(fg_color=self._button)
        if "button_hover_color" in kwargs:
            self._hover = kwargs.pop("button_hover_color")
            self._display.configure(hover_color=self._hover)
            self._arrow.configure(hover_color=self._hover)
        if "text_color" in kwargs:
            self._text_color = kwargs.pop("text_color")
            self._display.configure(text_color=self._text_color)
            self._arrow.configure(text_color=self._text_color)
        if "width" in kwargs:
            self._width = kwargs["width"]
        if "height" in kwargs:
            self._height = kwargs.pop("height")
            self._display.configure(height=self._height)
            self._arrow.configure(height=self._height)
        if "max_render" in kwargs:
            self._max_render = int(kwargs.pop("max_render"))
        super().configure(**kwargs)

    config = configure

    def _toggle(self):
        if self._popup is not None and self._popup.winfo_exists():
            self._close_popup()
        else:
            self._open_popup()

    def _scroll_units_for_count(self, count):
        if count <= self._max_items:
            rows_per_notch = 3
        else:
            rows_per_notch = max(5, min(18, round(count / 10)))
        return rows_per_notch * 24

    def _open_popup(self):
        if ScrollableOptionMenu._open_widget and ScrollableOptionMenu._open_widget is not self:
            ScrollableOptionMenu._open_widget._close_popup()
        ScrollableOptionMenu._open_widget = self

        values = self._values or ["(none)"]
        row_h = 32
        popup_w = max(self.winfo_width(), self._width or 180, 220)
        search_h = 38 if len(values) > self._max_items else 0
        popup_h = min(len(values), self._max_items) * row_h + search_h + 10

        self._popup = ctk.CTkToplevel(self)
        self._popup.overrideredirect(True)
        self._popup.transient(self.winfo_toplevel())
        self._popup.attributes("-topmost", True)
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height() + 2
        self._popup.geometry(f"{popup_w}x{popup_h}+{x}+{y}")

        shell = ctk.CTkFrame(
            self._popup, fg_color=COLOR_BG_INPUT, corner_radius=8,
            border_width=1, border_color=COLOR_BORDER)
        shell.pack(fill="both", expand=True)

        self._search_var = tk.StringVar(value="")
        if search_h:
            search = ctk.CTkEntry(
                shell, textvariable=self._search_var,
                placeholder_text="Search...",
                fg_color=COLOR_BG_TERMINAL,
                border_color=COLOR_BORDER,
                height=30)
            search.pack(fill="x", padx=6, pady=(6, 2))
            search.bind("<KeyRelease>", lambda _e: self._render_popup_values())

        self._popup_scroll = ctk.CTkScrollableFrame(
            shell, fg_color=COLOR_BG_INPUT, corner_radius=6,
            scrollbar_button_color=COLOR_HOVER,
            scrollbar_button_hover_color=COLOR_ACCENT)
        self._popup_scroll.pack(fill="both", expand=True, padx=2, pady=2)
        self._popup_scroll.grid_columnconfigure(0, weight=1)

        def on_wheel(event):
            canvas = getattr(self._popup_scroll, "_parent_canvas", None)
            if canvas is not None:
                delta = getattr(event, "delta", 0)
                if delta:
                    direction = -1 if delta > 0 else 1
                    notches = max(1, int(abs(delta) / 120))
                else:
                    direction = -1 if getattr(event, "num", None) == 4 else 1
                    notches = 1
                units = self._scroll_units_for_count(self._popup_value_count) * notches
                canvas.yview_scroll(direction * units, "units")
            return "break"

        self._popup.bind("<Escape>", lambda _e: self._close_popup())
        self._popup.bind("<MouseWheel>", on_wheel)
        self._popup.bind("<Button-4>", on_wheel)
        self._popup.bind("<Button-5>", on_wheel)
        shell.bind("<MouseWheel>", on_wheel)
        shell.bind("<Button-4>", on_wheel)
        shell.bind("<Button-5>", on_wheel)
        self._popup_scroll.bind("<MouseWheel>", on_wheel)
        self._popup_scroll.bind("<Button-4>", on_wheel)
        self._popup_scroll.bind("<Button-5>", on_wheel)
        self._popup_on_wheel = on_wheel
        self._render_popup_values()

        if search_h:
            search.focus_set()

    def _render_popup_values(self):
        if self._popup_scroll is None:
            return
        for child in self._popup_scroll.winfo_children():
            child.destroy()

        query = ""
        if self._search_var is not None:
            query = self._search_var.get().strip().lower()

        values = self._values or ["(none)"]
        if query:
            values = [v for v in values if query in str(v).lower()]
        self._popup_value_count = len(values)
        shown = values[:self._max_render]

        if not shown:
            empty = ctk.CTkLabel(
                self._popup_scroll, text="No matches",
                text_color=COLOR_DIM, height=30)
            empty.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
            if hasattr(self, "_popup_on_wheel"):
                empty.bind("<MouseWheel>", self._popup_on_wheel)
                empty.bind("<Button-4>", self._popup_on_wheel)
                empty.bind("<Button-5>", self._popup_on_wheel)
            return

        for i, value in enumerate(shown):
            selected = str(value) == self._value
            btn = ctk.CTkButton(
                self._popup_scroll, text=str(value), anchor="w", height=30,
                fg_color=COLOR_SELECTED if selected else "transparent",
                hover_color=COLOR_HOVER,
                text_color=COLOR_TEXT if selected else "#d6dde6",
                corner_radius=6,
                command=lambda v=value: self._select(v))
            btn.grid(row=i, column=0, sticky="ew", padx=2, pady=1)
            if hasattr(self, "_popup_on_wheel"):
                btn.bind("<MouseWheel>", self._popup_on_wheel)
                btn.bind("<Button-4>", self._popup_on_wheel)
                btn.bind("<Button-5>", self._popup_on_wheel)

        if len(values) > len(shown):
            note = ctk.CTkLabel(
                self._popup_scroll,
                text=f"Showing {len(shown)} of {len(values)}. Type to filter.",
                text_color=COLOR_DIM,
                font=ctk.CTkFont(size=10),
                height=24)
            note.grid(row=len(shown), column=0, sticky="ew", padx=4, pady=(4, 2))
            if hasattr(self, "_popup_on_wheel"):
                note.bind("<MouseWheel>", self._popup_on_wheel)
                note.bind("<Button-4>", self._popup_on_wheel)
                note.bind("<Button-5>", self._popup_on_wheel)

    def _select(self, value):
        self.set(value)
        self._close_popup()
        if self._command:
            self._command(value)

    def _close_popup(self):
        if self._popup is not None:
            try:
                self._popup.destroy()
            except Exception:
                pass
        self._popup = None
        self._popup_scroll = None
        self._search_var = None
        if ScrollableOptionMenu._open_widget is self:
            ScrollableOptionMenu._open_widget = None

    def destroy(self):
        self._close_popup()
        super().destroy()


# ============================================================
# Main Application
# ============================================================
class RLTradingStudio(ctk.CTk):

    # Build PAGES + PAGE_TITLES from NAV_ITEMS config
    PAGES = [(k, icon, label) for (k, icon, label, _) in NAV_ITEMS]
    PAGE_TITLES = {k: title for (k, _, _, title) in NAV_ITEMS}

    def __init__(self):
        super().__init__()
        self.title(BRANDING["window_title"])
        self.geometry("1280x800")
        self.minsize(1100, 700)
        # Window icon (titlebar/taskbar)
        self._set_window_icon()

        # State
        self.current_page = "train"
        self.runner = ProcessRunner()
        self.nav_buttons = {}
        self.pages = {}
        self._file_cache = {"csvs": None, "models": None, "mtimes": {}}
        self._model_rows_cache = None
        self._model_rows_cache_sig = None
        self._model_tree_sig = None
        self._pipeline_equity_image_sig = None
        self._pipeline_dataset_token = 0
        self._pipeline_row_count_q = queue.Queue()
        self.pipeline_selected_rows = None
        self.pipeline_suggested_steps = None
        self._entry_focus_bound = set()

        self._build_ui()
        self.bind("<Button-1>", self._close_popup_on_root_click, add="+")
        self.show_page("train")
        self._poll_queue()

    # ---------- Branding helpers ----------
    def _set_window_icon(self):
        """Set window icon from BRANDING['window_icon'] if specified"""
        icon_path = BRANDING.get("window_icon", "")
        if not icon_path:
            return
        full_path = Path(icon_path)
        if not full_path.is_absolute():
            full_path = WORK_DIR / icon_path
        if not full_path.exists():
            print(f"[branding] window_icon not found: {full_path}")
            return
        try:
            if full_path.suffix.lower() == ".ico":
                self.iconbitmap(str(full_path))
            else:
                # PNG/JPG → use iconphoto
                from PIL import Image, ImageTk
                img = ImageTk.PhotoImage(Image.open(full_path))
                self.iconphoto(True, img)
                self._icon_ref = img  # keep reference (avoid garbage collection)
        except Exception as e:
            print(f"[branding] Failed to set window icon: {e}")

    def _load_logo_image(self):
        """Load sidebar logo image as CTkImage. Returns None if not configured."""
        path = BRANDING.get("logo_image", "")
        if not path:
            return None
        full_path = Path(path)
        if not full_path.is_absolute():
            full_path = WORK_DIR / path
        if not full_path.exists():
            print(f"[branding] logo_image not found: {full_path}")
            return None
        try:
            from PIL import Image
            size = BRANDING.get("logo_image_size", (32, 32))
            if isinstance(size, int):
                size = (size, size)
            return ctk.CTkImage(
                light_image=Image.open(full_path),
                dark_image=Image.open(full_path),
                size=size,
            )
        except Exception as e:
            print(f"[branding] Failed to load logo image: {e}")
            return None

    # ---------- Lightweight file caches ----------
    def _path_signature(self, pattern):
        """Small signature for files matching a pattern; avoids repeated full rescans."""
        sig = []
        try:
            for path in WORK_DIR.glob(pattern):
                try:
                    stat = path.stat()
                    sig.append((path.name, stat.st_mtime_ns, stat.st_size))
                except OSError:
                    continue
        except Exception:
            return ()
        return tuple(sorted(sig))

    def _best_model_signature(self):
        sig = []
        try:
            for path in WORK_DIR.glob("*_best"):
                best = path / "best_model.zip"
                if not path.is_dir() or not best.exists():
                    continue
                try:
                    stat = best.stat()
                    sig.append((path.name, stat.st_mtime_ns, stat.st_size))
                except OSError:
                    continue
        except Exception:
            return ()
        return tuple(sorted(sig))

    def _list_csv_files(self):
        sig = self._path_signature("*.csv")
        if self._file_cache.get("csv_sig") != sig:
            self._file_cache["csv_sig"] = sig
            self._file_cache["csvs"] = [name for name, _, _ in sig] or ["(none)"]
        return self._file_cache["csvs"]

    def _list_model_names(self):
        zip_sig = self._path_signature("*.zip")
        best_sig = self._best_model_signature()
        sig = (zip_sig, best_sig)
        if self._file_cache.get("model_sig") != sig:
            models = {name[:-4] for name, _, _ in zip_sig if name.lower().endswith(".zip")}
            for name, _, _ in best_sig:
                if name.endswith("_best"):
                    models.add(name[:-5])
            self._file_cache["model_sig"] = sig
            self._file_cache["models"] = sorted(models) or ["(none)"]
        return self._file_cache["models"]

    def _csv_row_count(self, csv_name):
        path = WORK_DIR / csv_name
        if not path.exists() or not path.is_file():
            return None

        try:
            stat = path.stat()
        except OSError:
            return None

        sig = (stat.st_mtime_ns, stat.st_size)
        row_counts = self._file_cache.setdefault("row_counts", {})
        cached = row_counts.get(csv_name)
        if cached and cached.get("sig") == sig:
            return cached.get("rows")

        lines = 0
        last_byte = b""
        try:
            with path.open("rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    lines += chunk.count(b"\n")
                    last_byte = chunk[-1:]
            if stat.st_size > 0 and last_byte not in (b"\n", b"\r"):
                lines += 1
        except OSError:
            return None

        rows = max(lines - 1, 0)
        row_counts[csv_name] = {"sig": sig, "rows": rows}
        return rows

    def _pipeline_effective_rows(self, rows):
        if rows is None:
            return None
        try:
            train_pct = float(self.pipe_train_pct.get().strip() or "1.0")
        except Exception:
            train_pct = 1.0
        try:
            window = int(float(self.pipe_window.get().strip() or "0"))
        except Exception:
            window = 0
        train_pct = min(max(train_pct, 0.01), 1.0)
        return max(int(rows * train_pct) - max(window, 0), 1)

    def _round_steps(self, value):
        step = 50000
        return max(step, int(round(value / step) * step))

    def _suggest_train_steps(self, rows):
        effective = self._pipeline_effective_rows(rows)
        if not effective:
            return None

        if effective < 5000:
            target_passes, min_steps = 30, 100000
        elif effective < 30000:
            target_passes, min_steps = 12, 200000
        elif effective < 100000:
            target_passes, min_steps = 8, 400000
        else:
            target_passes, min_steps = 5, 600000

        suggested = self._round_steps(max(effective * target_passes, min_steps))
        approx_passes = suggested / effective
        return suggested, effective, approx_passes

    def _model_results_signature(self):
        return (
            self._path_signature("*_live_bt_trades.csv"),
            self._path_signature("*_trades.csv"),
            self._path_signature("*.zip"),
            self._path_signature("*_live_bt_equity.png"),
            self._path_signature("*_equity.png"),
            self._path_signature("*_filtered_equity.png"),
            self._path_signature("*_backtest_chart.html"),
        )

    # --------------------------------------------------------
    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main()

    def _build_sidebar(self):
        side = ctk.CTkFrame(self, width=250,
                             fg_color=BRANDING["sidebar_bg"],
                             corner_radius=0)
        side.grid(row=0, column=0, sticky="nsew")
        side.grid_propagate(False)
        side.grid_rowconfigure(99, weight=1)

        # === Logo container ===
        logo_frame = ctk.CTkFrame(side, fg_color="transparent")
        logo_frame.grid(row=0, column=0, sticky="ew", padx=18, pady=(22, 18))

        # Try to load image logo
        logo_image_obj = self._load_logo_image()

        # Inner row: image (optional) + text logo
        logo_row = ctk.CTkFrame(logo_frame, fg_color="transparent")
        logo_row.pack(anchor="w")

        if logo_image_obj is not None:
            # Image present — show image + text (no emoji)
            ctk.CTkLabel(logo_row, image=logo_image_obj, text=""
                          ).pack(side="left", padx=(0, 8))
            display_text = BRANDING["logo_text"]
        else:
            # No image — fall back to emoji + text
            display_text = f"{BRANDING['logo_emoji']} {BRANDING['logo_text']}".strip()

        ctk.CTkLabel(logo_row, text=display_text,
                      font=ctk.CTkFont(size=BRANDING["logo_size"], weight="bold"),
                      text_color=BRANDING["logo_color"]
                      ).pack(side="left")

        # Subtitle
        if BRANDING.get("subtitle"):
            ctk.CTkLabel(logo_frame, text=BRANDING["subtitle"],
                          font=ctk.CTkFont(size=BRANDING["subtitle_size"]),
                          text_color=BRANDING["subtitle_color"]
                          ).pack(anchor="w", pady=(2, 0))

        # Separator
        ctk.CTkFrame(side, height=1, fg_color=COLOR_BORDER
                      ).grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 10))

        # Nav items
        for i, (key, icon, label) in enumerate(self.PAGES):
            btn = ctk.CTkButton(side,
                text=f"  {icon}   {label}",
                anchor="w",
                fg_color="transparent",
                text_color=COLOR_DIM,
                hover_color=COLOR_HOVER,
                corner_radius=8,
                height=38,
                font=ctk.CTkFont(size=14),
                command=lambda k=key: self.show_page(k),
            )
            btn.grid(row=2 + i, column=0, sticky="ew", padx=12, pady=2)
            self.nav_buttons[key] = btn

        # Theme toggle (bottom)
        theme_frame = ctk.CTkFrame(side, fg_color="transparent")
        theme_frame.grid(row=100, column=0, sticky="ew", padx=15, pady=15)

        ctk.CTkFrame(theme_frame, height=1, fg_color=COLOR_BORDER
                      ).pack(fill="x", pady=(0, 10))

        self.theme_switch = ctk.CTkSwitch(theme_frame,
            text=BRANDING["theme_label"],
            command=self._toggle_theme,
            font=ctk.CTkFont(size=12),
            text_color=COLOR_DIM,
        )
        self.theme_switch.select()
        self.theme_switch.pack(anchor="w")

    def _build_main(self):
        # Container for top bar + content
        self.main = ctk.CTkFrame(self, fg_color=COLOR_BG_APP, corner_radius=0)
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_rowconfigure(1, weight=1)

        # Top bar
        topbar = ctk.CTkFrame(self.main, fg_color=COLOR_BG_APP,
                               border_width=0, corner_radius=0, height=76)
        topbar.grid(row=0, column=0, sticky="ew")
        topbar.grid_propagate(False)
        topbar.grid_columnconfigure(0, weight=1)

        self.page_title_label = ctk.CTkLabel(topbar, text="🎯 Train New Model",
            font=ctk.CTkFont(size=23, weight="bold"),
            text_color=COLOR_TEXT)
        self.page_title_label.grid(row=0, column=0, sticky="w", padx=30, pady=22)

        self.status_label = ctk.CTkLabel(topbar, text="● Idle",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_DIM,
            fg_color=COLOR_BG_PANEL,
            corner_radius=14,
            height=28,
            width=96)
        self.status_label.grid(row=0, column=1, sticky="e", padx=30, pady=20)

        # Separator
        ctk.CTkFrame(self.main, height=1, fg_color=COLOR_BORDER
                      ).grid(row=0, column=0, sticky="sew")

        # Content area (scrollable)
        self.content = ctk.CTkScrollableFrame(self.main,
            fg_color=COLOR_BG_APP, corner_radius=0)
        self.content.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        self.content.grid_columnconfigure(0, weight=1)

        # Pages are built on first open. This keeps startup responsive.
        self._page_builders = {
            "tools": self._build_tools_page,
            "pipeline": self._build_pipeline_page,
            "train": self._build_train_page,
            "backtest": self._build_backtest_page,
            "walkfwd": self._build_walkfwd_page,
            "finetune": self._build_finetune_page,
            "analyze": self._build_analyze_page,
            "models": self._build_models_page,
            "settings": self._build_settings_page,
        }

    def _toggle_theme(self):
        if self.theme_switch.get() == 1:
            ctk.set_appearance_mode("dark")
            self.theme_switch.configure(text="Dark Mode")
        else:
            ctk.set_appearance_mode("light")
            self.theme_switch.configure(text="Light Mode")

    # --------------------------------------------------------
    def show_page(self, key):
        self.current_page = key

        if key not in self.pages and key in getattr(self, "_page_builders", {}):
            self._page_builders[key]()

        # Update nav buttons
        for k, btn in self.nav_buttons.items():
            if k == key:
                btn.configure(fg_color=COLOR_SELECTED, text_color=COLOR_TEXT)
            else:
                btn.configure(fg_color="transparent", text_color=COLOR_DIM)

        # Hide all pages
        for k, page in self.pages.items():
            page.grid_remove()

        # Show selected
        if key in self.pages:
            self.pages[key].grid(row=0, column=0, sticky="nsew", padx=30, pady=20)
            self._bind_entry_focus_fix(self.pages[key])

        # Update title
        self.page_title_label.configure(text=self.PAGE_TITLES[key])

        # Refresh dynamic content
        if key == "models":
            self._refresh_models_list()
        elif key == "tools":
            self._refresh_tools_dropdowns()
        elif key == "pipeline":
            self._refresh_dropdowns()
            self._refresh_model_comparison()
        elif key in ("backtest", "walkfwd", "finetune", "analyze"):
            self._refresh_dropdowns()

    def _bind_entry_focus_fix(self, root):
        """Keep CTkEntry responsive inside scrollable pages and custom popups."""
        try:
            children = root.winfo_children()
        except Exception:
            return

        for widget in children:
            if isinstance(widget, ctk.CTkEntry):
                ident = str(widget)
                if ident not in self._entry_focus_bound:
                    self._entry_focus_bound.add(ident)

                    def focus_entry(event, entry=widget):
                        if ScrollableOptionMenu._open_widget is not None:
                            ScrollableOptionMenu._open_widget._close_popup()
                        inner = getattr(entry, "_entry", entry)
                        try:
                            entry.after_idle(inner.focus_force)
                        except Exception:
                            pass

                    widget.bind("<Button-1>", focus_entry, add="+")
                    widget.bind("<ButtonRelease-1>", focus_entry, add="+")
                    inner = getattr(widget, "_entry", None)
                    if inner is not None:
                        inner.bind("<Button-1>", focus_entry, add="+")
                        inner.bind("<ButtonRelease-1>", focus_entry, add="+")

            self._bind_entry_focus_fix(widget)

    def _is_widget_descendant(self, widget, parent):
        while widget is not None:
            if widget is parent:
                return True
            try:
                widget = widget.master
            except Exception:
                return False
        return False

    def _close_popup_on_root_click(self, event):
        open_widget = ScrollableOptionMenu._open_widget
        if open_widget is None:
            return
        if self._is_widget_descendant(event.widget, open_widget):
            return
        open_widget._close_popup()

    def _is_process_busy(self):
        return self.runner.is_running() or getattr(self, "pipeline_running", False)

    # --------------------------------------------------------
    # PAGE: FULL PIPELINE
    # --------------------------------------------------------
    def _build_pipeline_page(self):
        page = ctk.CTkFrame(self.content, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        self.pages["pipeline"] = page

        self.pipeline_running = False
        self.pipeline_stop_requested = False
        self.pipeline_proc = None
        self.pipeline_equity_image = None
        self.pipeline_stage_labels = []

        intro = ctk.CTkLabel(
            page,
            text="Run one flow: dataset -> optional relabel -> train PPO -> backtest -> chart/report.",
            text_color=COLOR_ACCENT,
            font=ctk.CTkFont(size=13),
            wraplength=900,
            justify="left",
        )
        intro.grid(row=0, column=0, sticky="w", padx=8, pady=(0, 16))

        setup = Card(page, title="1. Run Full Pipeline")
        setup.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        setup.grid_columnconfigure(1, weight=1)
        setup.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(setup, text="Train CSV", text_color=COLOR_DIM).grid(
            row=1, column=0, sticky="w", padx=18, pady=(10, 6))
        csvs = self._list_csv_files()
        train_csv_box = ctk.CTkFrame(setup, fg_color="transparent")
        train_csv_box.grid(row=1, column=1, sticky="ew", padx=(8, 18), pady=(10, 6))
        train_csv_box.grid_columnconfigure(0, weight=1)
        self.pipe_csv = ScrollableOptionMenu(
            train_csv_box, values=csvs, width=260,
            command=lambda _: self._on_pipeline_train_csv_change())
        self.pipe_csv.grid(row=0, column=0, sticky="ew")
        self.pipe_data_hint = ctk.CTkLabel(
            train_csv_box,
            text="Rows: - | Suggested steps: -",
            text_color=COLOR_DIM,
            font=ctk.CTkFont(size=11),
            anchor="w",
            justify="left",
        )
        self.pipe_data_hint.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        if csvs[0] != "(none)":
            self.pipe_csv.set(csvs[0])

        ctk.CTkLabel(setup, text="Model name", text_color=COLOR_DIM).grid(
            row=1, column=2, sticky="w", padx=18, pady=(10, 6))
        self.pipe_model_name = ctk.CTkEntry(setup, placeholder_text="rl_pipeline_v1")
        self.pipe_model_name.insert(0, "rl_pipeline_v1")
        self.pipe_model_name.grid(row=1, column=3, sticky="ew", padx=(8, 18), pady=(10, 6))

        ctk.CTkLabel(setup, text="Backtest CSV", text_color=COLOR_DIM).grid(
            row=2, column=0, sticky="w", padx=18, pady=6)
        self.pipe_bt_csv = ScrollableOptionMenu(setup, values=csvs, width=260)
        self.pipe_bt_csv.grid(row=2, column=1, sticky="ew", padx=(8, 18), pady=6)
        if csvs[0] != "(none)":
            self.pipe_bt_csv.set(csvs[0])

        ctk.CTkLabel(setup, text="Train pct", text_color=COLOR_DIM).grid(
            row=2, column=2, sticky="w", padx=18, pady=6)
        self.pipe_train_pct = ctk.CTkEntry(setup, width=120)
        self.pipe_train_pct.insert(0, "1.0")
        self.pipe_train_pct.grid(row=2, column=3, sticky="w", padx=(8, 18), pady=6)
        self.pipe_train_pct.bind("<KeyRelease>", lambda _e: self._update_pipeline_data_hint())

        ctk.CTkLabel(setup, text="Train steps", text_color=COLOR_DIM).grid(
            row=3, column=0, sticky="w", padx=18, pady=6)
        step_box = ctk.CTkFrame(setup, fg_color="transparent")
        step_box.grid(row=3, column=1, sticky="w", padx=(8, 18), pady=6)
        self.pipe_steps = ctk.CTkEntry(step_box, width=120)
        self.pipe_steps.insert(0, "200000")
        self.pipe_steps.grid(row=0, column=0, sticky="w")
        self.pipe_use_steps_btn = ctk.CTkButton(
            step_box,
            text="Use suggested",
            command=self._use_pipeline_suggested_steps,
            fg_color=COLOR_BG_INPUT,
            hover_color=COLOR_HOVER,
            width=110,
            height=30,
            state="disabled",
        )
        self.pipe_use_steps_btn.grid(row=0, column=1, sticky="w", padx=(8, 0))

        ctk.CTkLabel(setup, text="Window size", text_color=COLOR_DIM).grid(
            row=3, column=2, sticky="w", padx=18, pady=6)
        self.pipe_window = ctk.CTkEntry(setup, width=120)
        self.pipe_window.insert(0, "10")
        self.pipe_window.grid(row=3, column=3, sticky="w", padx=(8, 18), pady=6)
        self.pipe_window.bind("<KeyRelease>", lambda _e: self._update_pipeline_data_hint())

        ctk.CTkLabel(setup, text="Backtest confidence", text_color=COLOR_DIM).grid(
            row=4, column=0, sticky="w", padx=18, pady=6)
        self.pipe_conf = ctk.CTkEntry(setup, width=120)
        self.pipe_conf.insert(0, "0.85")
        self.pipe_conf.grid(row=4, column=1, sticky="w", padx=(8, 18), pady=6)

        ctk.CTkLabel(setup, text="Backtest mode", text_color=COLOR_DIM).grid(
            row=4, column=2, sticky="w", padx=18, pady=6)
        self.pipe_bt_mode = ctk.CTkOptionMenu(
            setup, values=["Agent + SL/TP", "Pure Agent"], width=180)
        self.pipe_bt_mode.set("Agent + SL/TP")
        self.pipe_bt_mode.grid(row=4, column=3, sticky="w", padx=(8, 18), pady=6)

        self.pipe_relabel = ctk.CTkCheckBox(
            setup,
            text="Relabel first with quantile 33/33/33",
            text_color=COLOR_DIM,
        )
        self.pipe_relabel.grid(row=5, column=0, columnspan=2, sticky="w",
                               padx=18, pady=(8, 12))

        btns = ctk.CTkFrame(setup, fg_color="transparent")
        btns.grid(row=5, column=2, columnspan=2, sticky="e", padx=18, pady=(8, 12))
        self.pipe_run_btn = ctk.CTkButton(
            btns, text="Run full pipeline", command=self._run_full_pipeline,
            fg_color=COLOR_GREEN, hover_color="#2ea043", width=170)
        self.pipe_run_btn.pack(side="left", padx=(0, 8))
        self.pipe_stop_btn = ctk.CTkButton(
            btns, text="Stop", command=self._stop_pipeline,
            fg_color=COLOR_RED, hover_color="#da3633", width=90, state="disabled")
        self.pipe_stop_btn.pack(side="left")

        self.pipe_hparam_frame = ctk.CTkFrame(
            setup, fg_color=COLOR_BG_INPUT, corner_radius=8,
            border_width=1, border_color=COLOR_BORDER)
        self.pipe_hparam_frame.grid(row=6, column=0, columnspan=4, sticky="ew",
                                    padx=18, pady=(0, 16))
        for col in range(4):
            self.pipe_hparam_frame.grid_columnconfigure(col, weight=1)

        h_head = ctk.CTkFrame(self.pipe_hparam_frame, fg_color="transparent")
        h_head.grid(row=0, column=0, columnspan=4, sticky="ew", padx=12, pady=(10, 6))
        h_head.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            h_head, text="PPO Hyperparameters",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=COLOR_TEXT,
        ).grid(row=0, column=0, sticky="w")

        preset_row = ctk.CTkFrame(h_head, fg_color="transparent")
        preset_row.grid(row=0, column=1, sticky="e")
        for label, preset in [
            ("Default", "default"),
            ("Stable", "stable"),
            ("Fast", "fast"),
            ("Explore", "explore"),
        ]:
            ctk.CTkButton(
                preset_row, text=label,
                command=lambda p=preset: self._apply_pipeline_preset(p),
                fg_color=COLOR_BG_CARD,
                hover_color=COLOR_HOVER,
                width=72,
                height=26,
                font=ctk.CTkFont(size=11),
            ).pack(side="left", padx=(0, 6))

        self.pipe_hparams = {}
        h_fields = [
            ("learning_rate", "3e-4", "lr", "ความแรงในการเรียนรู้"),
            ("clip_range", "0.2", "clip", "กัน policy เปลี่ยนแรงเกิน"),
            ("ent_coef", "0.01", "ent", "เพิ่ม/ลดการลองทางใหม่"),
            ("n_steps", "2048", "nsteps", "จำนวน step ก่อน update"),
            ("n_epochs", "10", "nepochs", "รอบเรียนซ้ำต่อ rollout"),
            ("batch_size", "64", "batch", "ขนาด minibatch"),
            ("gamma", "0.99", "gamma", "น้ำหนักผลลัพธ์อนาคต"),
            ("gae_lambda", "0.95", "gae", "ทำให้ advantage นิ่งขึ้น"),
            ("vf_coef", "0.5", "vf", "น้ำหนัก critic/value loss"),
            ("max_hold", "30", "max_hold", "ถือ position ได้กี่แท่ง"),
            ("ep_len", "2000", "ep_len", "ความยาว episode ตอน train"),
            ("net_arch", "auto", "net_arch", "ขนาด neural network"),
        ]
        for i, (label, default, key, desc) in enumerate(h_fields):
            row = 1 + (i // 4) * 3
            col = i % 4
            ctk.CTkLabel(
                self.pipe_hparam_frame,
                text=label,
                text_color=COLOR_DIM,
                font=ctk.CTkFont(size=11),
            ).grid(row=row, column=col, sticky="w", padx=12, pady=(6, 2))
            entry = ctk.CTkEntry(self.pipe_hparam_frame, width=120)
            entry.insert(0, default)
            entry.grid(row=row + 1, column=col, sticky="ew", padx=12, pady=(0, 2))
            self.pipe_hparams[key] = entry
            ctk.CTkLabel(
                self.pipe_hparam_frame,
                text=desc,
                text_color=COLOR_DIM,
                font=ctk.CTkFont(size=10),
                anchor="w",
                justify="left",
            ).grid(row=row + 2, column=col, sticky="ew", padx=12, pady=(0, 8))

        reward_box = ctk.CTkFrame(self.pipe_hparam_frame, fg_color="transparent")
        reward_box.grid(row=10, column=0, columnspan=4, sticky="ew", padx=12, pady=(0, 10))
        ctk.CTkLabel(
            reward_box, text="Reward mode", text_color=COLOR_DIM,
            font=ctk.CTkFont(size=11),
        ).pack(side="left", padx=(0, 8))
        self.pipe_reward_mode = ctk.CTkOptionMenu(
            reward_box,
            values=["realized (recommended)", "mtm"],
            width=180,
            fg_color=COLOR_BG_CARD,
            button_color=COLOR_BG_CARD,
            button_hover_color=COLOR_HOVER,
        )
        self.pipe_reward_mode.set("realized (recommended)")
        self.pipe_reward_mode.pack(side="left")
        ctk.CTkLabel(
            reward_box,
            text="realized = ให้ reward ตอนปิดไม้ | mtm = ให้ทุกแท่งตาม unrealized P/L",
            text_color=COLOR_DIM,
            font=ctk.CTkFont(size=10),
        ).pack(side="left", padx=(10, 0))

        progress = Card(page, title="2. Progress")
        progress.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        progress.grid_columnconfigure(0, weight=1)
        self.pipe_progress = ctk.CTkProgressBar(
            progress,
            height=12,
            progress_color=COLOR_ACCENT,
            fg_color=COLOR_HOVER)
        self.pipe_progress.set(0)
        self.pipe_progress.grid(row=1, column=0, sticky="ew", padx=18, pady=(12, 6))
        self.pipe_status = ctk.CTkLabel(progress, text="Idle", text_color=COLOR_DIM)
        self.pipe_status.grid(row=2, column=0, sticky="w", padx=18, pady=(0, 12))

        stage_row = ctk.CTkFrame(progress, fg_color="transparent")
        stage_row.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 12))
        for label in ["Relabel", "Train PPO", "Backtest", "Chart"]:
            pill = ctk.CTkLabel(
                stage_row, text=label, height=30, corner_radius=8,
                fg_color=COLOR_BG_INPUT, text_color=COLOR_DIM,
                font=ctk.CTkFont(size=11, weight="bold"))
            pill.pack(side="left", padx=(0, 8), ipadx=12)
            self.pipeline_stage_labels.append(pill)

        health = Card(page, title="3. Live Metric Health")
        health.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        health.grid_columnconfigure(0, weight=1)

        self.pipeline_health_frame = ctk.CTkFrame(
            health, fg_color=COLOR_BG_INPUT, corner_radius=8)
        self.pipeline_health_frame.grid(row=1, column=0, sticky="ew", padx=18, pady=(10, 12))
        self.pipeline_health_frame.grid_columnconfigure(0, weight=1)

        self.pipeline_health_pills_frame = ctk.CTkFrame(
            self.pipeline_health_frame, fg_color="transparent")
        self.pipeline_health_pills_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=8)
        self.pipeline_health_pills = {}
        self._build_pipeline_health_pills()

        pipe_graph = ctk.CTkFrame(
            self.pipeline_health_frame, fg_color="#0a0e14", corner_radius=8,
            border_width=1, border_color="#30363d")
        pipe_graph.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))
        pipe_graph.grid_columnconfigure(0, weight=1)

        pipe_graph_head = ctk.CTkFrame(pipe_graph, fg_color="transparent")
        pipe_graph_head.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))
        pipe_graph_head.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            pipe_graph_head, text="Reward Trend",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLOR_DIM).grid(row=0, column=0, sticky="w")
        self.pipeline_reward_trend_status = ctk.CTkLabel(
            pipe_graph_head, text="-",
            font=ctk.CTkFont(size=10, family="Consolas"),
            text_color=COLOR_DIM)
        self.pipeline_reward_trend_status.grid(row=0, column=1, sticky="e")

        self.pipeline_reward_canvas = tk.Canvas(
            pipe_graph, height=60, bg="#0a0e14", highlightthickness=0, bd=0)
        self.pipeline_reward_canvas.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))
        self.pipeline_reward_canvas.bind(
            "<Configure>", lambda e: self._draw_pipeline_reward_sparkline())

        self.pipeline_health_recs_frame = ctk.CTkFrame(
            self.pipeline_health_frame, fg_color="transparent")
        self.pipeline_health_recs_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(2, 10))
        self.pipeline_health_recs_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            self.pipeline_health_recs_frame,
            text="Waiting for PPO training metrics...",
            font=ctk.CTkFont(size=11, family="Consolas"),
            text_color=COLOR_DIM,
            anchor="w").grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        self._reset_pipeline_metrics()

        compare = Card(page, title="4. Model Comparison")
        compare.grid(row=4, column=0, sticky="ew", pady=(0, 12))
        compare.grid_columnconfigure(0, weight=1)

        compare_btns = ctk.CTkFrame(compare, fg_color="transparent")
        compare_btns.grid(row=1, column=0, sticky="ew", padx=18, pady=(10, 8))
        ctk.CTkButton(compare_btns, text="Refresh table", command=self._refresh_model_comparison,
                      fg_color=COLOR_BG_INPUT, hover_color=COLOR_HOVER,
                      corner_radius=8, height=34, width=130).pack(side="left", padx=(0, 8))
        ctk.CTkButton(compare_btns, text="Open equity image", command=self._open_selected_equity,
                      fg_color=COLOR_BG_INPUT, hover_color=COLOR_HOVER,
                      corner_radius=8, height=34, width=150).pack(side="left", padx=(0, 8))
        ctk.CTkButton(compare_btns, text="Open HTML chart", command=self._open_selected_chart,
                      fg_color=COLOR_BG_INPUT, hover_color=COLOR_HOVER,
                      corner_radius=8, height=34, width=140).pack(side="left", padx=(0, 8))
        ctk.CTkButton(compare_btns, text="Export report PDF", command=self._export_pipeline_pdf,
                      fg_color=COLOR_PURPLE, hover_color="#8957e5",
                      corner_radius=8, height=34, width=150).pack(side="right")

        tree_frame = tk.Frame(compare, bg=COLOR_BG_CARD)
        tree_frame.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 14))
        tree_frame.grid_columnconfigure(0, weight=1)

        style = ttk.Style()
        try:
            style.theme_use("default")
        except Exception:
            pass
        style.configure(
            "Pipeline.Treeview",
            background=COLOR_BG_TERMINAL,
            foreground="#c9d1d9",
            fieldbackground=COLOR_BG_TERMINAL,
            borderwidth=0,
            rowheight=30,
            font=("Segoe UI", 9),
        )
        style.configure(
            "Pipeline.Treeview.Heading",
            background=COLOR_BG_INPUT,
            foreground=COLOR_TEXT,
            font=("Segoe UI", 9, "bold"),
            relief="flat",
        )
        style.map("Pipeline.Treeview", background=[("selected", COLOR_SELECTED)])

        columns = ("model", "source", "trades", "win_rate", "profit_factor",
                   "return", "max_dd", "chart")
        self.model_tree = ttk.Treeview(
            tree_frame, columns=columns, show="headings", height=9,
            style="Pipeline.Treeview")
        headings = {
            "model": "Model",
            "source": "Result source",
            "trades": "Trades",
            "win_rate": "Win rate",
            "profit_factor": "PF",
            "return": "Return",
            "max_dd": "Max DD",
            "chart": "Chart",
        }
        widths = {
            "model": 190, "source": 170, "trades": 70, "win_rate": 80,
            "profit_factor": 70, "return": 80, "max_dd": 80, "chart": 70,
        }
        for col in columns:
            self.model_tree.heading(col, text=headings[col])
            self.model_tree.column(col, width=widths[col], anchor="center")
        self.model_tree.column("model", anchor="w")
        self.model_tree.column("source", anchor="w")

        yscroll = ttk.Scrollbar(tree_frame, orient="vertical",
                                command=self.model_tree.yview)
        self.model_tree.configure(yscrollcommand=yscroll.set)
        self.model_tree.grid(row=0, column=0, sticky="ew")
        yscroll.grid(row=0, column=1, sticky="ns")
        self.model_tree.bind("<<TreeviewSelect>>", self._refresh_equity_viewer)

        equity = Card(page, title="5. Equity Curve Viewer")
        equity.grid(row=5, column=0, sticky="ew", pady=(0, 12))
        equity.grid_columnconfigure(0, weight=1)
        self.pipeline_equity_label = ctk.CTkLabel(
            equity, text="Select a model result to preview equity curve.",
            text_color=COLOR_DIM, height=360)
        self.pipeline_equity_label.grid(row=1, column=0, sticky="ew", padx=18, pady=12)

        log_card = Card(page, title="6. Pipeline Log")
        log_card.grid(row=6, column=0, sticky="ew", pady=(0, 12))
        log_frame = ctk.CTkFrame(log_card, fg_color="transparent")
        log_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 12))
        self.pipeline_log = self._make_log_widget(log_frame, height=12)
        self._on_pipeline_train_csv_change()

    def _on_pipeline_train_csv_change(self):
        if not hasattr(self, "pipe_bt_csv"):
            return
        train_csv = self.pipe_csv.get().strip()
        if not train_csv or train_csv == "(none)":
            self.pipeline_selected_rows = None
            self.pipeline_suggested_steps = None
            if hasattr(self, "pipe_data_hint"):
                self.pipe_data_hint.configure(text="Rows: - | Suggested steps: -", text_color=COLOR_DIM)
            if hasattr(self, "pipe_use_steps_btn"):
                self.pipe_use_steps_btn.configure(state="disabled")
            return

        self._start_pipeline_row_count(train_csv)

        train_path = Path(train_csv)
        candidates = []
        stem = train_path.stem
        if stem.endswith("_train"):
            candidates.append(train_path.with_stem(stem[:-len("_train")] + "_test").name)
        candidates.append(train_csv)

        for name in candidates:
            if (WORK_DIR / name).exists():
                self.pipe_bt_csv.set(name)
                return

    def _start_pipeline_row_count(self, csv_name):
        if not hasattr(self, "pipe_data_hint"):
            return
        self._pipeline_dataset_token += 1
        token = self._pipeline_dataset_token
        self.pipeline_selected_rows = None
        self.pipeline_suggested_steps = None
        if hasattr(self, "pipe_use_steps_btn"):
            self.pipe_use_steps_btn.configure(state="disabled")
        self.pipe_data_hint.configure(
            text=f"Rows: counting {csv_name}...",
            text_color=COLOR_DIM,
        )

        def worker():
            rows = self._csv_row_count(csv_name)
            self._pipeline_row_count_q.put((token, csv_name, rows))

        threading.Thread(target=worker, daemon=True).start()

    def _drain_pipeline_row_counts(self):
        if not hasattr(self, "_pipeline_row_count_q"):
            return
        try:
            while True:
                token, csv_name, rows = self._pipeline_row_count_q.get_nowait()
                self._finish_pipeline_row_count(token, csv_name, rows)
        except queue.Empty:
            pass

    def _finish_pipeline_row_count(self, token, csv_name, rows):
        if token != getattr(self, "_pipeline_dataset_token", None):
            return
        if not hasattr(self, "pipe_data_hint"):
            return
        if rows is None:
            self.pipeline_selected_rows = None
            self.pipeline_suggested_steps = None
            self.pipe_data_hint.configure(
                text=f"Rows: unable to read {csv_name} | Suggested steps: -",
                text_color=COLOR_RED,
            )
            if hasattr(self, "pipe_use_steps_btn"):
                self.pipe_use_steps_btn.configure(state="disabled")
            return

        self.pipeline_selected_rows = rows
        self._update_pipeline_data_hint()

    def _update_pipeline_data_hint(self):
        if not hasattr(self, "pipe_data_hint"):
            return
        rows = getattr(self, "pipeline_selected_rows", None)
        if rows is None:
            return

        rec = self._suggest_train_steps(rows)
        if not rec:
            self.pipeline_suggested_steps = None
            self.pipe_data_hint.configure(
                text=f"Rows: {rows:,} | Suggested steps: -",
                text_color=COLOR_DIM,
            )
            if hasattr(self, "pipe_use_steps_btn"):
                self.pipe_use_steps_btn.configure(state="disabled")
            return

        suggested, effective, approx_passes = rec
        self.pipeline_suggested_steps = suggested
        self.pipe_data_hint.configure(
            text=(
                f"Rows: {rows:,} | Train bars: {effective:,} | "
                f"Suggested steps: {suggested:,} (~{approx_passes:.1f} passes)"
            ),
            text_color=COLOR_GREEN,
        )
        if hasattr(self, "pipe_use_steps_btn"):
            self.pipe_use_steps_btn.configure(state="normal")

    def _use_pipeline_suggested_steps(self):
        steps = getattr(self, "pipeline_suggested_steps", None)
        if not steps or not hasattr(self, "pipe_steps"):
            return
        self.pipe_steps.delete(0, "end")
        self.pipe_steps.insert(0, str(int(steps)))

    def _ppo_presets(self):
        return {
            "default": {
                "lr": "3e-4", "clip": "0.2", "ent": "0.01",
                "nsteps": "2048", "nepochs": "10", "batch": "64",
                "gamma": "0.99", "gae": "0.95", "vf": "0.5",
            },
            "stable": {
                "lr": "1e-4", "clip": "0.1", "ent": "0.01",
                "nsteps": "4096", "nepochs": "10", "batch": "128",
                "gamma": "0.99", "gae": "0.95", "vf": "0.5",
            },
            "fast": {
                "lr": "5e-4", "clip": "0.3", "ent": "0.01",
                "nsteps": "1024", "nepochs": "5", "batch": "64",
                "gamma": "0.99", "gae": "0.95", "vf": "0.5",
            },
            "explore": {
                "lr": "3e-4", "clip": "0.2", "ent": "0.05",
                "nsteps": "2048", "nepochs": "10", "batch": "64",
                "gamma": "0.99", "gae": "0.95", "vf": "0.5",
            },
        }

    def _apply_pipeline_preset(self, name):
        presets = self._ppo_presets()
        p = presets.get(name)
        if not p or not hasattr(self, "pipe_hparams"):
            return
        mapping = {
            "lr": p["lr"],
            "clip": p["clip"],
            "ent": p["ent"],
            "nsteps": p["nsteps"],
            "nepochs": p["nepochs"],
            "batch": p["batch"],
            "gamma": p["gamma"],
            "gae": p["gae"],
            "vf": p["vf"],
        }
        for key, val in mapping.items():
            entry = self.pipe_hparams.get(key)
            if entry is None:
                continue
            entry.delete(0, "end")
            entry.insert(0, val)

    def _get_pipeline_hparams(self):
        if not hasattr(self, "pipe_hparams"):
            return {}

        raw = {
            key: entry.get().strip()
            for key, entry in self.pipe_hparams.items()
        }
        defaults = {
            "lr": "3e-4", "clip": "0.2", "ent": "0.01",
            "nsteps": "2048", "nepochs": "10", "batch": "64",
            "gamma": "0.99", "gae": "0.95", "vf": "0.5",
            "max_hold": "30", "ep_len": "2000", "net_arch": "auto",
        }
        values = {k: (raw.get(k) or defaults[k]) for k in defaults}

        try:
            floats = {
                "learning_rate": float(values["lr"]),
                "clip_range": float(values["clip"]),
                "ent_coef": float(values["ent"]),
                "gamma": float(values["gamma"]),
                "gae_lambda": float(values["gae"]),
                "vf_coef": float(values["vf"]),
            }
            ints = {
                "n_steps": int(float(values["nsteps"])),
                "n_epochs": int(float(values["nepochs"])),
                "batch_size": int(float(values["batch"])),
                "max_hold": int(float(values["max_hold"])),
                "ep_len": int(float(values["ep_len"])),
            }
        except ValueError:
            messagebox.showerror("Invalid hyperparameters", "Hyperparameters must be numeric, except net_arch.")
            return None

        if floats["learning_rate"] <= 0:
            messagebox.showerror("Invalid hyperparameters", "learning_rate must be greater than 0.")
            return None
        if floats["clip_range"] <= 0:
            messagebox.showerror("Invalid hyperparameters", "clip_range must be greater than 0.")
            return None
        if floats["ent_coef"] < 0 or floats["vf_coef"] < 0:
            messagebox.showerror("Invalid hyperparameters", "ent_coef and vf_coef must be 0 or greater.")
            return None
        if not (0 < floats["gamma"] <= 1):
            messagebox.showerror("Invalid hyperparameters", "gamma must be > 0 and <= 1.")
            return None
        if not (0 < floats["gae_lambda"] <= 1):
            messagebox.showerror("Invalid hyperparameters", "gae_lambda must be > 0 and <= 1.")
            return None
        if any(v <= 0 for v in ints.values()):
            messagebox.showerror("Invalid hyperparameters", "n_steps, n_epochs, batch_size, max_hold, and ep_len must be greater than 0.")
            return None
        if ints["batch_size"] > ints["n_steps"]:
            messagebox.showerror("Invalid hyperparameters", "batch_size must be <= n_steps.")
            return None

        net_arch = values["net_arch"].strip() or "auto"
        if net_arch != "auto":
            try:
                layers = [int(x.strip()) for x in net_arch.split(",") if x.strip()]
            except ValueError:
                messagebox.showerror("Invalid hyperparameters", "net_arch must be 'auto' or comma-separated integers, e.g. 256,128,64.")
                return None
            if not layers or any(x <= 0 for x in layers):
                messagebox.showerror("Invalid hyperparameters", "net_arch layers must be positive integers.")
                return None

        reward_mode = "realized"
        if hasattr(self, "pipe_reward_mode"):
            reward_mode = self.pipe_reward_mode.get().split()[0]

        return {
            "learning_rate": str(floats["learning_rate"]),
            "clip_range": str(floats["clip_range"]),
            "ent_coef": str(floats["ent_coef"]),
            "n_steps": str(ints["n_steps"]),
            "n_epochs": str(ints["n_epochs"]),
            "batch_size": str(ints["batch_size"]),
            "gamma": str(floats["gamma"]),
            "gae_lambda": str(floats["gae_lambda"]),
            "vf_coef": str(floats["vf_coef"]),
            "max_hold": str(ints["max_hold"]),
            "ep_len": str(ints["ep_len"]),
            "net_arch": net_arch,
            "reward_mode": reward_mode,
        }

    def _run_full_pipeline(self):
        if self.runner.is_running() or getattr(self, "pipeline_running", False):
            messagebox.showwarning("Busy", "Another process is already running.")
            return

        csv_name = self.pipe_csv.get().strip()
        if not csv_name or csv_name == "(none)":
            messagebox.showerror("Missing CSV", "Please select a dataset CSV first.")
            return
        csv_path = WORK_DIR / csv_name
        if not csv_path.exists():
            messagebox.showerror("CSV not found", str(csv_path))
            return

        bt_csv = self.pipe_bt_csv.get().strip()
        if not bt_csv or bt_csv == "(none)":
            bt_csv = csv_name
        bt_csv_path = WORK_DIR / bt_csv
        if not bt_csv_path.exists():
            messagebox.showerror("Backtest CSV not found", str(bt_csv_path))
            return

        model_name = self.pipe_model_name.get().strip() or "rl_pipeline_v1"
        model_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("._-")
        if not model_name:
            messagebox.showerror("Invalid name", "Model name is empty after cleanup.")
            return
        self.pipe_model_name.delete(0, "end")
        self.pipe_model_name.insert(0, model_name)

        try:
            steps = int(float(self.pipe_steps.get().strip() or "200000"))
            window = int(float(self.pipe_window.get().strip() or "10"))
            conf = float(self.pipe_conf.get().strip() or "0.85")
            train_pct = float(self.pipe_train_pct.get().strip() or "1.0")
        except ValueError:
            messagebox.showerror("Invalid settings", "Steps/window/confidence/train_pct must be numeric.")
            return

        if steps <= 0 or window <= 0:
            messagebox.showerror("Invalid settings", "Steps and window must be greater than 0.")
            return
        if not (0 < train_pct <= 1.0):
            messagebox.showerror("Invalid settings", "Train pct must be > 0 and <= 1.0.")
            return

        hparams = self._get_pipeline_hparams()
        if hparams is None:
            return

        if (WORK_DIR / f"{model_name}.zip").exists():
            ok = messagebox.askyesno(
                "Overwrite model?",
                f"{model_name}.zip already exists. Continue and overwrite it?")
            if not ok:
                return

        use_relabel = bool(self.pipe_relabel.get())
        mode = "pure_agent" if "Pure" in self.pipe_bt_mode.get() else "agent_sltp"

        self.pipeline_running = True
        self.pipeline_stop_requested = False
        self.pipe_run_btn.configure(state="disabled")
        self.pipe_stop_btn.configure(state="normal")
        self.pipeline_log.configure(state="normal")
        self.pipeline_log.delete("1.0", "end")
        self.pipeline_log.configure(state="disabled")
        self._set_pipeline_progress(0, "Starting pipeline...")
        self._set_pipeline_stage(-1)
        self._reset_pipeline_metrics(steps)
        self.status_label.configure(text="Running pipeline", text_color=COLOR_GREEN)

        t = threading.Thread(
            target=self._pipeline_worker,
            args=(csv_name, bt_csv, use_relabel, model_name, steps, window, conf, mode, train_pct, hparams),
            daemon=True,
        )
        t.start()

    def _stop_pipeline(self):
        self.pipeline_stop_requested = True
        self._pipeline_log("Stop requested. Terminating current stage...", "warn")
        if self.pipeline_proc and self.pipeline_proc.poll() is None:
            try:
                self.pipeline_proc.terminate()
            except Exception:
                pass

    def _pipeline_worker(self, csv_name, bt_csv, use_relabel, model_name,
                         steps, window, conf, mode, train_pct, hparams):
        train_csv = csv_name
        stages_total = 3
        if use_relabel and not Path(csv_name).stem.endswith("_relabeled"):
            stages_total += 1

        stage = 1
        try:
            if use_relabel:
                if Path(csv_name).stem.endswith("_relabeled"):
                    self._pipeline_log("Relabel skipped because source already ends with _relabeled.", "warn")
                else:
                    relabel_cmd = [sys.executable, "relabel.py", csv_name, "--mode", "quantile"]
                    self._pipeline_run_cmd(relabel_cmd, stage, "Relabel", stages_total)
                    train_csv = self._expected_relabeled_path(csv_name).name
                    stage += 1

            train_cmd = [
                sys.executable, "rl_train.py", train_csv,
                "--steps", str(steps),
                "--window", str(window),
                "--name", model_name,
                "--train_pct", str(train_pct),
                "--eval_csv", bt_csv,
                "--algo", "ppo",
                "--reward_mode", hparams["reward_mode"],
                "--max_hold", hparams["max_hold"],
                "--ep_len", hparams["ep_len"],
                "--net_arch", hparams["net_arch"],
                "--learning_rate", hparams["learning_rate"],
                "--clip_range", hparams["clip_range"],
                "--ent_coef", hparams["ent_coef"],
                "--n_steps", hparams["n_steps"],
                "--n_epochs", hparams["n_epochs"],
                "--batch_size", hparams["batch_size"],
                "--gamma", hparams["gamma"],
                "--gae_lambda", hparams["gae_lambda"],
                "--vf_coef", hparams["vf_coef"],
            ]
            self._pipeline_run_cmd(train_cmd, stage, "Train PPO", stages_total)
            stage += 1

            backtest_cmd = [
                sys.executable, "backtest_live.py", model_name, bt_csv,
                "--conf", str(conf),
                "--window", str(window),
                "--mode", mode,
            ]
            self._pipeline_run_cmd(backtest_cmd, stage, "Backtest", stages_total)
            stage += 1

            chart_cmd = [
                sys.executable, "backtest_chart.py", model_name, bt_csv,
                "--limit", "5000",
            ]
            self._pipeline_run_cmd(chart_cmd, stage, "Chart", stages_total)

            self.after(0, lambda: self._finish_pipeline(True, "Pipeline complete."))
        except Exception as e:
            self.after(0, lambda err=e: self._finish_pipeline(False, str(err)))

    def _pipeline_run_cmd(self, cmd, stage_idx, stage_name, stages_total):
        if self.pipeline_stop_requested:
            raise RuntimeError("Pipeline stopped.")

        base = (stage_idx - 1) / max(stages_total, 1)
        span = 1 / max(stages_total, 1)
        self._set_pipeline_stage(stage_name)
        self._set_pipeline_progress(base, f"{stage_name} started")
        self._pipeline_log("")
        self._pipeline_log(f"=== {stage_name} ===", "metric")
        self._pipeline_log("$ " + self._format_cmd(cmd), "info")

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        self.pipeline_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(WORK_DIR),
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )

        assert self.pipeline_proc.stdout is not None
        for line in iter(self.pipeline_proc.stdout.readline, ""):
            line = line.rstrip()
            if not line:
                continue
            tag = self._classify_log(line)
            self._pipeline_log(line, tag)

            if stage_name == "Train PPO":
                self._handle_pipeline_metric_line(line)

            prog = self._parse_progress(line)
            if not prog and stage_name == "Train PPO":
                prog = self._parse_pipeline_progress(line)
            if prog:
                cur, tot = prog
                pct = min(max(cur / tot, 0), 1) if tot else 0
                if stage_name == "Train PPO":
                    self.pipeline_train_step_count = cur
                self._set_pipeline_progress(
                    base + span * pct,
                    f"{stage_name}: {cur:,} / {tot:,} ({pct * 100:.1f}%)",
                )

            if self.pipeline_stop_requested and self.pipeline_proc.poll() is None:
                self.pipeline_proc.terminate()

        self.pipeline_proc.wait()
        rc = self.pipeline_proc.returncode
        self.pipeline_proc = None

        if self.pipeline_stop_requested:
            raise RuntimeError("Pipeline stopped.")
        if rc != 0:
            raise RuntimeError(f"{stage_name} failed with exit code {rc}.")

        self._set_pipeline_progress(stage_idx / max(stages_total, 1), f"{stage_name} done")
        self._pipeline_log(f"{stage_name} done.", "success")

    def _finish_pipeline(self, ok, message):
        self.pipeline_running = False
        self.pipeline_proc = None
        self.pipe_run_btn.configure(state="normal")
        self.pipe_stop_btn.configure(state="disabled")
        self.status_label.configure(text="Idle", text_color=COLOR_DIM)

        if ok:
            self._set_pipeline_progress(1, message)
            self._set_pipeline_stage("done")
            self._pipeline_log(message, "success")
            self._refresh_dropdowns()
            self._refresh_model_comparison()
        else:
            self._set_pipeline_progress(self.pipe_progress.get(), "Pipeline stopped/failed")
            self._pipeline_log(message, "error")
            self._set_pipeline_stage("failed")

    def _pipeline_log(self, text, tag="info"):
        if threading.current_thread() is not threading.main_thread():
            self.after(0, lambda: self._pipeline_log(text, tag))
            return
        if hasattr(self, "pipeline_log"):
            self._log(self.pipeline_log, text, tag)

    def _set_pipeline_progress(self, pct, text=None):
        if threading.current_thread() is not threading.main_thread():
            self.after(0, lambda: self._set_pipeline_progress(pct, text))
            return
        pct = min(max(float(pct), 0.0), 1.0)
        if hasattr(self, "pipe_progress"):
            self.pipe_progress.set(pct)
        if text and hasattr(self, "pipe_status"):
            self.pipe_status.configure(text=f"{text} ({pct * 100:.1f}%)")

    def _set_pipeline_stage(self, active):
        if threading.current_thread() is not threading.main_thread():
            self.after(0, lambda: self._set_pipeline_stage(active))
            return
        if not hasattr(self, "pipeline_stage_labels"):
            return
        for lbl in self.pipeline_stage_labels:
            name = lbl.cget("text")
            if active == "done":
                lbl.configure(fg_color="#16351f", text_color=COLOR_GREEN)
            elif active == "failed":
                lbl.configure(fg_color="#3a1717", text_color=COLOR_RED)
            elif active == name:
                lbl.configure(fg_color=COLOR_SELECTED, text_color=COLOR_TEXT)
            else:
                lbl.configure(fg_color=COLOR_BG_INPUT, text_color=COLOR_DIM)

    def _format_cmd(self, cmd):
        parts = []
        for item in cmd:
            s = str(item)
            parts.append(f'"{s}"' if " " in s else s)
        return " ".join(parts)

    def _build_pipeline_health_pills(self):
        metrics_to_show = [
            ("ep_rew_mean", "Reward"),
            ("approx_kl", "KL"),
            ("clip_fraction", "Clip%"),
            ("explained_variance", "EV"),
            ("entropy_loss", "Entropy"),
            ("value_loss", "V-Loss"),
        ]
        for i, (key, label) in enumerate(metrics_to_show):
            self.pipeline_health_pills_frame.grid_columnconfigure(i, weight=1)
            pill = ctk.CTkFrame(
                self.pipeline_health_pills_frame,
                fg_color="#0a0e14",
                corner_radius=6,
                border_width=1,
                border_color="#30363d",
            )
            pill.grid(row=0, column=i, sticky="ew", padx=3)

            ctk.CTkLabel(
                pill,
                text=label,
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color=COLOR_DIM,
            ).pack(pady=(6, 0))
            value_lbl = ctk.CTkLabel(
                pill,
                text="-",
                font=ctk.CTkFont(size=14, family="Consolas", weight="bold"),
                text_color=COLOR_DIM,
            )
            value_lbl.pack(pady=(2, 6))
            self.pipeline_health_pills[key] = value_lbl

    def _reset_pipeline_metrics(self, total_steps=None):
        self.pipeline_current_metrics = {}
        self.pipeline_reward_history = []
        self.pipeline_reward_peak = None
        self.pipeline_reward_alert_shown = False
        self.pipeline_train_step_count = 0
        if total_steps is not None:
            self.pipeline_train_steps_total = int(total_steps)
            self._train_steps_total = int(total_steps)
        elif not hasattr(self, "pipeline_train_steps_total"):
            self.pipeline_train_steps_total = 0

        if hasattr(self, "pipeline_health_pills"):
            for lbl in self.pipeline_health_pills.values():
                lbl.configure(text="-", text_color=COLOR_DIM)

        if hasattr(self, "pipeline_reward_trend_status"):
            self.pipeline_reward_trend_status.configure(text="-", text_color=COLOR_DIM)
        self._draw_pipeline_reward_sparkline()

        if hasattr(self, "pipeline_health_recs_frame"):
            for w in self.pipeline_health_recs_frame.winfo_children():
                w.destroy()
            ctk.CTkLabel(
                self.pipeline_health_recs_frame,
                text="Waiting for PPO training metrics...",
                font=ctk.CTkFont(size=11, family="Consolas"),
                text_color=COLOR_DIM,
                anchor="w",
            ).grid(row=0, column=0, sticky="ew", padx=4, pady=4)

    def _handle_pipeline_metric_line(self, line):
        if threading.current_thread() is not threading.main_thread():
            self.after(0, lambda: self._handle_pipeline_metric_line(line))
            return

        mm = re.match(r'\|\s*([a-z_]+)\s*\|\s*([+\-\d\.eE]+)\s*\|', line)
        if not mm:
            return

        name = mm.group(1)
        if name not in METRIC_INFO:
            return

        try:
            value = float(mm.group(2))
        except ValueError:
            return

        self._update_pipeline_health_pill(name, value)
        self.pipeline_current_metrics[name] = value
        self._refresh_pipeline_recommendations()

        if name == "ep_rew_mean":
            self._track_pipeline_reward(value)

    def _parse_pipeline_progress(self, line):
        total = int(getattr(self, "pipeline_train_steps_total", 0) or 0)
        if total <= 0:
            return None

        m = re.search(r'\|\s*total_timesteps\s*\|\s*(\d[\d,]*)\s*\|', line)
        if m:
            try:
                return int(m.group(1).replace(",", "")), total
            except Exception:
                return None

        m = re.search(r'total_timesteps\s*[=:]\s*(\d[\d,]*)', line, re.I)
        if m:
            try:
                return int(m.group(1).replace(",", "")), total
            except Exception:
                return None

        m = re.search(r'\bstep\s+(\d[\d,]*)\b', line, re.I)
        if m:
            try:
                return int(m.group(1).replace(",", "")), total
            except Exception:
                return None

        return None

    def _update_pipeline_health_pill(self, name, value):
        if not hasattr(self, "pipeline_health_pills") or name not in self.pipeline_health_pills:
            return
        status = classify_metric(name, value)
        color = {
            "good": COLOR_GREEN,
            "warn": COLOR_YELLOW,
            "bad": COLOR_RED,
        }.get(status, COLOR_DIM)

        if abs(value) >= 100:
            txt = f"{value:.0f}"
        elif abs(value) >= 1:
            txt = f"{value:.2f}"
        elif abs(value) >= 0.001:
            txt = f"{value:.3f}"
        else:
            txt = f"{value:.2e}"
        self.pipeline_health_pills[name].configure(text=txt, text_color=color)

    def _track_pipeline_reward(self, value):
        self.pipeline_reward_history.append(float(value))
        if len(self.pipeline_reward_history) > 200:
            self.pipeline_reward_history = self.pipeline_reward_history[-200:]

        if self.pipeline_reward_peak is None or value > self.pipeline_reward_peak:
            self.pipeline_reward_peak = float(value)

        n = len(self.pipeline_reward_history)
        trend_txt, trend_color = "-", COLOR_DIM
        if n >= 5:
            recent = self.pipeline_reward_history[-5:]
            slope = recent[-1] - recent[0]
            if slope > 0.05:
                trend_txt, trend_color = "rising", COLOR_GREEN
            elif slope < -0.05:
                trend_txt, trend_color = "falling", COLOR_RED
            else:
                trend_txt, trend_color = "flat", COLOR_YELLOW

        peak = self.pipeline_reward_peak if self.pipeline_reward_peak is not None else value
        self.pipeline_reward_trend_status.configure(
            text=f"now {value:+.2f}  peak {peak:+.2f}  {trend_txt}",
            text_color=trend_color,
        )
        self._draw_pipeline_reward_sparkline()

        min_steps = max(10, int(getattr(self, "pipeline_train_steps_total", 0) * 0.3))
        if (self.pipeline_reward_peak is not None
                and self.pipeline_reward_peak > 0.5
                and self.pipeline_train_step_count >= min_steps
                and value < self.pipeline_reward_peak * 0.5
                and not self.pipeline_reward_alert_shown):
            self.pipeline_reward_alert_shown = True
            msg = (
                "Reward dropped below 50% of peak\n\n"
                f"Peak reward : {self.pipeline_reward_peak:+.3f}\n"
                f"Current     : {value:+.3f}\n\n"
                "Suggested actions:\n"
                "- Lower learning_rate, for example 3e-4 -> 1e-4\n"
                "- Lower clip_range if KL/Clip are high\n"
                "- Consider stopping and fine-tuning from the best checkpoint"
            )
            self.after(50, lambda: messagebox.showwarning("Reward Drop Detected", msg))

    def _draw_pipeline_reward_sparkline(self):
        c = getattr(self, "pipeline_reward_canvas", None)
        if c is None:
            return
        c.delete("all")
        try:
            w = c.winfo_width()
            h = c.winfo_height()
        except Exception:
            return
        if w < 4 or h < 4:
            return

        data = getattr(self, "pipeline_reward_history", [])
        if not data:
            c.create_text(
                w / 2, h / 2,
                text="(waiting for ep_rew_mean ...)",
                fill=COLOR_DIM,
                font=("Consolas", 10),
            )
            return

        c.create_line(0, h / 2, w, h / 2, fill="#30363d", dash=(2, 3))

        n = len(data)
        vmin = min(data)
        vmax = max(data)
        if vmax > 0 and vmin < 0:
            span = max(abs(vmin), abs(vmax)) or 1.0
            lo, hi = -span, span
        elif vmax > 0:
            lo, hi = 0, max(vmax, 0.001)
        else:
            lo, hi = min(vmin, -0.001), 0
        rng = (hi - lo) or 1.0
        lo -= rng * 0.1
        hi += rng * 0.1

        def to_xy(i, v):
            x = (w - 4) * (i / max(n - 1, 1)) + 2
            y = h - ((v - lo) / (hi - lo)) * (h - 4) - 2
            return x, y

        recent = data[-5:] if n >= 5 else data
        slope = recent[-1] - recent[0] if len(recent) > 1 else 0
        if slope > 0.05:
            line_color = COLOR_GREEN
        elif slope < -0.05:
            line_color = COLOR_RED
        else:
            line_color = COLOR_YELLOW

        flat = []
        for x, y in [to_xy(i, v) for i, v in enumerate(data)]:
            flat += [x, y]
        if len(flat) >= 4:
            c.create_line(*flat, fill=line_color, width=2, smooth=True)

        if self.pipeline_reward_peak is not None and n > 0:
            try:
                idx = data.index(self.pipeline_reward_peak)
                px, py = to_xy(idx, self.pipeline_reward_peak)
                c.create_oval(px - 3, py - 3, px + 3, py + 3,
                              fill=COLOR_ACCENT, outline="")
            except Exception:
                pass

        if self.pipeline_reward_peak is not None and self.pipeline_reward_peak > 0:
            thr = self.pipeline_reward_peak * 0.5
            if lo <= thr <= hi:
                ty = h - ((thr - lo) / (hi - lo)) * (h - 4) - 2
                c.create_line(0, ty, w, ty, fill=COLOR_RED, dash=(3, 3), width=1)
                c.create_text(w - 4, ty - 6, text="50% peak",
                              fill=COLOR_RED, anchor="ne", font=("Consolas", 8))

    def _refresh_pipeline_recommendations(self):
        if not hasattr(self, "pipeline_health_recs_frame"):
            return
        for w in self.pipeline_health_recs_frame.winfo_children():
            w.destroy()

        recs = []
        for name, value in getattr(self, "pipeline_current_metrics", {}).items():
            rec = get_recommendation(name, value)
            if rec:
                recs.append(rec)

        sev_order = {"high": 0, "med": 1, "low": 2}
        recs.sort(key=lambda r: sev_order.get(r.get("severity", "med"), 1))

        if not recs:
            ctk.CTkLabel(
                self.pipeline_health_recs_frame,
                text="All tracked metrics are in healthy range.",
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=COLOR_GREEN,
                anchor="w",
            ).grid(row=0, column=0, sticky="ew", padx=4, pady=8)
            return

        for i, rec in enumerate(recs):
            row = i // 2
            col = i % 2
            self.pipeline_health_recs_frame.grid_columnconfigure(col, weight=1)
            self._build_rec_card(self.pipeline_health_recs_frame, rec, row, col)

    def _expected_relabeled_path(self, csv_name):
        p = WORK_DIR / csv_name
        return p.with_stem(p.stem + "_relabeled")

    def _collect_model_rows(self):
        sig = self._model_results_signature()
        if self._model_rows_cache_sig == sig and self._model_rows_cache is not None:
            return list(self._model_rows_cache)

        rows = []
        seen = set()
        trade_files = list(WORK_DIR.glob("*_live_bt_trades.csv")) + list(WORK_DIR.glob("*_trades.csv"))

        for trades_path in sorted(trade_files, key=lambda p: p.stat().st_mtime, reverse=True):
            if trades_path in seen:
                continue
            seen.add(trades_path)
            name = trades_path.name
            if name.endswith("_live_bt_trades.csv"):
                model = name[:-len("_live_bt_trades.csv")]
                source = "live backtest"
            elif name.endswith("_trades.csv"):
                model = name[:-len("_trades.csv")]
                source = "env backtest"
            else:
                continue

            metrics = self._metrics_from_trades(trades_path)
            chart_path = WORK_DIR / f"{model}_backtest_chart.html"
            equity_path = self._equity_path_for_model(model)
            rows.append({
                "iid": str(trades_path),
                "model": model,
                "source": source,
                "trades_path": trades_path,
                "chart_path": chart_path if chart_path.exists() else None,
                "equity_path": equity_path,
                **metrics,
            })

        model_names = {r["model"] for r in rows}
        for zip_path in sorted(WORK_DIR.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True):
            model = zip_path.stem
            if model in model_names:
                continue
            rows.append({
                "iid": str(zip_path),
                "model": model,
                "source": "model only",
                "trades_path": None,
                "chart_path": None,
                "equity_path": self._equity_path_for_model(model),
                "trades": None,
                "win_rate": None,
                "profit_factor": None,
                "return_pct": None,
                "max_dd": None,
            })

        rows.sort(key=lambda r: (
            r.get("return_pct") is not None,
            r.get("return_pct") if r.get("return_pct") is not None else -1e9,
        ), reverse=True)
        self._model_rows_cache_sig = sig
        self._model_rows_cache = list(rows)
        return rows

    def _metrics_from_trades(self, trades_path):
        try:
            import numpy as np
            import pandas as pd
            df = pd.read_csv(trades_path)
            if df.empty:
                return {
                    "trades": 0, "win_rate": None, "profit_factor": None,
                    "return_pct": None, "max_dd": None,
                }

            if "pnl_dollars" in df.columns:
                pnl = pd.to_numeric(df["pnl_dollars"], errors="coerce").dropna()
                equity = 10000 + pnl.cumsum()
                return_pct = float(pnl.sum() / 10000 * 100)
            elif "pnl_pct" in df.columns:
                pnl = pd.to_numeric(df["pnl_pct"], errors="coerce").dropna()
                equity = (1 + pnl).cumprod()
                return_pct = float((equity.iloc[-1] - 1) * 100) if len(equity) else None
            elif "pnl" in df.columns:
                pnl = pd.to_numeric(df["pnl"], errors="coerce").dropna()
                if len(pnl) and pnl.abs().median() < 0.2:
                    equity = (1 + pnl).cumprod()
                    return_pct = float((equity.iloc[-1] - 1) * 100)
                else:
                    equity = 10000 + pnl.cumsum()
                    return_pct = float(pnl.sum() / 10000 * 100)
            else:
                pnl = pd.Series(dtype=float)
                equity = pd.Series(dtype=float)
                return_pct = None

            if "equity" in df.columns:
                eq = pd.to_numeric(df["equity"], errors="coerce").dropna()
                if len(eq):
                    equity = eq
                    return_pct = float((eq.iloc[-1] - 10000) / 10000 * 100)

            trades = int(len(pnl)) if len(pnl) else int(len(df))
            win_rate = float((pnl > 0).mean() * 100) if len(pnl) else None
            gross_profit = float(pnl[pnl > 0].sum()) if len(pnl) else 0.0
            gross_loss = float(abs(pnl[pnl < 0].sum())) if len(pnl) else 0.0
            profit_factor = float("inf") if gross_profit > 0 and gross_loss == 0 else (
                gross_profit / gross_loss if gross_loss > 0 else None)

            max_dd = None
            if len(equity):
                peak = equity.cummax()
                dd = equity / peak - 1
                max_dd = float(dd.min() * 100)

            return {
                "trades": trades,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "return_pct": return_pct,
                "max_dd": max_dd,
            }
        except Exception:
            return {
                "trades": None, "win_rate": None, "profit_factor": None,
                "return_pct": None, "max_dd": None,
            }

    def _fmt_pct(self, value):
        return "-" if value is None else f"{value:+.2f}%"

    def _fmt_num(self, value):
        if value is None:
            return "-"
        if value == float("inf"):
            return "inf"
        return f"{value:.2f}"

    def _refresh_model_comparison(self):
        if not hasattr(self, "model_tree"):
            return

        sig = self._model_results_signature()
        if self._model_tree_sig == sig and getattr(self, "pipeline_rows", None) and self.model_tree.get_children():
            self._refresh_equity_viewer()
            return

        for item in self.model_tree.get_children():
            self.model_tree.delete(item)

        rows = self._collect_model_rows()
        self.pipeline_rows = {row["iid"]: row for row in rows}
        self._model_tree_sig = sig
        for row in rows:
            values = (
                row["model"],
                row["source"],
                "-" if row.get("trades") is None else f"{row['trades']:,}",
                "-" if row.get("win_rate") is None else f"{row['win_rate']:.1f}%",
                self._fmt_num(row.get("profit_factor")),
                self._fmt_pct(row.get("return_pct")),
                self._fmt_pct(row.get("max_dd")),
                "yes" if row.get("chart_path") else "-",
            )
            self.model_tree.insert("", "end", iid=row["iid"], values=values)

        children = self.model_tree.get_children()
        if children and not self.model_tree.selection():
            self.model_tree.selection_set(children[0])
            self.model_tree.focus(children[0])
        self._refresh_equity_viewer()

    def _selected_pipeline_row(self):
        if not hasattr(self, "model_tree"):
            return None
        selection = self.model_tree.selection()
        if not selection:
            children = self.model_tree.get_children()
            selection = children[:1]
        if not selection:
            return None
        iid = selection[0]
        return getattr(self, "pipeline_rows", {}).get(iid)

    def _equity_path_for_model(self, model):
        candidates = [
            WORK_DIR / f"{model}_live_bt_equity.png",
            WORK_DIR / f"{model}_equity.png",
            WORK_DIR / f"{model}_filtered_equity.png",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _refresh_equity_viewer(self, event=None):
        if not hasattr(self, "pipeline_equity_label"):
            return
        row = self._selected_pipeline_row()
        if not row:
            self.pipeline_equity_image = None
            self._pipeline_equity_image_sig = None
            self.pipeline_equity_label.configure(
                image=None, text="No model results found yet.")
            return

        equity_path = row.get("equity_path") or self._equity_path_for_model(row["model"])
        if not equity_path:
            self.pipeline_equity_image = None
            self._pipeline_equity_image_sig = None
            self.pipeline_equity_label.configure(
                image=None,
                text=f"No equity image found for {row['model']}. Run backtest first.",
                text_color=COLOR_DIM,
            )
            return

        try:
            stat = equity_path.stat()
            image_sig = (str(equity_path), stat.st_mtime_ns, stat.st_size)
            if self._pipeline_equity_image_sig == image_sig and self.pipeline_equity_image is not None:
                self.pipeline_equity_label.configure(
                    image=self.pipeline_equity_image,
                    text="",
                )
                return

            from PIL import Image
            img = Image.open(equity_path)
            img.thumbnail((920, 360), Image.LANCZOS)
            self.pipeline_equity_image = ctk.CTkImage(
                light_image=img,
                dark_image=img,
                size=img.size,
            )
            self._pipeline_equity_image_sig = image_sig
            self.pipeline_equity_label.configure(
                image=self.pipeline_equity_image,
                text="",
            )
        except Exception as e:
            self.pipeline_equity_image = None
            self._pipeline_equity_image_sig = None
            self.pipeline_equity_label.configure(
                image=None,
                text=f"Equity image: {equity_path.name}\nPreview failed: {e}",
                text_color=COLOR_DIM,
            )

    def _open_selected_equity(self):
        row = self._selected_pipeline_row()
        if not row:
            messagebox.showinfo("No selection", "Select a model result first.")
            return
        path = row.get("equity_path") or self._equity_path_for_model(row["model"])
        if not path:
            messagebox.showinfo("No equity image", f"No equity image found for {row['model']}.")
            return
        try:
            os.startfile(str(path))
        except Exception as e:
            messagebox.showerror("Open failed", str(e))

    def _open_selected_chart(self):
        row = self._selected_pipeline_row()
        if not row:
            messagebox.showinfo("No selection", "Select a model result first.")
            return
        path = row.get("chart_path")
        if not path:
            messagebox.showinfo("No HTML chart", f"No HTML chart found for {row['model']}.")
            return
        try:
            os.startfile(str(path))
        except Exception as e:
            messagebox.showerror("Open failed", str(e))

    def _export_pipeline_pdf(self):
        rows = self._collect_model_rows()
        if not rows:
            messagebox.showinfo("No results", "No model/backtest results found to export.")
            return

        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.image as mpimg

            out = WORK_DIR / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            selected = self._selected_pipeline_row() or rows[0]

            with PdfPages(out) as pdf:
                fig, ax = plt.subplots(figsize=(11.69, 8.27))
                ax.axis("off")
                ax.text(0.02, 0.96, "RL Trading Pipeline Report",
                        fontsize=20, weight="bold", transform=ax.transAxes)
                ax.text(0.02, 0.92, f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
                        fontsize=10, color="#555555", transform=ax.transAxes)

                table_rows = []
                for row in rows[:18]:
                    table_rows.append([
                        row["model"],
                        row["source"],
                        "-" if row.get("trades") is None else f"{row['trades']:,}",
                        "-" if row.get("win_rate") is None else f"{row['win_rate']:.1f}%",
                        self._fmt_num(row.get("profit_factor")),
                        self._fmt_pct(row.get("return_pct")),
                        self._fmt_pct(row.get("max_dd")),
                    ])
                table = ax.table(
                    cellText=table_rows,
                    colLabels=["Model", "Source", "Trades", "WR", "PF", "Return", "Max DD"],
                    cellLoc="center",
                    colLoc="center",
                    bbox=[0.02, 0.08, 0.96, 0.78],
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.2)
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

                equity_path = selected.get("equity_path") or self._equity_path_for_model(selected["model"])
                if equity_path and Path(equity_path).exists():
                    fig, ax = plt.subplots(figsize=(11.69, 8.27))
                    ax.axis("off")
                    ax.set_title(f"Equity Curve - {selected['model']}", fontsize=16, weight="bold")
                    img = mpimg.imread(equity_path)
                    ax.imshow(img)
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)

            messagebox.showinfo("PDF exported", f"Saved report:\n{out}")
            try:
                os.startfile(str(out))
            except Exception:
                pass
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    # --------------------------------------------------------
    # PAGE: DATA TOOLS  (NEW!)
    # --------------------------------------------------------
    def _build_tools_page(self):
        page = ctk.CTkFrame(self.content, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        self.pages["tools"] = page

        # Intro
        intro = ctk.CTkLabel(page,
            text="🛠️ Pre-process data จาก MT5 ก่อนเทรน — ทำทุกอย่างที่นี่ ไม่ต้องใช้ command line",
            text_color=COLOR_ACCENT, font=ctk.CTkFont(size=13),
            wraplength=900, justify="left")
        intro.grid(row=0, column=0, sticky="w", padx=8, pady=(0, 16))

        # =====================================================
        # CARD 1: Import from MT5 Tester
        # =====================================================
        c1 = Card(page, title="① 📥 Import from MT5 Tester")
        c1.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        c1.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(c1,
            text="ดึงไฟล์ CSV จาก MT5 Strategy Tester → Convert UTF-16 → UTF-8 → เพิ่ม header → save ที่ project folder",
            text_color=COLOR_DIM, font=ctk.CTkFont(size=12),
            wraplength=900, justify="left"
            ).grid(row=1, column=0, sticky="w", padx=18, pady=(2, 12))

        ctk.CTkLabel(c1, text="MT5 Source CSV", text_color=COLOR_DIM,
                      font=ctk.CTkFont(size=12)
                      ).grid(row=2, column=0, sticky="w", padx=18, pady=(0, 4))

        path_frame = ctk.CTkFrame(c1, fg_color="transparent")
        path_frame.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 8))
        path_frame.grid_columnconfigure(0, weight=1)

        self.tool_mt5_path = ctk.CTkEntry(path_frame,
            placeholder_text="C:\\...\\Tester\\...\\MQL5\\Files\\training_data_v3.csv",
            font=ctk.CTkFont(family="Consolas", size=12))
        self.tool_mt5_path.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        ctk.CTkButton(path_frame, text="📂 Browse",
            command=self._browse_mt5_csv,
            fg_color=COLOR_BG_INPUT, hover_color="#2d333b",
            width=110
        ).grid(row=0, column=1)

        ctk.CTkButton(path_frame, text="🔍 Auto-detect",
            command=self._autodetect_mt5_csv,
            fg_color=COLOR_BG_INPUT, hover_color="#2d333b",
            width=130
        ).grid(row=0, column=2, padx=(8, 0))

        # Output options
        opts = ctk.CTkFrame(c1, fg_color="transparent")
        opts.grid(row=4, column=0, sticky="ew", padx=18, pady=(8, 12))
        opts.grid_columnconfigure(0, weight=1)
        opts.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(opts, text="Output filename", text_color=COLOR_DIM,
                      font=ctk.CTkFont(size=12)
                      ).grid(row=0, column=0, sticky="w", pady=(0, 4))
        ctk.CTkLabel(opts, text="Indicator config", text_color=COLOR_DIM,
                      font=ctk.CTkFont(size=12)
                      ).grid(row=0, column=1, sticky="w", padx=(8, 0), pady=(0, 4))

        self.tool_out_name = ctk.CTkEntry(opts, placeholder_text="training_data_v3.csv")
        self.tool_out_name.insert(0, "training_data_v3.csv")
        self.tool_out_name.grid(row=1, column=0, sticky="ew")

        self.tool_indi_mode = ctk.CTkOptionMenu(opts,
            values=[
                "Auto-detect (recommended)",
                "v3 — Standard (50 features)",
                "v4 — v3 + Candle Patterns (60 features)",
                "All AGGREGATE (4 indicators)",
                "All MULTI (4 periods each)",
                "Custom (manual)",
            ],
            fg_color=COLOR_BG_INPUT, button_color=COLOR_BG_INPUT)
        self.tool_indi_mode.grid(row=1, column=1, sticky="ew", padx=(8, 0))

        ctk.CTkButton(c1, text="📥 Import + Convert + Add Header",
            command=self._import_mt5_csv,
            fg_color=COLOR_ACCENT, hover_color="#4493f8",
            height=40, font=ctk.CTkFont(size=14, weight="bold")
            ).grid(row=5, column=0, sticky="ew", padx=18, pady=(4, 16))

        # =====================================================
        # CARD 2: Train/Test Split by Date
        # =====================================================
        c2 = Card(page, title="② ✂️ Split Train/Test by Date")
        c2.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        c2.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(c2,
            text="แบ่งข้อมูลเป็น Train (อดีต) + Test (อนาคต) — สำหรับ proper out-of-sample validation",
            text_color=COLOR_DIM, font=ctk.CTkFont(size=12),
            wraplength=900, justify="left"
            ).grid(row=1, column=0, sticky="w", padx=18, pady=(2, 12))

        ctk.CTkLabel(c2, text="Source CSV", text_color=COLOR_DIM,
                      font=ctk.CTkFont(size=12)
                      ).grid(row=2, column=0, sticky="w", padx=18, pady=(0, 4))
        self.tool_split_csv = ScrollableOptionMenu(c2, values=["(none)"],
            fg_color=COLOR_BG_INPUT, button_color=COLOR_BG_INPUT)
        self.tool_split_csv.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 12))

        # Date inputs
        dates = ctk.CTkFrame(c2, fg_color="transparent")
        dates.grid(row=4, column=0, sticky="ew", padx=18, pady=(0, 12))
        dates.grid_columnconfigure(0, weight=1)
        dates.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(dates, text="Split Date (YYYY-MM-DD)",
            text_color=COLOR_DIM, font=ctk.CTkFont(size=12)
            ).grid(row=0, column=0, sticky="w", pady=(0, 4))
        ctk.CTkLabel(dates, text="Quick presets",
            text_color=COLOR_DIM, font=ctk.CTkFont(size=12)
            ).grid(row=0, column=1, sticky="w", padx=(8, 0), pady=(0, 4))

        self.tool_split_date = ctk.CTkEntry(dates, placeholder_text="2021-01-01")
        self.tool_split_date.insert(0, "2021-01-01")
        self.tool_split_date.grid(row=1, column=0, sticky="ew")

        self.tool_split_preset = ctk.CTkOptionMenu(dates,
            values=["2021-01-01 (5y test)", "2020-01-01 (6y test)",
                     "2022-01-01 (4y test)", "2023-01-01 (3y test)"],
            command=self._apply_split_preset,
            fg_color=COLOR_BG_INPUT, button_color=COLOR_BG_INPUT)
        self.tool_split_preset.grid(row=1, column=1, sticky="ew", padx=(8, 0))

        # Output names
        out_names = ctk.CTkFrame(c2, fg_color="transparent")
        out_names.grid(row=5, column=0, sticky="ew", padx=18, pady=(0, 12))
        out_names.grid_columnconfigure(0, weight=1)
        out_names.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(out_names, text="Train output suffix",
            text_color=COLOR_DIM, font=ctk.CTkFont(size=12)
            ).grid(row=0, column=0, sticky="w", pady=(0, 4))
        ctk.CTkLabel(out_names, text="Test output suffix",
            text_color=COLOR_DIM, font=ctk.CTkFont(size=12)
            ).grid(row=0, column=1, sticky="w", padx=(8, 0), pady=(0, 4))

        self.tool_train_suffix = ctk.CTkEntry(out_names, placeholder_text="_train")
        self.tool_train_suffix.insert(0, "_train")
        self.tool_train_suffix.grid(row=1, column=0, sticky="ew")
        self.tool_test_suffix = ctk.CTkEntry(out_names, placeholder_text="_test")
        self.tool_test_suffix.insert(0, "_test")
        self.tool_test_suffix.grid(row=1, column=1, sticky="ew", padx=(8, 0))

        # ⭐ Auto-clean checkbox
        clean_frame = ctk.CTkFrame(c2, fg_color="transparent")
        clean_frame.grid(row=6, column=0, sticky="ew", padx=18, pady=(0, 4))
        clean_frame.grid_columnconfigure(0, weight=1)

        self.tool_clean_after_split = ctk.CTkCheckBox(clean_frame,
            text="🧹 Auto-remove redundant features after split",
            text_color=COLOR_DIM, font=ctk.CTkFont(size=12),
            checkbox_height=18, checkbox_width=18)
        self.tool_clean_after_split.grid(row=0, column=0, sticky="w")

        # threshold inline
        ctk.CTkLabel(clean_frame, text="threshold:",
            font=ctk.CTkFont(size=11), text_color=COLOR_DIM
            ).grid(row=0, column=1, sticky="e", padx=(8, 4))
        self.tool_split_clean_thr = ctk.CTkEntry(clean_frame, width=70,
            placeholder_text="0.85", font=ctk.CTkFont(family="Consolas", size=11))
        self.tool_split_clean_thr.insert(0, "0.85")
        self.tool_split_clean_thr.grid(row=0, column=2, sticky="e")

        ctk.CTkLabel(c2,
            text="(เปิดเช็กบ็อกซ์ → ลบ features ที่ correlate กันสูง > threshold หลัง split — ลด overfitting)",
            font=ctk.CTkFont(size=10), text_color=COLOR_DIM
            ).grid(row=7, column=0, sticky="w", padx=18, pady=(0, 8))

        ctk.CTkButton(c2, text="✂️ Split Now",
            command=self._split_csv,
            fg_color=COLOR_ACCENT, hover_color="#4493f8",
            height=40, font=ctk.CTkFont(size=14, weight="bold")
            ).grid(row=8, column=0, sticky="ew", padx=18, pady=(4, 16))

        # =====================================================
        # CARD 3: Relabel
        # =====================================================
        c3 = Card(page, title="③ ⚖️ Relabel (Fix class imbalance)")
        c3.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        c3.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(c3,
            text="ปรับสัดส่วน UP/DOWN/FLAT ให้สมดุล (ป้องกัน lazy classifier trap)",
            text_color=COLOR_DIM, font=ctk.CTkFont(size=12),
            wraplength=900, justify="left"
            ).grid(row=1, column=0, sticky="w", padx=18, pady=(2, 12))

        relabel_grid = ctk.CTkFrame(c3, fg_color="transparent")
        relabel_grid.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 12))
        relabel_grid.grid_columnconfigure(0, weight=1)
        relabel_grid.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(relabel_grid, text="Source CSV", text_color=COLOR_DIM,
            font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky="w", pady=(0, 4))
        ctk.CTkLabel(relabel_grid, text="Method", text_color=COLOR_DIM,
            font=ctk.CTkFont(size=12)).grid(row=0, column=1, sticky="w", padx=(8, 0), pady=(0, 4))

        self.tool_relabel_csv = ScrollableOptionMenu(relabel_grid, values=["(none)"],
            fg_color=COLOR_BG_INPUT, button_color=COLOR_BG_INPUT)
        self.tool_relabel_csv.grid(row=1, column=0, sticky="ew")

        self.tool_relabel_method = ctk.CTkOptionMenu(relabel_grid,
            values=["Quantile 33/33/33 (recommended)",
                     "Fixed threshold ±0.001",
                     "Binary UP/DOWN (drop FLAT)"],
            fg_color=COLOR_BG_INPUT, button_color=COLOR_BG_INPUT)
        self.tool_relabel_method.grid(row=1, column=1, sticky="ew", padx=(8, 0))

        ctk.CTkButton(c3, text="⚖️ Relabel Now",
            command=self._relabel_csv,
            fg_color=COLOR_ACCENT, hover_color="#4493f8",
            height=40, font=ctk.CTkFont(size=14, weight="bold")
            ).grid(row=3, column=0, sticky="ew", padx=18, pady=(4, 16))

        # =====================================================
        # CARD 4: Feature Analysis & Cleanup ⭐ NEW
        # =====================================================
        c4 = Card(page, title="④ 🔬 Feature Analysis & Cleanup")
        c4.grid(row=4, column=0, sticky="ew", pady=(0, 12))
        c4.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(c4,
            text="ตรวจ correlation ระหว่าง features → หา redundant features → ลบทิ้งเพื่อลด NN size + overfitting",
            text_color=COLOR_DIM, font=ctk.CTkFont(size=12),
            wraplength=900, justify="left"
            ).grid(row=1, column=0, sticky="w", padx=18, pady=(2, 12))

        # Source CSV + threshold
        feat_grid = ctk.CTkFrame(c4, fg_color="transparent")
        feat_grid.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 12))
        feat_grid.grid_columnconfigure(0, weight=2)
        feat_grid.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(feat_grid, text="Source CSV", text_color=COLOR_DIM,
            font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky="w", pady=(0, 4))
        ctk.CTkLabel(feat_grid, text="Correlation threshold", text_color=COLOR_DIM,
            font=ctk.CTkFont(size=12)).grid(row=0, column=1, sticky="w", padx=(8, 0), pady=(0, 4))

        self.tool_feat_csv = ScrollableOptionMenu(feat_grid, values=["(none)"],
            fg_color=COLOR_BG_INPUT, button_color=COLOR_BG_INPUT)
        self.tool_feat_csv.grid(row=1, column=0, sticky="ew")

        self.tool_feat_threshold = ctk.CTkEntry(feat_grid,
            placeholder_text="0.85",
            font=ctk.CTkFont(family="Consolas", size=12))
        self.tool_feat_threshold.insert(0, "0.85")
        self.tool_feat_threshold.grid(row=1, column=1, sticky="ew", padx=(8, 0))

        # 2-button row
        btn_row = ctk.CTkFrame(c4, fg_color="transparent")
        btn_row.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 16))
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(1, weight=1)

        ctk.CTkButton(btn_row, text="📊 Show Correlation Matrix",
            command=self._show_correlation,
            fg_color=COLOR_BG_INPUT, hover_color="#2d333b",
            height=40, font=ctk.CTkFont(size=13)
            ).grid(row=0, column=0, sticky="ew", padx=(0, 4))

        ctk.CTkButton(btn_row, text="🧹 Clean Redundant Features",
            command=self._clean_features,
            fg_color=COLOR_ACCENT, hover_color="#4493f8",
            height=40, font=ctk.CTkFont(size=13, weight="bold")
            ).grid(row=0, column=1, sticky="ew", padx=(4, 0))

        # =====================================================
        # CARD 5: Export to MT5 (ONNX) ⭐ NEW
        # =====================================================
        c5 = Card(page, title="⑤ 🚀 Export to MT5 (ONNX)")
        c5.grid(row=5, column=0, sticky="ew", pady=(0, 12))
        c5.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(c5,
            text="แปลง model PPO → ONNX + สร้างไฟล์ MT5 (EA + config + helpers) พร้อมใช้",
            text_color=COLOR_DIM, font=ctk.CTkFont(size=12),
            wraplength=900, justify="left"
            ).grid(row=1, column=0, sticky="w", padx=18, pady=(2, 12))

        # Source model + deploy name
        ex_grid = ctk.CTkFrame(c5, fg_color="transparent")
        ex_grid.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 12))
        ex_grid.grid_columnconfigure(0, weight=2)
        ex_grid.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(ex_grid, text="Source model", text_color=COLOR_DIM,
            font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky="w", pady=(0, 4))
        ctk.CTkLabel(ex_grid, text="Deploy name (filename prefix)",
            text_color=COLOR_DIM,
            font=ctk.CTkFont(size=12)).grid(row=0, column=1, sticky="w", padx=(8, 0), pady=(0, 4))

        self.tool_export_model = ScrollableOptionMenu(ex_grid, values=["(none)"],
            fg_color=COLOR_BG_INPUT, button_color=COLOR_BG_INPUT,
            command=self._on_export_model_change)
        self.tool_export_model.grid(row=1, column=0, sticky="ew")

        self.tool_export_name = ctk.CTkEntry(ex_grid,
            placeholder_text="rl_v10",
            font=ctk.CTkFont(family="Consolas", size=12))
        self.tool_export_name.grid(row=1, column=1, sticky="ew", padx=(8, 0))

        # Output dir
        ctk.CTkLabel(c5, text="Output folder",
            text_color=COLOR_DIM, font=ctk.CTkFont(size=12)
            ).grid(row=3, column=0, sticky="w", padx=18, pady=(0, 4))

        self.tool_export_outdir = ctk.CTkEntry(c5,
            font=ctk.CTkFont(family="Consolas", size=11))
        default_outdir = str(WORK_DIR / "mt5_files" / "MQL5")
        self.tool_export_outdir.insert(0, default_outdir)
        self.tool_export_outdir.grid(row=4, column=0, sticky="ew", padx=18, pady=(0, 8))

        ctk.CTkLabel(c5,
            text="(สร้าง Files/, Include/, Experts/ ภายใน folder นี้ — copy ไป MT5/MQL5/ ได้เลย)",
            font=ctk.CTkFont(size=10), text_color=COLOR_DIM
            ).grid(row=5, column=0, sticky="w", padx=18, pady=(0, 8))

        ctk.CTkButton(c5, text="🚀 Export to ONNX + Generate MT5 Files",
            command=self._export_onnx,
            fg_color=COLOR_PURPLE, hover_color="#8b5cf6",
            height=40, font=ctk.CTkFont(size=14, weight="bold")
            ).grid(row=6, column=0, sticky="ew", padx=18, pady=(4, 16))

        # =====================================================
        # CARD 6: Import from DataCollector_RL (Common\Files)
        # =====================================================
        c6_imp = Card(page, title="⑥ 🤖 Import from DataCollector_RL")
        c6_imp.grid(row=6, column=0, sticky="ew", pady=(0, 12))
        c6_imp.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(c6_imp,
            text="คัดลอก CSV + params.json จาก MT5 Common\\Files → project folder. "
                 "params.json รับประกัน parity ระหว่าง training และ live EA.",
            text_color=COLOR_DIM, font=ctk.CTkFont(size=12),
            wraplength=900, justify="left"
            ).grid(row=1, column=0, sticky="w", padx=18, pady=(2, 12))

        ctk.CTkLabel(c6_imp, text="Source CSV (Common\\Files)", text_color=COLOR_DIM,
                      font=ctk.CTkFont(size=12)
                      ).grid(row=2, column=0, sticky="w", padx=18, pady=(0, 4))

        src_row = ctk.CTkFrame(c6_imp, fg_color="transparent")
        src_row.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 8))
        src_row.grid_columnconfigure(0, weight=1)

        self.tool_collector_src = ScrollableOptionMenu(src_row, values=["(none)"], width=480)
        self.tool_collector_src.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        ctk.CTkButton(src_row, text="🔄 Refresh",
            command=self._refresh_collector_sources,
            fg_color=COLOR_BG_INPUT, hover_color="#2d333b",
            width=110
        ).grid(row=0, column=1)

        # Output name + build option
        opts2 = ctk.CTkFrame(c6_imp, fg_color="transparent")
        opts2.grid(row=4, column=0, sticky="ew", padx=18, pady=(4, 12))
        opts2.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(opts2, text="Output basename", text_color=COLOR_DIM,
                      font=ctk.CTkFont(size=12)
                      ).grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.tool_collector_out = ctk.CTkEntry(opts2,
            placeholder_text="training_data_<symbol>_rl.csv")
        self.tool_collector_out.grid(row=1, column=0, sticky="ew")

        self.tool_collector_build = ctk.CTkCheckBox(opts2,
            text="Run build_training_from_collector.py "
                 "(adds future_return + target, selects features)",
            text_color=COLOR_TEXT)
        self.tool_collector_build.select()
        self.tool_collector_build.grid(row=2, column=0, sticky="w", pady=(12, 0))

        ctk.CTkButton(c6_imp, text="📥 Import + Build training CSV",
            command=self._import_from_collector,
            fg_color=COLOR_ACCENT, hover_color="#4493f8",
            height=40, font=ctk.CTkFont(size=14, weight="bold")
            ).grid(row=5, column=0, sticky="ew", padx=18, pady=(4, 16))

        # =====================================================
        # CARD 7: Log
        # =====================================================
        c7 = Card(page, title="📝 Tools Log")
        c7.grid(row=7, column=0, sticky="ew")
        c7.grid_columnconfigure(0, weight=1)

        log_frame = ctk.CTkFrame(c7, fg_color="#0a0e14", corner_radius=8,
                                   border_width=1, border_color="#30363d")
        log_frame.grid(row=1, column=0, sticky="ew", padx=18, pady=(8, 16))

        self.tools_log = self._make_log_widget(log_frame, height=10)

        # populate collector sources after log widget is ready
        self._refresh_collector_sources()

    # --------------------------------------------------------
    # Tools functions
    # --------------------------------------------------------
    def _common_files_dir(self):
        """MT5 Common\\Files folder (where DataCollector_RL writes)."""
        return Path.home() / "AppData" / "Roaming" / "MetaQuotes" / "Terminal" \
               / "Common" / "Files"

    def _refresh_collector_sources(self):
        """List CSV files in Common\\Files for the import dropdown."""
        d = self._common_files_dir()
        files = []
        try:
            if d.exists():
                files = sorted([p.name for p in d.glob("*.csv")])
        except Exception:
            pass
        values = files or ["(none)"]
        try:
            self.tool_collector_src.configure(values=values)
            if self.tool_collector_src.get() in ("", "(none)"):
                self.tool_collector_src.set(values[0])
            # default output name from selected
            sel = self.tool_collector_src.get()
            if sel and sel != "(none)" and not self.tool_collector_out.get().strip():
                stem = Path(sel).stem
                self.tool_collector_out.delete(0, "end")
                self.tool_collector_out.insert(0, f"training_data_{stem}.csv")
        except Exception:
            pass
        self._log(self.tools_log,
                  f"[collector] {len(files)} CSV(s) in Common\\Files", "info")

    def _import_from_collector(self):
        """Copy CSV + params.json from Common\\Files → project, then
           optionally run build_training_from_collector.py."""
        if self._is_process_busy():
            messagebox.showwarning("Busy", "Another task is running")
            return

        src_name = self.tool_collector_src.get().strip()
        if not src_name or src_name == "(none)":
            messagebox.showwarning("No source",
                "Pick a CSV from Common\\Files (or check that DataCollector_RL has run)")
            return

        src_csv = self._common_files_dir() / src_name
        if not src_csv.exists():
            messagebox.showerror("Not found", f"File missing: {src_csv}")
            return

        src_params = src_csv.with_suffix(".params.json")
        out_name = (self.tool_collector_out.get().strip() or
                    f"training_data_{Path(src_name).stem}.csv")
        run_build = bool(self.tool_collector_build.get())

        import threading
        threading.Thread(
            target=self._import_from_collector_worker,
            args=(src_csv, src_params, out_name, run_build),
            daemon=True,
        ).start()

    def _import_from_collector_worker(self, src_csv, src_params, out_name, run_build):
        import shutil
        try:
            self.after(0, lambda: self.status_label.configure(
                text="Importing", text_color=COLOR_YELLOW))

            # 1) Copy raw CSV
            dst_csv = WORK_DIR / Path(out_name).with_suffix(".csv").name
            # If user said "training_data_xxx.csv" and we will run build,
            # the raw copy should land under a different name to keep both.
            if run_build:
                raw_copy = WORK_DIR / src_csv.name   # keep original filename
            else:
                raw_copy = dst_csv
            shutil.copy(src_csv, raw_copy)
            self.after(0, lambda p=raw_copy: self._log(
                self.tools_log, f"[copy] CSV  -> {p.name}", "success"))

            # 2) Copy params.json (if exists)
            if src_params.exists():
                dst_params = raw_copy.with_suffix(".params.json")
                shutil.copy(src_params, dst_params)
                self.after(0, lambda p=dst_params: self._log(
                    self.tools_log, f"[copy] params -> {p.name}", "success"))
            else:
                self.after(0, lambda: self._log(self.tools_log,
                    "[warn] no params.json sidecar — EA will use RL_Indicators defaults",
                    "warn"))

            # 3) Optionally run build_training_from_collector
            if run_build:
                cmd = [sys.executable, "build_training_from_collector.py",
                       "--in", str(raw_copy), "--out", out_name]
                self.after(0, lambda c=cmd: self._log(
                    self.tools_log, "$ " + " ".join(c), "info"))
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    cwd=str(WORK_DIR), text=True, encoding="utf-8",
                    errors="replace", bufsize=1, env=env)
                for line in iter(proc.stdout.readline, ""):
                    line = line.rstrip()
                    if not line: continue
                    tag = self._classify_log(line)
                    self.after(0, lambda l=line, t=tag: self._log(self.tools_log, l, t))
                proc.wait()
                if proc.returncode != 0:
                    raise RuntimeError(f"build script exit {proc.returncode}")

            self.after(0, lambda: self._refresh_dropdowns())
            self.after(0, lambda: self._log(self.tools_log,
                "[done] dataset ready for training", "success"))
        except Exception as e:
            self.after(0, lambda err=e: self._log(
                self.tools_log, f"[error] {err}", "error"))
        finally:
            self.after(0, lambda: self.status_label.configure(
                text="Idle", text_color=COLOR_DIM))

    def _browse_mt5_csv(self):
        path = filedialog.askopenfilename(
            title="Select MT5 CSV (UTF-16)",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=str(Path.home() / "AppData" / "Roaming" / "MetaQuotes"))
        if path:
            self.tool_mt5_path.delete(0, "end")
            self.tool_mt5_path.insert(0, path)

    def _autodetect_mt5_csv(self):
        """Search MT5 Tester folders for recent CSVs"""
        self._log(self.tools_log, "Searching MT5 Tester folders...", "info")
        mt5_root = Path.home() / "AppData" / "Roaming" / "MetaQuotes"
        if not mt5_root.exists():
            self._log(self.tools_log, "MT5 root not found", "error")
            return

        # Search recent CSVs
        candidates = []
        for csv in mt5_root.glob("**/MQL5/Files/*.csv"):
            stat = csv.stat()
            candidates.append((stat.st_mtime, csv))
        for csv in mt5_root.glob("**/Tester/Files/*.csv"):
            stat = csv.stat()
            candidates.append((stat.st_mtime, csv))

        if not candidates:
            self._log(self.tools_log, "No CSV found in MT5 folders", "warn")
            return

        # Sort by mtime, newest first
        candidates.sort(reverse=True)
        newest = candidates[0][1]
        size_mb = newest.stat().st_size / 1024 / 1024
        self._log(self.tools_log,
            f"Found {len(candidates)} CSVs · newest: {newest.name} ({size_mb:.1f} MB)",
            "success")

        self.tool_mt5_path.delete(0, "end")
        self.tool_mt5_path.insert(0, str(newest))

    def _import_mt5_csv(self):
        """Import + convert + add header in one step"""
        threading.Thread(target=self._import_mt5_csv_worker, daemon=True).start()

    def _import_mt5_csv_worker(self):
        try:
            src = self.tool_mt5_path.get().strip()
            if not src or not Path(src).exists():
                self._log(self.tools_log, "Source file not found", "error")
                return

            out_name = self.tool_out_name.get().strip() or "training_data_v3.csv"
            dst = WORK_DIR / out_name

            self._log(self.tools_log, f"Reading source ({Path(src).stat().st_size / 1024 / 1024:.1f} MB)...", "info")

            # Detect encoding (UTF-16 starts with FF FE or FE FF)
            with open(src, 'rb') as f:
                magic = f.read(2)
            is_utf16 = magic in (b'\xff\xfe', b'\xfe\xff')

            if is_utf16:
                self._log(self.tools_log, "Detected UTF-16 → converting to UTF-8...", "info")
                # Read all bytes and decode
                bytes_data = Path(src).read_bytes()
                if magic == b'\xff\xfe':
                    text = bytes_data.decode('utf-16-le').lstrip('﻿')
                else:
                    text = bytes_data.decode('utf-16-be').lstrip('﻿')
            else:
                self._log(self.tools_log, "Reading as UTF-8...", "info")
                text = Path(src).read_text(encoding='utf-8-sig')

            # Check if has header
            first_line = text.split('\n')[0]
            has_header = 'timestamp' in first_line.lower() or 'symbol' in first_line.lower()

            # Add header if missing
            if not has_header:
                self._log(self.tools_log, "No header detected, adding...", "info")
                first_cols = first_line.count(',') + 1
                mode = self.tool_indi_mode.get()

                # Auto-detect by column count
                if mode.startswith("Auto"):
                    if first_cols == 57:
                        mode = "v3"
                        self._log(self.tools_log,
                            f"  Auto-detected: v3 standard (57 cols)", "info")
                    elif first_cols == 67:
                        mode = "v4"
                        self._log(self.tools_log,
                            f"  Auto-detected: v4 with candle patterns (67 cols)", "info")
                    else:
                        self._log(self.tools_log,
                            f"  ⚠️ Unknown column count: {first_cols}",
                            "warn")
                        self._log(self.tools_log,
                            f"  Using generic header f0,f1,...", "warn")
                        mode = "generic"

                header = self._build_header_for_mode(mode, first_cols)

                # Verify column count
                if len(header) != first_cols:
                    self._log(self.tools_log,
                        f"⚠️ Column mismatch: header has {len(header)} but data has {first_cols}",
                        "warn")
                    self._log(self.tools_log,
                        "Falling back to generic column names f0,f1,f2...",
                        "warn")
                    header = ['timestamp', 'symbol'] + \
                             [f'f{i}' for i in range(first_cols - 4)] + \
                             ['future_return', 'target']
                text = ','.join(header) + '\n' + text
                self._log(self.tools_log,
                    f"  ✓ Header added: {len(header)} columns", "success")

            # Write
            dst.write_text(text, encoding='utf-8')
            size_mb = dst.stat().st_size / 1024 / 1024
            self._log(self.tools_log, f"✓ Saved: {dst.name} ({size_mb:.1f} MB)", "success")

            # Quick verify with pandas
            import pandas as pd
            df = pd.read_csv(dst, nrows=5)
            self._log(self.tools_log,
                f"  Columns: {len(df.columns)} · Sample target values: {df['target'].tolist() if 'target' in df.columns else 'N/A'}",
                "metric")

            # Refresh dropdowns
            self._refresh_dropdowns()
            self._refresh_tools_dropdowns()

        except Exception as e:
            import traceback
            self._log(self.tools_log, f"Error: {e}", "error")
            self._log(self.tools_log, traceback.format_exc(), "error")

    def _build_header_for_mode(self, mode, n_cols=None):
        """Build header columns for indicator mode preset.

        Supported modes:
          - "v3"      → 57 columns (DataCollector_v3.mq5 default)
          - "v4"      → 67 columns (v3 + 10 candle patterns)
          - "generic" → fallback f0, f1, ... when count unknown
          - else      → uses keyword in mode string to detect v3/v4
        """
        # === Common base (timestamp + symbol + OHLCV) ===
        meta = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]

        # === v3 indicator features (50 cols total without OHLCV/label) ===
        v3_features = [
            "rsi_min", "rsi_max", "rsi_mean", "rsi_std",
            "ema_20", "ema_50", "ema_100", "ema_200",
            "atr_min", "atr_max", "atr_mean", "atr_std",
            "stoch_min", "stoch_max", "stoch_mean", "stoch_std",
            "cci_min", "cci_max", "cci_mean", "cci_std",
            "wpr_min", "wpr_max", "wpr_mean", "wpr_std",
            "adx_min", "adx_max", "adx_mean", "adx_std",
            "ema_long", "macd", "macd_signal", "macd_hist", "bb_position",
            "hour", "dow", "session_london", "session_ny", "session_asia",
            "ret_1", "ret_3", "ret_5", "ret_10", "ret_20",
            "close_zscore", "pct_rank", "sharpe_20", "hl_range", "body_size",
        ]

        # === v4 candle pattern features (10 cols, added before label) ===
        candle_features = [
            "candle_hammer", "candle_engulfing",
            "candle_inside", "candle_outside",
            "candle_star", "candle_soldiers",
            "candle_marubozu", "candle_harami",
            "candle_piercing", "candle_mathold",
        ]

        label = ["future_return", "target"]

        # === Mode resolution ===
        mode_lc = mode.lower()
        if "v4" in mode_lc or "candle" in mode_lc:
            return meta + v3_features + candle_features + label
        elif "v3" in mode_lc or "default" in mode_lc or "aggregate" in mode_lc or "multi" in mode_lc:
            return meta + v3_features + label
        elif "generic" in mode_lc or "custom" in mode_lc:
            # Generic fallback: f0, f1, ...
            n = n_cols if n_cols else 57
            return meta[:2] + [f"f{i}" for i in range(n - 4)] + label

        # Default → v3 (most common)
        return meta + v3_features + label

    def _apply_split_preset(self, choice):
        """Apply quick preset to date entry"""
        date = choice.split()[0]
        self.tool_split_date.delete(0, "end")
        self.tool_split_date.insert(0, date)

    def _split_csv(self):
        threading.Thread(target=self._split_csv_worker, daemon=True).start()

    def _split_csv_worker(self):
        try:
            csv = self.tool_split_csv.get()
            if csv in ("(none)", ""):
                self._log(self.tools_log, "Please select a CSV", "error")
                return

            split_date = self.tool_split_date.get().strip()
            if not split_date:
                self._log(self.tools_log, "Please enter split date", "error")
                return

            train_suffix = self.tool_train_suffix.get().strip() or "_train"
            test_suffix = self.tool_test_suffix.get().strip() or "_test"

            self._log(self.tools_log, f"Loading {csv}...", "info")
            import pandas as pd
            df = self._smart_read_csv(WORK_DIR / csv)

            if 'timestamp' not in df.columns:
                self._log(self.tools_log, "No 'timestamp' column found", "error")
                return

            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
            if df.empty:
                self._log(self.tools_log, "No valid timestamp rows after parsing", "error")
                return

            split_dt = pd.to_datetime(split_date)

            train_df = df[df['timestamp'] < split_dt]
            test_df = df[df['timestamp'] >= split_dt]

            if train_df.empty or test_df.empty:
                first_date = df['timestamp'].min().date()
                last_date = df['timestamp'].max().date()
                self._log(self.tools_log,
                    "Split aborted: train/test would be empty.", "error")
                self._log(self.tools_log,
                    f"  Data range : {first_date} -> {last_date}", "metric")
                self._log(self.tools_log,
                    f"  Train rows : {len(train_df):,}",
                    "error" if train_df.empty else "metric")
                self._log(self.tools_log,
                    f"  Test rows  : {len(test_df):,}",
                    "error" if test_df.empty else "metric")
                self._log(self.tools_log,
                    "Choose a split date inside the data range.", "warn")
                return

            self._log(self.tools_log,
                f"Total: {len(df):,} rows | Split at {split_date}", "info")
            self._log(self.tools_log,
                f"  Train: {len(train_df):,} rows ({df['timestamp'].min().date()} → {(split_dt - pd.Timedelta(days=1)).date()})",
                "metric")
            self._log(self.tools_log,
                f"  Test : {len(test_df):,} rows ({split_dt.date()} → {df['timestamp'].max().date()})",
                "metric")

            # Save
            base = Path(csv).stem
            train_path = WORK_DIR / f"{base}{train_suffix}.csv"
            test_path  = WORK_DIR / f"{base}{test_suffix}.csv"

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            self._log(self.tools_log, f"✓ Saved: {train_path.name}", "success")
            self._log(self.tools_log, f"✓ Saved: {test_path.name}", "success")

            # ⭐ Auto-clean redundant features if checkbox is checked
            if self.tool_clean_after_split.get():
                try:
                    threshold = float(self.tool_split_clean_thr.get() or "0.85")
                except ValueError:
                    threshold = 0.85

                self._log(self.tools_log,
                    f"\n=== Auto-cleaning redundant features (corr > {threshold}) ===",
                    "info")

                # Compute kept features from TRAIN file (use same selection for both)
                train_features = self._get_feature_columns(train_df)
                kept, dropped, reasons = self._greedy_correlation_prune(
                    train_df, train_features, threshold)

                if dropped:
                    self._log(self.tools_log,
                        f"  Dropped {len(dropped)} features  (kept {len(kept)})",
                        "warn")
                    for col in dropped[:10]:
                        kept_col, r = reasons[col]
                        self._log(self.tools_log,
                            f"    - {col:25s}  (r={r:.3f} with {kept_col})",
                            "warn")
                    if len(dropped) > 10:
                        self._log(self.tools_log,
                            f"    ... +{len(dropped) - 10} more", "warn")

                    # Apply same column selection to both files
                    non_feat_train = [c for c in train_df.columns if c not in train_features]
                    out_cols_train = non_feat_train + kept

                    non_feat_test  = [c for c in test_df.columns if c not in train_features]
                    # filter: only kept that exist in test (should be all)
                    kept_in_test = [c for c in kept if c in test_df.columns]
                    out_cols_test = non_feat_test + kept_in_test

                    train_df[out_cols_train].to_csv(train_path, index=False)
                    test_df[out_cols_test].to_csv(test_path, index=False)

                    self._log(self.tools_log,
                        f"  ✓ Re-saved: {train_path.name} ({len(out_cols_train)} cols)",
                        "success")
                    self._log(self.tools_log,
                        f"  ✓ Re-saved: {test_path.name} ({len(out_cols_test)} cols)",
                        "success")
                else:
                    self._log(self.tools_log,
                        "  No redundant features found — files unchanged",
                        "info")

            self._refresh_dropdowns()
            self._refresh_tools_dropdowns()

        except Exception as e:
            import traceback
            self._log(self.tools_log, f"Error: {e}", "error")
            self._log(self.tools_log, traceback.format_exc(), "error")

    def _relabel_csv(self):
        threading.Thread(target=self._relabel_csv_worker, daemon=True).start()

    def _relabel_csv_worker(self):
        try:
            csv = self.tool_relabel_csv.get()
            if csv in ("(none)", ""):
                self._log(self.tools_log, "Please select a CSV", "error")
                return

            method = self.tool_relabel_method.get()

            self._log(self.tools_log, f"Loading {csv}...", "info")
            import pandas as pd
            import numpy as np

            df = self._smart_read_csv(WORK_DIR / csv)

            if 'future_return' not in df.columns:
                self._log(self.tools_log, "No 'future_return' column found", "error")
                return

            self._log(self.tools_log, f"Original distribution:", "info")
            if 'target' in df.columns:
                vc = df['target'].value_counts()
                for k, v in vc.items():
                    pct = v / len(df) * 100
                    self._log(self.tools_log, f"  {k}: {v:,} ({pct:.1f}%)", "info")

            fr = df['future_return']

            if method.startswith("Quantile"):
                q33 = fr.quantile(0.33)
                q67 = fr.quantile(0.67)
                self._log(self.tools_log,
                    f"Quantile thresholds: q33={q33:.5f}, q67={q67:.5f}", "info")
                df['target'] = pd.cut(fr,
                    bins=[-np.inf, q33, q67, np.inf],
                    labels=["DOWN", "FLAT", "UP"])
            elif method.startswith("Fixed"):
                df['target'] = "FLAT"
                df.loc[fr >= 0.001, 'target'] = "UP"
                df.loc[fr <= -0.001, 'target'] = "DOWN"
            elif method.startswith("Binary"):
                t = fr.abs().median()
                df = df[fr.abs() > t].copy()
                df['target'] = "DOWN"
                df.loc[df['future_return'] > 0, 'target'] = "UP"
                self._log(self.tools_log,
                    f"Filtered: kept {len(df):,} rows with |return| > {t:.5f}", "info")

            self._log(self.tools_log, f"\nNew distribution:", "info")
            vc = df['target'].value_counts()
            for k, v in vc.items():
                pct = v / len(df) * 100
                self._log(self.tools_log, f"  {k}: {v:,} ({pct:.1f}%)", "metric")

            # Save
            base = Path(csv).stem
            out_path = WORK_DIR / f"{base}_relabeled.csv"
            df.to_csv(out_path, index=False)
            self._log(self.tools_log, f"\n✓ Saved: {out_path.name}", "success")

            self._refresh_dropdowns()
            self._refresh_tools_dropdowns()

        except Exception as e:
            import traceback
            self._log(self.tools_log, f"Error: {e}", "error")
            self._log(self.tools_log, traceback.format_exc(), "error")

    def _refresh_tools_dropdowns(self):
        csvs = self._list_csv_files()
        menus = [self.tool_split_csv, self.tool_relabel_csv]
        if hasattr(self, 'tool_feat_csv'):
            menus.append(self.tool_feat_csv)
        for menu in menus:
            try:
                menu.configure(values=csvs)
                if menu.get() in ("(none)", "") and csvs[0] != "(none)":
                    menu.set(csvs[0])
            except: pass

        # Refresh model dropdown for export (find .zip + _best/)
        if hasattr(self, 'tool_export_model'):
            model_list = self._list_model_names()
            try:
                self.tool_export_model.configure(values=model_list)
                if self.tool_export_model.get() in ("(none)", "") and model_list[0] != "(none)":
                    self.tool_export_model.set(model_list[0])
            except: pass

    def _on_export_model_change(self, choice):
        """When user picks a model, default deploy_name to the same"""
        if choice in ("(none)", ""):
            return
        # Don't overwrite if user has typed something
        current = self.tool_export_name.get().strip()
        if not current or current in ("rl_v10", ""):
            self.tool_export_name.delete(0, "end")
            self.tool_export_name.insert(0, choice)

    def _export_onnx(self):
        threading.Thread(target=self._export_onnx_worker, daemon=True).start()

    def _export_onnx_worker(self):
        try:
            model_name = self.tool_export_model.get()
            if model_name in ("(none)", ""):
                self._log(self.tools_log, "Please select a source model", "error")
                return

            deploy_name = self.tool_export_name.get().strip() or model_name
            outdir = self.tool_export_outdir.get().strip()
            if not outdir:
                outdir = str(WORK_DIR / "mt5_files" / "MQL5")

            self._log(self.tools_log,
                f"\n=== Exporting {model_name} -> {deploy_name} ===", "info")
            self._log(self.tools_log, f"Output: {outdir}", "info")

            # Verify model exists
            zip_paths = [
                WORK_DIR / f"{model_name}_best" / "best_model.zip",
                WORK_DIR / f"{model_name}.zip",
            ]
            model_path = None
            for p in zip_paths:
                if p.exists():
                    model_path = p
                    break
            if model_path is None:
                self._log(self.tools_log,
                    f"❌ Model not found: {model_name}", "error")
                return
            self._log(self.tools_log, f"Found model: {model_path.name}", "info")

            # Run export script as subprocess (so it inherits clean python env)
            cmd = [
                sys.executable, "export_to_onnx.py",
                model_name,
                "--name", deploy_name,
                "--output_dir", outdir,
            ]
            self._log(self.tools_log, f"$ {' '.join(cmd)}", "info")

            import subprocess
            proc = subprocess.Popen(cmd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=str(WORK_DIR), text=True,
                bufsize=1, encoding='utf-8', errors='replace')

            for line in proc.stdout:
                line = line.rstrip()
                tag = "info"
                if "✅" in line or "[ok]" in line.lower(): tag = "success"
                elif "❌" in line or "error" in line.lower(): tag = "error"
                elif "⚠️" in line or "warn" in line.lower(): tag = "warn"
                elif "[" in line and "]" in line: tag = "metric"
                self._log(self.tools_log, line, tag)

            proc.wait()

            if proc.returncode == 0:
                # Show files generated
                out_path = Path(outdir)
                self._log(self.tools_log,
                    f"\n✅ Export complete! Files in {out_path}", "success")
                self._log(self.tools_log,
                    f"  Files/{deploy_name}.onnx", "metric")
                self._log(self.tools_log,
                    f"  Include/{deploy_name}_config.mqh", "metric")
                self._log(self.tools_log,
                    f"  Include/RL_Indicators.mqh", "metric")
                self._log(self.tools_log,
                    f"  Experts/{deploy_name}_EA.mq5", "metric")
                self._log(self.tools_log,
                    f"\n📋 Next: copy {out_path}/* to MT5's MQL5/ folder",
                    "info")
            else:
                self._log(self.tools_log,
                    f"❌ Export failed (exit code {proc.returncode})",
                    "error")

        except Exception as e:
            import traceback
            self._log(self.tools_log, f"Error: {e}", "error")
            self._log(self.tools_log, traceback.format_exc(), "error")

    # --------------------------------------------------------
    # ⭐ Smart CSV reader — auto-detect UTF-16 / UTF-8
    # --------------------------------------------------------
    def _smart_read_csv(self, path, **kwargs):
        """Read CSV with auto encoding detection.
        Handles raw MT4/MT5 UTF-16 LE files transparently.
        If file is UTF-16, converts in-place and saves as UTF-8."""
        import pandas as pd
        path = Path(path)
        try:
            # Try UTF-8 first (fast path)
            return pd.read_csv(path, **kwargs)
        except UnicodeDecodeError:
            pass

        # Check encoding
        with open(path, 'rb') as f:
            magic = f.read(2)

        if magic == b'\xff\xfe':
            encoding = 'utf-16-le'
        elif magic == b'\xfe\xff':
            encoding = 'utf-16-be'
        else:
            # Last-resort try cp1252 / latin-1
            encoding = 'cp1252'

        self._log(self.tools_log,
            f"⚠️  {path.name} is {encoding} — converting to UTF-8...",
            "warn")

        # Read raw bytes, decode, strip BOM
        text = path.read_bytes().decode(encoding, errors='replace').lstrip('﻿')

        # Save as UTF-8 (overwrite)
        backup = path.with_suffix(path.suffix + '.utf16_backup')
        if not backup.exists():
            path.rename(backup)
            self._log(self.tools_log,
                f"  Original backed up as: {backup.name}", "info")
        path.write_text(text, encoding='utf-8')
        self._log(self.tools_log,
            f"  ✓ Converted to UTF-8 ({path.stat().st_size / 1024 / 1024:.1f} MB)",
            "success")

        # Read again
        return pd.read_csv(path, **kwargs)

    # --------------------------------------------------------
    # ⭐ Feature analysis helpers
    # --------------------------------------------------------
    def _get_feature_columns(self, df):
        """Get numeric feature columns (skip OHLCV + leaky)"""
        import pandas as pd
        skip = {"timestamp", "symbol", "ticker", "open", "high", "low",
                "close", "volume"}
        leaky_kw = ("future_", "forward_", "next_", "target")
        return [c for c in df.columns
                if c not in skip
                and not any(k in c.lower() for k in leaky_kw)
                and pd.api.types.is_numeric_dtype(df[c])]

    def _greedy_correlation_prune(self, df, features, threshold):
        """Greedy: drop feature with highest avg correlation in each high-corr pair.
        Returns: (kept_features, dropped_features, drop_reasons)"""
        import numpy as np
        kept = list(features)
        dropped = []
        reasons = {}  # dropped_col -> (kept_col, corr_value)

        while True:
            corr = df[kept].corr().abs()
            np.fill_diagonal(corr.values, 0)
            max_val = corr.values.max()
            if max_val <= threshold:
                break
            # find highest pair
            i, j = np.unravel_index(np.argmax(corr.values), corr.values.shape)
            c1, c2 = corr.columns[i], corr.columns[j]
            # pick which to drop = higher avg correlation to others
            avg_c1 = corr[c1].mean()
            avg_c2 = corr[c2].mean()
            drop_col = c1 if avg_c1 > avg_c2 else c2
            keep_col = c2 if drop_col == c1 else c1
            kept.remove(drop_col)
            dropped.append(drop_col)
            reasons[drop_col] = (keep_col, max_val)
        return kept, dropped, reasons

    def _show_correlation(self):
        threading.Thread(target=self._show_correlation_worker, daemon=True).start()

    def _show_correlation_worker(self):
        try:
            csv = self.tool_feat_csv.get()
            if csv in ("(none)", ""):
                self._log(self.tools_log, "Please select a CSV", "error")
                return

            try:
                threshold = float(self.tool_feat_threshold.get() or "0.85")
            except ValueError:
                threshold = 0.85

            self._log(self.tools_log, f"Loading {csv}...", "info")
            import pandas as pd
            import numpy as np

            df = self._smart_read_csv(WORK_DIR / csv)
            features = self._get_feature_columns(df)
            self._log(self.tools_log,
                f"Found {len(features)} numeric features (after dropping OHLCV + leaky)",
                "info")

            if len(features) < 2:
                self._log(self.tools_log, "Not enough features to analyze", "error")
                return

            # Compute correlation
            self._log(self.tools_log, "Computing correlation matrix...", "info")
            corr = df[features].corr().abs()

            # Find high-correlation pairs
            pairs = []
            for i, c1 in enumerate(features):
                for j, c2 in enumerate(features):
                    if i >= j:
                        continue
                    r = corr.loc[c1, c2]
                    if r > threshold:
                        pairs.append((r, c1, c2))
            pairs.sort(reverse=True)

            # Print top 30 pairs
            self._log(self.tools_log,
                f"\n[Top correlated pairs] (threshold > {threshold}):",
                "metric")
            self._log(self.tools_log,
                f"  Found {len(pairs)} pairs above threshold", "metric")
            for r, c1, c2 in pairs[:30]:
                self._log(self.tools_log,
                    f"  r={r:.3f}  {c1:25s} <-> {c2}", "info")
            if len(pairs) > 30:
                self._log(self.tools_log,
                    f"  ... +{len(pairs) - 30} more", "info")

            # Save heatmap
            self._log(self.tools_log, "\nGenerating heatmap PNG...", "info")
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            n = len(features)
            fig_size = max(10, min(24, n * 0.3))
            fig, ax = plt.subplots(figsize=(fig_size, fig_size))
            im = ax.imshow(corr.values, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')

            ax.set_xticks(range(n))
            ax.set_yticks(range(n))
            ax.set_xticklabels(features, rotation=90, fontsize=8)
            ax.set_yticklabels(features, fontsize=8)

            # Annotate high-corr cells
            for i in range(n):
                for j in range(n):
                    if i != j and corr.values[i, j] > threshold:
                        ax.text(j, i, f"{corr.values[i,j]:.2f}",
                                ha='center', va='center',
                                color='white' if corr.values[i,j] > 0.6 else 'black',
                                fontsize=6)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f"Correlation Matrix — {csv} ({n} features, threshold={threshold})",
                          fontsize=14)
            plt.tight_layout()

            png_path = WORK_DIR / f"{Path(csv).stem}_correlation.png"
            plt.savefig(png_path, dpi=110, bbox_inches='tight')
            plt.close(fig)
            self._log(self.tools_log, f"✓ Saved: {png_path.name}", "success")

            # Auto-open
            try:
                import os
                os.startfile(str(png_path))
                self._log(self.tools_log, "Opened in default image viewer", "info")
            except Exception:
                self._log(self.tools_log,
                    f"Open manually: {png_path}", "info")

        except Exception as e:
            import traceback
            self._log(self.tools_log, f"Error: {e}", "error")
            self._log(self.tools_log, traceback.format_exc(), "error")

    def _clean_features(self):
        threading.Thread(target=self._clean_features_worker, daemon=True).start()

    def _clean_features_worker(self):
        try:
            csv = self.tool_feat_csv.get()
            if csv in ("(none)", ""):
                self._log(self.tools_log, "Please select a CSV", "error")
                return

            try:
                threshold = float(self.tool_feat_threshold.get() or "0.85")
            except ValueError:
                threshold = 0.85

            self._log(self.tools_log,
                f"\n=== Cleaning redundant features (corr > {threshold}) ===", "info")
            self._log(self.tools_log, f"Loading {csv}...", "info")
            import pandas as pd
            df = self._smart_read_csv(WORK_DIR / csv)
            features = self._get_feature_columns(df)
            self._log(self.tools_log,
                f"Original: {len(features)} features", "info")

            kept, dropped, reasons = self._greedy_correlation_prune(
                df, features, threshold)

            if not dropped:
                self._log(self.tools_log,
                    "✓ No redundant features found — already clean!",
                    "success")
                return

            self._log(self.tools_log,
                f"\nDropped {len(dropped)} features:", "warn")
            for col in dropped:
                kept_col, r = reasons[col]
                self._log(self.tools_log,
                    f"  - {col:25s}  (r={r:.3f} with {kept_col})", "warn")

            self._log(self.tools_log,
                f"\nKept {len(kept)} features:", "metric")
            for col in kept:
                self._log(self.tools_log, f"  + {col}", "info")

            # Save: keep all non-feature cols + kept features
            non_feature = [c for c in df.columns if c not in features]
            out_cols = non_feature + kept
            cleaned_df = df[out_cols]

            out_name = f"{Path(csv).stem}_clean.csv"
            out_path = WORK_DIR / out_name
            cleaned_df.to_csv(out_path, index=False)

            size_mb = out_path.stat().st_size / 1024 / 1024
            self._log(self.tools_log,
                f"\n✓ Saved: {out_name} ({size_mb:.1f} MB · {len(out_cols)} cols)",
                "success")
            self._log(self.tools_log,
                f"  Reduction: {len(features)} → {len(kept)} features ({(len(kept)/len(features))*100:.0f}% kept)",
                "metric")

            self._refresh_dropdowns()
            self._refresh_tools_dropdowns()

        except Exception as e:
            import traceback
            self._log(self.tools_log, f"Error: {e}", "error")
            self._log(self.tools_log, traceback.format_exc(), "error")

    # --------------------------------------------------------
    # PAGE: TRAIN
    # --------------------------------------------------------
    def _build_train_page(self):
        page = ctk.CTkFrame(self.content, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        self.pages["train"] = page

        # === Dataset card ===
        c1 = Card(page, title="📂 Dataset")
        c1.grid(row=0, column=0, sticky="ew", pady=(0, 12))

        file_frame = ctk.CTkFrame(c1, fg_color=COLOR_BG_INPUT, corner_radius=8,
                                    border_width=1, border_color="#30363d")
        file_frame.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 16))
        file_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(file_frame, text="📊", font=ctk.CTkFont(size=28)
                      ).grid(row=0, column=0, padx=14, pady=12)

        info_frame = ctk.CTkFrame(file_frame, fg_color="transparent")
        info_frame.grid(row=0, column=1, sticky="w", padx=4, pady=12)
        self.train_csv_label = ctk.CTkLabel(info_frame, text="No file selected",
            font=ctk.CTkFont(size=13, weight="bold"))
        self.train_csv_label.pack(anchor="w")
        self.train_csv_meta = ctk.CTkLabel(info_frame, text="Click Browse to select",
            font=ctk.CTkFont(size=11), text_color=COLOR_DIM)
        self.train_csv_meta.pack(anchor="w", pady=(2, 0))

        ctk.CTkButton(file_frame, text="📂 Browse",
            command=self._browse_train_csv,
            fg_color=COLOR_ACCENT, hover_color="#4493f8",
            width=110
        ).grid(row=0, column=2, padx=14, pady=12)

        # === Settings: 2-column ===
        settings = ctk.CTkFrame(page, fg_color="transparent")
        settings.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        settings.grid_columnconfigure(0, weight=1)
        settings.grid_columnconfigure(1, weight=1)

        # Left: Mode
        c2 = Card(settings, title="⚙️ Training Mode")
        c2.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        c2.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(c2, text="Algorithm", text_color=COLOR_DIM,
                      font=ctk.CTkFont(size=12)
                      ).grid(row=1, column=0, sticky="w", padx=18, pady=(8, 4))
        self.train_algo = ctk.CTkOptionMenu(c2,
            values=["PPO (recommended)", "DQN", "A2C"],
            fg_color=COLOR_BG_INPUT, button_color=COLOR_BG_INPUT,
            button_hover_color="#2d333b")
        self.train_algo.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 12))

        ctk.CTkLabel(c2, text="Reward Mode", text_color=COLOR_DIM,
                      font=ctk.CTkFont(size=12)
                      ).grid(row=3, column=0, sticky="w", padx=18, pady=(0, 4))
        self.train_reward = ctk.CTkOptionMenu(c2,
            values=["realized (recommended)", "mtm"],
            fg_color=COLOR_BG_INPUT, button_color=COLOR_BG_INPUT)
        self.train_reward.grid(row=4, column=0, sticky="ew", padx=18, pady=(0, 16))

        # Right: Hyperparameters
        c3 = Card(settings, title="🎚️ Hyperparameters")
        c3.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        c3.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(c3, text="Training Steps", text_color=COLOR_DIM,
                      font=ctk.CTkFont(size=12)
                      ).grid(row=1, column=0, sticky="w", padx=18, pady=(8, 4))
        self.train_steps = ctk.CTkEntry(c3, placeholder_text="200000")
        self.train_steps.insert(0, "200000")
        self.train_steps.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 12))

        # Window + max_hold (row)
        wm = ctk.CTkFrame(c3, fg_color="transparent")
        wm.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 12))
        wm.grid_columnconfigure(0, weight=1)
        wm.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(wm, text="Window Size", text_color=COLOR_DIM,
                      font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(wm, text="Max Hold", text_color=COLOR_DIM,
                      font=ctk.CTkFont(size=12)).grid(row=0, column=1, sticky="w", padx=(8, 0))

        self.train_window = ctk.CTkEntry(wm, placeholder_text="10")
        self.train_window.insert(0, "10")
        self.train_window.grid(row=1, column=0, sticky="ew", pady=(2, 0))

        self.train_maxhold = ctk.CTkEntry(wm, placeholder_text="30")
        self.train_maxhold.insert(0, "30")
        self.train_maxhold.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(2, 0))

        ctk.CTkLabel(c3, text="Model Name", text_color=COLOR_DIM,
                      font=ctk.CTkFont(size=12)
                      ).grid(row=4, column=0, sticky="w", padx=18, pady=(0, 4))
        self.train_name = ctk.CTkEntry(c3, placeholder_text="rl_prod_v1")
        self.train_name.insert(0, "rl_prod_v1")
        self.train_name.grid(row=5, column=0, sticky="ew", padx=18, pady=(0, 16))

        # === Advanced PPO settings (collapsible) ===
        c_adv = Card(page, title="⚙️ Advanced PPO Settings")
        c_adv.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        c_adv.grid_columnconfigure(0, weight=1)

        # Toggle button
        self.adv_visible = False
        self.adv_toggle_btn = ctk.CTkButton(c_adv,
            text="▶ Show Advanced",
            command=self._toggle_advanced,
            fg_color="transparent", hover_color="#2d333b",
            text_color=COLOR_ACCENT, height=30,
            anchor="w")
        self.adv_toggle_btn.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 8))

        # Advanced fields container (hidden by default)
        self.adv_frame = ctk.CTkFrame(c_adv, fg_color="transparent")
        self.adv_frame.grid_columnconfigure(0, weight=1)
        self.adv_frame.grid_columnconfigure(1, weight=1)
        self.adv_frame.grid_columnconfigure(2, weight=1)
        # Note: not gridded yet — toggled by _toggle_advanced

        # Learning Rate
        ctk.CTkLabel(self.adv_frame, text="Learning Rate ⭐", text_color=COLOR_DIM,
                      font=ctk.CTkFont(size=12)
                      ).grid(row=0, column=0, sticky="w", padx=18, pady=(8, 4))
        self.train_lr = ctk.CTkEntry(self.adv_frame, placeholder_text="3e-4")
        self.train_lr.insert(0, "3e-4")
        self.train_lr.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 8))
        ctk.CTkLabel(self.adv_frame,
                      text="ลดถ้า KL/Clip สูง (3e-4 → 1e-4)",
                      font=ctk.CTkFont(size=10),
                      text_color=COLOR_DIM
                      ).grid(row=2, column=0, sticky="w", padx=18, pady=(0, 12))

        # Clip Range
        ctk.CTkLabel(self.adv_frame, text="Clip Range", text_color=COLOR_DIM,
                      font=ctk.CTkFont(size=12)
                      ).grid(row=0, column=1, sticky="w", padx=8, pady=(8, 4))
        self.train_clip = ctk.CTkEntry(self.adv_frame, placeholder_text="0.2")
        self.train_clip.insert(0, "0.2")
        self.train_clip.grid(row=1, column=1, sticky="ew", padx=8, pady=(0, 8))
        ctk.CTkLabel(self.adv_frame,
                      text="PPO core: ห้าม policy เปลี่ยน > 20%",
                      font=ctk.CTkFont(size=10),
                      text_color=COLOR_DIM
                      ).grid(row=2, column=1, sticky="w", padx=8, pady=(0, 12))

        # Entropy Coef
        ctk.CTkLabel(self.adv_frame, text="Entropy Coef", text_color=COLOR_DIM,
                      font=ctk.CTkFont(size=12)
                      ).grid(row=0, column=2, sticky="w", padx=18, pady=(8, 4))
        self.train_ent = ctk.CTkEntry(self.adv_frame, placeholder_text="0.01")
        self.train_ent.insert(0, "0.01")
        self.train_ent.grid(row=1, column=2, sticky="ew", padx=18, pady=(0, 8))
        ctk.CTkLabel(self.adv_frame,
                      text="เพิ่มถ้า exploration หาย (0.01 → 0.05)",
                      font=ctk.CTkFont(size=10),
                      text_color=COLOR_DIM
                      ).grid(row=2, column=2, sticky="w", padx=18, pady=(0, 12))

        # ───── Row 2: Rollout & optimization ─────
        # n_steps
        ctk.CTkLabel(self.adv_frame, text="n_steps", text_color=COLOR_DIM,
                      font=ctk.CTkFont(size=12)
                      ).grid(row=3, column=0, sticky="w", padx=18, pady=(8, 4))
        self.train_nsteps = ctk.CTkEntry(self.adv_frame, placeholder_text="2048")
        self.train_nsteps.insert(0, "2048")
        self.train_nsteps.grid(row=4, column=0, sticky="ew", padx=18, pady=(0, 8))
        ctk.CTkLabel(self.adv_frame,
                      text="rollout buffer per env (เพิ่ม = stable)",
                      font=ctk.CTkFont(size=10),
                      text_color=COLOR_DIM
                      ).grid(row=5, column=0, sticky="w", padx=18, pady=(0, 12))

        # n_epochs
        ctk.CTkLabel(self.adv_frame, text="n_epochs", text_color=COLOR_DIM,
                      font=ctk.CTkFont(size=12)
                      ).grid(row=3, column=1, sticky="w", padx=8, pady=(8, 4))
        self.train_nepochs = ctk.CTkEntry(self.adv_frame, placeholder_text="10")
        self.train_nepochs.insert(0, "10")
        self.train_nepochs.grid(row=4, column=1, sticky="ew", padx=8, pady=(0, 8))
        ctk.CTkLabel(self.adv_frame,
                      text="SGD passes/rollout (10 = ปกติ)",
                      font=ctk.CTkFont(size=10),
                      text_color=COLOR_DIM
                      ).grid(row=5, column=1, sticky="w", padx=8, pady=(0, 12))

        # batch_size
        ctk.CTkLabel(self.adv_frame, text="batch_size", text_color=COLOR_DIM,
                      font=ctk.CTkFont(size=12)
                      ).grid(row=3, column=2, sticky="w", padx=18, pady=(8, 4))
        self.train_batch = ctk.CTkEntry(self.adv_frame, placeholder_text="64")
        self.train_batch.insert(0, "64")
        self.train_batch.grid(row=4, column=2, sticky="ew", padx=18, pady=(0, 8))
        ctk.CTkLabel(self.adv_frame,
                      text="minibatch (ต้อง ≤ n_steps)",
                      font=ctk.CTkFont(size=10),
                      text_color=COLOR_DIM
                      ).grid(row=5, column=2, sticky="w", padx=18, pady=(0, 12))

        # ───── Row 3: Discount / advantage / value ─────
        # gamma
        ctk.CTkLabel(self.adv_frame, text="gamma (γ)", text_color=COLOR_DIM,
                      font=ctk.CTkFont(size=12)
                      ).grid(row=6, column=0, sticky="w", padx=18, pady=(8, 4))
        self.train_gamma = ctk.CTkEntry(self.adv_frame, placeholder_text="0.99")
        self.train_gamma.insert(0, "0.99")
        self.train_gamma.grid(row=7, column=0, sticky="ew", padx=18, pady=(0, 8))
        ctk.CTkLabel(self.adv_frame,
                      text="discount factor (มอง future ไกล)",
                      font=ctk.CTkFont(size=10),
                      text_color=COLOR_DIM
                      ).grid(row=8, column=0, sticky="w", padx=18, pady=(0, 12))

        # gae_lambda
        ctk.CTkLabel(self.adv_frame, text="gae_lambda (λ)", text_color=COLOR_DIM,
                      font=ctk.CTkFont(size=12)
                      ).grid(row=6, column=1, sticky="w", padx=8, pady=(8, 4))
        self.train_gae = ctk.CTkEntry(self.adv_frame, placeholder_text="0.95")
        self.train_gae.insert(0, "0.95")
        self.train_gae.grid(row=7, column=1, sticky="ew", padx=8, pady=(0, 8))
        ctk.CTkLabel(self.adv_frame,
                      text="advantage smoothing (0.9–0.99)",
                      font=ctk.CTkFont(size=10),
                      text_color=COLOR_DIM
                      ).grid(row=8, column=1, sticky="w", padx=8, pady=(0, 12))

        # vf_coef
        ctk.CTkLabel(self.adv_frame, text="vf_coef", text_color=COLOR_DIM,
                      font=ctk.CTkFont(size=12)
                      ).grid(row=6, column=2, sticky="w", padx=18, pady=(8, 4))
        self.train_vf = ctk.CTkEntry(self.adv_frame, placeholder_text="0.5")
        self.train_vf.insert(0, "0.5")
        self.train_vf.grid(row=7, column=2, sticky="ew", padx=18, pady=(0, 8))
        ctk.CTkLabel(self.adv_frame,
                      text="critic loss weight (0.25–1.0)",
                      font=ctk.CTkFont(size=10),
                      text_color=COLOR_DIM
                      ).grid(row=8, column=2, sticky="w", padx=18, pady=(0, 12))

        # Quick presets
        preset_frame = ctk.CTkFrame(self.adv_frame, fg_color="transparent")
        preset_frame.grid(row=9, column=0, columnspan=3, sticky="ew",
                           padx=18, pady=(4, 12))

        ctk.CTkLabel(preset_frame, text="Quick presets:",
            font=ctk.CTkFont(size=11), text_color=COLOR_DIM
            ).pack(side="left", padx=(0, 8))

        ctk.CTkButton(preset_frame, text="🔄 Default",
            command=lambda: self._apply_preset("default"),
            fg_color=COLOR_BG_INPUT, hover_color="#2d333b",
            width=90, height=26, font=ctk.CTkFont(size=11)
            ).pack(side="left", padx=2)

        ctk.CTkButton(preset_frame, text="🛡️ Stable (low lr)",
            command=lambda: self._apply_preset("stable"),
            fg_color=COLOR_BG_INPUT, hover_color="#2d333b",
            width=120, height=26, font=ctk.CTkFont(size=11)
            ).pack(side="left", padx=2)

        ctk.CTkButton(preset_frame, text="🚀 Fast (high lr)",
            command=lambda: self._apply_preset("fast"),
            fg_color=COLOR_BG_INPUT, hover_color="#2d333b",
            width=120, height=26, font=ctk.CTkFont(size=11)
            ).pack(side="left", padx=2)

        ctk.CTkButton(preset_frame, text="🔍 Explorer",
            command=lambda: self._apply_preset("explore"),
            fg_color=COLOR_BG_INPUT, hover_color="#2d333b",
            width=100, height=26, font=ctk.CTkFont(size=11)
            ).pack(side="left", padx=2)

        # === Run + Progress + Log ===
        c4 = Card(page, title="🚀 Training")
        c4.grid(row=3, column=0, sticky="ew")
        c4.grid_columnconfigure(0, weight=1)

        btn_row = ctk.CTkFrame(c4, fg_color="transparent")
        btn_row.grid(row=1, column=0, sticky="ew", padx=18, pady=(8, 12))

        self.train_btn = ctk.CTkButton(btn_row, text="▶ Start Training",
            command=self._start_training,
            fg_color=COLOR_ACCENT, hover_color="#4493f8",
            height=42, font=ctk.CTkFont(size=14, weight="bold"))
        self.train_btn.pack(side="left", padx=(0, 8))

        self.train_stop_btn = ctk.CTkButton(btn_row, text="⏹ Stop",
            command=self._stop_training,
            fg_color=COLOR_RED, hover_color="#da3633",
            height=42, state="disabled", width=100)
        self.train_stop_btn.pack(side="left")

        # Progress
        prog_frame = ctk.CTkFrame(c4, fg_color=COLOR_BG_INPUT, corner_radius=8)
        prog_frame.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 12))
        prog_frame.grid_columnconfigure(0, weight=1)

        self.train_status = ctk.CTkLabel(prog_frame, text="Ready",
            font=ctk.CTkFont(size=13))
        self.train_status.grid(row=0, column=0, sticky="w", padx=14, pady=(10, 4))

        self.train_progress = ctk.CTkProgressBar(prog_frame,
            progress_color=COLOR_ACCENT, fg_color="#0f1419")
        self.train_progress.grid(row=1, column=0, sticky="ew", padx=14, pady=(0, 6))
        self.train_progress.set(0)

        self.train_meta = ctk.CTkLabel(prog_frame, text="",
            font=ctk.CTkFont(size=11), text_color=COLOR_DIM)
        self.train_meta.grid(row=2, column=0, sticky="w", padx=14, pady=(0, 10))

        # === Live Metrics Health Panel ⭐ NEW ===
        health_label = ctk.CTkLabel(c4, text="🩺 Live Metric Health",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_DIM)
        health_label.grid(row=3, column=0, sticky="w", padx=18, pady=(8, 4))

        # Container — 6 metric pills + recommendations
        self.health_frame = ctk.CTkFrame(c4, fg_color=COLOR_BG_INPUT, corner_radius=8)
        self.health_frame.grid(row=4, column=0, sticky="ew", padx=18, pady=(0, 8))
        self.health_frame.grid_columnconfigure(0, weight=1)

        # Pills row (will be populated dynamically)
        self.health_pills_frame = ctk.CTkFrame(self.health_frame, fg_color="transparent")
        self.health_pills_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=8)
        self.health_pills = {}  # name -> label widget
        self._build_health_pills()

        # ⭐ Reward trend graph (sparkline) ⭐
        graph_wrap = ctk.CTkFrame(self.health_frame, fg_color="#0a0e14",
                                    corner_radius=8, border_width=1,
                                    border_color="#30363d")
        graph_wrap.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))
        graph_wrap.grid_columnconfigure(0, weight=1)

        # Header row
        gh = ctk.CTkFrame(graph_wrap, fg_color="transparent")
        gh.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))
        gh.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(gh, text="📈 Reward Trend",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=COLOR_DIM
            ).grid(row=0, column=0, sticky="w")
        self.reward_trend_status = ctk.CTkLabel(gh, text="—",
            font=ctk.CTkFont(size=10, family="Consolas"),
            text_color=COLOR_DIM)
        self.reward_trend_status.grid(row=0, column=1, sticky="e")

        # Canvas for sparkline (tk.Canvas works well in CTk)
        self.reward_canvas = tk.Canvas(graph_wrap, height=60,
            bg="#0a0e14", highlightthickness=0, bd=0)
        self.reward_canvas.grid(row=1, column=0, sticky="ew",
            padx=10, pady=(0, 8))

        # State for reward trend
        self._reward_history = []           # list of ep_rew_mean values
        self._reward_peak = None            # highest seen
        self._reward_alert_shown = False    # debounce popup
        self._train_step_count = 0          # rough progress (for "Phase 1" guard)
        self.reward_canvas.bind("<Configure>",
            lambda e: self._draw_reward_sparkline())

        # Recommendations: container that holds multiple cards
        self.health_recs_frame = ctk.CTkFrame(self.health_frame, fg_color="transparent")
        self.health_recs_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(2, 10))
        self.health_recs_frame.grid_columnconfigure(0, weight=1)

        # Initial state — placeholder
        self.health_recs_placeholder = ctk.CTkLabel(self.health_recs_frame,
            text="⏳ Waiting for metrics...",
            font=ctk.CTkFont(size=11, family="Consolas"),
            text_color=COLOR_DIM, anchor="w")
        self.health_recs_placeholder.grid(row=0, column=0, sticky="ew", padx=4, pady=4)

        self.health_recs_cards = {}  # metric name -> card widget

        # Log (using tk.Text for color support)
        log_label = ctk.CTkLabel(c4, text="📝 Live Log",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=COLOR_DIM)
        log_label.grid(row=5, column=0, sticky="w", padx=18, pady=(8, 4))

        log_frame = ctk.CTkFrame(c4, fg_color="#0a0e14", corner_radius=8,
                                   border_width=1, border_color="#30363d")
        log_frame.grid(row=6, column=0, sticky="ew", padx=18, pady=(0, 16))

        self.train_log = self._make_log_widget(log_frame, height=14)

    def _toggle_advanced(self):
        """Show/hide advanced PPO settings"""
        if self.adv_visible:
            self.adv_frame.grid_remove()
            self.adv_toggle_btn.configure(text="▶ Show Advanced")
            self.adv_visible = False
        else:
            self.adv_frame.grid(row=2, column=0, sticky="ew")
            self.adv_toggle_btn.configure(text="▼ Hide Advanced")
            self.adv_visible = True

    def _apply_preset(self, name: str):
        """Apply hyperparameter preset"""
        presets = self._ppo_presets()
        p = presets.get(name)
        if not p:
            return
        mapping = [
            (self.train_lr, p["lr"]),
            (self.train_clip, p["clip"]),
            (self.train_ent, p["ent"]),
            (self.train_nsteps, p["nsteps"]),
            (self.train_nepochs, p["nepochs"]),
            (self.train_batch, p["batch"]),
            (self.train_gamma, p["gamma"]),
            (self.train_gae, p["gae"]),
            (self.train_vf, p["vf"]),
        ]
        for entry, val in mapping:
            entry.delete(0, "end")
            entry.insert(0, val)

    def _build_health_pills(self):
        """Build the 6-metric health pills"""
        metrics_to_show = [
            ('ep_rew_mean', 'Reward'),
            ('approx_kl', 'KL'),
            ('clip_fraction', 'Clip%'),
            ('explained_variance', 'EV'),
            ('entropy_loss', 'Entropy'),
            ('value_loss', 'V-Loss'),
        ]
        for i, (key, label) in enumerate(metrics_to_show):
            self.health_pills_frame.grid_columnconfigure(i, weight=1)
            pill = ctk.CTkFrame(self.health_pills_frame,
                fg_color="#0a0e14", corner_radius=6,
                border_width=1, border_color="#30363d")
            pill.grid(row=0, column=i, sticky="ew", padx=3)

            ctk.CTkLabel(pill, text=label,
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color=COLOR_DIM
                ).pack(pady=(6, 0))
            value_lbl = ctk.CTkLabel(pill, text="—",
                font=ctk.CTkFont(size=14, family="Consolas", weight="bold"),
                text_color=COLOR_DIM)
            value_lbl.pack(pady=(2, 6))
            self.health_pills[key] = value_lbl

    def _update_health_pill(self, name: str, value: float):
        """Update one pill + check status"""
        if name not in self.health_pills:
            return
        status = classify_metric(name, value)
        color = {
            'good': COLOR_GREEN,
            'warn': COLOR_YELLOW,
            'bad':  COLOR_RED,
        }.get(status, COLOR_DIM)
        # Format value
        if abs(value) >= 100:
            txt = f"{value:.0f}"
        elif abs(value) >= 1:
            txt = f"{value:.2f}"
        elif abs(value) >= 0.001:
            txt = f"{value:.3f}"
        else:
            txt = f"{value:.2e}"
        self.health_pills[name].configure(text=txt, text_color=color)

    # --------------------------------------------------------
    # Reward trend graph + threshold check
    # --------------------------------------------------------
    def _track_reward(self, value: float):
        """Append a new ep_rew_mean reading and update graph + threshold popup"""
        # Append + cap history (keep last 200 points)
        self._reward_history.append(float(value))
        if len(self._reward_history) > 200:
            self._reward_history = self._reward_history[-200:]

        # Update peak
        if self._reward_peak is None or value > self._reward_peak:
            self._reward_peak = float(value)

        # Decide trend label
        n = len(self._reward_history)
        trend_txt, trend_color = "—", COLOR_DIM
        if n >= 5:
            recent = self._reward_history[-5:]
            slope = recent[-1] - recent[0]
            if slope > 0.05:
                trend_txt, trend_color = "↑ rising", COLOR_GREEN
            elif slope < -0.05:
                trend_txt, trend_color = "↓ falling", COLOR_RED
            else:
                trend_txt, trend_color = "→ flat", COLOR_YELLOW
        peak = self._reward_peak if self._reward_peak is not None else value
        status = f"now {value:+.2f}  peak {peak:+.2f}  {trend_txt}"
        try:
            self.reward_trend_status.configure(text=status, text_color=trend_color)
        except Exception:
            pass

        # Draw sparkline
        self._draw_reward_sparkline()

        # ⚠️ Threshold check — popup if drops > 50% below peak
        # Guards: only after Phase 1 (>= 30% of total steps) AND peak meaningfully positive,
        # and only show ONCE per training run.
        try:
            min_steps = max(10, int(self._train_steps_total * 0.3))
        except Exception:
            min_steps = 10

        if (self._reward_peak is not None
                and self._reward_peak > 0.5
                and self._train_step_count >= min_steps
                and value < self._reward_peak * 0.5
                and not self._reward_alert_shown):
            self._reward_alert_shown = True
            try:
                pct = (value / self._reward_peak * 100.0) if self._reward_peak else 0
                msg = (
                    f"Reward dropped below 50% of peak\n\n"
                    f"  Peak reward : {self._reward_peak:+.3f}\n"
                    f"  Current     : {value:+.3f}  ({pct:.0f}% of peak)\n\n"
                    f"This often indicates:\n"
                    f"  • Concept drift / overfit on early data\n"
                    f"  • Catastrophic forgetting (lr ↑ too high)\n"
                    f"  • Unstable PPO update (KL/Clip spike)\n\n"
                    f"Suggested actions:\n"
                    f"  ⭐ ลด learning_rate (3e-4 → 1e-4)\n"
                    f"  ⭐ ลด clip_range (0.2 → 0.1) — ถ้า KL/Clip สูง\n"
                    f"  • พิจารณาหยุด train แล้ว fine-tune จาก best checkpoint"
                )
                # Non-blocking so training thread continues running
                self.after(50, lambda: messagebox.showwarning(
                    "⚠️ Reward Drop Detected", msg))
            except Exception:
                pass

    def _draw_reward_sparkline(self):
        """Render the reward history as a sparkline on the canvas"""
        c = getattr(self, 'reward_canvas', None)
        if c is None:
            return
        c.delete("all")
        try:
            w = c.winfo_width()
            h = c.winfo_height()
        except Exception:
            return
        if w < 4 or h < 4:
            return

        data = self._reward_history
        if not data:
            c.create_text(w / 2, h / 2,
                text="(waiting for ep_rew_mean ...)",
                fill=COLOR_DIM, font=("Consolas", 10))
            return

        # zero baseline
        c.create_line(0, h / 2, w, h / 2, fill="#30363d", dash=(2, 3))

        n = len(data)
        vmin = min(data)
        vmax = max(data)
        # symmetric range around 0 if peak positive
        if vmax > 0 and vmin < 0:
            span = max(abs(vmin), abs(vmax)) or 1.0
            lo, hi = -span, span
        elif vmax > 0:
            lo, hi = 0, max(vmax, 0.001)
        else:
            lo, hi = min(vmin, -0.001), 0
        # add 10% padding
        rng = (hi - lo) or 1.0
        lo -= rng * 0.1
        hi += rng * 0.1

        def to_xy(i, v):
            x = (w - 4) * (i / max(n - 1, 1)) + 2
            y = h - ((v - lo) / (hi - lo)) * (h - 4) - 2
            return x, y

        # area under curve (above baseline)
        pts = [to_xy(i, v) for i, v in enumerate(data)]
        # Line color based on slope
        recent = data[-5:] if n >= 5 else data
        slope = recent[-1] - recent[0] if len(recent) > 1 else 0
        if slope > 0.05:
            line_color = COLOR_GREEN
        elif slope < -0.05:
            line_color = COLOR_RED
        else:
            line_color = COLOR_YELLOW

        # draw line
        flat = []
        for x, y in pts:
            flat += [x, y]
        if len(flat) >= 4:
            c.create_line(*flat, fill=line_color, width=2, smooth=True)

        # peak marker
        if self._reward_peak is not None and n > 0:
            try:
                idx = data.index(self._reward_peak)
                px, py = to_xy(idx, self._reward_peak)
                c.create_oval(px - 3, py - 3, px + 3, py + 3,
                    fill=COLOR_ACCENT, outline="")
            except Exception:
                pass

        # threshold (50% of peak) horizontal line — only if peak > 0
        if self._reward_peak is not None and self._reward_peak > 0:
            thr = self._reward_peak * 0.5
            if lo <= thr <= hi:
                ty = h - ((thr - lo) / (hi - lo)) * (h - 4) - 2
                c.create_line(0, ty, w, ty, fill=COLOR_RED,
                    dash=(3, 3), width=1)
                c.create_text(w - 4, ty - 6, text="50% peak",
                    fill=COLOR_RED, anchor="ne",
                    font=("Consolas", 8))

    def _reset_reward_tracking(self):
        """Call when starting a new training run"""
        self._reward_history = []
        self._reward_peak = None
        self._reward_alert_shown = False
        self._train_step_count = 0
        try:
            self.reward_trend_status.configure(text="—", text_color=COLOR_DIM)
        except Exception:
            pass
        self._draw_reward_sparkline()

    def _refresh_recommendations(self):
        """Build recommendation cards (one per metric)"""
        if not hasattr(self, '_current_metrics'):
            return

        # Clear existing cards
        for w in self.health_recs_frame.winfo_children():
            w.destroy()
        self.health_recs_cards = {}

        # Collect recommendations
        recs = []
        for name, value in self._current_metrics.items():
            rec = get_recommendation(name, value)
            if rec:
                recs.append(rec)

        # Sort by severity (high first)
        sev_order = {'high': 0, 'med': 1, 'low': 2}
        recs.sort(key=lambda r: sev_order.get(r.get('severity', 'med'), 1))

        if not recs:
            ok = ctk.CTkLabel(self.health_recs_frame,
                text="✅ ทุก metric อยู่ใน healthy range — ไม่ต้องปรับอะไร",
                font=ctk.CTkFont(size=13, weight="bold"),
                text_color=COLOR_GREEN, anchor="w")
            ok.grid(row=0, column=0, sticky="ew", padx=4, pady=8)
            return

        # Layout: grid of cards, 2 per row
        for i, rec in enumerate(recs):
            row = i // 2
            col = i % 2
            self.health_recs_frame.grid_columnconfigure(col, weight=1)
            self._build_rec_card(self.health_recs_frame, rec, row, col)

    def _build_rec_card(self, parent, rec, row, col):
        """Build one recommendation card for a metric"""
        # Severity colors
        sev = rec.get('severity', 'med')
        sev_color = {
            'high': COLOR_RED,
            'med':  COLOR_YELLOW,
            'low':  COLOR_DIM,
        }.get(sev, COLOR_YELLOW)
        sev_icon = {'high': '🔴', 'med': '🟡', 'low': '🟢'}.get(sev, '🟡')

        card = ctk.CTkFrame(parent, fg_color="#0a0e14", corner_radius=8,
            border_width=2, border_color=sev_color)
        card.grid(row=row, column=col, sticky="nsew", padx=4, pady=4)
        card.grid_columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkFrame(card, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(header, text=rec.get('icon', '⚠️'),
            font=ctk.CTkFont(size=18)
            ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(header, text=rec['title'],
            font=ctk.CTkFont(size=13, family="Consolas", weight="bold"),
            text_color=sev_color, anchor="w"
            ).grid(row=0, column=1, sticky="w", padx=(8, 0))

        ctk.CTkLabel(header, text=sev_icon,
            font=ctk.CTkFont(size=14)
            ).grid(row=0, column=2, sticky="e")

        # Problem
        problem_label = ctk.CTkLabel(card, text="⚠️  " + rec['problem'],
            font=ctk.CTkFont(size=11),
            text_color=COLOR_DIM, anchor="w", wraplength=400)
        problem_label.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 6))

        # Actions
        for j, (arrow, param, change, tag) in enumerate(rec['actions']):
            action_row = ctk.CTkFrame(card, fg_color="transparent")
            action_row.grid(row=2 + j, column=0, sticky="ew", padx=10, pady=1)
            action_row.grid_columnconfigure(2, weight=1)

            arrow_color = {'main': COLOR_GREEN, 'warn': COLOR_RED}.get(tag, COLOR_DIM)

            ctk.CTkLabel(action_row, text=arrow,
                font=ctk.CTkFont(size=14, weight="bold"),
                text_color=arrow_color, width=20
                ).grid(row=0, column=0, sticky="w")

            ctk.CTkLabel(action_row, text=param,
                font=ctk.CTkFont(size=12, family="Consolas", weight="bold"),
                text_color=COLOR_ACCENT, anchor="w"
                ).grid(row=0, column=1, sticky="w", padx=(4, 6))

            ctk.CTkLabel(action_row, text=change,
                font=ctk.CTkFont(size=11, family="Consolas"),
                text_color=COLOR_DIM, anchor="w"
                ).grid(row=0, column=2, sticky="w")

            if tag == 'main':
                ctk.CTkLabel(action_row, text="⭐",
                    font=ctk.CTkFont(size=12)
                    ).grid(row=0, column=3, sticky="e")

        # Padding bottom
        ctk.CTkFrame(card, fg_color="transparent", height=8
            ).grid(row=10, column=0)

    def _browse_train_csv(self):
        path = filedialog.askopenfilename(
            title="Select training CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=str(WORK_DIR))
        if not path:
            return
        self._set_train_csv(path)

    def _set_train_csv(self, path):
        self.train_csv_path = path
        name = Path(path).name
        try:
            import pandas as pd
            df = pd.read_csv(path, nrows=1)
            cols = len(df.columns)
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                rows = sum(1 for _ in f) - 1
            meta = f"{rows:,} rows · {cols} columns"
        except Exception as e:
            meta = f"(error reading: {e})"

        self.train_csv_label.configure(text=name)
        self.train_csv_meta.configure(text=meta)

    def _start_training(self):
        if self._is_process_busy():
            messagebox.showwarning("Busy", "Already training")
            return
        if not hasattr(self, 'train_csv_path'):
            messagebox.showwarning("No data", "Please select a CSV file first")
            return

        steps = self.train_steps.get() or "200000"
        window = self.train_window.get() or "10"
        max_hold = self.train_maxhold.get() or "30"
        name = self.train_name.get() or "rl_prod_v1"
        reward = self.train_reward.get().split()[0]

        # Map algorithm dropdown to CLI value
        algo_map = {
            "PPO (recommended)": "ppo",
            "DQN": "dqn",
            "A2C": "a2c",
        }
        algo_label = self.train_algo.get()
        algo = algo_map.get(algo_label, "ppo")

        # Advanced PPO params
        lr = self.train_lr.get() or "3e-4"
        clip = self.train_clip.get() or "0.2"
        ent = self.train_ent.get() or "0.01"
        nsteps = self.train_nsteps.get() or "2048"
        nepochs = self.train_nepochs.get() or "10"
        batch = self.train_batch.get() or "64"
        gamma = self.train_gamma.get() or "0.99"
        gae = self.train_gae.get() or "0.95"
        vf = self.train_vf.get() or "0.5"

        cmd = [
            sys.executable, "rl_train.py", self.train_csv_path,
            "--steps", steps,
            "--window", window,
            "--max_hold", max_hold,
            "--reward_mode", reward,
            "--algo", algo,
            "--name", name,
            "--learning_rate", lr,
            "--clip_range", clip,
            "--ent_coef", ent,
            "--n_steps", nsteps,
            "--n_epochs", nepochs,
            "--batch_size", batch,
            "--gamma", gamma,
            "--gae_lambda", gae,
            "--vf_coef", vf,
        ]

        self._log(self.train_log, f"$ {' '.join(cmd)}", "info")
        self.train_btn.configure(state="disabled")
        self.train_stop_btn.configure(state="normal")
        self.status_label.configure(text="● Training...", text_color=COLOR_GREEN)
        self.train_status.configure(text="Starting...")

        self._train_steps_total = int(steps)
        # ⭐ reset reward graph + threshold tracker for new run
        self._reset_reward_tracking()
        self.runner.start(cmd)

    def _stop_training(self):
        if self._is_process_busy():
            self._log(self.train_log, "Stopping...", "warn")
            self.runner.stop()

    # --------------------------------------------------------
    # PAGE: BACKTEST
    # --------------------------------------------------------
    def _build_backtest_page(self):
        page = ctk.CTkFrame(self.content, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        self.pages["backtest"] = page

        # Tabs
        tab_view = ctk.CTkTabview(page, fg_color=COLOR_BG_CARD,
            segmented_button_fg_color=COLOR_BG_INPUT,
            segmented_button_selected_color=COLOR_ACCENT)
        tab_view.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        tab_view.add("Live Logic ⭐")
        tab_view.add("Standard")
        tab_view.add("Confidence Filter")

        # ===== Live Logic tab =====
        live_tab = tab_view.tab("Live Logic ⭐")
        live_tab.grid_columnconfigure(0, weight=1)
        live_tab.grid_columnconfigure(1, weight=1)

        # Setup card
        setup = Card(live_tab, title="📂 Setup")
        setup.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=14)
        setup.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(setup, text="Model", text_color=COLOR_DIM
                      ).grid(row=1, column=0, sticky="w", padx=18, pady=(8, 4))
        self.bt_model = ScrollableOptionMenu(setup, values=["(none)"],
            fg_color=COLOR_BG_INPUT, button_color=COLOR_BG_INPUT)
        self.bt_model.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 12))

        ctk.CTkLabel(setup, text="Test Dataset", text_color=COLOR_DIM
                      ).grid(row=3, column=0, sticky="w", padx=18, pady=(0, 4))
        self.bt_csv = ScrollableOptionMenu(setup, values=["(none)"],
            fg_color=COLOR_BG_INPUT, button_color=COLOR_BG_INPUT)
        self.bt_csv.grid(row=4, column=0, sticky="ew", padx=18, pady=(0, 12))

        ctk.CTkLabel(setup, text="Confidence Threshold", text_color=COLOR_DIM
                      ).grid(row=5, column=0, sticky="w", padx=18, pady=(0, 4))
        self.bt_conf = ctk.CTkEntry(setup)
        self.bt_conf.insert(0, "0.85")
        self.bt_conf.grid(row=6, column=0, sticky="ew", padx=18, pady=(0, 12))

        # Backtest mode toggle
        ctk.CTkLabel(setup, text="Backtest Mode", text_color=COLOR_DIM,
                      font=ctk.CTkFont(size=12)
                      ).grid(row=7, column=0, sticky="w", padx=18, pady=(0, 4))
        self.bt_mode = ctk.CTkOptionMenu(setup,
            values=[
                "Pure Agent (matches training) ⭐",
                "Agent + SL/TP (live-realistic)",
            ],
            fg_color=COLOR_BG_INPUT, button_color=COLOR_BG_INPUT)
        self.bt_mode.grid(row=8, column=0, sticky="ew", padx=18, pady=(0, 4))
        ctk.CTkLabel(setup,
            text="Pure = ตรงกับ env ตอน train · SL/TP = สำหรับ live broker",
            font=ctk.CTkFont(size=10), text_color=COLOR_DIM
            ).grid(row=9, column=0, sticky="w", padx=18, pady=(0, 16))

        # Risk card
        risk = Card(live_tab, title="🛡️ Risk Management")
        risk.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=14)
        risk.grid_columnconfigure(0, weight=1)
        risk.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(risk, text="Risk %", text_color=COLOR_DIM
                      ).grid(row=1, column=0, sticky="w", padx=18, pady=(8, 4))
        ctk.CTkLabel(risk, text="Max Positions", text_color=COLOR_DIM
                      ).grid(row=1, column=1, sticky="w", padx=18, pady=(8, 4))
        self.bt_risk = ctk.CTkEntry(risk); self.bt_risk.insert(0, "0.01")
        self.bt_risk.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 12))
        self.bt_max_pos = ctk.CTkEntry(risk); self.bt_max_pos.insert(0, "3")
        self.bt_max_pos.grid(row=2, column=1, sticky="ew", padx=18, pady=(0, 12))

        ctk.CTkLabel(risk, text="ATR × SL", text_color=COLOR_DIM
                      ).grid(row=3, column=0, sticky="w", padx=18, pady=(0, 4))
        ctk.CTkLabel(risk, text="ATR × TP", text_color=COLOR_DIM
                      ).grid(row=3, column=1, sticky="w", padx=18, pady=(0, 4))
        self.bt_sl = ctk.CTkEntry(risk); self.bt_sl.insert(0, "2.0")
        self.bt_sl.grid(row=4, column=0, sticky="ew", padx=18, pady=(0, 12))
        self.bt_tp = ctk.CTkEntry(risk); self.bt_tp.insert(0, "4.0")
        self.bt_tp.grid(row=4, column=1, sticky="ew", padx=18, pady=(0, 16))

        # Window size — auto-detected from the model; leave blank
        ctk.CTkLabel(risk, text="Window Size (auto-detected — leave blank)",
                      text_color=COLOR_DIM, font=ctk.CTkFont(size=12)
                      ).grid(row=5, column=0, columnspan=2, sticky="w", padx=18, pady=(0, 4))
        self.bt_window = ctk.CTkEntry(risk, placeholder_text="auto (from model)")
        self.bt_window.grid(row=6, column=0, columnspan=2, sticky="ew", padx=18, pady=(0, 16))

        # Run button + stats
        run_row = ctk.CTkFrame(page, fg_color="transparent")
        run_row.grid(row=1, column=0, sticky="ew", pady=(0, 12))

        self.bt_run_btn = ctk.CTkButton(run_row, text="▶ Run Backtest",
            command=self._run_backtest,
            fg_color=COLOR_ACCENT, hover_color="#4493f8",
            height=42, font=ctk.CTkFont(size=14, weight="bold"))
        self.bt_run_btn.pack(side="left", padx=(0, 8))

        # Generate Chart button
        self.bt_chart_btn = ctk.CTkButton(run_row, text="📊 Generate Chart",
            command=self._generate_chart,
            fg_color=COLOR_PURPLE, hover_color="#9333ea",
            height=42, font=ctk.CTkFont(size=14, weight="bold"))
        self.bt_chart_btn.pack(side="left", padx=(0, 8))

        self.bt_open_chart_btn = ctk.CTkButton(run_row, text="🌐 Open Chart",
            command=self._open_chart,
            fg_color=COLOR_BG_INPUT, hover_color="#2d333b",
            height=42, width=130)
        self.bt_open_chart_btn.pack(side="left")

        # Stats grid
        stats_card = Card(page, title="📊 Results")
        stats_card.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        stats_card.grid_columnconfigure(0, weight=1)

        stats_grid = ctk.CTkFrame(stats_card, fg_color="transparent")
        stats_grid.grid(row=1, column=0, sticky="ew", padx=18, pady=(8, 16))
        for i in range(4):
            stats_grid.grid_columnconfigure(i, weight=1)

        self.stat_wr = StatCard(stats_grid, "Win Rate", "—")
        self.stat_wr.grid(row=0, column=0, sticky="ew", padx=4)
        self.stat_pf = StatCard(stats_grid, "Profit Factor", "—")
        self.stat_pf.grid(row=0, column=1, sticky="ew", padx=4)
        self.stat_ret = StatCard(stats_grid, "Return", "—")
        self.stat_ret.grid(row=0, column=2, sticky="ew", padx=4)
        self.stat_dd = StatCard(stats_grid, "Max DD", "—")
        self.stat_dd.grid(row=0, column=3, sticky="ew", padx=4)

        # Log
        log_card = Card(page, title="📝 Backtest Log")
        log_card.grid(row=3, column=0, sticky="ew")
        log_card.grid_columnconfigure(0, weight=1)

        log_frame = ctk.CTkFrame(log_card, fg_color="#0a0e14", corner_radius=8,
                                   border_width=1, border_color="#30363d")
        log_frame.grid(row=1, column=0, sticky="ew", padx=18, pady=(8, 16))

        self.bt_log = self._make_log_widget(log_frame, height=12)

    def _generate_chart(self):
        """Generate backtest chart"""
        if self._is_process_busy():
            messagebox.showwarning("Busy", "Another task running")
            return

        model = self.bt_model.get()
        csv = self.bt_csv.get()
        if model in ("(none)", "") or csv in ("(none)", ""):
            messagebox.showwarning("Setup", "Please select model and dataset")
            return

        # Check trades file exists
        trades_path = WORK_DIR / f"{model}_live_bt_trades.csv"
        if not trades_path.exists():
            messagebox.showwarning("No Trades",
                f"ไม่พบไฟล์ {trades_path.name}\n\n"
                f"กรุณา Run Backtest ก่อน เพื่อสร้างไฟล์ trades")
            return

        cmd = [
            sys.executable, "backtest_chart.py", model, csv,
            "--limit", "5000",  # last 5000 bars (avoid huge HTML)
        ]

        self._log(self.bt_log, f"$ {' '.join(cmd)}", "info")
        self.bt_chart_btn.configure(state="disabled")
        self.status_label.configure(text="● Generating chart...", text_color=COLOR_PURPLE)

        # Run subprocess (no metric parsing needed for this)
        self.runner.start(cmd)

    def _open_chart(self):
        """Open chart HTML in default browser"""
        import webbrowser
        model = self.bt_model.get()
        if model in ("(none)", ""):
            messagebox.showwarning("Setup", "Please select model")
            return

        chart_path = WORK_DIR / f"{model}_backtest_chart.html"
        if not chart_path.exists():
            messagebox.showwarning("No Chart",
                f"ไม่พบไฟล์ {chart_path.name}\n\n"
                f"กรุณา 'Generate Chart' ก่อน")
            return

        webbrowser.open(chart_path.as_uri())
        self._log(self.bt_log, f"Opened {chart_path.name} in browser", "success")

    def _run_backtest(self):
        if self._is_process_busy():
            messagebox.showwarning("Busy", "Another task running")
            return

        model = self.bt_model.get()
        csv = self.bt_csv.get()
        if model in ("(none)", "") or csv in ("(none)", ""):
            messagebox.showwarning("Setup", "Please select model and dataset")
            return

        # Map mode dropdown to CLI value
        mode_label = self.bt_mode.get()
        mode = "pure_agent" if "Pure" in mode_label else "agent_sltp"

        # Use backtest_live.py for realistic results
        cmd = [
            sys.executable, "backtest_live.py", model, csv,
            "--conf", self.bt_conf.get() or "0.85",
            "--risk", self.bt_risk.get() or "0.01",
            "--max_positions", self.bt_max_pos.get() or "3",
            "--atr_sl", self.bt_sl.get() or "2.0",
            "--atr_tp", self.bt_tp.get() or "4.0",
            "--window", self.bt_window.get().strip() or "0",  # 0 = auto-detect from model
            "--mode", mode,
        ]

        self._log(self.bt_log, f"$ {' '.join(cmd)}", "info")
        self.bt_run_btn.configure(state="disabled")
        self.status_label.configure(text="● Backtesting...", text_color=COLOR_ACCENT)

        self.runner.start(cmd)

    # --------------------------------------------------------
    # PAGE: WALK-FORWARD
    # --------------------------------------------------------
    def _build_walkfwd_page(self):
        page = ctk.CTkFrame(self.content, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        self.pages["walkfwd"] = page

        info = ctk.CTkLabel(page,
            text="🔬 Walk-Forward Validation — เทรน + เทสต์ หลายช่วงเวลาเพื่อพิสูจน์ว่า model robust",
            text_color=COLOR_ACCENT, font=ctk.CTkFont(size=13),
            wraplength=900)
        info.grid(row=0, column=0, sticky="w", padx=8, pady=(0, 12))

        # Config card
        c1 = Card(page, title="⚙️ Configuration")
        c1.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        c1.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(c1, text="Test Dataset", text_color=COLOR_DIM
                      ).grid(row=1, column=0, sticky="w", padx=18, pady=(8, 4))
        self.wf_csv = ScrollableOptionMenu(c1, values=["(none)"],
            fg_color=COLOR_BG_INPUT, button_color=COLOR_BG_INPUT)
        self.wf_csv.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 12))

        # Sub frame
        sub = ctk.CTkFrame(c1, fg_color="transparent")
        sub.grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 12))
        sub.grid_columnconfigure(0, weight=1)
        sub.grid_columnconfigure(1, weight=1)
        sub.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(sub, text="Windows", text_color=COLOR_DIM
                      ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(sub, text="Steps/window", text_color=COLOR_DIM
                      ).grid(row=0, column=1, sticky="w", padx=8)
        ctk.CTkLabel(sub, text="Window size", text_color=COLOR_DIM
                      ).grid(row=0, column=2, sticky="w", padx=8)

        self.wf_windows = ctk.CTkEntry(sub); self.wf_windows.insert(0, "5")
        self.wf_windows.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        self.wf_steps = ctk.CTkEntry(sub); self.wf_steps.insert(0, "50000")
        self.wf_steps.grid(row=1, column=1, sticky="ew", padx=8, pady=(2, 0))
        self.wf_window = ctk.CTkEntry(sub); self.wf_window.insert(0, "10")
        self.wf_window.grid(row=1, column=2, sticky="ew", padx=8, pady=(2, 0))

        ctk.CTkLabel(c1, text="Output Name", text_color=COLOR_DIM
                      ).grid(row=4, column=0, sticky="w", padx=18, pady=(0, 4))
        self.wf_name = ctk.CTkEntry(c1); self.wf_name.insert(0, "wf_prod")
        self.wf_name.grid(row=5, column=0, sticky="ew", padx=18, pady=(0, 16))

        self.wf_run_btn = ctk.CTkButton(page, text="▶ Start Walk-Forward (~50 min)",
            command=self._run_walkforward,
            fg_color=COLOR_ACCENT, hover_color="#4493f8",
            height=42, font=ctk.CTkFont(size=14, weight="bold"))
        self.wf_run_btn.grid(row=2, column=0, sticky="ew", pady=(0, 12))

        # Results card with verdict + log
        c2 = Card(page, title="📊 Results")
        c2.grid(row=3, column=0, sticky="ew")
        c2.grid_columnconfigure(0, weight=1)

        self.wf_verdict_frame = ctk.CTkFrame(c2, fg_color=COLOR_BG_INPUT,
            corner_radius=8)
        self.wf_verdict_frame.grid(row=1, column=0, sticky="ew", padx=18, pady=8)

        self.wf_verdict_icon = ctk.CTkLabel(self.wf_verdict_frame, text="—",
            font=ctk.CTkFont(size=40))
        self.wf_verdict_icon.pack(pady=(16, 4))
        self.wf_verdict_text = ctk.CTkLabel(self.wf_verdict_frame, text="Not started",
            font=ctk.CTkFont(size=20, weight="bold"))
        self.wf_verdict_text.pack()
        self.wf_verdict_sub = ctk.CTkLabel(self.wf_verdict_frame, text="",
            font=ctk.CTkFont(size=12), text_color=COLOR_DIM)
        self.wf_verdict_sub.pack(pady=(2, 16))

        log_frame = ctk.CTkFrame(c2, fg_color="#0a0e14", corner_radius=8)
        log_frame.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 16))
        self.wf_log = self._make_log_widget(log_frame, height=10)

    def _run_walkforward(self):
        if self._is_process_busy():
            messagebox.showwarning("Busy", "Another task running")
            return
        csv = self.wf_csv.get()
        if csv in ("(none)", ""):
            messagebox.showwarning("Setup", "Please select dataset")
            return

        cmd = [
            sys.executable, "rl_walkforward.py", csv,
            "--windows", self.wf_windows.get() or "5",
            "--steps", self.wf_steps.get() or "50000",
            "--window", self.wf_window.get() or "10",
            "--name", self.wf_name.get() or "wf_prod",
        ]

        self._log(self.wf_log, f"$ {' '.join(cmd)}", "info")
        self.wf_run_btn.configure(state="disabled")
        self.status_label.configure(text="● Walk-Forward...", text_color=COLOR_PURPLE)
        self.runner.start(cmd)

    # --------------------------------------------------------
    # PAGE: FINE-TUNE
    # --------------------------------------------------------
    def _build_finetune_page(self):
        page = ctk.CTkFrame(self.content, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        self.pages["finetune"] = page

        info = ctk.CTkLabel(page,
            text="🔄 Fine-tune — โหลด model เดิม + train ต่อด้วย data ใหม่ (เร็วกว่า train scratch 4 เท่า)",
            text_color=COLOR_YELLOW, font=ctk.CTkFont(size=13), wraplength=900)
        info.grid(row=0, column=0, sticky="w", padx=8, pady=(0, 12))

        # Base model
        c1 = Card(page, title="🤖 Base Model")
        c1.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        c1.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(c1, text="Existing Model", text_color=COLOR_DIM
                      ).grid(row=1, column=0, sticky="w", padx=18, pady=(8, 4))
        self.ft_base = ScrollableOptionMenu(c1, values=["(none)"],
            fg_color=COLOR_BG_INPUT, button_color=COLOR_BG_INPUT)
        self.ft_base.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 16))

        # Data
        c2 = Card(page, title="📊 Data")
        c2.grid(row=2, column=0, sticky="ew", pady=(0, 12))
        c2.grid_columnconfigure(0, weight=1)
        c2.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(c2, text="Old Data CSV", text_color=COLOR_DIM
                      ).grid(row=1, column=0, sticky="w", padx=18, pady=(8, 4))
        ctk.CTkLabel(c2, text="New Data CSV", text_color=COLOR_DIM
                      ).grid(row=1, column=1, sticky="w", padx=18, pady=(8, 4))
        self.ft_old = ScrollableOptionMenu(c2, values=["(none)"],
            fg_color=COLOR_BG_INPUT, button_color=COLOR_BG_INPUT)
        self.ft_old.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 12))
        self.ft_new = ScrollableOptionMenu(c2, values=["(none)"],
            fg_color=COLOR_BG_INPUT, button_color=COLOR_BG_INPUT)
        self.ft_new.grid(row=2, column=1, sticky="ew", padx=18, pady=(0, 12))

        # Settings
        c3 = Card(page, title="⚙️ Settings")
        c3.grid(row=3, column=0, sticky="ew", pady=(0, 12))
        c3.grid_columnconfigure(0, weight=1)
        c3.grid_columnconfigure(1, weight=1)
        c3.grid_columnconfigure(2, weight=1)

        ctk.CTkLabel(c3, text="Mix Ratio (old %)", text_color=COLOR_DIM
                      ).grid(row=1, column=0, sticky="w", padx=18, pady=(8, 4))
        ctk.CTkLabel(c3, text="Steps", text_color=COLOR_DIM
                      ).grid(row=1, column=1, sticky="w", padx=8, pady=(8, 4))
        ctk.CTkLabel(c3, text="New Model Name", text_color=COLOR_DIM
                      ).grid(row=1, column=2, sticky="w", padx=18, pady=(8, 4))

        self.ft_mix = ctk.CTkEntry(c3); self.ft_mix.insert(0, "0.3")
        self.ft_mix.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 16))
        self.ft_steps = ctk.CTkEntry(c3); self.ft_steps.insert(0, "50000")
        self.ft_steps.grid(row=2, column=1, sticky="ew", padx=8, pady=(0, 16))
        self.ft_name = ctk.CTkEntry(c3); self.ft_name.insert(0, "rl_prod_v2")
        self.ft_name.grid(row=2, column=2, sticky="ew", padx=18, pady=(0, 16))

        self.ft_run_btn = ctk.CTkButton(page,
            text="▶ Start Fine-tuning (~10 min)",
            command=self._run_finetune,
            fg_color=COLOR_ACCENT, hover_color="#4493f8",
            height=42, font=ctk.CTkFont(size=14, weight="bold"))
        self.ft_run_btn.grid(row=4, column=0, sticky="ew", pady=(0, 12))

        # Log
        c4 = Card(page, title="📝 Log")
        c4.grid(row=5, column=0, sticky="ew")
        c4.grid_columnconfigure(0, weight=1)
        log_frame = ctk.CTkFrame(c4, fg_color="#0a0e14", corner_radius=8)
        log_frame.grid(row=1, column=0, sticky="ew", padx=18, pady=(8, 16))
        self.ft_log = self._make_log_widget(log_frame, height=10)

    def _run_finetune(self):
        if self._is_process_busy():
            messagebox.showwarning("Busy", "Another task running")
            return

        base = self.ft_base.get()
        old = self.ft_old.get()
        new = self.ft_new.get()
        if base in ("(none)", "") or old in ("(none)", "") or new in ("(none)", ""):
            messagebox.showwarning("Setup", "Please select all 3: base model + old + new CSV")
            return

        cmd = [
            sys.executable, "rl_finetune.py", base,
            "--old_csv", old,
            "--new_csv", new,
            "--mix_ratio", self.ft_mix.get() or "0.3",
            "--steps", self.ft_steps.get() or "50000",
            "--name", self.ft_name.get() or "rl_prod_v2",
        ]

        self._log(self.ft_log, f"$ {' '.join(cmd)}", "info")
        self.ft_run_btn.configure(state="disabled")
        self.status_label.configure(text="● Fine-tuning...", text_color=COLOR_YELLOW)
        self.runner.start(cmd)

    # --------------------------------------------------------
    # PAGE: ANALYZE
    # --------------------------------------------------------
    def _build_analyze_page(self):
        page = ctk.CTkFrame(self.content, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        self.pages["analyze"] = page

        # Setup
        c1 = Card(page, title="🔍 Confidence Analysis")
        c1.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        c1.grid_columnconfigure(0, weight=1)

        info = ctk.CTkLabel(c1,
            text="ดูว่า model ทายแม่นขึ้นเมื่อมั่นใจสูงไหม → ใช้หา threshold ที่ดี",
            text_color=COLOR_DIM, font=ctk.CTkFont(size=12), wraplength=900)
        info.grid(row=1, column=0, sticky="w", padx=18, pady=(4, 12))

        s = ctk.CTkFrame(c1, fg_color="transparent")
        s.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 12))
        s.grid_columnconfigure(0, weight=1)
        s.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(s, text="Model", text_color=COLOR_DIM
                      ).grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(s, text="CSV", text_color=COLOR_DIM
                      ).grid(row=0, column=1, sticky="w", padx=8)
        self.an_model = ScrollableOptionMenu(s, values=["(none)"],
            fg_color=COLOR_BG_INPUT, button_color=COLOR_BG_INPUT)
        self.an_model.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        self.an_csv = ScrollableOptionMenu(s, values=["(none)"],
            fg_color=COLOR_BG_INPUT, button_color=COLOR_BG_INPUT)
        self.an_csv.grid(row=1, column=1, sticky="ew", padx=8, pady=(2, 0))

        ctk.CTkButton(c1, text="▶ Analyze",
            command=self._run_analyze,
            fg_color=COLOR_ACCENT, hover_color="#4493f8", height=42
            ).grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 16))

        # Log
        c2 = Card(page, title="📊 Output")
        c2.grid(row=1, column=0, sticky="ew")
        c2.grid_columnconfigure(0, weight=1)

        log_frame = ctk.CTkFrame(c2, fg_color="#0a0e14", corner_radius=8)
        log_frame.grid(row=1, column=0, sticky="ew", padx=18, pady=(8, 16))
        self.an_log = self._make_log_widget(log_frame, height=20)

    def _run_analyze(self):
        if self._is_process_busy():
            messagebox.showwarning("Busy", "Another task running")
            return
        model = self.an_model.get()
        csv = self.an_csv.get()
        if model in ("(none)", "") or csv in ("(none)", ""):
            messagebox.showwarning("Setup", "Please select model and CSV")
            return

        cmd = [sys.executable, "rl_analyze.py", model, csv]
        self._log(self.an_log, f"$ {' '.join(cmd)}", "info")
        self.status_label.configure(text="● Analyzing...", text_color=COLOR_ACCENT)
        self.runner.start(cmd)

    # --------------------------------------------------------
    # PAGE: MODELS
    # --------------------------------------------------------
    def _build_models_page(self):
        page = ctk.CTkFrame(self.content, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        self.pages["models"] = page

        c1 = Card(page, title="🗂️ Saved Models")
        c1.grid(row=0, column=0, sticky="ew")
        c1.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(c1, text="🔄 Refresh",
            command=self._refresh_models_list,
            fg_color=COLOR_BG_INPUT, hover_color="#2d333b", width=110
            ).grid(row=1, column=0, sticky="w", padx=18, pady=(8, 8))

        # Container for model list
        self.models_container = ctk.CTkFrame(c1, fg_color="transparent")
        self.models_container.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 16))
        self.models_container.grid_columnconfigure(0, weight=1)

    def _refresh_models_list(self):
        # Clear
        for w in self.models_container.winfo_children():
            w.destroy()

        # Find .zip files
        zips = sorted(WORK_DIR.glob("*.zip"))
        if not zips:
            ctk.CTkLabel(self.models_container,
                text="No models found. Train one first!",
                text_color=COLOR_DIM, font=ctk.CTkFont(size=13)
                ).grid(row=0, column=0, padx=10, pady=20)
            return

        # Header row
        header = ctk.CTkFrame(self.models_container, fg_color=COLOR_BG_INPUT,
            corner_radius=6)
        header.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        header.grid_columnconfigure(0, weight=2)
        header.grid_columnconfigure(1, weight=1)
        header.grid_columnconfigure(2, weight=1)
        for i, txt in enumerate(["NAME", "SIZE", "MODIFIED"]):
            ctk.CTkLabel(header, text=txt, text_color=COLOR_DIM,
                font=ctk.CTkFont(size=11, weight="bold")
                ).grid(row=0, column=i, sticky="w", padx=14, pady=8)

        # Rows
        for i, zp in enumerate(zips):
            stat = zp.stat()
            size_kb = stat.st_size / 1024
            modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")

            row = ctk.CTkFrame(self.models_container, fg_color="transparent")
            row.grid(row=i + 1, column=0, sticky="ew", pady=2)
            row.grid_columnconfigure(0, weight=2)
            row.grid_columnconfigure(1, weight=1)
            row.grid_columnconfigure(2, weight=1)

            ctk.CTkLabel(row, text=f"🤖  {zp.stem}",
                font=ctk.CTkFont(size=13, family="Consolas")
                ).grid(row=0, column=0, sticky="w", padx=14, pady=6)
            ctk.CTkLabel(row, text=f"{size_kb:,.0f} KB",
                text_color=COLOR_DIM, font=ctk.CTkFont(size=12)
                ).grid(row=0, column=1, sticky="w", padx=14, pady=6)
            ctk.CTkLabel(row, text=modified,
                text_color=COLOR_DIM, font=ctk.CTkFont(size=12)
                ).grid(row=0, column=2, sticky="w", padx=14, pady=6)

    # --------------------------------------------------------
    # PAGE: SETTINGS
    # --------------------------------------------------------
    def _build_settings_page(self):
        page = ctk.CTkFrame(self.content, fg_color="transparent")
        page.grid_columnconfigure(0, weight=1)
        self.pages["settings"] = page

        # ============================================================
        # CARD 1: Branding Customization ⭐ NEW
        # ============================================================
        c0 = Card(page, title="🎨 Branding Customization")
        c0.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        c0.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(c0,
            text="ปรับ logo / ชื่อ / สี ของ app — กด Save แล้ว Restart เพื่อใช้งาน",
            text_color=COLOR_DIM, font=ctk.CTkFont(size=12),
            wraplength=900, justify="left"
            ).grid(row=1, column=0, sticky="w", padx=18, pady=(2, 12))

        # === Color Preset Dropdown ===
        preset_row = ctk.CTkFrame(c0, fg_color="transparent")
        preset_row.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 12))
        preset_row.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(preset_row, text="🎨 Color Preset:",
            text_color=COLOR_DIM, font=ctk.CTkFont(size=12)
            ).grid(row=0, column=0, sticky="w")

        self.brand_preset = ctk.CTkOptionMenu(preset_row,
            values=list(COLOR_PRESETS.keys()),
            command=self._apply_color_preset,
            fg_color=COLOR_BG_INPUT, button_color=COLOR_BG_INPUT,
            width=240)
        self.brand_preset.grid(row=0, column=1, sticky="w", padx=(8, 0))

        ctk.CTkLabel(preset_row,
            text="(เปลี่ยน preset จะอัปเดตช่องสีด้านล่างทันที)",
            text_color=COLOR_DIM, font=ctk.CTkFont(size=10)
            ).grid(row=0, column=2, sticky="w", padx=(12, 0))

        # === Branding Text Fields ===
        text_grid = ctk.CTkFrame(c0, fg_color="transparent")
        text_grid.grid(row=3, column=0, sticky="ew", padx=18, pady=(8, 8))
        text_grid.grid_columnconfigure(0, weight=1)
        text_grid.grid_columnconfigure(1, weight=1)

        # Helper to add labeled entries
        self.brand_inputs = {}
        text_fields = [
            ("window_title",    "Window Title",            "ชื่อหน้าต่าง app"),
            ("logo_text",       "Logo Text",                "ชื่อใน sidebar"),
            ("logo_emoji",      "Logo Emoji (ถ้าไม่มี image)", "เช่น 🤖 ⚡ 🚀"),
            ("subtitle",        "Subtitle",                 "ข้อความใต้ logo"),
            ("logo_image",      "Logo Image Path",          "เช่น assets/logo.png"),
            ("window_icon",     "Window Icon Path",         "เช่น assets/icon.ico"),
            ("theme_label",     "Theme Toggle Label",       "ข้อความปุ่มล่าง sidebar"),
        ]
        for i, (key, label, hint) in enumerate(text_fields):
            row = i // 2
            col = i % 2
            sub = ctk.CTkFrame(text_grid, fg_color="transparent")
            sub.grid(row=row, column=col, sticky="ew", padx=4, pady=4)
            sub.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(sub, text=label, text_color=COLOR_DIM,
                font=ctk.CTkFont(size=11)
                ).grid(row=0, column=0, sticky="w", pady=(0, 2))
            entry = ctk.CTkEntry(sub, placeholder_text=hint,
                font=ctk.CTkFont(size=11))
            entry.insert(0, str(BRANDING.get(key, "")))
            entry.grid(row=1, column=0, sticky="ew")
            ctk.CTkLabel(sub, text=hint,
                text_color=COLOR_DIM, font=ctk.CTkFont(size=9)
                ).grid(row=2, column=0, sticky="w")
            self.brand_inputs[key] = entry

        # === Color Pickers ===
        ctk.CTkLabel(c0, text="🎨 Custom Colors (hex):",
            text_color=COLOR_DIM,
            font=ctk.CTkFont(size=13, weight="bold")
            ).grid(row=4, column=0, sticky="w", padx=18, pady=(12, 4))

        color_grid = ctk.CTkFrame(c0, fg_color="transparent")
        color_grid.grid(row=5, column=0, sticky="ew", padx=18, pady=(0, 12))
        for i in range(4):
            color_grid.grid_columnconfigure(i, weight=1)

        self.brand_color_inputs = {}
        color_fields = [
            ("ACCENT",     "Primary",     "สี brand หลัก, ปุ่ม"),
            ("GREEN",      "Success",     "metric ดี, win trade"),
            ("RED",        "Error",       "metric แย่, loss"),
            ("YELLOW",     "Warning",     "metric เตือน"),
            ("PURPLE",     "Special",     "Export/Chart buttons"),
            ("DIM",        "Secondary",   "text รอง, hint"),
            ("BG_CARD",    "Card BG",     "พื้นหลังการ์ด"),
            ("BG_INPUT",   "Input BG",    "พื้นหลัง input"),
            ("SIDEBAR_BG", "Sidebar BG",  "พื้นหลัง sidebar"),
        ]

        # Build current color map (from globals + BRANDING)
        cur_colors = {
            "ACCENT": COLOR_ACCENT,  "GREEN": COLOR_GREEN,
            "RED": COLOR_RED,        "YELLOW": COLOR_YELLOW,
            "PURPLE": COLOR_PURPLE,  "DIM": COLOR_DIM,
            "BG_CARD": COLOR_BG_CARD, "BG_INPUT": COLOR_BG_INPUT,
            "SIDEBAR_BG": BRANDING.get("sidebar_bg", "#161b22"),
        }

        for i, (key, label, hint) in enumerate(color_fields):
            row = i // 4
            col = i % 4
            sub = ctk.CTkFrame(color_grid, fg_color="transparent")
            sub.grid(row=row, column=col, sticky="ew", padx=4, pady=4)
            sub.grid_columnconfigure(1, weight=1)

            # ⭐ Clickable color swatch — opens color picker dialog
            swatch = ctk.CTkFrame(sub, width=28, height=28,
                fg_color=cur_colors[key], corner_radius=4,
                border_width=1, border_color="#30363d",
                cursor="hand2")
            swatch.grid(row=0, column=0, padx=(0, 6), pady=(2, 0))
            swatch.grid_propagate(False)
            # Bind click → open color picker
            swatch.bind("<Button-1>",
                lambda e, k=key: self._open_color_picker(k))

            entry = ctk.CTkEntry(sub, placeholder_text="#xxxxxx",
                font=ctk.CTkFont(family="Consolas", size=11), width=90)
            entry.insert(0, cur_colors[key])
            entry.grid(row=0, column=1, sticky="ew")

            ctk.CTkLabel(sub, text=f"{label} — {hint}  (คลิก swatch)",
                text_color=COLOR_DIM, font=ctk.CTkFont(size=9),
                anchor="w"
                ).grid(row=1, column=0, columnspan=2, sticky="w", padx=(0, 0))

            self.brand_color_inputs[key] = (entry, swatch)
            # Update swatch when entry changes (typing hex)
            entry.bind("<KeyRelease>",
                lambda e, k=key: self._update_swatch(k))

        # === Action buttons ===
        btn_row = ctk.CTkFrame(c0, fg_color="transparent")
        btn_row.grid(row=6, column=0, sticky="ew", padx=18, pady=(8, 16))
        btn_row.grid_columnconfigure(0, weight=1)
        btn_row.grid_columnconfigure(1, weight=1)
        btn_row.grid_columnconfigure(2, weight=1)

        ctk.CTkButton(btn_row, text="💾 Save & Restart",
            command=self._save_branding,
            fg_color=COLOR_ACCENT, hover_color="#4493f8",
            height=42, font=ctk.CTkFont(size=13, weight="bold")
            ).grid(row=0, column=0, sticky="ew", padx=(0, 4))

        ctk.CTkButton(btn_row, text="↩ Reset to Default",
            command=self._reset_branding,
            fg_color=COLOR_BG_INPUT, hover_color="#2d333b",
            height=42, font=ctk.CTkFont(size=13)
            ).grid(row=0, column=1, sticky="ew", padx=4)

        ctk.CTkButton(btn_row, text="📂 Open Config File",
            command=self._open_branding_config,
            fg_color=COLOR_BG_INPUT, hover_color="#2d333b",
            height=42, font=ctk.CTkFont(size=13)
            ).grid(row=0, column=2, sticky="ew", padx=(4, 0))

        # ============================================================
        # CARD 2: Application Settings (existing)
        # ============================================================
        c1 = Card(page, title="⚙️ Application Settings")
        c1.grid(row=1, column=0, sticky="ew", pady=(0, 12))
        c1.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(c1, text="Working Directory", text_color=COLOR_DIM
                      ).grid(row=1, column=0, sticky="w", padx=18, pady=(8, 4))
        ctk.CTkLabel(c1, text=str(WORK_DIR),
            text_color=COLOR_ACCENT, font=ctk.CTkFont(size=12, family="Consolas")
            ).grid(row=2, column=0, sticky="w", padx=18, pady=(0, 16))

        # ============================================================
        # CARD 3: About
        # ============================================================
        c2 = Card(page, title="ℹ️ About")
        c2.grid(row=2, column=0, sticky="ew")
        c2.grid_columnconfigure(0, weight=1)

        about = """RL Trading Studio v1.0
Modern GUI for RL Trading Pipeline

Features:
• Train PPO/DQN/A2C agents
• Realistic backtest with risk management
• Walk-forward validation
• Smart fine-tuning with data mixing
• Confidence analysis
• Model library management

Built with: CustomTkinter + stable-baselines3
"""
        ctk.CTkLabel(c2, text=about, text_color=COLOR_DIM,
            font=ctk.CTkFont(size=12), justify="left", anchor="w"
            ).grid(row=1, column=0, sticky="w", padx=18, pady=(8, 16))

    # --------------------------------------------------------
    # Branding settings handlers
    # --------------------------------------------------------
    def _apply_color_preset(self, preset_name):
        """Load a preset's colors into the entry fields"""
        preset = COLOR_PRESETS.get(preset_name)
        if not preset:
            return
        for key, (entry, swatch) in self.brand_color_inputs.items():
            val = preset.get(key, "")
            if val:
                entry.delete(0, "end")
                entry.insert(0, val)
                try:
                    swatch.configure(fg_color=val)
                except: pass

    def _update_swatch(self, key):
        """Update color swatch when entry changes"""
        entry, swatch = self.brand_color_inputs[key]
        val = entry.get().strip()
        # Validate hex format
        if not (val.startswith("#") and len(val) in (4, 7)):
            return
        try:
            swatch.configure(fg_color=val)
        except: pass

    def _open_color_picker(self, key):
        """Open native color picker dialog → update entry + swatch"""
        entry, swatch = self.brand_color_inputs[key]
        current = entry.get().strip() or "#888888"
        # askcolor returns ((r,g,b), '#rrggbb') or (None, None) if cancelled
        try:
            result = colorchooser.askcolor(
                color=current,
                title=f"Pick color for {key}",
                parent=self)
        except Exception:
            return
        if result is None or result[1] is None:
            return  # user cancelled
        new_hex = result[1]   # already in #rrggbb format
        entry.delete(0, "end")
        entry.insert(0, new_hex)
        try:
            swatch.configure(fg_color=new_hex)
        except: pass

    def _save_branding(self):
        """Save branding to JSON + ask user to restart"""
        # Collect text inputs
        branding_overrides = {}
        for key, entry in self.brand_inputs.items():
            val = entry.get().strip()
            branding_overrides[key] = val

        # Collect color inputs
        colors = {}
        for key, (entry, _) in self.brand_color_inputs.items():
            val = entry.get().strip()
            if val.startswith("#") and len(val) in (4, 7):
                colors[key] = val

        # Save to JSON
        try:
            _save_branding_config(branding_overrides, colors)
            messagebox.showinfo("Saved",
                f"✓ Branding saved to:\n{BRANDING_CONFIG_FILE}\n\n"
                f"⚠️ ปิด + เปิด app ใหม่เพื่อให้สีและ branding มีผล")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")

    def _reset_branding(self):
        """Delete config file → revert to default on next start"""
        if messagebox.askyesno("Reset",
                "ลบไฟล์ branding_config.json แล้ว restart?\n"
                "(ค่า branding จะกลับเป็น default)"):
            try:
                if BRANDING_CONFIG_FILE.exists():
                    BRANDING_CONFIG_FILE.unlink()
                messagebox.showinfo("Reset",
                    "✓ ลบ config สำเร็จ\nปิด + เปิด app ใหม่")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def _open_branding_config(self):
        """Open config file in default editor"""
        if not BRANDING_CONFIG_FILE.exists():
            messagebox.showinfo("No file",
                "ยังไม่มีไฟล์ config — กด 'Save & Restart' ก่อน เพื่อสร้างไฟล์")
            return
        try:
            import os
            os.startfile(str(BRANDING_CONFIG_FILE))
        except Exception as e:
            messagebox.showerror("Error", str(e))

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    def _refresh_dropdowns(self):
        """Refresh model + csv dropdowns from current dir"""
        models = self._list_model_names()
        csvs = self._list_csv_files()

        model_menu_names = ("bt_model", "an_model", "ft_base")
        for menu_name in model_menu_names:
            menu = getattr(self, menu_name, None)
            if menu is None:
                continue
            try:
                menu.configure(values=models)
                if menu.get() in ("(none)", "") and models[0] != "(none)":
                    menu.set(models[0])
            except: pass

        csv_menu_names = (
            "pipe_csv", "pipe_bt_csv", "bt_csv", "wf_csv", "an_csv",
            "ft_old", "ft_new",
        )
        for menu_name in csv_menu_names:
            menu = getattr(self, menu_name, None)
            if menu is None:
                continue
            try:
                menu.configure(values=csvs)
                if menu.get() in ("(none)", "") and csvs[0] != "(none)":
                    menu.set(csvs[0])
            except: pass

    def _make_log_widget(self, parent, height=10):
        """Create a Text log widget with:
          - vertical scrollbar
          - isolated mousewheel (doesn't bubble up to parent scroll)
          - color tags configured
        Returns the Text widget (for backwards compat — still passed to _log)
        """
        # Container frame to hold Text + scrollbar side-by-side
        container = tk.Frame(parent, bg=COLOR_BG_TERMINAL, bd=0, highlightthickness=0)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        # Vertical scrollbar
        scrollbar = ttk.Scrollbar(container, orient="vertical")
        scrollbar.pack(side="right", fill="y")

        # Text widget
        text = tk.Text(container,
            bg=COLOR_BG_TERMINAL, fg=COLOR_DIM,
            font=("Consolas", 11), bd=0, height=height,
            insertbackground="white", highlightthickness=0,
            yscrollcommand=scrollbar.set)
        text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=text.yview)

        # Color tags
        text.tag_configure("info",    foreground=COLOR_DIM)
        text.tag_configure("success", foreground=COLOR_GREEN)
        text.tag_configure("warn",    foreground=COLOR_YELLOW)
        text.tag_configure("error",   foreground=COLOR_RED)
        text.tag_configure("metric",  foreground=COLOR_ACCENT)

        # ⭐ Isolate mousewheel: when hovering Text widget, scroll Text itself
        # and STOP propagation to parent scrollable frame.
        def _on_mousewheel(event):
            text.yview_scroll(int(-1 * (event.delta / 120)), "units")
            return "break"  # prevent parent from also scrolling

        # Bind both for Text and Scrollbar (so user can use either)
        text.bind("<MouseWheel>", _on_mousewheel)
        scrollbar.bind("<MouseWheel>", _on_mousewheel)
        # Linux compatibility (Button-4 = up, Button-5 = down)
        text.bind("<Button-4>", lambda e: (text.yview_scroll(-1, "units"), "break")[1])
        text.bind("<Button-5>", lambda e: (text.yview_scroll(1, "units"), "break")[1])

        return text

    def _log(self, widget, text, tag="info"):
        widget.configure(state="normal")
        ts = datetime.now().strftime("%H:%M:%S")
        widget.insert("end", f"[{ts}] ", "info")
        widget.insert("end", text + "\n", tag)
        widget.see("end")
        widget.configure(state="disabled")

    def _classify_log(self, line):
        """Auto-detect log line color"""
        # Priority 1: SB3 metric line — color by health
        metric_match = re.match(r'\|\s*([a-z_]+)\s*\|\s*([+\-\d\.eE]+)\s*\|', line)
        if metric_match:
            name = metric_match.group(1)
            try:
                value = float(metric_match.group(2))
                status = classify_metric(name, value)
                if status == 'good':  return 'success'
                if status == 'warn':  return 'warn'
                if status == 'bad':   return 'error'
                # Unknown metric — use neutral
                return 'metric'
            except (ValueError, KeyError):
                pass

        l = line.lower()
        if any(w in l for w in ["error", "fail", "exception", "traceback"]):
            return "error"
        if any(w in l for w in ["warn", "skip", "abort"]):
            return "warn"
        if any(w in l for w in ["✓", "ok", "success", "[save]", "complete", "done"]):
            return "success"
        if re.search(r"step\s+\d+", l) or re.search(r"\d+%", l):
            return "metric"
        return "info"

    def _parse_progress(self, line):
        """Try to extract step / total from training logs

        Supports multiple formats from stable-baselines3:
          1. tqdm progress: "36864/200000 [03:42<..."
          2. SB3 table: "| total_timesteps | 36864 |"
          3. Generic: "Step 36864 / 200000"
        """
        # Format 1: tqdm "X/Y ["
        m = re.search(r'(\d[\d,]*)/(\d[\d,]*)\s*\[', line)
        if m:
            try:
                cur = int(m.group(1).replace(',', ''))
                tot = int(m.group(2).replace(',', ''))
                return cur, tot
            except:
                pass

        # Format 2: SB3 log table "| total_timesteps |    36864 |"
        m = re.search(r'\|\s*total_timesteps\s*\|\s*(\d+)\s*\|', line)
        total = self.__dict__.get('_train_steps_total') or self.__dict__.get(
            'pipeline_train_steps_total')
        if m and total:
            try:
                cur = int(m.group(1))
                return cur, int(total)
            except:
                pass

        return None

    def _parse_stats(self, line):
        """Parse final backtest stats"""
        stats = {}
        m = re.search(r'Win rate\s*:\s*([\d.]+)%', line)
        if m: stats['wr'] = m.group(1)
        m = re.search(r'Profit factor\s*:\s*([\d.]+|inf)', line)
        if m: stats['pf'] = m.group(1)
        m = re.search(r'Return\s*:\s*([+\-][\d.]+)%', line)
        if m: stats['ret'] = m.group(1)
        m = re.search(r'Max drawdown\s*:\s*([\-\d.]+)%', line)
        if m: stats['dd'] = m.group(1)
        return stats

    def _poll_queue(self):
        """Read subprocess output queue + dispatch to UI"""
        self._drain_pipeline_row_counts()
        try:
            while True:
                kind, data = self.runner.q.get_nowait()
                if kind == 'line':
                    self._handle_log_line(data)
                elif kind == 'done':
                    self._handle_done(data)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _handle_log_line(self, line):
        page = self.current_page
        log_widget = {
            'train': getattr(self, 'train_log', None),
            'backtest': getattr(self, 'bt_log', None),
            'walkfwd': getattr(self, 'wf_log', None),
            'finetune': getattr(self, 'ft_log', None),
            'analyze': getattr(self, 'an_log', None),
        }.get(page)
        if log_widget:
            tag = self._classify_log(line)
            self._log(log_widget, line, tag)

        # Page-specific updates
        if page == 'train':
            prog = self._parse_progress(line)
            if prog:
                cur, tot = prog
                pct = cur / tot if tot > 0 else 0
                self.train_progress.set(min(pct, 1.0))
                self.train_status.configure(
                    text=f"Step {cur:,} / {tot:,}  ({pct*100:.1f}%)")
                # remember current step for reward-threshold guard
                self._train_step_count = cur

            # ⭐ Parse metric line and update health pill
            mm = re.match(r'\|\s*([a-z_]+)\s*\|\s*([+\-\d\.eE]+)\s*\|', line)
            if mm:
                name = mm.group(1)
                try:
                    value = float(mm.group(2))
                    if name in METRIC_INFO:
                        self._update_health_pill(name, value)
                        # Track current metrics
                        if not hasattr(self, '_current_metrics'):
                            self._current_metrics = {}
                        self._current_metrics[name] = value
                        # Refresh recommendations
                        self._refresh_recommendations()

                        # ⭐ Reward graph + threshold popup
                        if name == 'ep_rew_mean':
                            self._track_reward(value)
                except (ValueError, KeyError):
                    pass

        elif page in ('backtest',):
            stats = self._parse_stats(line)
            if stats.get('wr'):
                self.stat_wr = self._update_stat(self.stat_wr, stats['wr'] + '%')
            if stats.get('pf'):
                color = COLOR_GREEN if float(stats['pf']) > 1.0 else COLOR_RED
                self.stat_pf = self._update_stat(self.stat_pf, stats['pf'], color)
            if stats.get('ret'):
                color = COLOR_GREEN if stats['ret'].startswith('+') else COLOR_RED
                self.stat_ret = self._update_stat(self.stat_ret, stats['ret'] + '%', color)
            if stats.get('dd'):
                self.stat_dd = self._update_stat(self.stat_dd, stats['dd'] + '%', COLOR_RED)

        elif page == 'walkfwd':
            if 'ROBUST' in line and 'NOT' not in line:
                self.wf_verdict_icon.configure(text="✅")
                self.wf_verdict_text.configure(text="ROBUST", text_color=COLOR_GREEN)
                self.wf_verdict_sub.configure(text="All windows passed gate")
            elif 'UNSTABLE' in line:
                self.wf_verdict_icon.configure(text="⚠️")
                self.wf_verdict_text.configure(text="UNSTABLE", text_color=COLOR_YELLOW)
                self.wf_verdict_sub.configure(text="Some windows failed")
            elif 'NOT ROBUST' in line:
                self.wf_verdict_icon.configure(text="❌")
                self.wf_verdict_text.configure(text="NOT ROBUST", text_color=COLOR_RED)
                self.wf_verdict_sub.configure(text="Most windows failed")

    def _update_stat(self, old_card, new_value, color=None):
        """Replace stat card content"""
        # Find label
        for w in old_card.winfo_children():
            if isinstance(w, ctk.CTkLabel):
                txt = w.cget("text")
                # value is the big number
                if any(c.isdigit() for c in txt) and len(txt) > 3:
                    if color:
                        w.configure(text=new_value, text_color=color)
                    else:
                        w.configure(text=new_value)
                    return old_card
                # if it's "—"
                if txt == "—":
                    if color:
                        w.configure(text=new_value, text_color=color)
                    else:
                        w.configure(text=new_value)
                    return old_card
        return old_card

    def _handle_done(self, rc):
        if rc == 0:
            msg = "✓ Done"
            color = COLOR_GREEN
        else:
            msg = f"✗ Failed (exit {rc})"
            color = COLOR_RED

        # Re-enable buttons
        for btn_attr in ['train_btn', 'bt_run_btn', 'bt_chart_btn',
                          'wf_run_btn', 'ft_run_btn']:
            if hasattr(self, btn_attr):
                getattr(self, btn_attr).configure(state="normal")
        if hasattr(self, 'train_stop_btn'):
            self.train_stop_btn.configure(state="disabled")

        self.status_label.configure(text="● Idle", text_color=COLOR_DIM)

        # Log
        page = self.current_page
        log_widget = {
            'train': getattr(self, 'train_log', None),
            'backtest': getattr(self, 'bt_log', None),
            'walkfwd': getattr(self, 'wf_log', None),
            'finetune': getattr(self, 'ft_log', None),
            'analyze': getattr(self, 'an_log', None),
        }.get(page)
        if log_widget:
            self._log(log_widget, msg,
                "success" if rc == 0 else "error")

        # Refresh model list if was training
        if page in ('train', 'finetune'):
            self._refresh_dropdowns()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    app = RLTradingStudio()
    app.mainloop()
