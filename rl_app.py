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
from tkinter import filedialog, messagebox, ttk
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

WORK_DIR = Path(__file__).parent.resolve()


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
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace',
                cwd=str(WORK_DIR), bufsize=1,
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
        super().__init__(master, fg_color=COLOR_BG_CARD, corner_radius=12,
                          border_width=1, border_color="#30363d", **kwargs)
        self.grid_columnconfigure(0, weight=1)
        if title:
            self.title_label = ctk.CTkLabel(self, text=title,
                font=ctk.CTkFont(size=15, weight="bold"))
            self.title_label.grid(row=0, column=0, sticky="w", padx=18, pady=(14, 4))


class StatCard(ctk.CTkFrame):
    """Metric stat card"""
    def __init__(self, master, label, value, change="", color=None, **kwargs):
        super().__init__(master, fg_color=COLOR_BG_INPUT, corner_radius=8,
                          border_width=1, border_color="#30363d", **kwargs)
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


# ============================================================
# Main Application
# ============================================================
class RLTradingStudio(ctk.CTk):

    PAGES = [
        ("tools",    "🛠️", "Data Tools"),
        ("train",    "🎯", "Train"),
        ("backtest", "📊", "Backtest"),
        ("walkfwd",  "🔬", "Walk-Forward"),
        ("finetune", "🔄", "Fine-tune"),
        ("analyze",  "🔍", "Analyze"),
        ("models",   "🗂️", "Models"),
        ("settings", "⚙️", "Settings"),
    ]

    PAGE_TITLES = {
        "tools":    "🛠️ Data Tools",
        "train":    "🎯 Train New Model",
        "backtest": "📊 Backtest Model",
        "walkfwd":  "🔬 Walk-Forward Validation",
        "finetune": "🔄 Fine-tune Existing Model",
        "analyze":  "🔍 Confidence Analysis",
        "models":   "🗂️ Model Library",
        "settings": "⚙️ Settings",
    }

    def __init__(self):
        super().__init__()
        self.title("RL Trading Studio")
        self.geometry("1280x800")
        self.minsize(1100, 700)

        # State
        self.current_page = "train"
        self.runner = ProcessRunner()
        self.nav_buttons = {}
        self.pages = {}

        self._build_ui()
        self.show_page("train")
        self._poll_queue()

    # --------------------------------------------------------
    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main()

    def _build_sidebar(self):
        side = ctk.CTkFrame(self, width=240, fg_color="#161b22",
                             corner_radius=0)
        side.grid(row=0, column=0, sticky="nsew")
        side.grid_propagate(False)
        side.grid_rowconfigure(99, weight=1)

        # Logo
        logo_frame = ctk.CTkFrame(side, fg_color="transparent")
        logo_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 16))
        ctk.CTkLabel(logo_frame, text="🤖 RL Studio",
                      font=ctk.CTkFont(size=20, weight="bold"),
                      text_color=COLOR_ACCENT
                      ).pack(anchor="w")
        ctk.CTkLabel(logo_frame, text="v1.0 · Trading AI",
                      font=ctk.CTkFont(size=11),
                      text_color=COLOR_DIM
                      ).pack(anchor="w")

        # Separator
        ctk.CTkFrame(side, height=1, fg_color="#30363d"
                      ).grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))

        # Nav items
        for i, (key, icon, label) in enumerate(self.PAGES):
            btn = ctk.CTkButton(side,
                text=f"  {icon}   {label}",
                anchor="w",
                fg_color="transparent",
                text_color=COLOR_DIM,
                hover_color="#2d333b",
                corner_radius=0,
                height=42,
                font=ctk.CTkFont(size=14),
                command=lambda k=key: self.show_page(k),
            )
            btn.grid(row=2 + i, column=0, sticky="ew", padx=0, pady=1)
            self.nav_buttons[key] = btn

        # Theme toggle (bottom)
        theme_frame = ctk.CTkFrame(side, fg_color="transparent")
        theme_frame.grid(row=100, column=0, sticky="ew", padx=15, pady=15)

        ctk.CTkFrame(theme_frame, height=1, fg_color="#30363d"
                      ).pack(fill="x", pady=(0, 10))

        self.theme_switch = ctk.CTkSwitch(theme_frame,
            text="Dark Mode",
            command=self._toggle_theme,
            font=ctk.CTkFont(size=12),
            text_color=COLOR_DIM,
        )
        self.theme_switch.select()
        self.theme_switch.pack(anchor="w")

    def _build_main(self):
        # Container for top bar + content
        self.main = ctk.CTkFrame(self, fg_color="#0f1419", corner_radius=0)
        self.main.grid(row=0, column=1, sticky="nsew")
        self.main.grid_columnconfigure(0, weight=1)
        self.main.grid_rowconfigure(1, weight=1)

        # Top bar
        topbar = ctk.CTkFrame(self.main, fg_color="#0f1419",
                               border_width=0, corner_radius=0, height=70)
        topbar.grid(row=0, column=0, sticky="ew")
        topbar.grid_propagate(False)
        topbar.grid_columnconfigure(0, weight=1)

        self.page_title_label = ctk.CTkLabel(topbar, text="🎯 Train New Model",
            font=ctk.CTkFont(size=22, weight="bold"))
        self.page_title_label.grid(row=0, column=0, sticky="w", padx=30, pady=20)

        self.status_label = ctk.CTkLabel(topbar, text="● Idle",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=COLOR_DIM)
        self.status_label.grid(row=0, column=1, sticky="e", padx=30, pady=20)

        # Separator
        ctk.CTkFrame(self.main, height=1, fg_color="#30363d"
                      ).grid(row=0, column=0, sticky="sew")

        # Content area (scrollable)
        self.content = ctk.CTkScrollableFrame(self.main,
            fg_color="#0f1419", corner_radius=0)
        self.content.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
        self.content.grid_columnconfigure(0, weight=1)

        # Build pages
        self._build_tools_page()
        self._build_train_page()
        self._build_backtest_page()
        self._build_walkfwd_page()
        self._build_finetune_page()
        self._build_analyze_page()
        self._build_models_page()
        self._build_settings_page()

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

        # Update nav buttons
        for k, btn in self.nav_buttons.items():
            if k == key:
                btn.configure(fg_color="#2d333b", text_color=COLOR_ACCENT)
            else:
                btn.configure(fg_color="transparent", text_color=COLOR_DIM)

        # Hide all pages
        for k, page in self.pages.items():
            page.grid_remove()

        # Show selected
        if key in self.pages:
            self.pages[key].grid(row=0, column=0, sticky="nsew", padx=30, pady=20)

        # Update title
        self.page_title_label.configure(text=self.PAGE_TITLES[key])

        # Refresh dynamic content
        if key == "models":
            self._refresh_models_list()
        elif key == "tools":
            self._refresh_tools_dropdowns()
        elif key in ("backtest", "walkfwd", "finetune", "analyze"):
            self._refresh_dropdowns()

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
        self.tool_split_csv = ctk.CTkOptionMenu(c2, values=["(none)"],
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

        self.tool_relabel_csv = ctk.CTkOptionMenu(relabel_grid, values=["(none)"],
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

        self.tool_feat_csv = ctk.CTkOptionMenu(feat_grid, values=["(none)"],
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

        self.tool_export_model = ctk.CTkOptionMenu(ex_grid, values=["(none)"],
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
        # CARD 6: Log
        # =====================================================
        c6 = Card(page, title="📝 Tools Log")
        c6.grid(row=6, column=0, sticky="ew")
        c6.grid_columnconfigure(0, weight=1)

        log_frame = ctk.CTkFrame(c6, fg_color="#0a0e14", corner_radius=8,
                                   border_width=1, border_color="#30363d")
        log_frame.grid(row=1, column=0, sticky="ew", padx=18, pady=(8, 16))

        self.tools_log = self._make_log_widget(log_frame, height=10)

    # --------------------------------------------------------
    # Tools functions
    # --------------------------------------------------------
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

            split_dt = pd.to_datetime(split_date)

            train_df = df[df['timestamp'] < split_dt]
            test_df = df[df['timestamp'] >= split_dt]

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
        csvs = sorted([p.name for p in WORK_DIR.glob("*.csv")]) or ["(none)"]
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
            models = set()
            for p in WORK_DIR.glob("*.zip"):
                models.add(p.stem)
            for p in WORK_DIR.glob("*_best"):
                if p.is_dir() and (p / "best_model.zip").exists():
                    models.add(p.name.replace("_best", ""))
            model_list = sorted(models) or ["(none)"]
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
        presets = {
            "default": {"lr": "3e-4", "clip": "0.2", "ent": "0.01",
                         "nsteps": "2048", "nepochs": "10", "batch": "64",
                         "gamma": "0.99", "gae": "0.95", "vf": "0.5"},
            "stable":  {"lr": "1e-4", "clip": "0.1", "ent": "0.01",
                         "nsteps": "4096", "nepochs": "10", "batch": "128",
                         "gamma": "0.99", "gae": "0.95", "vf": "0.5"},
            "fast":    {"lr": "5e-4", "clip": "0.3", "ent": "0.01",
                         "nsteps": "1024", "nepochs": "5",  "batch": "64",
                         "gamma": "0.99", "gae": "0.95", "vf": "0.5"},
            "explore": {"lr": "3e-4", "clip": "0.2", "ent": "0.05",
                         "nsteps": "2048", "nepochs": "10", "batch": "64",
                         "gamma": "0.99", "gae": "0.95", "vf": "0.5"},
        }
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
        if self.runner.is_running():
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
        if self.runner.is_running():
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
        self.bt_model = ctk.CTkOptionMenu(setup, values=["(none)"],
            fg_color=COLOR_BG_INPUT, button_color=COLOR_BG_INPUT)
        self.bt_model.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 12))

        ctk.CTkLabel(setup, text="Test Dataset", text_color=COLOR_DIM
                      ).grid(row=3, column=0, sticky="w", padx=18, pady=(0, 4))
        self.bt_csv = ctk.CTkOptionMenu(setup, values=["(none)"],
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

        # Window size — ⚠️ ต้องตรงกับตอน train!
        ctk.CTkLabel(risk, text="Window Size (must match training)",
                      text_color=COLOR_DIM, font=ctk.CTkFont(size=12)
                      ).grid(row=5, column=0, columnspan=2, sticky="w", padx=18, pady=(0, 4))
        self.bt_window = ctk.CTkEntry(risk, placeholder_text="10 (default)")
        self.bt_window.insert(0, "10")
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
        if self.runner.is_running():
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
        if self.runner.is_running():
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
            "--window", self.bt_window.get() or "10",
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
        self.wf_csv = ctk.CTkOptionMenu(c1, values=["(none)"],
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
        if self.runner.is_running():
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
        self.ft_base = ctk.CTkOptionMenu(c1, values=["(none)"],
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
        self.ft_old = ctk.CTkOptionMenu(c2, values=["(none)"],
            fg_color=COLOR_BG_INPUT, button_color=COLOR_BG_INPUT)
        self.ft_old.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 12))
        self.ft_new = ctk.CTkOptionMenu(c2, values=["(none)"],
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
        if self.runner.is_running():
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
        self.an_model = ctk.CTkOptionMenu(s, values=["(none)"],
            fg_color=COLOR_BG_INPUT, button_color=COLOR_BG_INPUT)
        self.an_model.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        self.an_csv = ctk.CTkOptionMenu(s, values=["(none)"],
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
        if self.runner.is_running():
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

        c1 = Card(page, title="⚙️ Application Settings")
        c1.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        c1.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(c1, text="Working Directory", text_color=COLOR_DIM
                      ).grid(row=1, column=0, sticky="w", padx=18, pady=(8, 4))
        ctk.CTkLabel(c1, text=str(WORK_DIR),
            text_color=COLOR_ACCENT, font=ctk.CTkFont(size=12, family="Consolas")
            ).grid(row=2, column=0, sticky="w", padx=18, pady=(0, 16))

        c2 = Card(page, title="ℹ️ About")
        c2.grid(row=1, column=0, sticky="ew")
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
    # Helpers
    # --------------------------------------------------------
    def _refresh_dropdowns(self):
        """Refresh model + csv dropdowns from current dir"""
        models = sorted([p.stem for p in WORK_DIR.glob("*.zip")]) or ["(none)"]
        csvs = sorted([p.name for p in WORK_DIR.glob("*.csv")]) or ["(none)"]

        for menu in [self.bt_model, self.an_model, self.ft_base]:
            try:
                menu.configure(values=models)
                if menu.get() in ("(none)", "") and models[0] != "(none)":
                    menu.set(models[0])
            except: pass

        for menu in [self.bt_csv, self.wf_csv, self.an_csv,
                     self.ft_old, self.ft_new]:
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
        container = tk.Frame(parent, bg="#0a0e14", bd=0, highlightthickness=0)
        container.pack(fill="both", expand=True, padx=10, pady=10)

        # Vertical scrollbar
        scrollbar = ttk.Scrollbar(container, orient="vertical")
        scrollbar.pack(side="right", fill="y")

        # Text widget
        text = tk.Text(container,
            bg="#0a0e14", fg=COLOR_DIM,
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
        if m and hasattr(self, '_train_steps_total'):
            try:
                cur = int(m.group(1))
                return cur, self._train_steps_total
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
            'train': self.train_log,
            'backtest': self.bt_log,
            'walkfwd': self.wf_log,
            'finetune': self.ft_log,
            'analyze': self.an_log,
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
            'train': self.train_log,
            'backtest': self.bt_log,
            'walkfwd': self.wf_log,
            'finetune': self.ft_log,
            'analyze': self.an_log,
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
