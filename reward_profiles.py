"""Reward profile presets for TradingEnv.

The profile changes how the agent is rewarded while training. It does not
change backtest/live execution after a model has already been trained.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

from reward_formula import validate_reward_formula


REWARD_PROFILES = {
    "balanced": {
        "label": "Balanced (default)",
        "description": "Current V4 honest reward. Balanced between real P/L, trade cost, and timely exits.",
        "params": {
            "close_pnl_mult": 50.0,
            "loss_penalty_mult": 1.0,
            "profit_bonus_threshold": 0.005,
            "profit_bonus": 0.01,
            "unrealized_scale": 0.1,
            "giveback_trigger": 0.002,
            "giveback_fraction": 0.5,
            "giveback_penalty": 0.001,
            "trade_penalty": 0.005,
            "idle_penalty": 0.0001,
            "time_decay_start": 0.7,
            "time_decay_penalty": 0.0002,
            "mtm_profit_bonus": 0.001,
            "mtm_long_hold_bars": 50,
            "mtm_long_hold_penalty": 0.0005,
            "blowup_penalty": 1.0,
        },
    },
    "anti_overtrade": {
        "label": "Anti Overtrade",
        "description": "Raises the cost of opening trades so the agent needs a clearer edge.",
        "params": {
            "close_pnl_mult": 55.0,
            "loss_penalty_mult": 1.0,
            "profit_bonus_threshold": 0.007,
            "profit_bonus": 0.012,
            "unrealized_scale": 0.06,
            "giveback_trigger": 0.002,
            "giveback_fraction": 0.5,
            "giveback_penalty": 0.0012,
            "trade_penalty": 0.010,
            "idle_penalty": 0.00005,
            "time_decay_start": 0.7,
            "time_decay_penalty": 0.00025,
            "mtm_profit_bonus": 0.001,
            "mtm_long_hold_bars": 45,
            "mtm_long_hold_penalty": 0.0006,
            "blowup_penalty": 1.0,
        },
    },
    "low_drawdown": {
        "label": "Low Drawdown",
        "description": "Punishes losses, give-back, and stale positions more aggressively.",
        "params": {
            "close_pnl_mult": 45.0,
            "loss_penalty_mult": 1.25,
            "profit_bonus_threshold": 0.006,
            "profit_bonus": 0.010,
            "unrealized_scale": 0.05,
            "giveback_trigger": 0.0015,
            "giveback_fraction": 0.6,
            "giveback_penalty": 0.0020,
            "trade_penalty": 0.006,
            "idle_penalty": 0.0001,
            "time_decay_start": 0.6,
            "time_decay_penalty": 0.0004,
            "mtm_profit_bonus": 0.001,
            "mtm_long_hold_bars": 40,
            "mtm_long_hold_penalty": 0.0008,
            "blowup_penalty": 1.25,
        },
    },
    "trend_follower": {
        "label": "Trend Follower",
        "description": "Allows longer holds and gives a little more direction signal while in position.",
        "params": {
            "close_pnl_mult": 50.0,
            "loss_penalty_mult": 1.0,
            "profit_bonus_threshold": 0.008,
            "profit_bonus": 0.014,
            "unrealized_scale": 0.15,
            "giveback_trigger": 0.003,
            "giveback_fraction": 0.45,
            "giveback_penalty": 0.0008,
            "trade_penalty": 0.004,
            "idle_penalty": 0.00008,
            "time_decay_start": 0.9,
            "time_decay_penalty": 0.00005,
            "mtm_profit_bonus": 0.001,
            "mtm_long_hold_bars": 80,
            "mtm_long_hold_penalty": 0.0002,
            "blowup_penalty": 1.0,
        },
    },
    "scalper": {
        "label": "Scalper",
        "description": "Rewards quicker exits and applies time decay earlier.",
        "params": {
            "close_pnl_mult": 65.0,
            "loss_penalty_mult": 1.05,
            "profit_bonus_threshold": 0.002,
            "profit_bonus": 0.006,
            "unrealized_scale": 0.05,
            "giveback_trigger": 0.001,
            "giveback_fraction": 0.6,
            "giveback_penalty": 0.0015,
            "trade_penalty": 0.003,
            "idle_penalty": 0.00012,
            "time_decay_start": 0.4,
            "time_decay_penalty": 0.0005,
            "mtm_profit_bonus": 0.0008,
            "mtm_long_hold_bars": 20,
            "mtm_long_hold_penalty": 0.0010,
            "blowup_penalty": 1.0,
        },
    },
}

REWARD_PARAM_SPECS = [
    {
        "key": "close_pnl_mult",
        "label": "Close PnL Mult",
        "min": 10.0,
        "max": 100.0,
        "steps": 90,
        "decimals": 0,
        "description": "น้ำหนักกำไร/ขาดทุนตอนปิดไม้",
    },
    {
        "key": "loss_penalty_mult",
        "label": "Loss Penalty Mult",
        "min": 0.5,
        "max": 2.0,
        "steps": 30,
        "decimals": 2,
        "description": "คูณเพิ่มเฉพาะไม้ที่ขาดทุน",
    },
    {
        "key": "profit_bonus_threshold",
        "label": "Bonus Threshold",
        "min": 0.0,
        "max": 0.02,
        "steps": 40,
        "decimals": 4,
        # GUI displays percent (raw × 100); stored/JSON values stay raw fractions
        "display_scale": 100,
        "display_decimals": 2,
        "display_suffix": "%",
        "description": "ต้องกำไรเกินกี่ % จึงได้ bonus",
    },
    {
        "key": "profit_bonus",
        "label": "Profit Bonus",
        "min": 0.0,
        "max": 0.05,
        "steps": 50,
        "decimals": 4,
        "description": "bonus เมื่อปิดกำไรได้มากพอ",
    },
    {
        "key": "unrealized_scale",
        "label": "Unrealized Scale",
        "min": 0.0,
        "max": 0.30,
        "steps": 60,
        "decimals": 3,
        "description": "direction hint ระหว่างถือไม้",
    },
    {
        "key": "giveback_trigger",
        "label": "Giveback Trigger",
        "min": 0.0,
        "max": 0.01,
        "steps": 50,
        "decimals": 4,
        "display_scale": 100,
        "display_decimals": 2,
        "display_suffix": "%",
        "description": "กำไรลอยตัวขั้นต่ำกี่ % ก่อนจับ give-back",
    },
    {
        "key": "giveback_fraction",
        "label": "Giveback Fraction",
        "min": 0.2,
        "max": 0.9,
        "steps": 70,
        "decimals": 2,
        "display_scale": 100,
        "display_decimals": 0,
        "display_suffix": "%",
        "description": "เหลือกี่ % ของ peak จึงโดน penalty",
    },
    {
        "key": "giveback_penalty",
        "label": "Giveback Penalty",
        "min": 0.0,
        "max": 0.01,
        "steps": 50,
        "decimals": 4,
        "description": "โทษเมื่อคืนกำไรลอยตัวมากเกินไป",
    },
    {
        "key": "trade_penalty",
        "label": "Trade Penalty",
        "min": 0.0,
        "max": 0.02,
        "steps": 80,
        "decimals": 4,
        "description": "ต้นทุน reward ตอนเปิดไม้",
    },
    {
        "key": "idle_penalty",
        "label": "Idle Penalty",
        "min": 0.0,
        "max": 0.001,
        "steps": 50,
        "decimals": 5,
        "description": "โทษตอน Hold ว่างไม่มี position",
    },
    {
        "key": "time_decay_start",
        "label": "Time Decay Start",
        "min": 0.1,
        "max": 1.0,
        "steps": 90,
        "decimals": 2,
        "display_scale": 100,
        "display_decimals": 0,
        "display_suffix": "%",
        "description": "เริ่ม decay หลังถือเกินกี่ % ของ max hold",
    },
    {
        "key": "time_decay_penalty",
        "label": "Time Decay Penalty",
        "min": 0.0,
        "max": 0.002,
        "steps": 80,
        "decimals": 5,
        "description": "โทษต่อ bar หลังเข้าเขต time decay",
    },
]
REWARD_PARAM_SPEC_BY_KEY = {spec["key"]: spec for spec in REWARD_PARAM_SPECS}
REWARD_PROFILE_JSON_SCHEMA = "metafxclub.reward_profile.v1"
DEFAULT_REWARD_PROFILE_CONFIG_DIR = Path(__file__).resolve().parent / "reward_profile_configs"
REWARD_PROFILE_LABELS = [profile["label"] for profile in REWARD_PROFILES.values()]
_LABEL_TO_KEY = {profile["label"]: key for key, profile in REWARD_PROFILES.items()}


def reward_profile_key_from_label(value: str | None) -> str:
    """Return a stable preset key from a GUI label or CLI key."""
    if not value:
        return "balanced"
    value = str(value).strip()
    if value in REWARD_PROFILES:
        return value
    if value in _LABEL_TO_KEY:
        return _LABEL_TO_KEY[value]
    normalized = value.split()[0].strip().lower().replace("-", "_")
    return normalized if normalized in REWARD_PROFILES else "balanced"


def _coerce_reward_param(key: str, value) -> float:
    spec = REWARD_PARAM_SPEC_BY_KEY.get(key)
    if spec is None:
        raise ValueError(f"unknown reward override: {key}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"reward override {key} must be numeric") from None
    if not math.isfinite(number):
        raise ValueError(f"reward override {key} must be finite")
    if number < spec["min"] or number > spec["max"]:
        raise ValueError(
            f"reward override {key}={number} outside safe range "
            f"{spec['min']}..{spec['max']}"
        )
    return number


def coerce_reward_overrides(overrides: dict | None) -> dict:
    """Validate and normalize user-adjustable reward slider overrides."""
    if not overrides:
        return {}
    if not isinstance(overrides, dict):
        raise ValueError("reward overrides must be a JSON object")
    return {key: _coerce_reward_param(key, value) for key, value in overrides.items()}


def get_reward_profile(value: str | dict | None = "balanced", overrides: dict | None = None) -> tuple[str, dict]:
    """Resolve a profile name or custom dict into ``(profile_name, params)``."""
    if isinstance(value, dict):
        params = deepcopy(REWARD_PROFILES["balanced"]["params"])
        params.update(coerce_reward_overrides(value))
        params.update(coerce_reward_overrides(overrides))
        return "custom", params

    key = reward_profile_key_from_label(value)
    params = deepcopy(REWARD_PROFILES[key]["params"])
    params.update(coerce_reward_overrides(overrides))
    return key, params


def reward_profile_label(key: str | None) -> str:
    key = reward_profile_key_from_label(key)
    return REWARD_PROFILES[key]["label"]


def reward_profile_json_payload(
    name: str,
    base_profile: str,
    overrides: dict | None = None,
    formula: str = "",
    notes: str = "",
) -> dict:
    """Build a portable JSON payload for a reward profile recipe."""
    base_key = reward_profile_key_from_label(base_profile)
    clean_overrides = coerce_reward_overrides(overrides)
    _, final_params = get_reward_profile(base_key, clean_overrides)
    formula = (formula or "").strip()
    if formula:
        validate_reward_formula(formula)
    payload = {
        "schema": REWARD_PROFILE_JSON_SCHEMA,
        "name": name or base_key,
        "base_profile": base_key,
        "base_profile_label": reward_profile_label(base_key),
        "overrides": clean_overrides,
        "params": final_params,
        "notes": notes,
    }
    if formula:
        payload["developer_mode"] = {
            "enabled": True,
            "formula": formula,
        }
    return payload


def load_reward_profile_json(path: str | Path) -> dict:
    """Load a reward profile JSON file and return normalized metadata."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("reward profile JSON must contain an object")
    schema = data.get("schema", "")
    if schema and schema != REWARD_PROFILE_JSON_SCHEMA:
        raise ValueError(f"unsupported reward profile schema: {schema}")

    base = (
        data.get("base_profile")
        or data.get("reward_profile")
        or data.get("profile")
        or "balanced"
    )
    base_key = reward_profile_key_from_label(base)
    raw_overrides = data.get("overrides")
    if raw_overrides is None:
        raw_params = data.get("params", {})
        if not isinstance(raw_params, dict):
            raw_params = {}
        raw_overrides = {
            key: raw_params[key]
            for key in raw_params
            if key in REWARD_PARAM_SPEC_BY_KEY
        }
    clean_overrides = coerce_reward_overrides(raw_overrides)
    _, final_params = get_reward_profile(base_key, clean_overrides)
    formula = ""
    developer = data.get("developer_mode")
    if isinstance(developer, dict) and developer.get("enabled", True):
        formula = str(developer.get("formula") or "").strip()
    elif isinstance(data.get("custom_formula"), str):
        formula = str(data.get("custom_formula") or "").strip()
    elif isinstance(data.get("formula"), str):
        formula = str(data.get("formula") or "").strip()
    if formula:
        validate_reward_formula(formula)
    return {
        "path": str(path),
        "schema": schema or REWARD_PROFILE_JSON_SCHEMA,
        "name": data.get("name") or path.stem,
        "base_profile": base_key,
        "base_profile_label": reward_profile_label(base_key),
        "overrides": clean_overrides,
        "params": final_params,
        "formula": formula,
        "raw": data,
    }
