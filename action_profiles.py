"""Action profile presets for the RL trading flow.

Action profiles define what the model is allowed to do. They affect the
training action space, the backtest execution layer, and the generated MT5 EA.
Changing the profile changes the model output dimension, so old models cannot
be reused with a different action profile.
"""
from copy import deepcopy
import json
import math
from pathlib import Path


ACTION_PROFILE_JSON_SCHEMA = "metafxclub.action_profile.v1"
DEFAULT_ACTION_PROFILE_CONFIG_DIR = Path(__file__).resolve().parent / "action_profile_configs"


ACTION_PRIMITIVES = {
    "hold": {
        "id": 0,
        "label": "Hold",
        "description": "Do nothing.",
        "role": "hold",
    },
    "buy": {
        "id": 1,
        "label": "Buy",
        "description": "Open a long position when flat.",
        "role": "entry",
    },
    "sell": {
        "id": 2,
        "label": "Sell",
        "description": "Open a short position when flat.",
        "role": "entry",
    },
    "close": {
        "id": 3,
        "label": "Close",
        "description": "Close current position.",
        "role": "exit",
    },
    "move_sl_breakeven": {
        "id": 4,
        "label": "Move SL to BE",
        "description": "Move stop loss to entry once floating profit is high enough.",
        "role": "manage",
    },
    "trail_sl_atr": {
        "id": 5,
        "label": "Trail SL by ATR",
        "description": "Trail stop loss behind price using ATR distance.",
        "role": "manage",
    },
}

ACTION_PARAM_SPECS = [
    {
        "key": "breakeven_min_profit",
        "label": "BE Min Profit",
        "min": 0.0,
        "max": 0.05,
        "steps": 100,
        "decimals": 4,
        "unit": "สัดส่วนกำไรเทียบราคาเข้า (0.0010 = 0.10%)",
        "description": "Minimum floating P/L before Move SL to BE is allowed.",
    },
    {
        "key": "trail_atr_period",
        "label": "Trail ATR Period",
        "min": 2,
        "max": 100,
        "steps": 98,
        "decimals": 0,
        "unit": "จำนวนแท่งย้อนหลัง (bars)",
        "description": "Bars used for ATR proxy in trail action.",
    },
    {
        "key": "trail_atr_mult",
        "label": "Trail ATR Mult",
        "min": 0.1,
        "max": 10.0,
        "steps": 99,
        "decimals": 2,
        "unit": "ตัวคูณ ATR (เช่น 2.0 = 2 x ATR)",
        "description": "ATR multiplier used as trailing distance.",
    },
    {
        "key": "trail_min_profit",
        "label": "Trail Min Profit",
        "min": 0.0,
        "max": 0.05,
        "steps": 100,
        "decimals": 4,
        "unit": "สัดส่วนกำไรเทียบราคาเข้า (0.0010 = 0.10%)",
        "description": "Minimum floating P/L before ATR trailing is allowed.",
    },
    {
        "key": "management_action_penalty",
        "label": "Manage Penalty",
        "min": 0.0,
        "max": 0.01,
        "steps": 100,
        "decimals": 5,
        "unit": "คะแนน reward ที่หักต่อ management action",
        "description": "Small reward cost when the agent uses a management action.",
    },
    {
        "key": "invalid_action_penalty",
        "label": "Invalid Action Penalty",
        "min": 0.0,
        "max": 0.02,
        "steps": 100,
        "decimals": 5,
        "unit": "คะแนน reward ที่หักต่อ invalid action",
        "description": "Reward cost when a management action cannot be applied.",
    },
]
ACTION_PARAM_SPEC_BY_KEY = {spec["key"]: spec for spec in ACTION_PARAM_SPECS}


DEFAULT_ACTION_PARAMS = {
    "breakeven_min_profit": 0.0010,
    "trail_atr_period": 14,
    "trail_atr_mult": 2.0,
    "trail_min_profit": 0.0010,
    "management_action_penalty": 0.0002,
    "invalid_action_penalty": 0.0010,
}


ACTION_PROFILES = {
    "basic_4": {
        "label": "Basic 4 (Hold/Buy/Sell/Close)",
        "description": "Current default action space. Safest baseline and compatible with existing behavior.",
        "actions": ["hold", "buy", "sell", "close"],
        "params": deepcopy(DEFAULT_ACTION_PARAMS),
    },
    "manage_6": {
        "label": "Manage 6 (+ BE/Trailing)",
        "description": "Adds stop-management actions: move SL to breakeven and trail SL by ATR.",
        "actions": ["hold", "buy", "sell", "close", "move_sl_breakeven", "trail_sl_atr"],
        "params": deepcopy(DEFAULT_ACTION_PARAMS),
    },
}

ACTION_PROFILE_LABELS = [profile["label"] for profile in ACTION_PROFILES.values()]
_LABEL_TO_KEY = {profile["label"]: key for key, profile in ACTION_PROFILES.items()}


def action_profile_key_from_label(value: str | None) -> str:
    """Return a stable action profile key from a GUI label or CLI key."""
    if not value:
        return "basic_4"
    value = str(value).strip()
    if value in ACTION_PROFILES:
        return value
    if value in _LABEL_TO_KEY:
        return _LABEL_TO_KEY[value]
    normalized = value.split()[0].strip().lower().replace("-", "_")
    return normalized if normalized in ACTION_PROFILES else "basic_4"


def action_profile_label(key: str | None) -> str:
    key = action_profile_key_from_label(key)
    return ACTION_PROFILES[key]["label"]


def _coerce_action_param(key: str, value):
    spec = ACTION_PARAM_SPEC_BY_KEY.get(key)
    if spec is None:
        raise ValueError(f"unknown action parameter: {key}")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"action parameter {key} must be numeric") from None
    if not math.isfinite(number):
        raise ValueError(f"action parameter {key} must be finite")
    if number < spec["min"] or number > spec["max"]:
        raise ValueError(
            f"action parameter {key}={number} outside safe range "
            f"{spec['min']}..{spec['max']}"
        )
    if spec.get("decimals", 0) == 0:
        return int(round(number))
    return number


def coerce_action_params(params: dict | None) -> dict:
    """Validate and normalize user-adjustable action parameters."""
    if not params:
        return {}
    if not isinstance(params, dict):
        raise ValueError("action parameters must be a JSON object")
    return {key: _coerce_action_param(key, value) for key, value in params.items()}


def _build_actions(action_names: list[str]) -> list[dict]:
    if not action_names:
        raise ValueError("action profile must include at least one action")
    clean = []
    seen = set()
    for name in action_names:
        name = str(name).strip()
        if name not in ACTION_PRIMITIVES:
            raise ValueError(f"unsupported action primitive: {name}")
        if name in seen:
            raise ValueError(f"duplicate action primitive: {name}")
        seen.add(name)
        item = deepcopy(ACTION_PRIMITIVES[name])
        item["name"] = name
        clean.append(item)

    ids = [action["id"] for action in clean]
    expected = list(range(len(clean)))
    if ids != expected:
        raise ValueError(
            "actions must use contiguous canonical ids starting at 0; "
            f"got {ids}, expected {expected}"
        )
    return clean


def get_action_profile(value: str | dict | None = "basic_4", params: dict | None = None) -> tuple[str, dict]:
    """Resolve a profile key or JSON-like profile into ``(profile_name, profile)``."""
    if isinstance(value, dict):
        data = value
        base_key = action_profile_key_from_label(data.get("base_profile") or data.get("profile"))
        profile = deepcopy(ACTION_PROFILES[base_key])
        action_names = data.get("actions") or profile["actions"]
        if action_names and isinstance(action_names[0], dict):
            action_names = [item.get("name") for item in action_names]
        profile["actions"] = _build_actions(action_names)
        clean_params = deepcopy(DEFAULT_ACTION_PARAMS)
        clean_params.update(coerce_action_params(profile.get("params", {})))
        clean_params.update(coerce_action_params(data.get("params", {})))
        clean_params.update(coerce_action_params(params))
        profile["params"] = clean_params
        profile["label"] = data.get("label") or data.get("name") or "Custom Action Profile"
        profile["description"] = data.get("description") or profile.get("description", "")
        return "custom", profile

    key = action_profile_key_from_label(value)
    profile = deepcopy(ACTION_PROFILES[key])
    profile["actions"] = _build_actions(profile["actions"])
    profile["params"].update(coerce_action_params(params))
    return key, profile


def action_names_for_profile(profile: dict) -> dict[int, str]:
    return {int(action["id"]): str(action["label"]) for action in profile["actions"]}


def action_ids_by_name(profile: dict) -> dict[str, int]:
    """Return primitive name -> action id for a resolved profile."""
    result = {}
    for action in profile["actions"]:
        for name, primitive in ACTION_PRIMITIVES.items():
            if int(primitive["id"]) == int(action["id"]):
                result[name] = int(action["id"])
                break
    return result


def profile_for_action_count(n_actions: int) -> tuple[str, dict]:
    """Best-effort fallback for older models without train metadata."""
    for key, profile in ACTION_PROFILES.items():
        if len(profile["actions"]) == int(n_actions):
            return get_action_profile(key)
    raise ValueError(
        f"model has {n_actions} actions but no matching built-in action profile; "
        "provide --action_profile or retrain with metadata"
    )


def action_profile_json_payload(
    name: str,
    base_profile: str,
    params: dict | None = None,
    notes: str = "",
) -> dict:
    """Build a portable JSON payload for a limited developer action profile."""
    base_key = action_profile_key_from_label(base_profile)
    _, profile = get_action_profile(base_key, params)
    return {
        "schema": ACTION_PROFILE_JSON_SCHEMA,
        "name": name or base_key,
        "base_profile": base_key,
        "base_profile_label": action_profile_label(base_key),
        "actions": [
            {
                "id": int(action["id"]),
                "label": action["label"],
                "role": action["role"],
                "description": action["description"],
            }
            for action in profile["actions"]
        ],
        "params": profile["params"],
        "notes": notes,
    }


def load_action_profile_json(path: str | Path) -> dict:
    """Load a limited action-profile JSON file and return normalized metadata."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError("action profile JSON must contain an object")
    schema = data.get("schema", "")
    if schema and schema != ACTION_PROFILE_JSON_SCHEMA:
        raise ValueError(f"unsupported action profile schema: {schema}")

    base_key = action_profile_key_from_label(
        data.get("base_profile") or data.get("action_profile") or data.get("profile")
    )
    raw_actions = data.get("actions")
    if raw_actions is None:
        raw_actions = ACTION_PROFILES[base_key]["actions"]
    elif raw_actions and isinstance(raw_actions[0], dict):
        raw_actions = [
            item.get("name") or item.get("key") or _primitive_name_for_id(item.get("id"))
            for item in raw_actions
        ]
    _, profile = get_action_profile(
        {
            "base_profile": base_key,
            "actions": raw_actions,
            "params": data.get("params", {}),
            "name": data.get("name") or path.stem,
            "description": data.get("description", ""),
        }
    )
    return {
        "path": str(path),
        "schema": schema or ACTION_PROFILE_JSON_SCHEMA,
        "name": data.get("name") or path.stem,
        "base_profile": base_key,
        "base_profile_label": action_profile_label(base_key),
        "action_profile": "custom",
        "profile": profile,
        "params": profile["params"],
        "raw": data,
    }


def _primitive_name_for_id(action_id) -> str:
    for name, primitive in ACTION_PRIMITIVES.items():
        if int(primitive["id"]) == int(action_id):
            return name
    raise ValueError(f"unsupported action id: {action_id}")
