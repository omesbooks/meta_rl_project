"""Safe expression support for developer-mode reward formulas."""
import ast
import math


DEFAULT_REWARD_FORMULA = (
    "(unrealized_delta * unrealized_scale)"
    " - (giveback_penalty if giveback else 0)"
    " + (trade_closed_pnl * close_pnl_mult * "
    "(loss_penalty_mult if trade_closed_pnl < 0 else 1) if trade_closed else 0)"
    " + (profit_bonus if trade_closed and trade_closed_pnl > profit_bonus_threshold else 0)"
    " - (trade_penalty if just_opened else 0)"
    " - (idle_penalty if is_idle else 0)"
    " - (time_decay_penalty if time_decay_active else 0)"
)

SAFE_FORMULA_FUNCTIONS = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "tanh": math.tanh,
    "clip": lambda value, low, high: min(max(value, low), high),
    "sign": lambda value: -1.0 if value < 0 else (1.0 if value > 0 else 0.0),
}

REWARD_FORMULA_VARIABLES = {
    "action",
    "position",
    "just_opened",
    "is_idle",
    "in_position",
    "trade_closed",
    "trade_closed_pnl",
    "current_unrealized",
    "unrealized_delta",
    "peak_unrealized",
    "giveback",
    "bars_in_position",
    "max_hold_bars",
    "time_decay_active",
    "prev_equity",
    "current_equity",
    "equity_delta",
    "close_pnl_mult",
    "loss_penalty_mult",
    "profit_bonus_threshold",
    "profit_bonus",
    "unrealized_scale",
    "giveback_trigger",
    "giveback_fraction",
    "giveback_penalty",
    "trade_penalty",
    "idle_penalty",
    "time_decay_start",
    "time_decay_penalty",
    "blowup_penalty",
}

_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


def validate_reward_formula(formula: str) -> ast.Expression:
    """Validate a developer reward formula and return its parsed AST."""
    formula = (formula or "").strip()
    if not formula:
        raise ValueError("custom reward formula is empty")
    if len(formula) > 4000:
        raise ValueError("custom reward formula is too long")

    try:
        tree = ast.parse(formula, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"invalid formula syntax: {exc.msg}") from None

    nodes = list(ast.walk(tree))
    if len(nodes) > 300:
        raise ValueError("custom reward formula is too complex")

    allowed_names = REWARD_FORMULA_VARIABLES | set(SAFE_FORMULA_FUNCTIONS)
    for node in nodes:
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"formula uses unsupported syntax: {type(node).__name__}")
        if isinstance(node, ast.Name):
            if node.id.startswith("__") or node.id not in allowed_names:
                raise ValueError(f"unknown formula name: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_FORMULA_FUNCTIONS:
                raise ValueError("formula calls are limited to safe helper functions")
            if node.keywords:
                raise ValueError("formula helper functions do not accept keyword arguments")
    return tree


def compile_reward_formula(formula: str):
    tree = validate_reward_formula(formula)
    return compile(tree, "<reward_formula>", "eval")


def eval_reward_formula(compiled_formula, variables: dict) -> float:
    namespace = dict(SAFE_FORMULA_FUNCTIONS)
    namespace.update(variables)
    value = eval(compiled_formula, {"__builtins__": {}}, namespace)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("custom reward formula returned non-finite reward")
    return value
