"""Source-contract checks for MQL entry stops (not broker execution tests)."""
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "mt5_files/MQL5/Experts/ML_RL_Trader_template.mq5"


def function_body(source, name):
    start = re.search(r"\b(?:void|double|bool) " + name + r"\([^)]*\)\s*\{", source).end()
    depth = 1
    for index in range(start, len(source)):
        depth += (source[index] == "{") - (source[index] == "}")
        if depth == 0:
            return source[start:index]
    raise AssertionError(f"Unclosed function: {name}")


def assert_entry_contract(case, source):
    body = function_body(source, "OpenPosition")
    case.assertIn("double sl = 0.0, tp = 0.0;", body)
    guard = re.search(r"if\(InpUseSLTP\)\s*\{([^{}]*)\}", body)
    case.assertIsNotNone(guard)
    case.assertIn("CalcSLTP((action == 1) ? 1 : -1, price, sl, tp);", guard[1])
    # No other branch may initialize entry brackets or overwrite the zeros.
    outside = body[:guard.start()] + body[guard.end():]
    case.assertNotIn("CalcSLTP(", outside)
    case.assertEqual(len(re.findall(r"\bsl\s*=(?!=)", body)), 1)
    case.assertEqual(len(re.findall(r"\btp\s*=(?!=)", body)), 1)
    for side in ("Buy", "Sell"):
        case.assertIn(f"g_trade.{side}(lot, _Symbol, price, sl, tp, InpComment)", body)


class EntryStopsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = TEMPLATE.read_text(encoding="utf-8")

    def test_entry_flag_controls_both_buy_and_sell_brackets(self):
        assert_entry_contract(self, self.source)

    def test_no_sl_uses_zero_distance_and_balance_sizing(self):
        body = function_body(self.source, "OpenPosition")
        self.assertIn("double sl_dist = (sl > 0.0) ? MathAbs(price - sl) : 0.0;", body)
        self.assertIn("if(!InpUseSLTP || sl_distance <= 0) return CalcLotByBalance();",
                      function_body(self.source, "CalcLot"))

    def test_enabled_brackets_keep_directional_atr_calculation(self):
        body = function_body(self.source, "CalcSLTP")
        for statement in ("out_sl = price - sl_dist;", "out_tp = price + tp_dist;",
                          "out_sl = price + sl_dist;", "out_tp = price - tp_dist;"):
            self.assertIn(statement, body)
        self.assertIn("CopyBuffer(g_h_atr14_ref", body)

    def test_managed_sl_remains_independent_and_preserves_existing_tp(self):
        for name in ("MoveSLToBreakeven", "TrailSLByATR"):
            body = function_body(self.source, name)
            self.assertIn("if(!InpEnableActionManagement) return;", body)
            self.assertNotIn("InpUseSLTP", body)
        body = function_body(self.source, "ImproveSelectedSL")
        self.assertIn("double tp = PositionGetDouble(POSITION_TP);", body)
        self.assertIn("g_trade.PositionModify(ticket, proposed_sl, tp)", body)


if __name__ == "__main__":
    unittest.main()
