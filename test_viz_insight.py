"""tools/viz/insight.js — layer health and regime map maths, run through node (ES module).

Synthetic snapshot-like objects: a layer with constant neurons must be reported dead, one with all
activations at ±1 pinned, PCA must separate two clusters, and the trade-result join must attach each
entry bar to the P&L of the trade it opened. Skipped when node is not on PATH."""
import json
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JS = ROOT / "tools" / "viz" / "insight.js"
NODE = shutil.which("node")

# one hidden layer of 6 neurons over 40 frames; q is int8 scaled by 127
SCRIPT = r"""
import { layerHealth, healthAdvice, pca2, centerCols, entryResults, moveHorizon, forwardMove, sampleRows, participation, DEAD_STD,
         discountedReturns, explainedVariance, criticCheck, criticVerdict, learningAdvice, usageAdvice } from __JS__;
const NF = 40, NIN = 2, L = { tower: 0, idx: 0, size: 6, off: NIN }, W = NIN + 6;
const Q = new Int8Array(NF * W);
for (let f = 0; f < NF; f++) {
  const o = f * W + NIN, s = Math.sin(f / 3);
  Q[o + 0] = Math.round(s * 100);          // live
  Q[o + 1] = Math.round(-s * 60);          // live
  Q[o + 2] = 50;                           // dead (constant)
  Q[o + 3] = 50;                           // dead
  Q[o + 4] = f % 2 ? 127 : -127;           // pinned, but not dead
  Q[o + 5] = Math.round(s * 30);           // live
}
const s = { Q, W, NF, NIN, layers: [L], towerLayers: [[L], []], D: {} };
const H = layerHealth(s);
const adv = healthAdvice(H);
// pca: two clusters along a line
const m = 20, n = 3, X = new Float64Array(m * n);
for (let t = 0; t < m; t++) { const c = t < 10 ? -1 : 1; X[t * n] = c * 2 + (t % 3) * 0.01; X[t * n + 1] = c * 1; X[t * n + 2] = (t % 2) * 0.02; }
const P = pca2(centerCols({ X, m, n }));
const sepOK = [...Array(10).keys()].every(t => Math.sign(P.xy[t * 2]) === Math.sign(P.xy[0]) && Math.sign(P.xy[(t + 10) * 2]) === -Math.sign(P.xy[0]));
// participation ratio: 4 iid dimensions ~ 4, one dimension = 1
const R = 400, A = new Float64Array(R * 4); let seed = 7; const rnd = () => (seed = (seed * 48271) % 2147483647) / 2147483647 - 0.5;
for (let i = 0; i < A.length; i++) A[i] = rnd();
const pr4 = participation(centerCols({ X: A, m: R, n: 4 }));
const B = new Float64Array(R * 4); for (let t = 0; t < R; t++) { const v = rnd(); for (let j = 0; j < 4; j++) B[t * 4 + j] = v * (j + 1); }
const pr1 = participation(centerCols({ X: B, m: R, n: 4 }));
// trades: long entry at 3 closed at 6 (+10bp), short entry at 8 still open
const D = { pos: [0,0,0,1,1,1,0,0,-1,-1], tpnl: [0,0,0,0,0,0,0.001,0,0,0], c: [1,1.01,1.02,1.03,1.02,1.01,1.0,1.01,1.02,1.03] };
const E = entryResults(D), h = moveHorizon(D, E), F = forwardMove(D, h);
// critic: returns, explained variance, GAE path on a rollout-like snapshot
const disc = Array.from(discountedReturns([1, 0, 1], [0, 0, 2], 0.5));            // bootstrap V_last = 2
const evPerfect = explainedVariance([1, 2, 3, 4], [1, 2, 3, 4]), evMean = explainedVariance([2.5, 2.5, 2.5, 2.5], [1, 2, 3, 4]);
const NFc = 5, Wc = 3, Qc = new Int8Array(NFc * Wc); for (let f = 0; f < NFc; f++) Qc[f * Wc + 2] = 127 * (f % 2 ? 1 : -1);
const sc = { Q: Qc, W: Wc, NF: NFc, OFF_VAL: 2, D: { value_max: 2, adv: [0.5, 0.5, 0.5, 0.5, 0.5] } };
const cc = criticCheck(sc), noR = criticCheck({ ...sc, D: { value_max: 2 } });
// learning: every layer slowed to < 25% of its first change and the best reward is 3+ evals old -> stalled
const L1 = [{ tower: 0, idx: 0, size: 8, sat: 0.9 }, { tower: 0, idx: 1, size: 4, sat: 0.1 }];
const mk = (i, d, r) => ({ i, step: (i + 1) * 10000, reward: r, layers: L1, delta: { actor: d, critic: [] } });
const stalled = learningAdvice([mk(0, [0.4, 0.3], 1), mk(1, [0.2, 0.2], 3), mk(2, [0.05, 0.05], 2), mk(3, [0.04, 0.04], 2), mk(4, [0.03, 0.03], 1)]);
const moving = learningAdvice([mk(0, [0.4, 0.3], 1), mk(1, [0.3, 0.3], 2), mk(2, [0.3, 0.25], 3)]);
const frozen = learningAdvice([mk(0, [0.4, 0.3], 1), mk(1, [0.001, 0.3], 2)]);
const noDelta = learningAdvice([{ ...mk(0, null, 1), delta: null }]);
// usage over evals: never used / dropped / learned
const U = (i, use) => ({ i, use, names: ["a", "b", "c", "d"] });
const ua = usageAdvice([U(0, [0, 0.2, 0, 0.3]), U(1, [0.001, 0.1, 0.02, 0.3]), U(2, [0, 0, 0.09, 0.3])]);
console.log(JSON.stringify({ disc, evPerfect, evMean, ccKind: cc.kind, ccR: Array.from(cc.R), ccEv: cc.ev, noR,
  vBad: criticVerdict(-0.1).cls, vOk: criticVerdict(0.7).cls,
  stalled: stalled.map(a => a.t.slice(0, 22)), moving: moving.map(a => a.t.slice(0, 22)), frozen: frozen.map(a => a.t.slice(0, 30)), noDelta: noDelta[0].t.slice(0, 30),
  ua: ua.map(a => a.t) }));
console.log(JSON.stringify({ dead: Array.from(H[0].dead), nDead: H[0].nDead, sat: H[0].satShare, dims: H[0].dims, cls: H[0].cls,
  adv: adv.map(a => a.cls), explained: P.explained, sepOK, pr4, pr1,
  side: Array.from(E.side), pnl: Array.from(E.pnl).map(v => Number.isFinite(v) ? v : null), close: Array.from(E.close), h, f0: F[0],
  rows: Array.from(sampleRows(1000, 400)).slice(0, 3), rowsN: sampleRows(1000, 400).length, rowsAll: sampleRows(50, 400).length, deadStd: DEAD_STD }));
"""


@unittest.skipUnless(NODE, "node not installed")
class InsightJsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        src = SCRIPT.replace("__JS__", json.dumps(JS.resolve().as_uri()))
        out = subprocess.run([NODE, "--input-type=module", "-e", src], capture_output=True, text=True, cwd=str(ROOT), timeout=60)
        if out.returncode:
            raise AssertionError(out.stderr)
        lines = out.stdout.strip().splitlines()
        cls.r, cls.b = json.loads(lines[-1]), json.loads(lines[-2])

    def test_dead_and_pinned_neurons(self):
        r = self.r
        self.assertEqual(r["dead"], [0, 0, 1, 1, 0, 0])
        self.assertEqual(r["nDead"], 2)
        self.assertAlmostEqual(r["sat"], 1 / 6, places=3)        # one of six neurons always at ±1
        self.assertEqual(r["cls"], "bad")                         # 2/6 dead >= 25%
        self.assertIn("bad", r["adv"])

    def test_effective_dimension(self):
        r = self.r
        self.assertGreater(r["pr4"], 3.5); self.assertLessEqual(r["pr4"], 4.0 + 1e-9)
        self.assertAlmostEqual(r["pr1"], 1.0, places=6)
        self.assertLess(r["dims"], 3)                             # the synthetic layer is one sine + a square wave

    def test_pca_separates_clusters(self):
        self.assertTrue(self.r["sepOK"])
        self.assertGreater(self.r["explained"], 0.99)

    def test_entry_results_and_horizon(self):
        r = self.r
        self.assertEqual(r["side"], [0, 0, 0, 1, 0, 0, 0, 0, -1, 0])
        self.assertAlmostEqual(r["pnl"][3], 0.001)
        self.assertIsNone(r["pnl"][8])                            # still open
        self.assertEqual(r["close"][3], 6)
        self.assertEqual(r["h"], 3)                               # median hold = 3 bars
        self.assertAlmostEqual(r["f0"], 0.03, places=9)

    def test_row_sampling(self):
        r = self.r
        self.assertEqual(r["rowsN"], 400); self.assertEqual(r["rowsAll"], 50)
        self.assertEqual(r["rows"][0], 0); self.assertEqual(r["deadStd"], 0.02)


    def test_critic_returns_and_explained_variance(self):
        b = self.b
        self.assertEqual(b["disc"], [1 + 0.5 * (0 + 0.5 * (1 + 0.5 * 2)), 0.5 * (1 + 0.5 * 2), 1 + 0.5 * 2])
        self.assertAlmostEqual(b["evPerfect"], 1.0)
        self.assertAlmostEqual(b["evMean"], 0.0)
        self.assertEqual(b["ccKind"], "gae")                       # rollout: R = advantage + V
        self.assertEqual(b["ccR"], [-1.5, 2.5, -1.5, 2.5, -1.5])
        self.assertIsNone(b["noR"])                                # no reward, no advantage -> nothing to check
        self.assertEqual((b["vBad"], b["vOk"]), ("bad", "ok"))

    def test_learning_advice(self):
        b = self.b
        self.assertTrue(any(s.startswith("learning has stalled") for s in b["stalled"]))
        self.assertFalse(any(s.startswith("learning has stalled") for s in b["moving"]))   # still moving: no stall
        self.assertTrue(any(s.startswith("ACTOR L1 moves") for s in b["frozen"]))
        self.assertTrue(b["noDelta"].startswith("weight change per layer"))

    def test_usage_advice(self):
        ua = " | ".join(self.b["ua"])
        self.assertIn("1 of 4 inputs never changed", ua)
        self.assertIn("(a)", ua)                                   # a: never >= 1%
        self.assertIn("1 inputs mattered early and are ignored now (b)", ua)
        self.assertIn("1 inputs the policy learned to use during training (c)", ua)


class TemplateWiringTests(unittest.TestCase):
    def test_viewer_imports_and_controls(self):
        html = (ROOT / "tools/viz/network3d.html").read_text(encoding="utf-8")
        self.assertIn('"viz-insight": "./insight.js"', html)
        self.assertIn('id="iIns"', html)
        for k in ("health", "learning", "inputs", "critic", "regime"):
            self.assertIn(f'<option value="{k}">', html)
        self.assertIn('id="insHealth"', html); self.assertIn('id="insRegime"', html)
        self.assertIn("S.health = layerHealth(S);", html)
        self.assertIn("body.show #insight { display: none !important; }", html)
        self.assertIn("#insight { left: 24px; top: 150px;", html)          # 2026-11-08: left side, drivers list off
        self.assertIn('const DRIVERS = params.get("drivers") === "1";', html)
        # 2026-11-08: candle chart off by default (?chart=1), switch rings / faint lines (HEALTH), Δ frames + changed lines (LEARNING)
        self.assertIn('const CHART = params.get("chart") === "1";', html)
        self.assertIn('if (params.get("chart") !== "1") chartPass.enabled = false;', html)
        self.assertIn("if (vR > 0.5)", html)
        self.assertIn("else if (FAINT && FAINT[N.eSrc[e]]) sg *= 0.22;", html)
        self.assertIn("function layerFrames()", html)

    def test_snapshot_records_layer_activation(self):
        src = (ROOT / "tools/viz/snapshot.py").read_text(encoding="utf-8")
        self.assertIn('"layer_act":', src)
        train = (ROOT / "rl_train.py").read_text(encoding="utf-8")
        self.assertIn('meta["layer_delta"] = viz_snapshot.layer_delta(prev_w, w)', train)
        self.assertIn("def _init_callback(self):", train)
        self.assertIn("prev_weights=prev_w)", train)


if __name__ == "__main__":
    unittest.main()
