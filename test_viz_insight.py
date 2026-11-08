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
import { layerHealth, healthAdvice, pca2, centerCols, entryResults, moveHorizon, forwardMove, sampleRows, participation, DEAD_STD } from __JS__;
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
        cls.r = json.loads(out.stdout.strip().splitlines()[-1])

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


class TemplateWiringTests(unittest.TestCase):
    def test_viewer_imports_and_controls(self):
        html = (ROOT / "tools/viz/network3d.html").read_text(encoding="utf-8")
        self.assertIn('"viz-insight": "./insight.js"', html)
        self.assertIn('id="iIns"', html)
        self.assertIn('id="insHealth"', html); self.assertIn('id="insRegime"', html)
        self.assertIn("S.health = layerHealth(S);", html)
        self.assertIn("body.show #insight { display: none !important; }", html)
        self.assertIn("#insight { left: 24px; top: 150px;", html)          # 2026-11-08: left side, drivers list off
        self.assertIn('const DRIVERS = params.get("drivers") === "1";', html)

    def test_snapshot_records_layer_activation(self):
        src = (ROOT / "tools/viz/snapshot.py").read_text(encoding="utf-8")
        self.assertIn('"layer_act":', src)


if __name__ == "__main__":
    unittest.main()
