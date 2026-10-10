"""obs_ablation --train_start must trim TRAIN rows only: the Validation/Test
rows of the trimmed file have to be exactly the rows the untrimmed split gives
(rl_train splits at int(n * train_pct); backtest_live --start cuts at
int(n * start))."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools" / "analysis"))
import obs_ablation as oa  # noqa: E402


def _csv(tmp_path, n):
    ts = pd.date_range("2015-01-01", periods=n, freq="4h")
    df = pd.DataFrame({"timestamp": ts.strftime("%Y.%m.%d %H:%M"),
                       "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 1,
                       "f1": np.arange(n, dtype=float)})
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    (tmp_path / "data.params.json").write_text("{}", encoding="utf-8")
    return p, df


@pytest.mark.parametrize("n,pct,start", [(17143, 0.70, "2019-01-01"),
                                         (5000, 0.70, "2016-03-01"),
                                         (9001, 0.80, "2017-07-15")])
def test_trim_keeps_validation_and_test_rows(tmp_path, n, pct, start):
    src, df = _csv(tmp_path, n)
    split = int(n * pct)
    test0 = split + (n - split) // 2
    out, new_pct, new_oos, info = oa.trim_for_train_start(src, start, pct)
    trimmed = pd.read_csv(out)
    m = len(trimmed)
    first = info["rows_dropped"]
    assert first > 0 and m == n - first
    assert int(m * new_pct) == split - first          # rl_train split row
    assert int(m * new_oos) == test0 - first          # backtest --start row
    assert list(trimmed.f1[int(m * new_pct):]) == list(df.f1[split:])
    assert pd.Timestamp(trimmed.timestamp.iloc[0].replace(".", "-")) >= pd.Timestamp(start)
    assert Path(out).with_suffix(".params.json").exists()


def test_trim_rejects_start_after_split(tmp_path):
    src, _ = _csv(tmp_path, 5000)
    with pytest.raises(ValueError):
        oa.trim_for_train_start(src, "2030-01-01", 0.70)


def test_frac_for_row_is_exact():
    for n in (101, 5767, 11910, 17143):
        for k in (1, n // 3, n // 2, n - 1):
            assert int(n * oa._frac_for_row(n, k)) == k
