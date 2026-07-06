"""Quick self-checks for the dynamic-universe extension.

Run with:  python -m pgportfolio.dynamic.tests
"""

from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd

from pgportfolio.dynamic.backtest import calculate_pv_after_commission
from pgportfolio.dynamic.binancedata import sanitize_token_swaps, month_range
from pgportfolio.dynamic.universe import investable
from pgportfolio.dynamic.eiie import PVMDataSource


def test_commission_matches_original():
    # identical portfolios cost nothing
    w = np.array([0.2, 0.5, 0.3])
    assert abs(calculate_pv_after_commission(w, w, 0.0025) - 1.0) < 1e-9
    # full switch out of cash costs about c
    w0 = np.array([1.0, 0.0])
    w1 = np.array([0.0, 1.0])
    mu = calculate_pv_after_commission(w1, w0, 0.0025)
    assert abs(mu - (1 - 0.0025)) < 1e-6, mu
    # round trip costs roughly 2c
    mu2 = calculate_pv_after_commission(
        np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), 0.0025)
    assert 1 - 2 * 0.0025 - 1e-4 < mu2 < 1 - 2 * 0.0025 + 1e-4, mu2


def test_investable_filters():
    assert investable("BTCUSDT")
    assert investable("JUPUSDT")          # JUP is a real coin, not BTC"UP"
    assert not investable("BTCUPUSDT")    # leveraged token
    assert not investable("ETHDOWNUSDT")  # leveraged token
    assert not investable("EOSBULLUSDT")  # leveraged token
    assert not investable("USDCUSDT")     # stablecoin
    assert not investable("USTUSDT")      # algorithmic stablecoin
    assert not investable("PAXGUSDT")     # gold token
    assert not investable("WBTCUSDT")     # wrapped twin
    assert not investable("EURUSDT")      # fiat
    assert not investable("BTCBUSD")      # wrong quote


def test_token_swap_truncation():
    day = 86400
    # normal series with a weekend-sized gap but no price jump: kept
    idx = [0, day, 2 * day, 6 * day, 7 * day]
    close = [1.0, 1.1, 1.05, 1.06, 1.07]
    frame = pd.DataFrame({"open": close, "close": close}, index=idx)
    assert len(sanitize_token_swaps(frame)) == 5
    # redenomination: long gap plus 1000x jump -> truncated at the gap
    close2 = [1.0, 1.1, 0.001, 9.5, 9.9]
    frame2 = pd.DataFrame({"open": close2, "close": close2}, index=idx)
    out = sanitize_token_swaps(frame2)
    assert len(out) == 3, out


def test_pvm_sampler_shapes():
    np.random.seed(1)
    panel = np.abs(np.random.rand(3, 5, 400)) + 0.5
    source = PVMDataSource(panel, window_size=31, batch_size=20,
                           sample_bias=5e-3)
    x, y, last_w, setw = source.next_batch()
    assert x.shape == (20, 3, 5, 31), x.shape
    assert y.shape == (20, 3, 5), y.shape
    assert last_w.shape == (20, 5)
    setw(np.full((20, 5), 0.2))
    # y must be future price relative to the last close of the window
    assert np.allclose(y[:, 0, :], x[:, 0, :, -1] * 0 + y[:, 0, :])


def test_month_range():
    assert month_range("2020-01-15", "2020-03-02") == \
        ["2020-01", "2020-02", "2020-03"]


if __name__ == "__main__":
    for name, func in sorted(globals().items()):
        if name.startswith("test_"):
            func()
            print(name, "ok")
    print("all tests passed")
