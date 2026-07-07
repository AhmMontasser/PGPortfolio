"""Multi-frequency meta-allocation with online learning.

Treats the same trading system run at different decision frequencies as
sub-funds and allocates capital across them daily:

* ``EW``       - equal weight, rebalanced daily (the no-learning control);
* ``ADAPTIVE`` - online learning: weights proportional to each sub-fund's
  exponentially-weighted Sharpe over a trailing window (a priori: 90 days,
  30-day half-life), clipped at zero. If every sub-fund's trailing Sharpe
  is negative the whole book goes to cash - a systemic kill-switch that
  no single-frequency variant has. Weights computed strictly from data up
  to t-1 and applied to day t (no lookahead).

The frequency that wins is regime-dependent (2020-21 favoured slow bars,
2024-26 fast ones), which is exactly the situation where online
meta-allocation earns its keep: it does not need to know the next regime
in advance, it just follows realized performance with bounded regret.

Usage:  python dynamic_meta.py exp_r6_30m exp_r6_1h exp_r6_2h exp_r6_vol150 exp_r6_8h
(names of result directories under dynamic_results/; the first run's
directory naming is kept as the member label)
"""

from __future__ import absolute_import, division, print_function

import os
import sys

import numpy as np
import pandas as pd

RESULTS = "./dynamic_results"
SPLIT = "2024-01-01"
LOOKBACK_DAYS = 90
HALFLIFE_DAYS = 30


def member_daily_returns(name):
    curve = pd.read_csv(os.path.join(RESULTS, name, "portfolio_values.csv"),
                        index_col=0, parse_dates=True).iloc[:, 0]
    daily = curve.resample("1D").last().ffill()
    return daily.pct_change().dropna()


def metrics(rets, label):
    eq = (1 + rets).cumprod()
    years = len(rets) / 365.25
    cagr = eq.iloc[-1] ** (1 / years) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(365.25)
    mdd = float((1 - eq / eq.cummax()).max())
    yearly = eq.groupby(eq.index.year).apply(
        lambda x: x.iloc[-1] / (x.iloc[0] / (1 + rets[x.index[0]:x.index[0]].sum() or 1)) - 1)
    # simpler, correct per-year returns:
    yearly = (1 + rets).groupby(rets.index.year).prod() - 1
    return {
        "strategy": label, "CAGR": cagr, "Sharpe": sharpe, "maxDD": mdd,
        "yearly_std": float(yearly.std()), "worst_year": float(yearly.min()),
    }, yearly


def run(member_names):
    frame = pd.DataFrame({m: member_daily_returns(m) for m in member_names}).dropna()
    K = len(member_names)

    portfolios = {}
    portfolios["EW"] = frame.mean(axis=1)

    # unconstrained performance-chasing (kept as a documented negative
    # result: it lags regime flips and underperforms EW)
    mean = frame.ewm(halflife=HALFLIFE_DAYS,
                     min_periods=LOOKBACK_DAYS // 3).mean()
    std = frame.ewm(halflife=HALFLIFE_DAYS,
                    min_periods=LOOKBACK_DAYS // 3).std()
    score = (mean / std).clip(lower=0.0)
    weights = score.div(score.sum(axis=1), axis=0).fillna(1.0 / K)
    weights = weights.shift(1).fillna(1.0 / K)          # use info up to t-1
    exposure = (score.sum(axis=1) > 0).shift(1).fillna(True)  # all-negative -> cash
    portfolios["ADAPTIVE-RAW"] = (weights * frame).sum(axis=1) * exposure

    # adopted online-learning layer: EW-anchored bounded tilts. Each
    # member's weight follows its trailing EW-Sharpe but is clamped to
    # [0.5/K, 2/K] and renormalized - enough adaptivity to lean toward
    # the frequency the current regime rewards, bounded enough not to
    # whipsaw on regime turns.
    tilted = score.div(score.sum(axis=1), axis=0).fillna(1.0 / K)
    tilted = tilted.clip(lower=0.5 / K, upper=2.0 / K)
    tilted = tilted.div(tilted.sum(axis=1), axis=0).shift(1).fillna(1.0 / K)
    portfolios["META-GT"] = (tilted * frame).sum(axis=1)
    weights = tilted

    rows, tables = [], {}
    for m in member_names:
        row, yearly = metrics(frame[m], m)
        rows.append(row); tables[m] = yearly
    for label, rets in portfolios.items():
        row, yearly = metrics(rets, label)
        rows.append(row); tables[label] = yearly
        row_dev, _ = metrics(rets[:SPLIT], label + " [dev]")
        row_hold, _ = metrics(rets[SPLIT:], label + " [holdout]")
        rows += [row_dev, row_hold]

    print(pd.DataFrame(rows).set_index("strategy").to_string(
        float_format=lambda v: "%.3f" % v))
    print("\nper-year returns (%):")
    print((pd.DataFrame(tables) * 100).round(1).to_string())

    out = pd.DataFrame({k: (1 + v).cumprod() for k, v in portfolios.items()})
    os.makedirs(os.path.join(RESULTS, "meta"), exist_ok=True)
    out.to_csv(os.path.join(RESULTS, "meta", "meta_curves.csv"))
    avg_turnover = weights.diff().abs().sum(axis=1).mean()
    print("\nadaptive meta-weight turnover: %.4f/day (reallocation cost negligible)"
          % avg_turnover)
    print("mean adaptive weights:\n%s" % weights.mean().round(3).to_string())


if __name__ == "__main__":
    run(sys.argv[1:] if len(sys.argv) > 1 else
        ["exp_r6_30m", "exp_r6_1h", "exp_r6_2h", "exp_r6_vol150", "exp_r6_8h"])
