"""Round-8 portfolio-construction researches (R9-R20).

Operates on the stored daily return streams of the five frequency members
(no new backtests): each research is a different allocation/overlay rule.
All rules use only trailing data (shift(1)) - no lookahead. Verdicts are
formed on the dev window; the holdout column is reported for the record.
"""

import numpy as np
import pandas as pd

MEMBERS = ["exp_r6_30m", "exp_r6_1h", "exp_r6_2h", "exp_r6_vol150", "exp_r6_8h"]
SPLIT = "2024-01-01"


def load():
    def daily(name):
        c = pd.read_csv("dynamic_results/%s/portfolio_values.csv" % name,
                        index_col=0, parse_dates=True).iloc[:, 0]
        return c.resample("1D").last().ffill().pct_change().dropna()
    return pd.DataFrame({m: daily(m) for m in MEMBERS}).dropna()


def meta_gt(frame):
    K = frame.shape[1]
    mean = frame.ewm(halflife=30, min_periods=30).mean()
    std = frame.ewm(halflife=30, min_periods=30).std()
    s = (mean / std).clip(lower=0.0)
    w = s.div(s.sum(axis=1), axis=0).fillna(1.0 / K)
    w = w.clip(lower=0.5 / K, upper=2.0 / K)
    return w.div(w.sum(axis=1), axis=0).shift(1).fillna(1.0 / K)


def stats(r):
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    out = {}
    for win, c in (("dev", r[:SPLIT]), ("hold", r[SPLIT:])):
        e = (1 + c).cumprod()
        y = len(c) / 365.25
        out[win + "_cagr"] = (e.iloc[-1] ** (1 / y) - 1) * 100
        out[win + "_sharpe"] = c.mean() / c.std() * np.sqrt(365.25)
        out[win + "_mdd"] = (1 - e / e.cummax()).max() * 100
    out["worst_yr"] = yearly.min() * 100
    out["yr_std"] = yearly.std() * 100
    return out


def main():
    frame = load()
    K = frame.shape[1]
    gt_weights = meta_gt(frame)
    meta = (gt_weights * frame).sum(axis=1)
    ledger = {}
    ledger["CONTROL META-GT"] = meta

    # R9 risk parity: inverse trailing 60d vol
    inv = (1.0 / frame.rolling(60).std()).replace(np.inf, np.nan)
    w9 = inv.div(inv.sum(axis=1), axis=0).shift(1).fillna(1.0 / K)
    ledger["R9 risk-parity"] = (w9 * frame).sum(axis=1)

    # R10 Calmar tilt: trailing 180d return/MDD, bounded like META-GT
    def trailing_calmar(col):
        eq = (1 + col).cumprod()
        ret = eq / eq.shift(180) - 1
        roll_max = eq.rolling(180).max()
        mdd = (1 - eq / roll_max).rolling(180).max()
        return ret / mdd.clip(lower=0.02)
    cal = frame.apply(trailing_calmar).clip(lower=0.0)
    w10 = cal.div(cal.sum(axis=1), axis=0).fillna(1.0 / K)
    w10 = w10.clip(lower=0.5 / K, upper=2.0 / K)
    w10 = w10.div(w10.sum(axis=1), axis=0).shift(1).fillna(1.0 / K)
    ledger["R10 calmar-tilt"] = (w10 * frame).sum(axis=1)

    # R11 downside-vol targeting on META-GT (target = trailing median)
    downside = meta.clip(upper=0.0).rolling(60).std() * np.sqrt(365.25)
    target = downside.expanding(120).median()
    scale11 = (target / downside).clip(upper=1.0).shift(1).fillna(1.0)
    ledger["R11 downside-vol target"] = meta * scale11

    # R12 correlation gate: high average pairwise member corr -> de-gross
    corr = frame.rolling(60).corr().groupby(level=0).apply(
        lambda c: (c.values.sum() - K) / (K * (K - 1)))
    scale12 = pd.Series(np.where(corr > 0.85, 0.7, 1.0), index=frame.index)
    ledger["R12 corr-gate"] = meta * scale12.shift(1).fillna(1.0)

    # R13 vol-of-vol brake: fast vol >> slow vol -> half size
    fast = meta.rolling(10).std()
    slow = meta.rolling(60).std()
    scale13 = pd.Series(np.where(fast > 1.5 * slow, 0.5, 1.0), index=meta.index)
    ledger["R13 vol-of-vol brake"] = meta * scale13.shift(1).fillna(1.0)

    # R14 monthly (not daily) meta-weight updates
    w14 = gt_weights.resample("1MS").first().reindex(frame.index).ffill() \
        .fillna(1.0 / K)
    ledger["R14 monthly meta-rebal"] = (w14 * frame).sum(axis=1)

    # R16 kill-switch: every member's trailing 60d Sharpe negative -> cash
    sharpe60 = frame.rolling(60).mean() / frame.rolling(60).std()
    all_neg = (sharpe60 < 0).all(axis=1).shift(1).fillna(False)
    ledger["R16 kill-switch"] = meta * (~all_neg)

    # R17 skew tilt: prefer members with positive trailing 90d skew
    skew = frame.rolling(90).skew().clip(lower=0.0) + 0.1
    w17 = skew.div(skew.sum(axis=1), axis=0).fillna(1.0 / K)
    w17 = w17.clip(lower=0.5 / K, upper=2.0 / K)
    w17 = w17.div(w17.sum(axis=1), axis=0).shift(1).fillna(1.0 / K)
    ledger["R17 skew-tilt"] = (w17 * frame).sum(axis=1)

    # R18 self-momentum: meta's own trailing 90d return negative -> half
    own = (1 + meta).rolling(90).apply(np.prod, raw=True) - 1
    scale18 = pd.Series(np.where(own < 0, 0.5, 1.0), index=meta.index)
    ledger["R18 self-momentum"] = meta * scale18.shift(1).fillna(1.0)

    # R19 equal risk contribution (full covariance, monthly refit)
    def erc(cov):
        w = np.ones(K) / K
        for _ in range(200):
            m = cov @ w
            w = w * (w @ m) / (K * w * m).clip(1e-12)
            w = np.clip(w, 0, None); w /= w.sum()
        return w
    w19 = pd.DataFrame(index=frame.index, columns=MEMBERS, dtype=float)
    for month, chunk in frame.groupby(pd.Grouper(freq="1MS")):
        history = frame[:chunk.index[0]].iloc[:-1].tail(120)
        if len(history) >= 60:
            w19.loc[chunk.index] = erc(history.cov().values)
    w19 = w19.fillna(1.0 / K)
    ledger["R19 ERC"] = (w19 * frame).sum(axis=1)

    rows = {name: stats(r.dropna()) for name, r in ledger.items()}
    table = pd.DataFrame(rows).T
    print(table.to_string(float_format=lambda v: "%7.2f" % v))
    table.to_csv("dynamic_results/research_ledger_portfolio.csv")


if __name__ == "__main__":
    main()
