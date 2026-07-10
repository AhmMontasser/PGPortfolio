"""Build the B3 system curve from the five re-run frequency sleeves.

Pipeline (mirrors the repo's own code exactly):
  1. sleeve daily returns   -- research_portfolio.load()'s resampling
  2. META-GT weights        -- research_portfolio.meta_gt() verbatim
  3. R20c overlay stack     -- R11 downside-vol target, R12 corr gate,
                               R13 vol-of-vol brake (all shift(1) causal),
                               multiplied onto META-GT
  4. risk-budget scaling    -- constant k solved so the FULL-period max
                               drawdown equals the locked base budget 18.2%
                               (the round-8 'base budget' definition)
Validates the result against the committed dynamic_results/meta/
final_r8_capped.csv (same construction run by the original authors).
"""
import os
import numpy as np
import pandas as pd

PG = "/home/user/PGPortfolio"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "b3_out")
os.makedirs(OUT, exist_ok=True)

MEMBERS = ["exp_r6_30m", "exp_r6_1h", "exp_r6_2h", "exp_r6_vol150", "exp_r6_8h"]
TARGET_MDD = None  # solved from committed curve below


def daily(name):
    c = pd.read_csv(os.path.join(PG, "dynamic_results", name, "portfolio_values.csv"),
                    index_col=0, parse_dates=True).iloc[:, 0]
    return c.resample("1D").last().ffill().pct_change().dropna()


def meta_gt(frame):
    K = frame.shape[1]
    mean = frame.ewm(halflife=30, min_periods=30).mean()
    std = frame.ewm(halflife=30, min_periods=30).std()
    s = (mean / std).clip(lower=0.0)
    w = s.div(s.sum(axis=1), axis=0).fillna(1.0 / K)
    w = w.clip(lower=0.5 / K, upper=2.0 / K)
    return w.div(w.sum(axis=1), axis=0).shift(1).fillna(1.0 / K)


def overlay_stack(frame, meta):
    K = frame.shape[1]
    downside = meta.clip(upper=0.0).rolling(60).std() * np.sqrt(365.25)
    target = downside.expanding(120).median()
    s11 = (target / downside).clip(upper=1.0).shift(1).fillna(1.0)
    corr = frame.rolling(60).corr().groupby(level=0).apply(
        lambda c: (c.values.sum() - K) / (K * (K - 1)))
    s12 = pd.Series(np.where(corr > 0.85, 0.7, 1.0), index=frame.index) \
        .shift(1).fillna(1.0)
    fast = meta.rolling(10).std()
    slow = meta.rolling(60).std()
    s13 = pd.Series(np.where(fast > 1.5 * slow, 0.5, 1.0), index=meta.index) \
        .shift(1).fillna(1.0)
    return meta * s11 * s12 * s13


def mdd_of(r, k):
    eq = (1 + k * r).cumprod()
    return float((1 - eq / eq.cummax()).max())


def solve_k(r, target):
    lo, hi = 0.01, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if mdd_of(r, mid) > target:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def stats(r, label):
    eq = (1 + r).cumprod()
    years = len(r) / 365.25
    cagr = eq.iloc[-1] ** (1 / years) - 1
    sh = r.mean() / r.std() * np.sqrt(365.25)
    mdd = float((1 - eq / eq.cummax()).max())
    print(f"  {label:28s} x{eq.iloc[-1]:7.2f}  CAGR {cagr*100:+6.1f}%  "
          f"MDD {mdd*100:5.1f}%  Sharpe {sh:5.2f}")
    return eq


def main():
    frame = pd.DataFrame({m: daily(m) for m in MEMBERS}).dropna()
    print(f"sleeve daily-return frame: {frame.shape[0]} days x {frame.shape[1]}")
    meta = (meta_gt(frame) * frame).sum(axis=1)
    stack = overlay_stack(frame, meta)

    committed = pd.read_csv(os.path.join(PG, "dynamic_results/meta/final_r8_capped.csv"),
                            index_col=0, parse_dates=True).iloc[:, 0].dropna()
    target_mdd = float((1 - committed / committed.cummax()).max())
    print(f"committed final_r8_capped MDD (budget target): {target_mdd*100:.2f}%")

    k = solve_k(stack, target_mdd)
    b3 = k * stack
    print(f"solved budget scale k = {k:.4f}")

    print("--- my reconstruction:")
    eq_b3 = stats(b3, "B3 (rebuilt)")
    print("--- committed reference:")
    stats(committed.pct_change().dropna(), "B3 (committed)")

    # correlation check on the overlap
    mine = eq_b3
    both = pd.DataFrame({"mine": mine, "ref": committed / committed.iloc[0]}).dropna()
    logr_corr = both["mine"].pct_change().corr(both["ref"].pct_change())
    print(f"daily-return correlation rebuilt-vs-committed: {logr_corr:.4f}")

    b3.to_frame("ret").to_csv(os.path.join(OUT, "b3_daily.csv"))
    meta.to_frame("ret").to_csv(os.path.join(OUT, "metagt_daily.csv"))
    pd.DataFrame({"k": [k], "target_mdd": [target_mdd],
                  "return_corr_vs_committed": [logr_corr]}) \
        .to_csv(os.path.join(OUT, "b3_build_info.csv"), index=False)


if __name__ == "__main__":
    main()
