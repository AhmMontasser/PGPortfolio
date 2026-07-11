"""Full pack test of the 50/50 PG-B3 + M10 configuration.

Independent implementation (does NOT reuse allocation_study's combiner):
simulates an actual account with two sub-books, daily compounding, explicit
rebalancing transfers, costs charged on transferred notional, and optional
execution lag. Sweeps rebalance frequency and cost; splits dev/holdout;
block-bootstraps the Sharpe/MDD advantage of the blend over M10 alone.

Match target (from ALLOCATION_STUDY.md, frictionless monthly 50/50):
    CAGR +43.6% | vol 16.4% | Sharpe 2.29 | Sortino 4.31 | MDD 9.2%
    Calmar 4.73 | worst month -5.0%
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_report as B
from unified_metrics import daily_curve

APY = 365.25
W0, W1 = pd.Timestamp("2020-02-01"), pd.Timestamp("2026-06-01")
SPLIT = pd.Timestamp("2024-01-01")
START_CAPITAL = 10_000.0

# ---- load the two books' daily returns (independent path re-derivation) ----
b3_eq = B.ret_curve(os.path.join(HERE, "b3_out/b3_daily.csv"))
m10_eq = B.ret_curve(os.path.join(HERE, "m10_out/m10_full_daily.csv"))
frame = {}
for name, eq in [("B3", b3_eq), ("M10", m10_eq)]:
    e = eq.copy()
    if e.index.tz is not None:
        e.index = e.index.tz_localize(None)
    e = e[(e.index >= W0) & (e.index <= W1)]
    frame[name] = daily_curve(e).pct_change().dropna()
R = pd.DataFrame(frame).dropna()
print(f"grid: {len(R)} days  {R.index[0].date()} .. {R.index[-1].date()}")


def simulate(R, w_target=(0.5, 0.5), rebal="MS", cost_bps=0.0, lag_days=0,
             start=START_CAPITAL):
    """Account simulation. Returns (equity Series, stats dict of turnover/cost)."""
    if rebal == "never":
        rb_dates = set()
    else:
        rb_dates = set(R.resample(rebal).first().index)
    v = np.array(w_target, float) * start
    cost_rate = cost_bps / 1e4
    pending = None            # (execute_date_index_position)
    eq, transfers, costs_paid = [], [], 0.0
    dates = R.index
    rb_signal_days = []
    for i, t in enumerate(dates):
        # grow books with today's returns
        v = v * (1.0 + R.iloc[i].values)
        # schedule rebalance signal
        if t in rb_dates:
            rb_signal_days.append(i + lag_days)
        # execute any due rebalance (after growth, end of day)
        if rb_signal_days and i >= rb_signal_days[0]:
            rb_signal_days.pop(0)
            total = v.sum()
            target = np.array(w_target) * total
            moved = np.abs(target - v).sum() / 2.0
            fee = moved * cost_rate * 2.0        # exit one book + enter the other
            v = target
            v = v * (1.0 - fee / total)
            transfers.append(moved / total)
            costs_paid += fee
        eq.append(v.sum())
    equity = pd.Series(eq, index=dates)
    info = dict(avg_monthly_transfer_pct=100 * np.mean(transfers) if transfers else 0.0,
                n_rebalances=len(transfers), total_cost_usd=costs_paid,
                cost_drag_bps_per_year=(costs_paid / start) /
                ((dates[-1] - dates[0]).days / APY) * 1e4 if costs_paid else 0.0)
    return equity, info


def metrics(equity, label):
    r = equity.pct_change().dropna()
    yrs = (equity.index[-1] - equity.index[0]).days / APY
    eqn = equity / equity.iloc[0]
    cagr = eqn.iloc[-1] ** (1 / yrs) - 1
    vol = r.std() * np.sqrt(APY)
    sh = r.mean() / r.std() * np.sqrt(APY)
    dn = r[r < 0].std()
    so = r.mean() / dn * np.sqrt(APY)
    mdd = float((1 - eqn / eqn.cummax()).max())
    mo = (1 + r).resample("1MS").prod() - 1
    return dict(label=label, cagr=cagr * 100, vol=vol * 100, sharpe=sh, sortino=so,
                mdd=mdd * 100, calmar=cagr / mdd, worst_mo=mo.min() * 100,
                final_usd=equity.iloc[-1])


rows = []

# 1) exact frictionless reproduction attempt (monthly, 0 cost, no lag)
eq0, info0 = simulate(R, rebal="MS", cost_bps=0, lag_days=0)
rows.append(metrics(eq0, "monthly, 0 cost, lag 0  [match target]"))

# 2) execution lag
eq1, _ = simulate(R, rebal="MS", cost_bps=0, lag_days=1)
rows.append(metrics(eq1, "monthly, 0 cost, lag 1d"))

# 3) cost sweep on transferred notional (per side)
for bps in (5, 10, 20, 50):
    eqc, infoc = simulate(R, rebal="MS", cost_bps=bps, lag_days=1)
    rows.append(metrics(eqc, f"monthly, {bps}bps/side, lag 1d"))

# 4) rebalance frequency
for freq, lab in (("W-MON", "weekly"), ("QS", "quarterly"), ("never", "never")):
    eqf, infof = simulate(R, rebal=freq, cost_bps=10, lag_days=1)
    rows.append(metrics(eqf, f"{lab}, 10bps/side, lag 1d"))

# components for reference
rows.append(metrics(START_CAPITAL * (1 + R["M10"]).cumprod(), "M10 alone"))
rows.append(metrics(START_CAPITAL * (1 + R["B3"]).cumprod(), "PG-B3 alone"))

T = pd.DataFrame(rows).set_index("label")
print("\n=== PACK TEST: full account simulation, $10,000 start ===")
print(T.round(2).to_string())

print("\nrebalance mechanics (monthly, 10bps, lag1): "
      f"avg transfer {simulate(R, rebal='MS', cost_bps=10, lag_days=1)[1]['avg_monthly_transfer_pct']:.2f}% of NAV/month")

# 5) dev / holdout of the realistic variant
eqr, _ = simulate(R, rebal="MS", cost_bps=10, lag_days=1)
rr = eqr.pct_change().dropna()
for lab, seg in (("dev 2020-02..2023-12", rr[rr.index < SPLIT]),
                 ("holdout 2024-01..2026-05", rr[rr.index >= SPLIT])):
    e = (1 + seg).cumprod()
    print(f"{lab}: CAGR {100*(e.iloc[-1]**(APY/ (seg.index[-1]-seg.index[0]).days)-1):+.1f}%  "
          f"Sharpe {seg.mean()/seg.std()*np.sqrt(APY):.2f}  "
          f"MDD {100*(1-e/e.cummax()).max():.1f}%")

# 6) yearly returns of the realistic variant
print("\nyearly returns (monthly rebal, 10bps/side, lag 1d):")
yr = (1 + rr).groupby(rr.index.year).prod() - 1
print((yr * 100).round(1).to_string())

# 7) stationary block bootstrap: blend edge vs M10 (Sharpe and MDD)
rng = np.random.default_rng(7)
rb = eq0.pct_change().dropna().values
rm = R["M10"].reindex(eq0.pct_change().dropna().index).values
n = len(rb)
MEAN_BLOCK = 20
N_BOOT = 5000
d_sharpe, d_mdd = [], []
for _ in range(N_BOOT):
    idx = []
    while len(idx) < n:
        s0 = rng.integers(0, n)
        L = rng.geometric(1 / MEAN_BLOCK)
        idx.extend(((s0 + np.arange(L)) % n).tolist())
    idx = np.array(idx[:n])
    a, m = rb[idx], rm[idx]
    sh_a = a.mean() / a.std() * np.sqrt(APY)
    sh_m = m.mean() / m.std() * np.sqrt(APY)
    ea, em = np.cumprod(1 + a), np.cumprod(1 + m)
    mdd_a = (1 - ea / np.maximum.accumulate(ea)).max()
    mdd_m = (1 - em / np.maximum.accumulate(em)).max()
    d_sharpe.append(sh_a - sh_m)
    d_mdd.append(mdd_a - mdd_m)
d_sharpe, d_mdd = np.array(d_sharpe), np.array(d_mdd)
print(f"\n=== block bootstrap ({N_BOOT} draws, mean block {MEAN_BLOCK}d) ===")
print(f"Sharpe(blend) - Sharpe(M10): median {np.median(d_sharpe):+.2f}  "
      f"90% CI [{np.percentile(d_sharpe,5):+.2f}, {np.percentile(d_sharpe,95):+.2f}]  "
      f"P(blend>M10) = {(d_sharpe>0).mean():.1%}")
print(f"MDD(blend) - MDD(M10):       median {100*np.median(d_mdd):+.1f}pp  "
      f"90% CI [{100*np.percentile(d_mdd,5):+.1f}, {100*np.percentile(d_mdd,95):+.1f}]pp  "
      f"P(blend shallower) = {(d_mdd<0).mean():.1%}")

T.to_csv(os.path.join(HERE, "report_out", "pack_test_5050.csv"))
eq0.to_frame("equity").to_csv(os.path.join(HERE, "report_out", "pack_test_5050_equity.csv"))
print("\nsaved: pack_test_5050.csv, pack_test_5050_equity.csv")
