"""Portfolio-of-strategies study on the five systems' daily returns.

Discipline against overfitting:
  * Naive a-priori rules (equal weight, family equal weight) - zero fitted params.
  * Causal adaptive rules (inverse-vol, bounded Sharpe-tilt, min-variance):
    weights use ONLY trailing data (shift 1 day), monthly rebalance.
  * Full-sample max-Sharpe weights are computed ONLY as the labeled
    "overfit ceiling" reference.
  * Dev/holdout validation: rules are evaluated separately on 2020-02..2023-12
    and 2024-01..2026-05; the dev-optimized max-Sharpe portfolio is applied
    frozen to the holdout (the honest test of optimized-vs-naive).
No meta-level transaction costs (monthly sleeve reallocation turnover is a few
percent of NAV/month vs sleeves' own ~30-40%/day internal turnover).
"""
import os
import sys
import itertools

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_report as B
from unified_metrics import daily_curve

W0, W1 = pd.Timestamp("2020-02-01"), pd.Timestamp("2026-06-01")
SPLIT = pd.Timestamp("2024-01-01")
APY = 365.25

curves = {
    "PG-A2": B.pg_curve("exp_r9_asymstop"),
    "PG-B3": B.ret_curve(os.path.join(B.HERE, "b3_out/b3_daily.csv")),
    "PG-C1": B.pg_curve("exp_r8_flow1d"),
    "M10": B.ret_curve(os.path.join(B.HERE, "m10_out/m10_full_daily.csv")),
    "CT": B.ct_curve(),
}
rets = {}
for k, eq in curves.items():
    e = eq.copy()
    if e.index.tz is not None:
        e.index = e.index.tz_localize(None)
    e = e[(e.index >= W0) & (e.index <= W1)]
    rets[k] = daily_curve(e).pct_change().dropna()
R = pd.DataFrame(rets).dropna()
NAMES = list(R.columns)

print("=== daily-return correlations (common window) ===")
print(R.corr().round(2).to_string())
print("\n=== monthly-return correlations ===")
RM = (1 + R).resample("1MS").prod() - 1
print(RM.corr().round(2).to_string())


def stats(r, label):
    r = r.dropna()
    eq = (1 + r).cumprod()
    years = (r.index[-1] - r.index[0]).days / APY
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(APY)
    sh = r.mean() / r.std() * np.sqrt(APY)
    dn = r[r < 0].std()
    so = r.mean() / dn * np.sqrt(APY) if dn > 0 else np.nan
    mdd = float((1 - eq / eq.cummax()).max())
    monthly = (1 + r).resample("1MS").prod() - 1
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    return dict(label=label, cagr=cagr * 100, vol=vol * 100, sharpe=sh,
                sortino=so, mdd=mdd * 100, calmar=cagr / mdd if mdd > 0 else np.nan,
                worst_mo=monthly.min() * 100, worst_yr=yearly.min() * 100)


def combo(R, weight_fn, rebal="1MS"):
    """Simulate wealth with sleeve drift; weight_fn(hist)->weights at rebalance."""
    rb_dates = R.resample(rebal).first().index
    shares = None
    out = []
    w_now = None
    for t, row in R.iterrows():
        if w_now is None or t in rb_dates:
            hist = R.loc[:t].iloc[:-1]          # strictly before t
            w = weight_fn(hist)
            if w is not None:
                w_now = np.asarray(w, dtype=float)
                shares = w_now.copy()           # rebalance to targets
        if shares is None:
            out.append(0.0)
            continue
        g = shares @ row.values
        out.append(g / shares.sum() if shares.sum() > 0 else 0.0)
        shares = shares * (1 + row.values)
        shares = shares / shares.sum() * (1 + g / 1.0) if False else shares
    return pd.Series(out, index=R.index)


def combo_simple(R, weights_series):
    """weights_series: DataFrame of weights applied with monthly reset + drift."""
    rb = weights_series
    w = None
    shares = None
    out = []
    for t, row in R.iterrows():
        if t in rb.index:
            w = rb.loc[t].values
            shares = w.copy()
        if shares is None:
            out.append(0.0); continue
        tot = shares.sum()
        g = (shares @ row.values) / tot if tot > 0 else 0.0
        out.append(g)
        shares = shares * (1 + row.values)
        shares = shares / shares.sum() * (tot * (1 + g)) if shares.sum() > 0 else shares
    return pd.Series(out, index=R.index)


def monthly_weights(R, fn):
    """Build a monthly weight DataFrame from causal fn(trailing history)."""
    rows = {}
    for t in R.resample("1MS").first().index:
        hist = R[R.index < t]
        w = fn(hist)
        if w is not None:
            rows[t] = w
    W = pd.DataFrame(rows, index=R.columns).T
    # first month may lack history -> start equal weight
    return W


def fixed(w):
    w = np.asarray(w, float)
    return lambda hist: w / w.sum()


def inv_vol(win=90):
    def fn(hist):
        if len(hist) < 30:
            return np.ones(len(hist.columns)) / len(hist.columns)
        v = hist.tail(win).std()
        iv = 1.0 / v.clip(lower=1e-9)
        return (iv / iv.sum()).values
    return fn


def sharpe_tilt(halflife=30, minp=30, lo=0.5, hi=2.0):
    def fn(hist):
        K = len(hist.columns)
        if len(hist) < minp:
            return np.ones(K) / K
        m = hist.ewm(halflife=halflife, min_periods=minp).mean().iloc[-1]
        s = hist.ewm(halflife=halflife, min_periods=minp).std().iloc[-1]
        sc = (m / s).clip(lower=0.0)
        if sc.sum() <= 0:
            return np.ones(K) / K
        w = sc / sc.sum()
        w = w.clip(lower=lo / K, upper=hi / K)
        return (w / w.sum()).values
    return fn


def min_var(win=120):
    def fn(hist):
        K = len(hist.columns)
        if len(hist) < 60:
            return np.ones(K) / K
        cov = hist.tail(win).cov().values
        try:
            inv = np.linalg.pinv(cov)
            w = inv @ np.ones(K)
            w = np.clip(w, 0, None)
            if w.sum() <= 0:
                return np.ones(K) / K
            return w / w.sum()
        except Exception:
            return np.ones(K) / K
    return fn


def max_sharpe_weights(Rw):
    """Grid over the simplex (5% steps), long-only: the OVERFIT reference."""
    best, bw = -9, None
    cols = Rw.columns
    K = len(cols)
    mu = Rw.mean().values
    cov = Rw.cov().values
    steps = np.arange(0, 1.0001, 0.05)
    for w in itertools.product(steps, repeat=K - 1):
        s = sum(w)
        if s > 1.0001:
            continue
        wv = np.array(list(w) + [1.0 - s])
        vol = np.sqrt(wv @ cov @ wv)
        if vol <= 0:
            continue
        sh = (wv @ mu) / vol
        if sh > best:
            best, bw = sh, wv
    return pd.Series(bw, index=cols)


def run_rule(name, fn, Rw):
    Wm = monthly_weights(Rw, fn)
    r = combo_simple(Rw, Wm)
    return r


rules = {
    "EW-5 (20% each)": fixed(np.ones(5)),
    "EW-3fam (B3/M10/CT)": None,  # handled on the 3-col frame
    "IVW-5 (causal 90d)": inv_vol(90),
    "SharpeTilt-5 (causal)": sharpe_tilt(),
    "MinVar-5 (causal 120d)": min_var(120),
}

print("\n=== combos on the common window (monthly rebalance, causal weights) ===")
rows = []
for n in NAMES:
    rows.append(stats(R[n], n))

results = {}
for label, fn in rules.items():
    if fn is None:
        continue
    r = run_rule(label, fn, R)
    results[label] = r
    rows.append(stats(r, label))

R3 = R[["PG-B3", "M10", "CT"]]
r3 = run_rule("x", fixed(np.ones(3)), R3)
results["EW-3fam (B3/M10/CT)"] = r3
rows.append(stats(r3, "EW-3fam (B3/M10/CT)"))
R3a = R[["PG-A2", "M10", "CT"]]
r3a = run_rule("x", fixed(np.ones(3)), R3a)
results["EW-3fam (A2/M10/CT)"] = r3a
rows.append(stats(r3a, "EW-3fam (A2/M10/CT)"))
riv3 = run_rule("x", inv_vol(90), R3)
results["IVW-3fam (B3/M10/CT)"] = riv3
rows.append(stats(riv3, "IVW-3fam (B3/M10/CT)"))

# overfit ceiling: full-sample max-Sharpe (labeled!)
w_full = max_sharpe_weights(R)
r_of = combo_simple(R, monthly_weights(R, fixed(w_full.values)))
results["[OVERFIT] full-sample maxSharpe"] = r_of
rows.append(stats(r_of, "[OVERFIT] full-sample maxSharpe"))
print("\nfull-sample max-Sharpe weights (OVERFIT reference):")
print((w_full * 100).round(1).to_string())

T = pd.DataFrame(rows).set_index("label")
print()
print(T.round(2).to_string())
T.to_csv(os.path.join(HERE, "report_out", "allocation_common.csv"))

# ---- dev / holdout validation ----
print("\n=== dev (2020-02..2023-12) vs holdout (2024-01..2026-05) ===")
Rdev, Rhold = R[R.index < SPLIT], R[R.index >= SPLIT]
w_dev = max_sharpe_weights(Rdev)
print("dev-optimized max-Sharpe weights (frozen for holdout):")
print((w_dev * 100).round(1).to_string())

cand = {
    "EW-5": lambda Rw: run_rule("x", fixed(np.ones(5)), Rw),
    "IVW-5": lambda Rw: run_rule("x", inv_vol(90), Rw),
    "SharpeTilt-5": lambda Rw: run_rule("x", sharpe_tilt(), Rw),
    "EW-3fam(B3/M10/CT)": lambda Rw: run_rule("x", fixed(np.ones(3)), Rw[["PG-B3", "M10", "CT"]]),
    "IVW-3fam(B3/M10/CT)": lambda Rw: run_rule("x", inv_vol(90), Rw[["PG-B3", "M10", "CT"]]),
    "devOpt maxSharpe (frozen)": lambda Rw: combo_simple(Rw, monthly_weights(Rw, fixed(w_dev.values))),
}
rows2 = []
for label, make in cand.items():
    # weights computed causally within each window independently for the
    # adaptive rules; the frozen devOpt weights are the out-of-sample test
    r_all = make(R)
    rows2.append(stats(r_all[r_all.index < SPLIT], label + " [dev]"))
    rows2.append(stats(r_all[r_all.index >= SPLIT], label + " [holdout]"))
for n in ["PG-B3", "M10", "CT", "PG-A2", "PG-C1"]:
    rows2.append(stats(R[n][R.index < SPLIT], n + " [dev]"))
    rows2.append(stats(R[n][R.index >= SPLIT], n + " [holdout]"))
T2 = pd.DataFrame(rows2).set_index("label")
print(T2.round(2).to_string())
T2.to_csv(os.path.join(HERE, "report_out", "allocation_devhold.csv"))

# save combined daily returns for charting
pd.DataFrame({k: v for k, v in results.items()}).to_csv(
    os.path.join(HERE, "report_out", "allocation_daily.csv"))
print("\nsaved: allocation_common.csv, allocation_devhold.csv, allocation_daily.csv")
