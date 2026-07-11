"""Dump M10's daily per-coin portfolio weights (fraction of M10 book equity).

Rebuilds the full-span holdout-basis composition exactly as
compose_holdout_m10.py does, but additionally extracts:
  - EW book per-coin exposure per day (sum of direction x note of live trades)
  - each M2 sleeve's per-coin weight frame x exposure scaler (return_weights)
  - the vol-floored risk-parity weights (EW vs M2; and across M2 sleeves)
  - live_scale and the CPPI ddcap exposure scaler
and writes the combined daily coin-weight panel to parquet.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/home/user/EW_trader"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "m10_out")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
for sub in ("opt", "opt7", "opt8", "opt9", "opt10"):
    sys.path.insert(0, os.path.join(ROOT, "scripts", sub))

import m10lib
from m10lib import L, FC
from m8_data import build_futures_membership

R = L.R
CACHE = L.CACHE

# ---- full-span panels (holdout basis, 2019-09 .. 2026-05-31) ----
memS, close, qvol, fraw = L.load_universe(end="2026-05-31")
P = R.build(close, qvol, fraw, memS, require_perp=True)
idx = P["idx"]
mem = build_futures_membership("2026-05-31")[0]

# ---- EW book: daily coin exposure ----
trh = pd.read_parquet(os.path.join(CACHE, "trades_stop04_fut_hold.parquet")).copy()
dch = pd.read_parquet(os.path.join(CACHE, "dclose_stop04_fut_hold.parquet"))
trh["target_price"] = np.where(trh.direction > 0, 1e12, 1e-12)
trh = m10lib.resim_fut(trh, end="2026-06-01", gap_fill=True, **m10lib.C4)
wts = FC.trade_weights(trh, fraw)
ew_expo = pd.DataFrame(0.0, index=idx, columns=sorted(trh.symbol.unique()))
n_booked = 0
for i, t in trh.iterrows():
    note = L.NOTION.get(t.state, 0.0) * float(wts.loc[i])
    if note == 0.0 or not mem.contains(t.symbol, t.fill_time.value):
        continue
    ed = pd.Timestamp(t.fill_time).normalize()
    xd = pd.Timestamp(t.exit_time).normalize()
    days = idx[(idx >= ed) & (idx <= xd)]
    if len(days) == 0:
        continue
    ew_expo.loc[days, t.symbol] += t.direction * note
    n_booked += 1
print(f"EW book: {n_booked} booked trades -> daily exposure panel {ew_expo.shape}")

# EW daily returns (for vfrp weights)
pEW = L.ew_pnl(trh, dch, idx, P["fund_d"], mem, weights=wts)

# ---- M2 sleeves with weights ----
mom, w_mom, e_mom = R.momentum(P, fund_select="net", voltarget=0.50, maxlev=1.5,
                               ddstop=0.20, return_weights=True)
car, w_car, e_car = R.carry(P, return_weights=True)
idi, w_idi, e_idi = R.idio_sleeve(P, return_weights=True)
crowd_out = R.crowd_sleeve(P, return_weights=True)
sleeves = {"mom": (mom, w_mom, e_mom), "carry": (car, w_car, e_car),
           "idio": (idi, w_idi, e_idi)}
if crowd_out is not None:
    cro, w_cro, e_cro = crowd_out
    sleeves["crowd"] = (cro.reindex(idx), w_cro.reindex(idx), e_cro.reindex(idx))
print("M2 sleeves:", list(sleeves))

sl_rets = pd.DataFrame({k: v[0] for k, v in sleeves.items()}).reindex(idx).fillna(0.0)


def vfrp_weights(df, win=90, lev=1.0, floor_q=0.5, floor_win=180):
    sd = df.rolling(win, min_periods=30).std()
    floor = sd.rolling(floor_win, min_periods=30).median() * floor_q
    sdf = sd.clip(lower=floor).shift(1)
    ivw = 1.0 / sdf
    ivw = ivw.div(ivw.sum(axis=1), axis=0).fillna(0.0)
    return lev * ivw


w_sl = vfrp_weights(sl_rets)                      # weights across the 4 M2 sleeves
m2_ret = (w_sl * sl_rets).sum(axis=1)

# M2 per-coin weights = sum_s vfrp_s x w_s x exp_s
cols = sorted(set().union(*[set(v[1].columns) for v in sleeves.values()]))
m2_coin = pd.DataFrame(0.0, index=idx, columns=cols)
for k, (ret, w, e) in sleeves.items():
    contrib = w.reindex(idx).fillna(0.0).mul(e.reindex(idx).fillna(0.0), axis=0) \
               .mul(w_sl[k], axis=0)
    m2_coin = m2_coin.add(contrib.reindex(columns=cols, fill_value=0.0), fill_value=0.0)

# ---- blend: vfrp(EW, M2) x lev4 x live_scale, then CPPI ddcap scaler ----
two = pd.DataFrame({"ew": pEW.reindex(idx).fillna(0.0), "m2": m2_ret})
w_two = vfrp_weights(two, lev=FC.LEV)             # already x4
ls = FC.live_scale(pEW, {k: v[0] for k, v in sleeves.items()}, idx)
blend = (w_two["ew"] * two["ew"] + w_two["m2"] * two["m2"]) * ls
blend = blend.fillna(0.0)

# CPPI exposure scaler (replicates R.ddcap internals)
v = blend.values
eq, peak = 1.0, 1.0
scale = np.empty(len(v))
for i in range(len(v)):
    cushion = max(0.0, (eq - peak * (1 - FC.CAP)) / eq) if eq > 0 else 0.0
    s = min(1.0, FC.M * cushion)
    scale[i] = s
    re = v[i] * s
    eq *= (1 + re)
    peak = max(peak, eq)
cppi = pd.Series(scale, index=idx)

# sanity: capped return must equal R.ddcap(blend)
capped = blend * cppi
ref = R.ddcap(blend, cap=FC.CAP, m=FC.M)
err = float((capped - ref).abs().max())
print(f"ddcap replication max error: {err:.2e}")

# ---- final M10 daily coin weights (fraction of book equity) ----
all_cols = sorted(set(ew_expo.columns) | set(m2_coin.columns))
m10_w = pd.DataFrame(0.0, index=idx, columns=all_cols)
m10_w = m10_w.add(ew_expo.reindex(columns=all_cols, fill_value=0.0)
                  .mul(w_two["ew"] * ls * cppi, axis=0), fill_value=0.0)
m10_w = m10_w.add(m2_coin.reindex(columns=all_cols, fill_value=0.0)
                  .mul(w_two["m2"] * ls * cppi, axis=0), fill_value=0.0)

gross = m10_w.abs().sum(axis=1)
print(f"M10 gross exposure: mean {gross.mean():.2f}x  p95 {gross.quantile(0.95):.2f}x  "
      f"max {gross.max():.2f}x")

m10_w.index = m10_w.index.tz_localize(None)
m10_w.to_parquet(os.path.join(OUT, "m10_daily_coin_weights.parquet"))

# verification: weight-implied return vs actual blend return
rc = P["rc"].copy()
rc.index = rc.index.tz_localize(None)
implied = (m10_w.shift(1) * 0).sum(axis=1)  # placeholder to keep memory low
print(f"saved m10_daily_coin_weights.parquet  {m10_w.shape}")
