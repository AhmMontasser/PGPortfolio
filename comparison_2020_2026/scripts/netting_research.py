"""Netting-bot research: one wallet running 0.5*B3 + 0.5*M10 as a single book.

Quantifies, from the two systems' EXACT daily coin-weight panels:
  1. gross-exposure netting (margin saved by internal offsets)
  2. order-flow netting (turnover and fee savings vs two separate books)
  3. direction-conflict frequency (same coin, opposite sides)
  4. B3 financing swap: modeled 10% APR borrow -> real perp funding,
     computed on B3's actual daily weights against actual funding prints.
"""
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PG = "/home/user/PGPortfolio"
sys.path.insert(0, HERE)

APY = 365.25
K_SCALE = 0.4012
MEMBERS = ["exp_r6_30m", "exp_r6_1h", "exp_r6_2h", "exp_r6_vol150", "exp_r6_8h"]
W0, W1 = pd.Timestamp("2020-02-01"), pd.Timestamp("2026-05-31")

# ---------------------------------------------------------------- B3 panel
def sleeve_daily_weights(name):
    rec = pd.read_parquet(os.path.join(HERE, "pg_positions", f"{name}.parquet"))
    rec["date"] = pd.to_datetime(rec["time"], unit="s").dt.normalize()
    last = rec.sort_values("time").groupby(["date", "coin"]).tail(1)
    return last.pivot(index="date", columns="coin", values="weight").fillna(0.0)


def daily_rets(name):
    c = pd.read_csv(os.path.join(PG, "dynamic_results", name, "portfolio_values.csv"),
                    index_col=0, parse_dates=True).iloc[:, 0]
    return c.resample("1D").last().ffill().pct_change().dropna()


frame = pd.DataFrame({m: daily_rets(m) for m in MEMBERS}).dropna()
K = frame.shape[1]
mean = frame.ewm(halflife=30, min_periods=30).mean()
std = frame.ewm(halflife=30, min_periods=30).std()
s = (mean / std).clip(lower=0.0)
gtw = s.div(s.sum(axis=1), axis=0).fillna(1.0 / K)
gtw = gtw.clip(lower=0.5 / K, upper=2.0 / K)
gtw = gtw.div(gtw.sum(axis=1), axis=0).shift(1).fillna(1.0 / K)

meta = (gtw * frame).sum(axis=1)
downside = meta.clip(upper=0.0).rolling(60).std() * np.sqrt(APY)
target = downside.expanding(120).median()
s11 = (target / downside).clip(upper=1.0).shift(1).fillna(1.0)
corr = frame.rolling(60).corr().groupby(level=0).apply(
    lambda c: (c.values.sum() - K) / (K * (K - 1)))
s12 = pd.Series(np.where(corr > 0.85, 0.7, 1.0), index=frame.index).shift(1).fillna(1.0)
fast, slow = meta.rolling(10).std(), meta.rolling(60).std()
s13 = pd.Series(np.where(fast > 1.5 * slow, 0.5, 1.0), index=meta.index).shift(1).fillna(1.0)
scale = (K_SCALE * s11 * s12 * s13).rename("scale")

panels = {m: sleeve_daily_weights(m) for m in MEMBERS}
cols = sorted(set().union(*[set(p.columns) for p in panels.values()]))
b3 = pd.DataFrame(0.0, index=frame.index, columns=cols)
for m in MEMBERS:
    p = panels[m].reindex(index=frame.index, columns=cols).fillna(0.0)
    b3 = b3.add(p.mul(gtw[m], axis=0), fill_value=0.0)
b3 = b3.mul(scale, axis=0)
print(f"B3 daily coin-weight panel {b3.shape}; gross mean {b3.abs().sum(1).mean():.3f}x  "
      f"p95 {b3.abs().sum(1).quantile(0.95):.2f}x")

# validation: weight-implied return vs engine curve
b3_actual = pd.read_csv(os.path.join(HERE, "b3_out/b3_daily.csv"),
                        index_col=0, parse_dates=True)["ret"]
db = sqlite3.connect(os.path.join(PG, "database/dynamic_data.db"))
closes = pd.read_sql("select symbol, ts, close from klines30m", db)
closes["date"] = pd.to_datetime(closes.ts, unit="s").dt.normalize()
dclose = closes.sort_values("ts").groupby(["date", "symbol"]).close.last().unstack()
dclose = dclose.reindex(index=b3.index, columns=b3.columns)
y = dclose.pct_change().clip(-0.9, 9.0)
implied = (b3.shift(1) * y).sum(axis=1)
common = pd.concat([implied.rename("imp"), b3_actual.rename("act")], axis=1).dropna()
print(f"B3 panel validation: corr(weight-implied vs engine daily ret) = "
      f"{common.imp.corr(common.act):.3f}")

# ---------------------------------------------------------------- M10 panel
m10 = pd.read_parquet(os.path.join(HERE, "m10_out/m10_daily_coin_weights.parquet"))
allc = sorted(set(b3.columns) | set(m10.columns))
b3a = b3.reindex(columns=allc, fill_value=0.0)
m10a = m10.reindex(index=b3.index, columns=allc).fillna(0.0)
win = (b3a.index >= W0) & (b3a.index <= W1)
b3a, m10a = b3a[win], m10a[win]

half_b3, half_m10 = 0.5 * b3a, 0.5 * m10a
net = half_b3 + half_m10

# 1) gross netting
g_sep = half_b3.abs().sum(1) + half_m10.abs().sum(1)
g_net = net.abs().sum(1)
print("\n=== gross exposure (fraction of total account equity) ===")
print(f"separate books: mean {g_sep.mean():.3f}x  p95 {g_sep.quantile(.95):.3f}x  max {g_sep.max():.2f}x")
print(f"netted book   : mean {g_net.mean():.3f}x  p95 {g_net.quantile(.95):.3f}x  max {g_net.max():.2f}x")
print(f"netting saves {100*(1-g_net.mean()/g_sep.mean()):.1f}% of average gross "
      f"(margin requirement proxy)")

# 2) turnover + fee savings
t_sep = half_b3.diff().abs().sum(1) + half_m10.diff().abs().sum(1)
t_net = net.diff().abs().sum(1)
saved = t_sep - t_net
print("\n=== order flow (daily turnover, fraction of account) ===")
print(f"separate: {t_sep.mean():.4f}/day   netted: {t_net.mean():.4f}/day   "
      f"crossed internally: {saved.mean():.4f}/day ({100*saved.mean()/t_sep.mean():.1f}%)")
for fee_bps in (5, 7, 10):
    print(f"  fee savings at {fee_bps}bps/side: "
          f"{saved.mean() * fee_bps * 1e-4 * APY * 1e4:.1f} bps/year of account equity")

# 3) direction conflicts
both = (half_b3 != 0) & (half_m10 != 0)
opp = both & (np.sign(half_b3) != np.sign(half_m10))
coin_days = int(both.sum().sum())
print("\n=== same-coin overlap ===")
print(f"coin-days both books hold the same coin: {coin_days} "
      f"({100*both.sum().sum()/ (b3a!=0).sum().sum():.1f}% of B3's coin-days)")
print(f"  of which OPPOSITE direction: {int(opp.sum().sum())} "
      f"({100*opp.sum().sum()/max(1,coin_days):.1f}%)")
offset = np.minimum(half_b3.abs(), half_m10.abs())[opp].sum(1).fillna(0.0)
print(f"  average offsetting notional when opposed: {offset[offset>0].mean():.4f} of equity")

# 4) B3 financing swap: modeled borrow vs real perp funding
fund = pd.read_sql("select symbol, ts, rate from funding", db)
fund["date"] = pd.to_datetime(fund.ts, unit="s").dt.normalize()
fday = fund.groupby(["date", "symbol"]).rate.sum().unstack()
fday = fday.reindex(index=b3a.index, columns=allc).fillna(0.0)

b3w = b3a  # full B3 book (per B3 equity, not halved)
perp_funding_pnl = -(b3w * fday).sum(axis=1)              # long pays +rate
short_gross = b3w.clip(upper=0.0).abs().sum(axis=1)
gross_b3 = b3w.abs().sum(axis=1)
borrow_model = -(0.10 / 365.0) * (short_gross + (gross_b3 - 1).clip(lower=0.0))
delta = perp_funding_pnl - borrow_model
print("\n=== B3 financing: modeled margin-borrow -> real perp funding ===")
print(f"modeled borrow cost   : {borrow_model.mean()*APY*100:+.2f}%/yr of B3 equity")
print(f"real perp funding pnl : {perp_funding_pnl.mean()*APY*100:+.2f}%/yr of B3 equity")
print(f"delta (perp - model)  : {delta.mean()*APY*100:+.2f}%/yr on B3  "
      f"=> {delta.mean()*APY*100/2:+.2f}%/yr on the 50/50 blend")
by_side = {
    "long book funding": -(b3w.clip(lower=0.0) * fday).sum(1).mean() * APY * 100,
    "short book funding": -(b3w.clip(upper=0.0) * fday).sum(1).mean() * APY * 100,
}
print(f"  breakdown: longs {by_side['long book funding']:+.2f}%/yr, "
      f"shorts {by_side['short book funding']:+.2f}%/yr")

# save panels for the record
b3a.to_parquet(os.path.join(HERE, "report_out", "b3_daily_coin_weights.parquet"))
pd.DataFrame({"g_sep": g_sep, "g_net": g_net, "t_sep": t_sep, "t_net": t_net,
              "fund_delta_b3": delta}).to_csv(
    os.path.join(HERE, "report_out", "netting_daily_stats.csv"))
print("\nsaved: b3_daily_coin_weights.parquet, netting_daily_stats.csv")
