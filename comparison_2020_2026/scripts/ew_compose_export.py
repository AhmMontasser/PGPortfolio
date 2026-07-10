"""M10 composition exporter.

Replicates scripts/opt10/compose_holdout_m10.py verbatim (same functions,
same constants, same windows), then exports:
  - daily return series of the M10 blend (dev book and holdout book)
  - the EW trade book with the exact per-trade multipliers/gates ew_pnl uses
    (membership + perp/fast-fill weights), with per-trade net PnL
    (price move x direction x notional - entry/exit cost - funding)

No repo code is modified. Outputs go to the scratchpad m10_out/ dir.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/home/user/EW_trader"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "m10_out")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
for sub in ("opt", "opt7", "opt8", "opt9", "opt10"):
    sys.path.insert(0, os.path.join(ROOT, "scripts", sub))

import m10lib                                    # noqa: E402
from m10lib import L, FC                         # noqa: E402
from m8_data import build_futures_membership     # noqa: E402

R = L.R
CACHE = L.CACHE
COST = L.COST


def mom_ddstop(P):
    return R.momentum(P, fund_select="net", voltarget=0.50, maxlev=1.5, ddstop=0.20)


def compose(sl_base, P):
    sl = dict(sl_base)
    sl["mom"] = mom_ddstop(P).reindex(pd.DataFrame(sl_base).index).fillna(0.0)
    return sl


def curve_from(pEW, sl, idx, ddcap_from=None):
    m2 = R.vol_floored_risk_parity(sl).reindex(idx).fillna(0.0)
    blend = R.vol_floored_risk_parity(
        {"ew": pEW.reindex(idx).fillna(0.0), "m2": m2}, lev=FC.LEV)
    blend = (blend * FC.live_scale(pEW, sl, idx)).fillna(0.0)
    if ddcap_from is not None:
        blend = blend[blend.index >= ddcap_from]
    return R.ddcap(blend, cap=FC.CAP, m=FC.M)


def trade_book(tr, dclose, idx, fund_d, mem, weights):
    """Per-trade net PnL with EXACTLY ew_pnl's inclusion rules and accounting."""
    rows = []
    fcols = set(fund_d.columns)
    dc_cols = set(dclose.columns)
    for i, t in tr.iterrows():
        note = L.NOTION.get(t.state, 0.0)
        w = float(weights.loc[i])
        note = note * w
        if note == 0.0:
            continue
        if not mem.contains(t.symbol, t.fill_time.value):
            continue
        ed = pd.Timestamp(t.fill_time).normalize()
        xd = pd.Timestamp(t.exit_time).normalize()
        days = idx[(idx >= ed) & (idx <= xd)]
        if len(days) == 0:
            continue
        dcs = dclose[t.symbol] if t.symbol in dc_cols else None
        fser = fund_d[t.symbol] if t.symbol in fcols else None
        prev = t.entry_price
        d = t.direction
        pnl = 0.0
        funding = 0.0
        for day in days:
            if day == days[-1]:
                px = t.exit_price
            else:
                px = float(dcs.loc[day]) if (dcs is not None and day in dcs.index) else prev
            if not np.isfinite(px) or prev <= 0:
                prev = px
                continue
            r = d * (px / prev - 1)
            prev = px
            if fser is not None and day in fser.index:
                f = float(fser.loc[day])
                if np.isfinite(f):
                    r -= d * f
                    funding += d * f * note
            pnl += note * r
        pnl -= 2 * note * COST
        rows.append(dict(symbol=t.symbol, direction=int(d),
                         entry_time=t.fill_time, exit_time=t.exit_time,
                         entry_price=t.entry_price, exit_price=t.exit_price,
                         state=t.state, note=note,
                         pnl=pnl, funding_paid=funding,
                         price_ret=d * (t.exit_price / t.entry_price - 1)))
    return pd.DataFrame(rows)


def main():
    # ---- DEV (verbatim compose_holdout_m10) ----
    ctx = m10lib.load_ctx()
    idx = ctx["idx"]
    tr9 = pd.read_parquet(os.path.join(CACHE, "m9_stop04_C4gh.parquet"))
    dc9 = pd.read_parquet(os.path.join(CACHE, "dclose_stop04_fut.parquet"))
    w9 = FC.trade_weights(tr9, ctx["fund_raw"])
    pEW = L.ew_pnl(tr9, dc9, idx, ctx["fund_d"], ctx["mem"], weights=w9)
    memS, close, qvol, fraw = L.load_universe()
    P = R.build(close, qvol, fraw, memS, require_perp=True)
    sl_c4 = compose(ctx["sl"], P)
    p_c4 = curve_from(pEW, sl_c4, idx)
    print("DEV metrics:", L.fmt("M10 dev", L.metrics(p_c4)))
    p_c4.to_frame("ret").to_csv(os.path.join(OUT, "m10_dev_daily.csv"))
    tb_dev = trade_book(tr9, dc9, idx, ctx["fund_d"], ctx["mem"], w9)
    tb_dev.to_csv(os.path.join(OUT, "m10_trades_dev.csv"), index=False)
    print(f"dev trades: {len(tb_dev)}")

    # ---- HOLDOUT (verbatim) ----
    memSh, closeh, qvolh, frawh = L.load_universe(end="2026-05-31")
    PH = R.build(closeh, qvolh, frawh, memSh, require_perp=True)
    idxh, fund_dh = PH["idx"], PH["fund_d"]
    slh = L.m2_with_crowd(PH)
    mem_h, _ = build_futures_membership("2026-05-31")
    trh = pd.read_parquet(os.path.join(CACHE, "trades_stop04_fut_hold.parquet")).copy()
    dch = pd.read_parquet(os.path.join(CACHE, "dclose_stop04_fut_hold.parquet"))
    trh["target_price"] = np.where(trh.direction > 0, 1e12, 1e-12)
    trh = m10lib.resim_fut(trh, end="2026-06-01", gap_fill=True, **m10lib.C4)
    wh = FC.trade_weights(trh, frawh)
    pEWh = L.ew_pnl(trh, dch, idxh, fund_dh, mem_h, weights=wh)
    H0 = pd.Timestamp("2025-01-01", tz="UTC")
    slh_c4 = compose(slh, PH)
    ph_c4 = curve_from(pEWh, slh_c4, idxh, ddcap_from=H0)
    eq = (1 + ph_c4).cumprod()
    sh = ph_c4.mean() / ph_c4.std() * np.sqrt(365)
    print(f"HOLDOUT: {100*(eq.iloc[-1]-1):+.1f}%  maxDD "
          f"{100*(eq/eq.cummax()-1).min():+.1f}%  Sharpe {sh:+.2f}")
    ph_c4.to_frame("ret").to_csv(os.path.join(OUT, "m10_holdout_daily.csv"))
    tb_hold = trade_book(trh, dch, idxh, fund_dh, mem_h, wh)
    tb_hold.to_csv(os.path.join(OUT, "m10_trades_holdout.csv"), index=False)
    print(f"holdout-book trades: {len(tb_hold)}")

    # per-year of the stitched curve for reference
    dev_part = p_c4[p_c4.index < H0]
    hold_part = ph_c4
    full = pd.concat([dev_part, hold_part])
    full.to_frame("ret").to_csv(os.path.join(OUT, "m10_full_daily.csv"))
    eqf = (1 + full).cumprod()
    print("stitched full-curve total:", f"x{eqf.iloc[-1]:.2f}")
    for y in sorted(full.index.year.unique()):
        seg = full[full.index.year == y]
        e = (1 + seg).cumprod()
        print(f"  {y}: {100*(e.iloc[-1]-1):+7.1f}%  yearMDD "
              f"{100*(e/e.cummax()-1).min():6.1f}%")


if __name__ == "__main__":
    main()
