"""Assemble the unified 5-system comparison from all backtest outputs.

Systems:
  PG-A2   PGPortfolio exp_r9_asymstop      (4h funding-gated don+regime, short stops)
  PG-B3   five-sleeve META-GT + R20c stack @18.2% MDD budget
  PG-C1   PGPortfolio exp_r8_flow1d        (A2 rules + taker-flow entry filter)
  M10     EW_trader M10 blend (EW book + M2 sleeves, CPPI cap)
  CT      Crypto_Trader baseline trend-follower

Outputs: report tables as CSV + a printable text block per table.
Windows: native (each system's full curve) and common (identical slice).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from unified_metrics import curve_metrics, trade_metrics, daily_curve  # noqa: E402

PG = "/home/user/PGPortfolio"
CT = "/home/user/Crypto_Trader"
COMMON = ("2020-02-01", "2026-05-31")
OUT = os.path.join(HERE, "report_out")
os.makedirs(OUT, exist_ok=True)


# ----------------------------------------------------------------- curves
def pg_curve(exp):
    c = pd.read_csv(os.path.join(PG, "dynamic_results", exp, "portfolio_values.csv"),
                    index_col=0, parse_dates=True).iloc[:, 0]
    return c


def ret_curve(path, col="ret"):
    r = pd.read_csv(path, index_col=0, parse_dates=True)[col]
    if r.index.tz is not None:
        r.index = r.index.tz_localize(None)
    return (1.0 + r).cumprod()


def ct_curve():
    c = pd.read_csv(os.path.join(CT, "reports/compare_2020_2026/equity_daily.csv"),
                    index_col=0, parse_dates=True)["equity"]
    return c


# ----------------------------------------------------------------- trades
def ct_trades():
    t = pd.read_csv(os.path.join(CT, "reports/compare_2020_2026/trades.csv"),
                    parse_dates=["entry_t", "exit_t"])
    return pd.DataFrame(dict(entry_time=t.entry_t, exit_time=t.exit_t,
                             direction=t["dir"], pnl=t.pnl, symbol=t.symbol))


def pg_episodes(name):
    p = os.path.join(HERE, "pg_positions", f"{name}_episodes.csv")
    t = pd.read_csv(p, parse_dates=["entry_time", "exit_time"])
    return pd.DataFrame(dict(entry_time=t.entry_time, exit_time=t.exit_time,
                             direction=t.direction, pnl=t.pnl, symbol=t.coin))


def b3_episodes():
    parts = []
    for s in ["exp_r6_30m", "exp_r6_1h", "exp_r6_2h", "exp_r6_vol150", "exp_r6_8h"]:
        try:
            parts.append(pg_episodes(s).assign(sleeve=s))
        except FileNotFoundError:
            pass
    return pd.concat(parts, ignore_index=True)


def m10_trades():
    dev = pd.read_csv(os.path.join(HERE, "m10_out/m10_trades_dev.csv"),
                      parse_dates=["entry_time", "exit_time"])
    hold = pd.read_csv(os.path.join(HERE, "m10_out/m10_trades_holdout.csv"),
                       parse_dates=["entry_time", "exit_time"])
    h0 = pd.Timestamp("2025-01-01", tz="UTC")
    dev = dev[dev.entry_time < h0]
    hold = hold[hold.entry_time >= h0]
    t = pd.concat([dev, hold], ignore_index=True)
    for c in ("entry_time", "exit_time"):
        t[c] = pd.to_datetime(t[c], utc=True).dt.tz_localize(None)
    return pd.DataFrame(dict(entry_time=t.entry_time, exit_time=t.exit_time,
                             direction=t.direction, pnl=t.pnl, symbol=t.symbol))


def window_trades(t, start, end):
    return t[(t.entry_time >= pd.Timestamp(start)) &
             (t.entry_time <= pd.Timestamp(end))]


def main():
    curves = {
        "PG-A2": pg_curve("exp_r9_asymstop"),
        "PG-B3": ret_curve(os.path.join(HERE, "b3_out/b3_daily.csv")),
        "PG-C1": pg_curve("exp_r8_flow1d"),
        "M10": ret_curve(os.path.join(HERE, "m10_out/m10_full_daily.csv")),
        "CT": ct_curve(),
    }
    trades = {
        "PG-A2": pg_episodes("exp_r9_asymstop"),
        "PG-B3": b3_episodes(),
        "PG-C1": pg_episodes("exp_r8_flow1d"),
        "M10": m10_trades(),
        "CT": ct_trades(),
    }

    for window_name, (w0, w1) in [("common", COMMON), ("native", (None, None))]:
        curve_rows, trade_rows, yearly = {}, {}, {}
        for name, eq in curves.items():
            e = eq.copy()
            if e.index.tz is not None:
                e.index = e.index.tz_localize(None)
            if w0 is not None:
                e = e[(e.index >= pd.Timestamp(w0)) &
                      (e.index <= pd.Timestamp(w1) + pd.Timedelta(days=1))]
            m = curve_metrics(e, name)
            yearly[name] = (m.pop("_yearly") * 100).round(1)
            m.pop("_monthly"); m.pop("_daily")
            curve_rows[name] = m
            t = trades[name]
            if w0 is not None:
                t = window_trades(t, w0, w1)
            trade_rows[name] = trade_metrics(t)

        cdf = pd.DataFrame(curve_rows)
        tdf = pd.DataFrame(trade_rows)
        ydf = pd.DataFrame(yearly)
        cdf.to_csv(os.path.join(OUT, f"curve_metrics_{window_name}.csv"))
        tdf.to_csv(os.path.join(OUT, f"trade_metrics_{window_name}.csv"))
        ydf.to_csv(os.path.join(OUT, f"yearly_returns_{window_name}.csv"))
        print("=" * 100)
        print(f"WINDOW: {window_name}" + (f"  [{w0} .. {w1}]" if w0 else "  [each system's native span]"))
        print("=" * 100)
        with pd.option_context("display.float_format", lambda v: f"{v:,.2f}"):
            print(cdf.to_string())
            print("\n--- yearly returns (%):")
            print(ydf.to_string())
            print("\n--- trade statistics:")
            print(tdf.to_string())
        print()


if __name__ == "__main__":
    main()
