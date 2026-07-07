"""Compare dynamic-backtest iterations side by side.

Usage:
    python dynamic_compare.py dynamic_results/margin_v1 dynamic_results/margin_v2 ...

Prints overall metrics plus per-year return / max-drawdown for every run
directory (using the full-resolution portfolio_values.csv), and flags
whether each year meets the target of >=120% annual return with <=15%
max drawdown.
"""

from __future__ import absolute_import, division, print_function

import os
import sys

import numpy as np
import pandas as pd

TARGET_RETURN = 1.20
TARGET_MDD = 0.15


def yearly_stats(curve):
    """per-year return and max drawdown from an equity curve series."""
    rows = []
    for year, chunk in curve.groupby(curve.index.year):
        base = chunk.iloc[0]
        equity = chunk / base
        peaks = equity.cummax()
        rows.append({
            "year": year,
            "return": float(equity.iloc[-1]) - 1.0,
            "mdd": float((1.0 - equity / peaks).max()),
        })
    return pd.DataFrame(rows).set_index("year")


def describe(directory):
    curves = pd.read_csv(os.path.join(directory, "portfolio_values.csv"),
                         index_col=0, parse_dates=True)
    name = curves.columns[0] if len(curves.columns) == 1 else None
    frames = {}
    for column in curves.columns:
        if name and column != name:
            continue
        table = yearly_stats(curves[column].dropna())
        table["ok"] = ((table["return"] >= TARGET_RETURN) &
                       (table["mdd"] <= TARGET_MDD))
        frames[column] = table
    summary = pd.read_csv(os.path.join(directory, "summary.csv"), index_col=0)
    return summary, frames


def split_stats(directory, split="2024-01-01"):
    """Development vs holdout metrics for the first (or only) curve."""
    curves = pd.read_csv(os.path.join(directory, "portfolio_values.csv"),
                         index_col=0, parse_dates=True)
    curve = curves[curves.columns[0]].dropna()
    rows = {}
    for label, chunk in (("dev", curve[:split]), ("holdout", curve[split:])):
        if len(chunk) < 10:
            continue
        equity = chunk / chunk.iloc[0]
        rets = equity.pct_change().dropna()
        years = (chunk.index[-1] - chunk.index[0]).days / 365.25
        periods_per_year = len(rets) / years
        peaks = equity.cummax()
        rows[label] = {
            "total": float(equity.iloc[-1]) - 1.0,
            "annualized": float(equity.iloc[-1]) ** (1 / years) - 1.0,
            "sharpe": float(rets.mean() / rets.std() * (periods_per_year ** 0.5)),
            "mdd": float((1 - equity / peaks).max()),
        }
    return curves.columns[0], pd.DataFrame(rows).T


def main():
    args = sys.argv[1:]
    split = None
    if args and args[0].startswith("--split"):
        split = args.pop(0).split("=")[1] if "=" in args[0] else "2024-01-01"
    if split:
        table = {}
        for directory in args:
            try:
                name, stats = split_stats(directory, split)
            except Exception as e:
                print("%-24s ERROR %s" % (directory, e))
                continue
            for period_label, row in stats.iterrows():
                table[(os.path.basename(directory), period_label)] = row
        frame = pd.DataFrame(table).T
        frame.index.names = ["run", "window"]
        print(frame.to_string(float_format=lambda v: "%8.3f" % v))
        return
    for directory in args:
        print("=" * 70)
        print(directory)
        print("=" * 70)
        summary, frames = describe(directory)
        print(summary.to_string(float_format=lambda v: "%.4f" % v))
        for column, table in frames.items():
            print("\nper-year (%s):  [target: return >= %.0f%%, MDD <= %.0f%%]" %
                  (column, TARGET_RETURN * 100, TARGET_MDD * 100))
            print(table.to_string(
                formatters={"return": lambda v: "%+8.1f%%" % (v * 100),
                            "mdd": lambda v: "%6.1f%%" % (v * 100),
                            "ok": lambda v: " PASS" if v else " fail"}))
        print()


if __name__ == "__main__":
    main()
