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


def main():
    for directory in sys.argv[1:]:
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
