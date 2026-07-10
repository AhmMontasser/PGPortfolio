"""Equity + drawdown comparison chart for the 5 systems, common window."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_report as B  # noqa: E402
from unified_metrics import daily_curve  # noqa: E402

W0, W1 = pd.Timestamp("2020-02-01"), pd.Timestamp("2026-06-01")

SERIES = [  # fixed categorical order (validated palette)
    ("PG-A2", "#2a78d6"),
    ("PG-B3", "#1baf7a"),
    ("PG-C1", "#eda100"),
    ("M10",   "#008300"),
    ("CT",    "#4a3aa7"),
]

curves = {
    "PG-A2": B.pg_curve("exp_r9_asymstop"),
    "PG-B3": B.ret_curve(os.path.join(B.HERE, "b3_out/b3_daily.csv")),
    "PG-C1": B.pg_curve("exp_r8_flow1d"),
    "M10": B.ret_curve(os.path.join(B.HERE, "m10_out/m10_full_daily.csv")),
    "CT": B.ct_curve(),
}

daily = {}
for name, eq in curves.items():
    e = eq.copy()
    if e.index.tz is not None:
        e.index = e.index.tz_localize(None)
    e = e[(e.index >= W0) & (e.index <= W1)]
    d = daily_curve(e)
    daily[name] = d

fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.5), sharex=True,
                         gridspec_kw={"height_ratios": [2.4, 1]})
fig.patch.set_facecolor("#fcfcfb")
for ax in axes:
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color="#e4e3df", linewidth=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors="#52514e", labelsize=9)

ax1, ax2 = axes
for name, color in SERIES:
    d = daily[name]
    ax1.plot(d.index, d.values, color=color, linewidth=2.0, label=name)
    ax1.annotate(f" {name}  ×{d.iloc[-1]:,.1f}", xy=(d.index[-1], d.iloc[-1]),
                 xytext=(4, 0), textcoords="offset points", va="center",
                 fontsize=9, fontweight="bold", color="#0b0b0b")
    dd = (d / d.cummax() - 1.0) * 100
    ax2.plot(dd.index, dd.values, color=color, linewidth=1.4)

ax1.set_yscale("log")
ax1.set_ylabel("growth of 1 unit (log scale)", color="#52514e", fontsize=10)
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"×{v:g}"))
ax1.set_title("Five trading systems, identical window 2020-02-01 → 2026-05-31",
              fontsize=13, color="#0b0b0b", loc="left", fontweight="bold")
ax1.legend(loc="upper left", frameon=False, fontsize=9, labelcolor="#0b0b0b")
ax1.margins(x=0.09)

ax2.set_ylabel("drawdown (%)", color="#52514e", fontsize=10)
ax2.set_ylim(-72, 2)
ax2.margins(x=0.09)

fig.tight_layout()
out = os.path.join(HERE, "report_out", "comparison_equity.png")
fig.savefig(out, dpi=140, facecolor=fig.get_facecolor())
print("wrote", out)
