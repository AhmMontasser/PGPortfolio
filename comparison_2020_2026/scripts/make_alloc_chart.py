"""Chart: 50/50 B3+M10 blend vs components + CT variant, common window."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import allocation_study as al

R = al.R

series = {}
series["50/50 B3+M10"] = al.run_rule("x", al.fixed(np.ones(2)), R[["PG-B3", "M10"]])
series["45/45/10 +CT"] = al.run_rule("x", al.fixed(np.array([0.45, 0.45, 0.10])),
                                     R[["PG-B3", "M10", "CT"]])
series["M10"] = R["M10"]
series["PG-B3"] = R["PG-B3"]
series["CT"] = R["CT"]

COLORS = {  # validated categorical order; blend gets slot 1
    "50/50 B3+M10": "#2a78d6",
    "45/45/10 +CT": "#1baf7a",
    "M10": "#008300",
    "PG-B3": "#eda100",
    "CT": "#4a3aa7",
}
LW = {"50/50 B3+M10": 2.6, "45/45/10 +CT": 1.6, "M10": 1.6, "PG-B3": 1.6, "CT": 1.6}

fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.0), sharex=True,
                         gridspec_kw={"height_ratios": [2.2, 1]})
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
for name, r in series.items():
    eq = (1 + r).cumprod()
    ax1.plot(eq.index, eq.values, color=COLORS[name], linewidth=LW[name], label=name)
    ax1.annotate(f" {name}  ×{eq.iloc[-1]:,.1f}", xy=(eq.index[-1], eq.iloc[-1]),
                 xytext=(4, 0), textcoords="offset points", va="center",
                 fontsize=9, fontweight="bold", color="#0b0b0b")
    dd = (eq / eq.cummax() - 1) * 100
    ax2.plot(dd.index, dd.values, color=COLORS[name], linewidth=LW[name] * 0.8)

ax1.set_yscale("log")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"×{v:g}"))
ax1.set_ylabel("growth of 1 unit (log scale)", color="#52514e", fontsize=10)
ax1.set_title("Strategy allocation: 50/50 PG-B3 + M10 (ρ≈0.10) vs components — 2020-02 → 2026-05",
              fontsize=12.5, color="#0b0b0b", loc="left", fontweight="bold")
ax1.legend(loc="upper left", frameon=False, fontsize=9, labelcolor="#0b0b0b")
ax1.margins(x=0.10)
ax2.set_ylabel("drawdown (%)", color="#52514e", fontsize=10)
ax2.set_ylim(-40, 2)
ax2.margins(x=0.10)

fig.tight_layout()
out = os.path.join(HERE, "report_out", "allocation_blend.png")
fig.savefig(out, dpi=140, facecolor=fig.get_facecolor())
print("wrote", out)
