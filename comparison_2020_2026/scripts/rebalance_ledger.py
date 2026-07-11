"""Option A operational backtest: the monthly master-transfer ledger.

Simulates the two sub-accounts (B3, M10) at $10k total start, 50/50,
monthly rebalance executed via master transfers on the first trading day
of each month (1-day lag, 10bps/side re-positioning cost on moved capital).
Outputs the full transfer ledger and summary stats + a chart.
"""
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
import build_report as B
from unified_metrics import daily_curve

APY = 365.25
W0, W1 = pd.Timestamp("2020-02-01"), pd.Timestamp("2026-06-01")
START = 10_000.0
COST_BPS = 10.0        # per side, on moved capital (position resize cost)
DEADBAND_PP = 0.0      # transfer every month; deadband variant reported too

frame = {}
for name, eq in [("B3", B.ret_curve(os.path.join(HERE, "b3_out/b3_daily.csv"))),
                 ("M10", B.ret_curve(os.path.join(HERE, "m10_out/m10_full_daily.csv")))]:
    e = eq.copy()
    if e.index.tz is not None:
        e.index = e.index.tz_localize(None)
    e = e[(e.index >= W0) & (e.index <= W1)]
    frame[name] = daily_curve(e).pct_change().dropna()
R = pd.DataFrame(frame).dropna()

rb_dates = set(R.groupby([R.index.year, R.index.month]).apply(lambda g: g.index[0]).values)


def run(deadband_pp=0.0):
    v = np.array([0.5, 0.5]) * START
    ledger = []
    equities = []
    pend = []
    for i, (t, row) in enumerate(R.iterrows()):
        v = v * (1 + row.values)
        if t in rb_dates and i > 0:
            pend.append(i + 1)                      # execute next day
        if pend and i >= pend[0]:
            pend.pop(0)
            total = v.sum()
            split = v[0] / total
            drift_pp = (split - 0.5) * 100
            if abs(drift_pp) > deadband_pp:
                target = np.array([0.5, 0.5]) * total
                moved = float(abs(target - v)[0])
                sender = "B3" if v[0] > target[0] else "M10"
                fee = moved * (COST_BPS / 1e4) * 2
                v = target * (1 - fee / total)
                ledger.append(dict(date=t, from_sub=sender,
                                   to_sub="M10" if sender == "B3" else "B3",
                                   amount_usd=moved, pct_of_nav=100 * moved / total,
                                   pct_of_sender=100 * moved / (target[0] + moved),
                                   drift_pp=drift_pp, nav=total, fee_usd=fee))
        equities.append((t, v[0], v[1]))
    eq = pd.DataFrame(equities, columns=["date", "B3_sub", "M10_sub"]).set_index("date")
    led = pd.DataFrame(ledger)
    return eq, led


eq, led = run(DEADBAND_PP)
led.to_csv(os.path.join(HERE, "report_out", "rebalance_ledger.csv"), index=False)

print(f"=== Option A monthly transfer ledger: {len(led)} transfers over "
      f"{(R.index[-1]-R.index[0]).days/APY:.1f} years ===")
print(f"start $10,000 -> final ${eq.iloc[-1].sum():,.0f}")
amt = led.amount_usd
pct = led.pct_of_nav
print(f"transfer size: mean ${amt.mean():,.0f} ({pct.mean():.2f}% NAV) | "
      f"median ${amt.median():,.0f} ({pct.median():.2f}%) | "
      f"max ${amt.max():,.0f} ({pct.max():.2f}% NAV on {led.loc[amt.idxmax(),'date'].date()})")
print(f"direction: B3->M10 in {(led.from_sub=='B3').sum()} months, "
      f"M10->B3 in {(led.from_sub=='M10').sum()} months")
print(f"max single transfer as % of sending sub's equity: {led.pct_of_sender.max():.1f}%")
print(f"total repositioning fees paid: ${led.fee_usd.sum():,.0f} over the whole period")

# deadband variant
_, led2 = run(2.0)
print(f"\nwith a 2pp deadband: {len(led2)} transfers "
      f"({len(led)-len(led2)} months skipped), max {led2.pct_of_nav.max():.2f}% NAV")

print("\nlargest 8 transfers:")
cols = ["date", "from_sub", "to_sub", "amount_usd", "pct_of_nav", "drift_pp"]
print(led.nlargest(8, "amount_usd")[cols].to_string(index=False,
      formatters={"amount_usd": lambda v: f"${v:,.0f}",
                  "pct_of_nav": lambda v: f"{v:.2f}%",
                  "drift_pp": lambda v: f"{v:+.1f}pp"}))

yearly_flow = led.assign(year=led.date.dt.year).groupby("year").agg(
    n=("amount_usd", "size"), total_moved=("amount_usd", "sum"),
    avg_pct_nav=("pct_of_nav", "mean"))
print("\nper-year flow:")
print(yearly_flow.to_string(formatters={"total_moved": lambda v: f"${v:,.0f}",
                                        "avg_pct_nav": lambda v: f"{v:.2f}%"}))

# ---- chart: sub equities + transfer bars ----
fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.5), sharex=True,
                         gridspec_kw={"height_ratios": [2.2, 1]})
fig.patch.set_facecolor("#fcfcfb")
for ax in axes:
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color="#e4e3df", linewidth=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors="#52514e", labelsize=9)
ax1, ax2 = axes
ax1.plot(eq.index, eq.B3_sub, color="#eda100", lw=1.8, label="sub-account B3")
ax1.plot(eq.index, eq.M10_sub, color="#008300", lw=1.8, label="sub-account M10")
ax1.set_yscale("log")
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax1.set_title("Option A: the two sub-accounts under monthly 50/50 master transfers ($10k start)",
              fontsize=12.5, loc="left", fontweight="bold", color="#0b0b0b")
ax1.legend(loc="upper left", frameon=False, fontsize=9)
ax1.annotate(f" B3  ${eq.B3_sub.iloc[-1]:,.0f}", xy=(eq.index[-1], eq.B3_sub.iloc[-1]),
             xytext=(4, 0), textcoords="offset points", va="center", fontsize=9,
             fontweight="bold", color="#0b0b0b")
ax1.annotate(f" M10  ${eq.M10_sub.iloc[-1]:,.0f}", xy=(eq.index[-1], eq.M10_sub.iloc[-1]),
             xytext=(4, 0), textcoords="offset points", va="center", fontsize=9,
             fontweight="bold", color="#0b0b0b")
ax1.margins(x=0.08)

sign = np.where(led.from_sub == "M10", 1.0, -1.0)     # + = flows INTO B3
ax2.bar(led.date, sign * led.pct_of_nav, width=18, color=np.where(sign > 0, "#eda100", "#008300"))
ax2.axhline(0, color="#c9c8c2", lw=0.8)
ax2.set_ylabel("monthly transfer (% of NAV)\n+ into B3 / − into M10", color="#52514e", fontsize=9)
ax2.margins(x=0.08)
fig.tight_layout()
out = os.path.join(HERE, "report_out", "rebalance_ledger.png")
fig.savefig(out, dpi=140, facecolor=fig.get_facecolor())
print(f"\nwrote {out}")
