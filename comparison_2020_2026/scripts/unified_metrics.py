"""Unified performance metrics computed identically for every system.

Inputs:
  equity : pd.Series (datetime index, any resolution; resampled to daily)
  trades : optional pd.DataFrame with columns
           [entry_time, exit_time, direction (+1/-1), pnl (system units),
            ret (per-trade net return on traded notional, optional)]

All curve metrics use daily returns, 365.25-day annualization, rf = 0.
"""
import numpy as np
import pandas as pd

APY = 365.25


def daily_curve(equity: pd.Series) -> pd.Series:
    eq = equity.dropna().sort_index()
    if eq.index.tz is not None:
        eq.index = eq.index.tz_localize(None)
    daily = eq.resample("1D").last().ffill()
    return daily / daily.iloc[0]


def curve_metrics(equity: pd.Series, label="") -> dict:
    eq = daily_curve(equity)
    r = eq.pct_change().dropna()
    years = (eq.index[-1] - eq.index[0]).days / APY
    total = eq.iloc[-1] / eq.iloc[0] - 1.0
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1.0
    vol = r.std() * np.sqrt(APY)
    sharpe = r.mean() / r.std() * np.sqrt(APY) if r.std() > 0 else np.nan
    downside = r[r < 0]
    sortino = (r.mean() / downside.std() * np.sqrt(APY)
               if len(downside) > 1 and downside.std() > 0 else np.nan)
    peaks = eq.cummax()
    dd = 1.0 - eq / peaks
    mdd = float(dd.max())
    calmar = cagr / mdd if mdd > 0 else np.nan

    # drawdown durations: longest peak-to-recovery stretch, in days
    under = dd > 0
    longest, cur = 0, 0
    for u in under.values:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)

    monthly = eq.resample("1MS").last().pct_change().dropna()
    yearly = eq.groupby(eq.index.year).apply(lambda x: x.iloc[-1] / x.iloc[0] - 1.0)

    win_days = float((r > 0).mean())
    t_stat = float(r.mean() / r.std() * np.sqrt(len(r))) if r.std() > 0 else np.nan
    var95 = float(np.percentile(r, 5))
    cvar95 = float(r[r <= var95].mean()) if (r <= var95).any() else np.nan

    return {
        "label": label,
        "start": eq.index[0].date().isoformat(),
        "end": eq.index[-1].date().isoformat(),
        "years": round(years, 2),
        "total_return_pct": total * 100,
        "cagr_pct": cagr * 100,
        "ann_vol_pct": vol * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown_pct": mdd * 100,
        "calmar": calmar,
        "longest_dd_days": longest,
        "pct_positive_days": win_days * 100,
        "pct_positive_months": float((monthly > 0).mean()) * 100,
        "best_month_pct": float(monthly.max()) * 100 if len(monthly) else np.nan,
        "worst_month_pct": float(monthly.min()) * 100 if len(monthly) else np.nan,
        "best_year_pct": float(yearly.max()) * 100,
        "worst_year_pct": float(yearly.min()) * 100,
        "daily_skew": float(r.skew()),
        "daily_kurtosis": float(r.kurtosis()),
        "var95_daily_pct": var95 * 100,
        "cvar95_daily_pct": cvar95 * 100,
        "t_stat_daily_mean": t_stat,
        "n_days": len(r),
        "_yearly": yearly,
        "_monthly": monthly,
        "_daily": r,
    }


def trade_metrics(trades: pd.DataFrame) -> dict:
    """trades: entry_time, exit_time, direction (+1/-1), pnl. Optional: ret."""
    t = trades.copy()
    out = {"n_trades": len(t)}
    if len(t) == 0:
        return out
    t["win"] = t["pnl"] > 0
    months = max(1e-9, (pd.Timestamp(t["exit_time"].max()) -
                        pd.Timestamp(t["entry_time"].min())).days / 30.4375)
    out["trades_per_month"] = len(t) / months

    def block(sub, prefix):
        if len(sub) == 0:
            return {f"{prefix}n": 0}
        wins, losses = sub[sub.pnl > 0], sub[sub.pnl <= 0]
        gross_w, gross_l = wins.pnl.sum(), -losses.pnl.sum()
        d = {
            f"{prefix}n": len(sub),
            f"{prefix}win_rate_pct": 100.0 * len(wins) / len(sub),
            f"{prefix}profit_factor": gross_w / gross_l if gross_l > 0 else np.inf,
            f"{prefix}avg_pnl": sub.pnl.mean(),
            f"{prefix}avg_win": wins.pnl.mean() if len(wins) else np.nan,
            f"{prefix}avg_loss": losses.pnl.mean() if len(losses) else np.nan,
            f"{prefix}payoff": (wins.pnl.mean() / -losses.pnl.mean()
                                if len(wins) and len(losses) and losses.pnl.mean() != 0
                                else np.nan),
            f"{prefix}total_pnl": sub.pnl.sum(),
        }
        if "entry_time" in sub and "exit_time" in sub:
            hold = (pd.to_datetime(sub.exit_time).values -
                    pd.to_datetime(sub.entry_time).values) / np.timedelta64(1, "D")
            d[f"{prefix}avg_hold_days"] = float(np.mean(hold))
            d[f"{prefix}med_hold_days"] = float(np.median(hold))
        return d

    out.update(block(t, ""))
    out.update(block(t[t.direction > 0], "long_"))
    out.update(block(t[t.direction < 0], "short_"))
    out["pct_long"] = 100.0 * (t.direction > 0).mean()
    out["pct_short"] = 100.0 * (t.direction < 0).mean()
    return out


def slice_window(equity: pd.Series, start, end) -> pd.Series:
    eq = equity.dropna().sort_index()
    if eq.index.tz is not None:
        eq.index = eq.index.tz_localize(None)
    return eq[(eq.index >= pd.Timestamp(start)) & (eq.index <= pd.Timestamp(end))]
