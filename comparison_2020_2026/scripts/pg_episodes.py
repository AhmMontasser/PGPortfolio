"""Convert recorded PGPortfolio decision weights into position episodes.

An episode = maximal run of consecutive decision periods for one coin with
constant position sign (non-zero). A gap in records (coin left the monthly
universe) closes the episode at its last record (the engine force-sells at
month boundaries).

Episode PnL is decision-level, close-to-close, per unit of account equity:
    pnl = sum_t w_t * (close_{t+1}/close_t - 1)   over the episode
          - fee * (|w_entry| + sum|dw| + |w_exit|)
          - borrow costs (shorts: short_apr, longs: proportional share of
            USDT borrow is portfolio-level and skipped here)
Headline system metrics come from the engine equity curve, not from this;
this exists for trade/direction statistics.
"""
import os
import sys

import numpy as np
import pandas as pd

FEE = 0.001
SHORT_APR = 0.10


def episodes_for(rec: pd.DataFrame, period_seconds: int) -> pd.DataFrame:
    rows = []
    for coin, g in rec.groupby("coin"):
        g = g.sort_values("time").reset_index(drop=True)
        t = g["time"].values
        w = g["weight"].values
        px = g["close"].values
        n = len(g)
        i = 0
        while i < n:
            if w[i] == 0.0:
                i += 1
                continue
            sign = np.sign(w[i])
            j = i
            pnl = 0.0
            turnover = abs(w[i])
            borrow = 0.0
            while j + 1 < n and np.sign(w[j + 1]) == sign \
                    and t[j + 1] - t[j] == period_seconds:
                if px[j] > 0 and np.isfinite(px[j]) and np.isfinite(px[j + 1]):
                    pnl += w[j] * (px[j + 1] / px[j] - 1.0)
                turnover += abs(w[j + 1] - w[j])
                if sign < 0:
                    borrow += abs(w[j]) * SHORT_APR * period_seconds / (365.25 * 86400)
                j += 1
            turnover += abs(w[j])  # exit
            pnl_net = pnl - FEE * turnover - borrow
            rows.append(dict(
                coin=coin, direction=int(sign),
                entry_time=pd.Timestamp(int(t[i]), unit="s"),
                exit_time=pd.Timestamp(int(t[j]), unit="s") +
                pd.Timedelta(seconds=period_seconds),
                periods=j - i + 1,
                avg_weight=float(np.mean(np.abs(w[i:j + 1]))),
                pnl=float(pnl_net),
                pnl_gross=float(pnl),
            ))
            i = j + 1
    return pd.DataFrame(rows)


def main():
    name = sys.argv[1]                    # e.g. exp_r9_asymstop
    period = int(sys.argv[2])             # trade period in seconds
    here = os.path.dirname(os.path.abspath(__file__))
    rec = pd.read_parquet(os.path.join(here, "pg_positions", f"{name}.parquet"))
    ep = episodes_for(rec, period)
    out = os.path.join(here, "pg_positions", f"{name}_episodes.csv")
    ep.to_csv(out, index=False)
    n_l = (ep.direction > 0).sum()
    n_s = (ep.direction < 0).sum()
    wr = lambda d: 100.0 * (d.pnl > 0).mean() if len(d) else float("nan")
    print(f"{name}: {len(ep)} episodes  (long {n_l}, short {n_s})  "
          f"win {wr(ep):.1f}%  long-win {wr(ep[ep.direction>0]):.1f}%  "
          f"short-win {wr(ep[ep.direction<0]):.1f}%")


if __name__ == "__main__":
    main()
