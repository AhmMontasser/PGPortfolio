# Deployment audit — A2 / B3 / C1 (pre-paper-trading)

## 1. Edge-truth audit (lookahead / leakage)

**Empirical tests (run, not argued):**

| test | result |
|---|---|
| **Negative control** — engine deliberately peeks 1 bar ahead (`decision_lag=-1`) | Sharpe explodes to 4.2 dev / 3.6 holdout (+1,341%/yr). The honest system (1.64/1.25) sits nowhere near the leak signature; if the base were already leaking, injected peeking could not multiply performance ~10×. **PASS** |
| **Delay sensitivity** — decide 1 bar late (`decision_lag=1`) | Holdout 66→28%/yr: edge is real but concentrated in the first bar; requires prompt execution (minutes after bar close). Live performance bounded by [delay-0, delay-1]. |
| **Bar-close identity** — 60 random (coin, bar) samples vs raw 30m DB | Panel bar t equals the last 30m close inside bucket [t, t+period); nothing from the next bucket. **PASS** |
| **Funding causality** — 40 samples | Funding value stamped at bar t always originates from an event ≤ t, never the next print. **PASS** |
| **Universe boundary** — 3 months × 10 coins | Selection statistics use daily candles strictly before month start. **PASS** |
| Survivorship (round-4 audit) | LUNA in-universe through its May-2022 collapse; 72 distinct coins; delisted symbols retained. **PASS** |
| Overlay causality (code audit) | Every meta/overlay signal (tilts, downside-vol, corr gate, vol-of-vol, dispersion) is `shift(1)`-lagged. **PASS** |

**Disclosed idealizations (all conservative or bounded):** token-swap
truncation uses load-time knowledge (only *removes* data — conservative);
pre-listing history is flat-backfilled (signal-free by construction);
fills at decision-bar close with 0.1% fee (slippage +5bps costs ~6pts/yr
— measured); financing fixed at 10% APR and borrow availability assumed
(both become *logged live quantities* below). Residual, non-testable
risk: accumulated holdout looks across rounds (documented in report) —
which is precisely why paper trading is the next instrument.

## 2. Exchange-minimums compliance (Binance)

Mechanics implemented in the backtester (`trading.initial_capital_usdt`,
`trading.min_notional`): any order below minNotional is skipped and the
position persists as dust (matches exchange behavior; dust conversion not
assumed).

| account size | dev Sharpe delta | holdout delta |
|---|---|---|
| $10,000 | 0.000 | 0.000 |
| $2,000 (≈ B3 per-frequency sleeve at $10k) | +0.000 | 0.000 |
| $1,000 | +0.002 | 0.000 |

Why it's a non-issue: the 5% equity deadband makes every rebalance
chunky, so per-coin orders are almost always far above 10 USDT.
LOT_SIZE/step rounding error is bounded by one step per order (<0.1% of
notional at ≥$1k) — negligible. **Recommended minimums: ≥$5k for A2/C1;
≥$10k for B3 (five ≥$2k sleeves). Margin-borrow availability per coin
must be checked at entry and logged (assumed in backtest).**

## 3. Paper-trading logging specification (triple-checked)

**The question to answer:** is live performance consistent with the
backtest OOS estimate — and if not, is the gap *implementation*
(slippage, latency, fees, borrow) or *alpha decay*? Every log item below
exists to separate those two.

**A. Decision log (every bar, every system):** timestamp; config hash;
universe list; per coin: raw member weights, post-gate weights,
post-overlay target, current drifted position; which gate/overlay fired
(funding gate, deadband, min-notional skip, stop, vol/corr/dd scale
values); equity mark.

**B. Execution log (every order):** intended vs submitted vs filled
notional; decision-bar close vs fill price (slippage, bps); submit and
fill timestamps (latency vs bar close); fees paid; partial fills.

**C. Financing log (daily):** actual margin borrow rate per held coin;
funding prints used by gates; any borrow-availability failure (entry
skipped because the coin wasn't borrowable — backtest assumes never).

**D. Shadow backtest (nightly, the key instrument):** run the identical
frozen config through this repo's pipeline on the archive's data for the
same days; log shadow-vs-paper daily return difference. Tracking error
isolates implementation gap; shadow-vs-history drift isolates alpha
decay. Without this single item the final question is unanswerable.

**E. Market context (daily):** BTC return, universe dispersion, breadth,
realized vol — so a flat six months can be classified "no trend supply"
vs "edge gone" (the system's PnL is regime-conditional by design).

**F. Account state (daily):** equity (unit-based, so deposits/withdrawals
don't distort returns), gross/net exposure, positions, dust inventory,
margin level; exchange-incident/downtime notes; NTP clock-skew check.

**Pre-registered evaluation criteria (fixed now, before any live data):**

1. *Implementation health (weekly):* mean |slippage| ≤ 10 bps; median
   decision→fill latency < 60 s; shadow tracking error RMS < 0.15%/day;
   zero unlogged borrow failures. Failing these means fix operations —
   it says nothing about alpha.
2. *Alpha health (monthly, from month 3):* realized Sharpe must stay
   above the 5th percentile of same-length windows bootstrapped from the
   backtest's OOS returns; de-risk 50% after 2 consecutive failing
   months; stop if live drawdown exceeds the backtest's 99th-percentile
   same-horizon drawdown.
3. *Verdict timeline:* ≥6 months before any scale-up decision; ~12
   months for a "system is great" claim (power analysis: distinguishing
   Sharpe ≈1.2 from 0 at 95% confidence needs ≈2 years — six months
   yields directional evidence plus full implementation validation, not
   statistical proof; the criteria above are designed to catch *failure*
   fast, not to certify success fast).
