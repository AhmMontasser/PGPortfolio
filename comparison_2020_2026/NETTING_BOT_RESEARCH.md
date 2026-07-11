# Netting-Bot Research — One Bot, One Wallet, Both Strategies

Question: can a single bot run B3 + M10 as one netted strategy (one futures wallet, one public track record) with the **same performance** as the 50/50 two-account blend?

**Answer: yes — identical performance up to three measured deltas, all of which are a few basis points per year and net slightly *in favor* of the netted bot.** The blend's return is a linear combination of the two books' returns; internal netting changes only fees, margin, and B3's financing line. All three were quantified below from the systems' exact daily coin-weight panels (B3: rebuilt from the recorded decision weights of all five sleeves × META-GT tilts × overlay scales × k; M10: rebuilt from the booked EW trade exposures + all four M2 sleeves' live weight frames × risk-parity × live-scale × CPPI — CPPI replication error 0.0, M10 mean gross 0.89×, B3 mean gross 0.18×).

## 1. The three deltas, measured (common window 2020-02 → 2026-05, 50/50)

| Effect | Separate books | Netted single wallet | Impact on blend performance |
|---|---:|---:|---|
| **Gross exposure** (margin proxy) | 0.578× mean / 1.48× p95 / 3.72× max | 0.564× / 1.43× / 3.72× | 2.5% less average margin use — immaterial at these levels |
| **Order flow** | 22.0%/day of equity | 21.7%/day | 1.6% of flow crosses internally → **fees saved: +6 bps/yr** @5bps-side (+13 bps/yr @10bps) |
| **B3 financing swap** (10% APR margin-borrow model → real perp funding, on B3's actual daily weights vs actual funding prints) | −0.59%/yr of B3 equity (modeled) | −0.68%/yr (real funding: longs pay −0.80%/yr, shorts *receive* +0.12%/yr) | **−9 bps/yr on B3 → −4 bps/yr on the blend** |

**Net effect: ≈ +2 to +9 bps/yr in favor of the netted bot** — statistically invisible next to the blend's ~45%/yr CAGR and its ±bootstrap bands. The Sharpe-2.3 / MDD-13% profile carries over unchanged. The biggest feared deviation — moving B3 from its spot-margin backtest onto perps — measures out to almost exactly zero on average, because B3's funding-gated entries avoid crowded-funding regimes by design and its short book flips financing from a cost (borrow) to a small credit (received funding).

Overlap structure behind these numbers: on 64% of B3's coin-days M10 holds the same coin, but only 25% of those are opposite-direction, and the average offsetting notional is just 2.8% of equity — the two books are uncorrelated *in time*, not mirror images, so netting is real but small.

Validation note: the B3 weight panel reproduces the engine's daily returns with correlation 0.92 (residual = daily snapshot vs 4h intraday decisions + cost lines), so the overlap/turnover figures are estimates at daily resolution; conclusions are robust to this.

## 2. Architecture of the single netted bot

```
                     ┌──────────────────────────── one process ───────────────────────────┐
 market data ──────▶ │  B3 engine (5 sleeves 30m..8h → META-GT tilt → overlays → k)       │
 (klines, funding)   │        → target weights w_B3(sym)   [virtual capital: 50%]         │
                     │  M10 engine (EW 5m entries + M2 daily sleeves → vfrp → CPPI)       │
                     │        → target weights w_M10(sym)  [virtual capital: 50%]         │
                     │  NETTING EXECUTOR: net(sym) = ½·w_B3 + ½·w_M10                      │
                     │        → delta vs exchange position → orders (deadband, minNotional)│
                     └───────────────────────────────┬──────────────────────────────────--┘
                                                     ▼
                                     ONE USDT-M wallet (= one lead portfolio)
```

Load-bearing design rules:
1. **Virtual books are the isolation.** Each engine reads *its virtual equity only* (virtual positions marked to market + its share of costs). B3's vol-target/overlays and M10's CPPI must never see the wallet's combined equity, or you're live-trading an unbacktested strategy.
2. **The monthly 50/50 rebalance becomes free**: reset the two virtual-capital numbers on the first of the month. No transfers exist.
3. **Stops move into the bot.** One net exchange position cannot carry two strategies' reduce-only stops; every stop (B3's 8%/5% trailing stops, M10's C4 exits) executes as bot logic against virtual positions, with the delta sent to the wallet. Mitigate bot-death risk with a dead-man's switch (heartbeat-monitored cancel-all/flatten) — this is the one genuine new risk vs sub-accounts, where the exchange enforces isolation for you.
4. **Reconciliation loop**: assert |virtual_B3 + virtual_M10 − exchange position| < ε per symbol every cycle; alarm and halt on drift. Both repos already prescribe shadow-backtest parity checking (EW's replay-parity framework is the template — it was verified byte-identical for M10's paper deployment).
5. **Execution cadence**: the executor batches at each engine's native decision times (5m granularity for EW entries; 30m–8h bar closes for B3's sleeves; daily for M2/tilts). Same-timestamp decisions net before ordering.
6. One wallet = one lead-trading portfolio ⇒ **copiers receive the blend automatically** (ties to the copy-trading constraint that lead portfolios support API trading but sub-accounts are excluded from copy trading).

## 3. What you give up vs two sub-accounts

| | Two sub-accounts | One netted bot |
|---|---|---|
| Isolation enforced by | **the exchange** (margin, liquidation, API walls) | **your software** (virtual books) |
| Strategy-logic contamination risk | none by construction | none *if* the virtual-book discipline holds — a bug here is silent and dangerous |
| Stops | can be exchange-side per book | bot-managed only + dead-man's switch |
| Liquidation domain | per book | shared (mean gross 0.56×, max 3.72× — keep margin headroom for the max day) |
| Copy-trading | invisible to copiers | the whole point: copiers get the blend |
| Rebalance | real monthly transfers (~1.8% NAV) | free virtual reset |
| Performance | pack-test blend | same +2..9 bps/yr |

**Recommendation:** if the goal is *private* running, keep sub-accounts — exchange-enforced isolation is strictly safer than software-enforced isolation. If the goal is *one copyable strategy with the blend's track record*, the netted single bot is the correct architecture, the performance is the same by construction (and marginally better after netting), and the engineering effort concentrates in exactly two places: the virtual-book state machine and the reconciliation/dead-man safety layer.

## Files
`data/netting_daily_stats.csv`, `data/b3_daily_coin_weights.parquet`, `m10_daily_coin_weights.parquet` (EW side), `scripts/netting_research.py`, `scripts/ew_m10_weights_dump.py`.
