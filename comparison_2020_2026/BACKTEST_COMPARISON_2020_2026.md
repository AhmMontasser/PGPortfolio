# Five-System Backtest Comparison — 2020→2026, Binance USDT-M Futures Basis

**Date of study:** 2026-07-10 · **Process:** read-only — every system was run exactly as committed on its branch; the only unification is the evaluation window. All metrics below were recomputed from each system's raw equity curve and trade records by one shared script, so definitions are identical across systems.

## 1. The five systems

| ID | Repo / branch | What it is | Traded book |
|---|---|---|---|
| **PG-A2** | PGPortfolio `claude/dynamic-universe-backtesting-ansci0` (`experiments/r9_asymstop.json`) | Funding-gated Donchian 20/10 breakout long book (70%) + 100-day-regime short book (30%), monthly top-10 dynamic universe, 4h decisions, max 3.5× gross, vol-target 1.5 overlay, 8% trailing stop / 5% short stop | Long + short, ~0.70× avg gross |
| **PG-B3** | same repo (5 sleeves + META-GT + R20c overlays) | The A2 rule run at 30m/1h/2h/4h/8h as five sleeves, combined daily by bounded online-Sharpe tilts (META-GT), plus downside-vol targeting, correlation gate, vol-of-vol brake, scaled to the 18.2%-MDD risk budget (k=0.401) | Long + short, risk-capped |
| **PG-C1** | same repo (`experiments/r8_flow1d.json`) | A2 rule stack + 1-day taker-order-flow entry confirmation (blocks entries that fight recent taker flow) | Long + short |
| **M10** | EW_trader `m10-universe-signal-expansion` (`scripts/opt10/compose_holdout_m10.py`) | Two-book blend: Elliott-Wave breakout engine (1d/4h/1h context, 5m execution, W2→W3 / W4→W5 / C-fade sleeves) + M2 daily cross-sectional sleeves (momentum, carry, idio-momentum, crowd-contrarian), vol-floored risk parity at lev 4, live-scale gate, CPPI 25% drawdown cap; M10 = M9 + momentum-sleeve ddstop 0.20 | Long + short, top-50 futures universe |
| **CT** | Crypto_Trader `claude/binance-futures-bot-nlt69c` (`config/config.yaml`, LOCKED) | 4h trend-follower: EMA50>EMA200 + MACD gate, trend-flip/MACD-recross entries, 2×ATR stop, 4×ATR chandelier trail, monthly top-50 universe, 0.5% equity risk per unit, ≤10 units, portfolio heat cap, drawdown throttle; shorts gated by daily EMA100 macro filter | Long + short, ~0.40× avg gross |

## 2. Data basis (all from Binance's official archive, `data.binance.vision`)

| System | Price data | Futures-specific costs |
|---|---|---|
| PG-A2 / B3 / C1 | Spot 30m klines (universe from spot dailies, survivorship-free incl. delisted) | UM-futures funding-rate gates; 10% APR short borrow + USDT borrow financing; 0.1% taker fee per side |
| M10 | **UM-futures 5m klines** (survivorship-free top-50 futures membership) ; M2 sleeve panels on spot dailies (audited ≤0.2% divergence) | Real per-stamp funding charged per held day (long pays positive funding); 0.11% cost per side; gap-honest stop fills (worse of level/open) |
| CT | **UM-futures 1h klines** resampled to 4h/1d | Flat 0.01%/8h funding on open notional both directions; 5 bps taker fee + 2 bps adverse slippage per fill; adverse gap fills on stops |

## 3. Reproduction validation (before comparing anything)

Every system was first reproduced against its repo's own committed results:

* **PG-A2, PG-C1, all five B3 sleeves** — regenerated `summary.csv`/`yearly.csv`/`universes.csv` are **byte-identical** to the committed files (deterministic rules + identical archive data).
* **PG-B3** — the rebuilt curve matches the committed `final_r8_capped.csv` with **daily-return correlation 1.0000** and identical stats (×4.97, +28.0%/yr, MDD 18.2%, Sharpe 1.52).
* **M10** — dev window ×11.9 / Sharpe 1.84 / MDD −23.1% / 90d-DD −22.8% vs committed reference ×11.4 / 1.82 / −23.1% / −22.8%; holdout **+95.6% / −17.2% / Sharpe 1.87** vs committed **+95.5% / −17.2% / 1.87** (essentially exact; small dev-side PnL delta consistent with archive revisions).
* **CT** — IS segment 2020-02→2025-01 reproduces the committed locked baseline **exactly** (+124.8% / −26.2% MDD).

## 4. Headline comparison — identical window 2020-02-01 → 2026-05-31 (6.33 years)

Common window = intersection of all systems' tradable spans (CT's futures universe begins 2020-02; M10's canonical span ends 2026-05-31). All metrics from daily equity, 365.25-day annualization, rf=0.

| Metric | PG-A2 | PG-B3 | PG-C1 | M10 | CT |
|---|---:|---:|---:|---:|---:|
| **Total PnL %** | **+8,682%** | +316% | +5,186% | +2,232% | +120% |
| **CAGR (PnL%/yr)** | **+102.8%** | +25.3% | +87.2% | +64.5% | +13.2% |
| Ann. volatility | 68.2% | 16.1% | 68.9% | 27.4% | 30.1% |
| **Max drawdown** | 57.4% | **18.2%** | 65.3% | 23.1% | 35.6% |
| **Sharpe** | 1.36 | 1.48 | 1.24 | **1.95** | 0.56 |
| **Sortino** | 2.04 | 2.28 | 1.85 | **3.18** | 0.73 |
| **Calmar** | 1.79 | 1.39 | 1.33 | **2.79** | 0.37 |
| Longest drawdown | 514 d | 509 d | 624 d | 721 d | 1,216 d |
| Positive days | 36.7% | 40.2% | 38.0% | **52.3%** | 36.8% |
| Positive months | 52.6% | 57.9% | 53.9% | **62.7%** | 42.1% |
| Best / worst month | +95.0 / −27.5% | +17.4 / −6.5% | +94.2 / −30.6% | +44.8 / −9.6% | +38.6 / −9.1% |
| Best / worst year | +280 / +0.0% | +47.8 / +4.9% | +194.8 / +10.2% | +122.3 / −3.7% | +36.7 / −13.8% |
| Daily VaR95 / CVaR95 | −4.7 / −7.0% | −1.0 / −1.7% | −4.7 / −7.2% | −1.4 / −2.6% | −2.2 / −3.7% |
| Daily skew / kurtosis | +2.3 / 21.8 | +1.5 / 9.4 | +2.2 / 22.3 | +3.9 / 59.0 | +0.5 / 5.6 |
| t-stat of daily mean | 3.43 | 3.73 | 3.12 | **4.90** | 1.41 |

### Per-year returns (%, common window; 2020 = Feb–Dec, 2026 = Jan–May)

| Year | PG-A2 | PG-B3 | PG-C1 | M10 | CT |
|---|---:|---:|---:|---:|---:|
| 2020 | +280.0 | +47.8 | +194.8 | +92.5 | +24.9 |
| 2021 | +100.1 | +33.2 | +104.3 | +59.2 | +30.5 |
| 2022 | +0.0 | +5.6 | +10.2 | +8.2 | +36.7 |
| 2023 | +180.6 | +32.0 | +96.1 | +122.3 | +9.8 |
| 2024 | +94.4 | +26.5 | +110.2 | +61.9 | −10.2 |
| 2025 | +73.8 | +15.7 | +63.1 | +115.0 | −13.8 |
| 2026 (Jan–May) | +28.6 | +4.9 | +22.6 | −3.7 | +20.3 |

Every system was profitable in 2022's bear market (CT the standout at +36.7% via its short book); 2024–25 split the field — the PG systems and M10 compounded strongly while CT lost money both years; 2026 YTD reverses that (CT +20.3%, M10 −3.7%).

## 5. Trade & direction statistics — common window

Trade semantics differ by system and are **not** unit-comparable across systems (see notes): PG systems = per-coin position *episodes* (decision-level, fee- and short-borrow-adjusted, PnL as fraction of account equity); M10 = the Elliott-Wave trade book exactly as its P&L layer books it (per-unit-notional PnL net of 2×0.11% cost and funding; the M2 sleeves are daily-rebalanced overlays with no discrete trades); CT = discrete unit trades in USDT (net of fees).

| Metric | PG-A2 | PG-B3 | PG-C1 | M10 (EW book) | CT |
|---|---:|---:|---:|---:|---:|
| Trades | 1,202 | 5,669 | 993 | 592 | 2,751 |
| Trades / month | 15.8 | 74.6 | 13.1 | 8.1 | 35.8 |
| Long / short split | 62% / 38% | 67% / 33% | 64% / 36% | **31% / 69%** | 55% / 45% |
| **Win rate** | 52.7% | 51.5% | 52.6% | 34.6% | 31.7% |
| Win rate — longs | 51.8% | 51.1% | 51.8% | 35.3% | 28.8% |
| Win rate — shorts | 54.2% | 52.3% | 53.9% | 34.3% | 35.2% |
| **Profit factor** | 3.78 | 2.98 | 3.51 | 1.91 | 1.18 |
| PF — longs | 3.41 | 2.88 | 3.40 | 1.72 | 1.16 |
| PF — shorts | 4.66 | 3.15 | 3.70 | 1.98 | 1.20 |
| Payoff (avg win / avg loss) | 3.39 | 2.80 | 3.16 | 3.60 | 2.54 |
| Avg / median hold (days) | 4.2 / 2.3 | 4.7 / 2.7 | 5.2 / 3.2 | 1.1 / 0.6 | 5.2 / 3.7 |
| Hold — longs vs shorts (avg d) | 5.0 vs 2.8 | 4.7 vs 4.8 | 5.2 vs 5.1 | 1.2 vs 1.0 | 4.7 vs 5.7 |
| Long-book share of trade PnL | 62% | 61% | 61% | 22% | 51% |

Notable structure: the three PG systems win >50% of episodes with ~3× payoff (breakout entries with funding gates); M10 and CT are classic low-win-rate/high-payoff trend books (~32–35% win, 2.5–3.6× payoff). M10's EW book is shorts-dominant (69%) yet earns most of its blend return in 2023–25 from both books plus the M2 overlays; every system's short book has a *higher* win rate and PF than its long book.

## 6. Native-window results (each system's full span, for reference)

| Metric | PG-A2 | PG-B3 | PG-C1 | M10 | CT |
|---|---:|---:|---:|---:|---:|
| Span | 2020-01→2026-06 | 2020-01→2026-06 | 2020-01→2026-06 | 2019-06→2026-05 | 2020-02→2026-06 |
| Total PnL | +14,831% | +397% | +8,058% | +2,232% | +158% |
| CAGR | +116.2% | +28.0% | +96.9% | +56.8% | +15.9% |
| Max DD | 57.4% | 18.2% | 65.3% | 23.1% | 35.6% |
| Sharpe | 1.44 | 1.52 | 1.30 | 1.85 | 0.64 |
| Calmar | 2.02 | 1.54 | 1.48 | 2.46 | 0.45 |

(M10's native span includes a flat 2019 ramp-up; its common-window CAGR is higher because the zeros drop out. PG systems' January 2020 was a +55%-class month, which the common window excludes — hence native CAGR > common CAGR for A2/C1.)

## 7. How to read this comparison honestly

1. **Risk-adjusted vs absolute.** On the identical window, **M10 dominates risk-adjusted** (Sharpe 1.95, Sortino 3.18, Calmar 2.79, MDD 23%, 52% positive days). **PG-A2 dominates absolute return** (+102.8%/yr) but at 57% max drawdown and −27.5% worst month. **PG-B3 is the tame configuration** (16% vol, 18.2% MDD — its design budget) with the second-best Sharpe. CT is the weakest on this window on every risk-adjusted measure, though it was the only system that made 2022 its *best* year.
2. **Same period ≠ same risk.** Gross leverage, vol targets and cost models differ by design (PG at 3.5× max gross vs CT at ~0.4× realized gross). Sharpe/Sortino/Calmar are leverage-insensitive and are the fairest cross-system numbers here; raw PnL% is not.
3. **These are backtests, run by each project's own engine.** Each repo's own audits disclose the residual risks: PGPortfolio's deflated-Sharpe analysis puts the post-search probability that its edge beats best-of-N-random at ≈0.50, and its delay test shows the edge concentrates in the first bar (prompt execution is load-bearing); M10's holdout has been read 5 times across milestones (evidentiary weight decayed, per its own docs); CT's committed OOS segment (+21–26% depending on equity-state handling) is modest. None of this is visible in a single backtest table.
4. **Cost realism differs.** M10 charges real per-stamp funding and gap-honest stop fills; CT charges flat funding both directions plus slippage; PG charges 0.1%/side + financing APRs but fills at decision-bar close with no slippage (its own audit prices +5 bps slippage at ≈6 pts/yr). The PG numbers are therefore the most optimistic on execution.
5. **Trade-stat semantics differ** (episodes vs trade book vs unit trades) — compare *within* a column, and across columns only for rates/ratios (win rate, PF, payoff, hold), not PnL magnitudes.

## 8. Reproducibility

All runs used each repo's committed code at the stated branches with data from `data.binance.vision` (spot + UM futures, including delisted symbols). Binance REST APIs are geo-blocked from the study machine; every system's own data layer already targets the public archive, so no code was modified. Comparison artifacts (unified metric CSVs for both windows, the chart, and the glue scripts for the B3 rebuild, M10 export, PG episode extraction, and this report) are committed alongside this file under `comparison_2020_2026/`.
