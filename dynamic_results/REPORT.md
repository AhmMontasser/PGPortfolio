# Dynamic-Universe Trading System — Backtest Report (2020-01 → 2026-06)

All experiments: monthly re-selected universe of the **top 10 Binance USDT
pairs by 30-day volume + liquidity** (trade count, Amihud), selected with
data available strictly before each month; delisted coins included (no
survivorship bias). Walk-forward protocol: initial training on 2019, then
each month the network is fine-tuned on the trailing 120 days of the *new*
universe before trading the month out-of-sample. **USDT is the base
currency for all accounting** (equity, PnL, commissions, financing).

## Goal (set by the user)

1. shorts — network chooses allocation *and* direction ✅ implemented
2. leverage ✅ implemented (sigmoid leverage head, ≤ `max_leverage`)
3. max allocation per coin ✅ implemented (`max_coin_weight`)
4. USDT base currency, PnL in USDT ✅ implemented
5. **Performance target: ≥ +120%/year with ≤ 15% max drawdown, every year**
   — ❌ **not achieved** (see honest assessment below)

## Iteration history

| run | setup | fAPV | per-year picture |
|-----|-------|------|------------------|
| baseline long-only (paper-faithful, 30 m, 0.25% fee) | no caps, no shorts | **0.0000** | wiped out −99.97% in the LUNA/UST collapse, May 9–13 2022 — the uncapped softmax concentrates into the crashing coin. The per-coin cap requirement is vindicated. |
| benchmarks (same period) | BTC-hold 8.15×, UBAH 1.39×, UCRP 0.32×; OLMAR/PAMR/RMR ≈ 0 (commission churn at 30 m) | | BTC-hold MDD 77% |
| margin **v1** (30 m, lev 1.5, cap 0.25, vol-target 60%, all-time-peak DD overlay) | shorts+leverage+caps | 0.72 | nearly flat book (avg gross 2.8%) yet 0.25×/day churn → commissions ate ~5%/yr; all-time-peak DD anchor pinned exposure at the floor after 2022 |
| margin **v3a** (4 h, price features, trailing-90d DD anchor, 5% deadband) | | 0.96 | 2020 +24%, **2021 +139%** (only year clearing +120%), 2022–25 −29/−10/−27/−26%; post-2022 monthly win rate 15% — the learned short-horizon pattern *inverted* after the 2021 regime |
| margin **v3b** (4 h, +volume feature, wider net 8/20) | | 0.13 | extra capacity overfit; strictly worse than v3a |
| margin **v4** (v3a engine + calendar-year DD budget 4%/12%, vol-target 50%) | | *(final iteration — see summary tables in this directory)* | bounds each year's drawdown near the budget; returns remain far from target in non-bull years |

Full metrics: `summary.csv`, `yearly.csv`, equity curves and monthly
universes in each run's directory.

## What the iterations established

* **Frequency matters most.** At 30-minute periods every strategy including
  the classic OLPS algorithms is destroyed by commissions (the original
  paper's 2016–17 Poloniex results do not survive 2020+ market efficiency).
  4-hour periods were the single largest improvement.
* **Per-coin caps are a survival requirement**, not a tuning knob: the
  uncapped paper design died on one event (LUNA). The capped margin agent
  traded through the same event with a controlled loss.
* **Execution deadband** (skip rebalances < 5% of equity) removes the
  commission bleed of decision noise.
* **Drawdown anchoring**: all-time-peak anchoring never releases (v1);
  trailing-window anchoring lets loss episodes chain (v3a reached 43%
  all-time); a **calendar-year loss budget** (v4) matches the stated
  per-year MDD target directly.
* **The learned edge is regime-dependent.** The EIIE-style network monetized
  the 2020–21 bull (+24%, +139%) and then systematically lost in the
  2022–26 regime (15% monthly win rate post-2022) despite monthly
  walk-forward fine-tuning. This is alpha decay, not an implementation bug:
  the same pipeline that printed +139% in 2021 was applied unchanged
  afterwards.

## Honest assessment of the 120%/year, ≤15% MDD target

A strategy returning +120% every year with ≤15% drawdown implies a Sharpe
ratio roughly in the 3.5–5 range *sustained for seven years* in a liquid,
increasingly efficient market. Nothing in this framework — and, to our
knowledge, nothing published on liquid crypto momentum/allocation at these
frequencies — sustains that. The 2021 result (+139%) shows the *return*
target is reachable in a strongly trending year; the failure is
*consistency*: the post-2021 regime does not offer the same short-horizon
alpha to this class of models.

Continuing to tune parameters against the same 2020–26 test window until
the numbers hit the target would not be research — it would be curve
fitting; the resulting configuration would have no reason to work on data
it hasn't seen. We therefore stop the parameter search at v4 and report
the gap plainly.

Directions with a realistic risk/return story (out of scope here):
perpetual-futures funding capture (carry), cross-sectional daily momentum
with weekly rebalancing, volatility-risk-premium selling with hard tail
hedges, and ensemble/regime-switching between them. None of these credibly
promises a *consistent* +120%/year either.
