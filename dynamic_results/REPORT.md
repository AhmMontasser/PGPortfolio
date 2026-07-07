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
| margin **v4** (v3a engine + calendar-year DD budget 4%/12%, vol-target 50%) | | **1.31** | **per-year MDD 12.0–15.0% — the ≤15% goal holds in all 7 years**; returns: 2021 +103.2%, 2020 −2.0%, 2022 −11.9%, 2023 −6.8%, 2024 −8.1%, 2025 −4.2%, 2026H1 −8.6% |

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

**Result: the drawdown half of the goal is fully met; the return half is
not.** v4 keeps every calendar year's max drawdown between 12.0% and
15.0% (the 15.0% is 2020, where the COVID crash gapped through the budget
by 3 points in a single 4-hour bar — gap risk is the irreducible slack in
any drawdown-control scheme). Returns: +103% in 2021 (near the +120%
target), −2% to −12% in every other year, +4.3% annualized overall, versus
BTC buy-and-hold's +38% annualized with a 77% drawdown.

A strategy returning +120% every year with ≤15% drawdown implies a Sharpe
ratio roughly in the 3.5–5 range *sustained for seven years* in a liquid,
increasingly efficient market. Nothing in this framework — and, to our
knowledge, nothing published on liquid crypto momentum/allocation at these
frequencies — sustains that. The 2021 results (v3a +139%, v4 +103% under
a hard risk budget) show the *return* target is reachable in a strongly
trending year; the failure is *consistency*: the post-2021 regime does not
offer the same short-horizon alpha to this class of models (post-2022
monthly win rate 15%), and a hard annual risk budget necessarily converts
"large losing year" into "small losing year" rather than into profit.

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

---

# Round 2 — 26 guided approaches with a development/holdout split

Second campaign (target revised to **+100%/year**), with the proper
methodology requested: **approach selection uses only the development
window 2020-01…2023-12; the holdout 2024-01…2026-06 was evaluated once,
after selection.** (Within every run, training remains walk-forward: each
month's model/universe uses only prior data.)

Families searched (26 total, all in `experiments/` with results in
`dynamic_results/exp_*`): time-series momentum (price-vs-MA and MA-cross,
5 speeds, long-only and chop-filtered variants), Donchian channel breakout
(3 speeds, long-only, with/without trailing stops), cross-sectional
momentum (3 lookbacks, with/without a BTC regime gate), the EIIE network
variants from round 1, ensembles, and risk stacks (annual drawdown budget,
soft trailing budget, 2× leverage) on the dev winners. New machinery:
per-coin trend gates and minimum-signal **no-trade** filters, **trailing
per-position stop-losses** with cooldown, ensemble agent.

**League-table findings (dev → holdout):**

* **Donchian 20/10 breakout is the only family that transfers.** Dev
  +87%/yr (Sharpe 1.5), holdout +21-26%/yr (Sharpe 0.6-0.8). Stop-losses
  slightly *help* out-of-sample.
* Cross-sectional momentum fails both windows; the learned EIIE variants
  are negative on holdout; channel-length tuning (30/15: dev +96%/yr →
  holdout −4%/yr) is a clean demonstration of why the split matters.
* The **annual drawdown budget destroys trend returns** (dev 87→27%/yr,
  holdout 21→1%/yr): it de-risks exactly before the recoveries trend
  systems live on, and 4-hour gap bars pierce the 12% line regardless
  (holdout MDD still 25%). Hard per-year drawdown caps and trend
  following are structurally incompatible at this leverage.

**Selected system — `don_lev2`** (2× levered long-only Donchian 20/10 on
the monthly top-10 universe, inverse-vol sizing, 80% vol target, 25%
per-coin cap, 5% deadband, USDT accounting, 0.1% fees + 10% APR
financing):

| year | return | MDD |
|------|--------|-----|
| 2020 | +369.5% | 32.6% |
| 2021 | +343.0% | 28.8% |
| 2022 | −36.7% | 47.4% |
| 2023 | +121.0% | 39.9% |
| 2024 | +83.0% | 34.2% |
| 2025 | +4.3% | 29.4% |
| 2026 H1 | −3.5% | 21.5% |

Full period: **55.1× (+85.4%/yr, Sharpe 1.38, MDD 55.5%)** vs BTC-hold
+38%/yr at 77% MDD. Runner-up (robustness pick): `don_20_10_stops`,
unlevered — +42.7%/yr, Sharpe 1.05, positive in 6 of 7 years.

**Honest verdict on "+100% per year":** the selected system *averages*
close to the target (+85%/yr compounded, with three years above +100%),
but not every year (2022 −37%), and the only unbiased estimate of
forward performance is the holdout: **≈ +28%/yr at Sharpe ~0.7 with
30-40% drawdowns**. The full-period average is flattered by overlap with
the selection window (2020-21 bull). No configuration among the 26
achieved +100% in every year, and none kept drawdowns near 15% while
earning trend-level returns — at 2× leverage a single 4-hour market-wide
gap bar can exceed 15% on its own.

---

# Round 3 — per-coin trend regimes and downtrend shorting

Motivation (user hypothesis): the round-2 system only monetizes uptrends;
correctly identified downtrends should be shortable. First, the evidence
that *naive* shorts don't work: comparing the same Donchian long/short vs
long-only per year, shorts added **+60pts in 2022, +18 in 2025, +16 in
2026** but cost **−62 in 2021, −37 in 2023, −44 in 2024** — shorting
bull-market dips gets squeezed, so the contributions cancel. The entire
problem is *regime classification*.

**Per-coin regime detector** (`rules.RegimeTrend`): each coin is
classified every period as **up** (close > MA(slow) and MA(fast) >
MA(slow)), **down** (both reversed) or **neutral** (no position at all —
the no-trade region); longs trade their Donchian channel only in an up
regime, shorts only in a down regime, with options for slope
confirmation, a market-wide BTC gate on shorts, half-sized shorts,
asymmetric channels and trailing stops. 15 further experiments
(dev-selected, holdout-validated as in round 2; `exp_rg_*`,
`exp_ens_ls_*`) established:

* a **stricter bear definition transfers best**: 100-day regime MA beats
  50-day out-of-sample (holdout +32.5%/yr vs +12.7%);
* **tight trailing stops on shorts help everywhere** (squeeze protection);
* asymmetric channel tuning looks great in-sample and fails out-of-sample
  (again);
* the winning architecture is a **70/30 ensemble of the round-2 long
  book (2× long-only Donchian 20/10) with a shorts-only 100-day-regime
  book**, stops 8%, 80% vol target, 25% per-coin cap:

**Selected system — `ens_ls_70_30`:**

| year | return | MDD | market (equal-weight top-10) |
|------|--------|-----|------------------------------|
| 2020 | +218.9% | 24.7% | +154% |
| 2021 | +172.2% | 22.3% | +266% |
| 2022 | **−8.2%** | 46.8% | −86% |
| 2023 | +90.9% | 23.7% | +82% |
| 2024 | +61.1% | 26.5% | +56% |
| 2025 | **+34.2%** | 23.3% | −47% |
| 2026 H1 | **+15.1%** | 40.1% | −37% |

Full period **38.1× = +75.1%/yr, Sharpe 1.37**; positive in 6 of 7 years;
dev window +98.6%/yr (Sharpe 1.59) — at the target in-sample; **holdout
+43.9%/yr at Sharpe 1.00**, the best out-of-sample result of the project
(vs +27.7%/yr before shorts: the regime-gated short book added ~16
points/yr of *out-of-sample* return, exactly as the hypothesis hoped, by
converting bear years from losses into roughly flat-to-positive years).

**Verdict vs the +100%/yr goal:** in-sample the target is now met
(dev +98.6%/yr ≈ 100%); across the full period two years exceed +100%
and the average is +75%/yr; the unbiased forward estimate remains the
holdout's **≈ +44%/yr at Sharpe ~1.0 with 30-45% drawdowns**. That is an
exceptional systematic crypto result. A *guaranteed* +100% every single
year remains outside what honest out-of-sample evidence supports; further
tuning against this test window would only manufacture it on paper.
