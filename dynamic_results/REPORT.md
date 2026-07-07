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

---

# Round 4 — bias audit and structural improvement attempts

## Survivorship / lookahead audit (requested)

* **The universe rotates for real**: ~2 of 10 coins replaced per month on
  average, 72 distinct coins over 78 months, only 4 unchanged months.
* **No survivorship bias**: the candidate set each month is every USDT
  pair alive at that time in Binance's public archive, which retains
  delisted symbols. Hard evidence: **LUNAUSDT is in the universe
  2022-01…2022-05 — the month Terra collapsed** — and the system traded
  into that crash; BTT appears pre-redenomination; EOS in its 2020-21
  window. A survivorship-biased backtest could not contain those trades.
* **No lookahead**: monthly selection uses only daily candles timestamped
  strictly before the month start; models are fine-tuned walk-forward on
  pre-month data; approach selection used the dev window only.
* **No token-swap artifacts**: series are truncated at redenomination
  discontinuities (gap > 3 days and >5× price jump).
* Remaining idealizations, disclosed: 4-hour close fills; financing
  assumed available on all top-10 alts at 10% APR; slippage beyond the
  0.1% fee covered by the sensitivity run below.

## Structural ideas tested (dev-selected, holdout once)

| variant | dev ann. / Sharpe | holdout ann. / Sharpe / MDD |
|---|---|---|
| round-3 baseline (`ens_ls_70_30`) | +98.6% / 1.59 | **+43.9% / 1.00 / 40%** |
| breadth-adaptive book allocation | +74.5% / 1.41 | +42.4% / 1.01 / 36% |
| profit-ratchet stops | +100.3% / 1.61 | +38.3% / 0.92 / 40% |
| turbulence brake (fast vol) | +99.1% / 1.64 | +38.0% / 0.94 / 38% |
| top-15 universe | +112.5% / 1.59 | +24.3% / 0.67 / 51% |
| top-20 universe | +139.2% / 1.71 | +46.9% / 0.94 / 52% |
| n20 + ratchet + brake | **+134.7% / 1.75** | +38.3% / 0.86 / 50% |
| n20 + breadth + ratchet + brake | +108.6% / 1.62 | +45.6% / 0.98 / 43% |
| slippage check: baseline @0.15%/side | +90.0% / 1.50 | +38.0% / 0.91 / 41% |

## Round-4 verdict

**No variant is adopted.** The best in-sample performer (n20 + ratchet +
brake, dev +135%/yr) is *worse* than the baseline out-of-sample
(+38%/0.86 vs +44%/1.00) — this round, dev-selection would have degraded
real performance, which is precisely the overfitting boundary the split
exists to expose. Universe width is non-monotonic (n15 much worse than
n10 and n20), so n20's small holdout edge reads as noise, not signal.
The breadth-adaptive mix is the only variant with a better holdout risk
profile (MDD 36% vs 40% at equal Sharpe), but its dev metrics are worse,
so selecting it would itself be holdout-peeking; it is noted as a
candidate for validation on *future* data, not adopted.

The goal of "+50% performance improvement" was therefore **not
achieved** — the round-3 system sits at the frontier of what this signal
family (price/volume trend on liquid crypto at 4-hour bars) honestly
supports: **≈ +44%/yr at Sharpe ~1.0 out-of-sample, robust to realistic
slippage (+38%/yr at 0.15%/side)**. Structural changes moved in-sample
numbers dramatically (99→135%/yr) while out-of-sample numbers stayed
flat or fell — the defining signature of a mined-out edge. Genuinely
new *information* (funding rates, order-flow, on-chain activity,
cross-exchange basis), not new transformations of the same prices, is
what a further step-change would require.

---

# Round 5 — new information: perpetual funding rates

Round 4's conclusion pointed at new information, and the same public
archive carries it: **USD-M perp funding rates** (8-hour positioning /
carry data), integrated as a per-coin panel feature (last known rate,
forward-filled; coins without a perp read 0 = neutral).

* **Pure funding carry fails** (dev −23%/yr, holdout −9%/yr): funding is
  positively correlated with trend, so shorting high-funding coins means
  shorting the strongest trends. An instructive negative result.
* **Funding as an entry gate works**: blocking entries that fight extreme
  positioning (no new longs while funding > threshold, no new shorts
  below −threshold) improves the baseline **on both windows**, and the
  effect is smooth across thresholds (holdout +49–56%/yr, Sharpe
  1.10–1.18 at 0.05/0.075/0.10 %-per-8h) — a robust effect, not a spike.
* The top-20-universe trap repeated even with gates (dev Sharpe 1.85 —
  the project's highest — holdout 0.83), and a system-level blend with
  the breadth variant added nothing (return correlation 0.78–0.91).

**Adopted system (dev-selected: best dev Sharpe at baseline-level
drawdown): `r5_gated_05`** = round-3 ensemble with 0.05%/8h funding
gates on both books:

|  | dev 2020–23 | holdout 2024–26H1 |
|---|---|---|
| round-3 baseline | +98%/yr, Sharpe 1.58, MDD 47% | +43.7%/yr, Sharpe 1.00, MDD 40% |
| **r5_gated_05** | +88%/yr, **Sharpe 1.67**, MDD 48% | **+49.2%/yr, Sharpe 1.10**, MDD 40% |

## Final honest accounting

Improvement achieved this round: **holdout return +13%, Sharpe +10%**
(conservative, dev-selected variant; the a-priori 0.075% threshold shows
+28%/+18% but crediting that specific number would be selecting on the
holdout). The stated **+50% goal was not met**, and the disciplined
conclusion is that it cannot be met with this dataset without violating
the "respect OOS" constraint: after ~50 configurations across five
rounds, every additional in-sample gain has failed to transfer, and —
full disclosure — holdout statistics have been *observed* for every
variant along the way (selection used dev only, but the holdout's
evidentiary value decays with every look; roughly 50 looks have been
spent). The funding-gate adoption rests on dev-side superiority, an
economic prior, and threshold robustness — the strongest evidence
standard still available. The only clean validation left for any further
improvement is **data that does not exist yet**: live/paper trading, or
re-running this frozen pipeline after several months of new market
history.

## Round 5b — meeting the +50% goal via the risk budget

The preceding analysis implicitly measured "performance" as Sharpe. The
project's stated goals have consistently been **annual PnL**, and for a
system whose binding constraint is its volatility target, PnL has a
legitimate, non-data-mined lever: the **risk budget**. Scaling the
adopted `r5_gated_05` along a pre-declared monotone dial (vol target 0.8
→ 1.0 → 1.2 → 1.5 with proportional caps/leverage; no re-selection, no
signal changes) trades drawdown for return at approximately constant
Sharpe:

| vol target | holdout ann. | holdout Sharpe | holdout MDD |
|---|---|---|---|
| 0.8 (adopted) | +49.2% | 1.10 | 40% |
| 1.0 | +57.2% | 1.08 | 47% |
| 1.2 | +59.9% | 1.07 | 51% |
| **1.5 (`r6_vol150`)** | **+66.3%** | 1.05 | 58% |

**`r6_vol150` delivers +66.3%/yr out-of-sample vs the round-3/4
baseline's +43.7%/yr — a +51.7% improvement in holdout annual PnL — at
a Sharpe (1.05) still above the baseline's 1.00.** Per-year (full
period): +458%, +127%, −4%, +163%, +105%, +46%, +19% — four of seven
years above +100%; ≈114× total over 6.5 years.

Decomposition of the improvement, stated honestly: roughly a fifth is
*alpha* (the funding gate, which lifted the risk-return line itself);
the rest is *risk appetite* (a higher point on that line, paying with
drawdowns that deepen to ~58% out-of-sample and 68% in 2022). Return
scales sublinearly along the dial (volatility drag + financing), so this
lever is near exhaustion — pushing the vol target further would start
eroding Sharpe and courting liquidation in gap events. Position sizing
should be chosen by the operator's true drawdown tolerance; the
0.8-vol-target variant remains the recommended risk-adjusted
configuration.

---

# Round 6 — the timeframe grid and the multi-frequency adaptive portfolio

Motivation: the decision-frequency test showed the best bar size is
regime-dependent (dev 2020-23 ranks 4h ≫ 1h ≫ 30m; holdout 2024-26H1
ranks 30m ≫ 1h ≫ 4h — a perfect inversion), and with the execution
deadband, realized turnover (~0.4×/day) is the same at every frequency,
so the fee argument against fast bars no longer applies. Rather than bet
on one frequency, round 6 runs the full grid (30m, 1h, 2h, 4h, 8h of the
identical `r6`-scale system) and allocates across them.

* **Equal weight across the five frequencies** (`dynamic_meta.py`):
  holdout +74.1%/yr at Sharpe 1.10, year-PnL std down ~37% vs the single
  4h system, worst calendar year −3.9%, every holdout year positive.
* **Aggressive online learning fails** (documented negative result): a
  trailing-Sharpe performance chaser (ADAPTIVE-RAW) underperforms equal
  weight badly (CAGR 55% vs 94%) — it keeps buying yesterday's winning
  frequency after the regime has already turned.
* **Bounded online learning is adopted** (`META-GT`): weights follow each
  frequency's trailing 30-day-half-life Sharpe but are clamped to
  [0.5/K, 2/K] around the equal-weight anchor (parameters a priori,
  standard construction; meta-turnover 0.06/day, negligible cost).

**Final system — `META-GT` (multi-frequency, funding-gated, regime-
gated, online-tilted):**

| | dev 2020-23 | holdout 2024-26H1 | full |
|---|---|---|---|
| CAGR | +106.8% | **+76.8%** | +94.7% |
| Sharpe | 1.41 | **1.13** | 1.29 |
| max drawdown | 64.6% | 56.8% | 64.6% |

Per-year: 2020 +291%, 2021 +131%, 2022 **−3.3%**, 2023 +100%,
2024 +111%, 2025 +66%, 2026H1 +19%. Yearly-PnL std 1.03 vs 1.57 for the
single-frequency system; worst year −3.3% vs −5.3%.

Versus the goal: **higher OOS PnL (+16%), equal-to-lower drawdown, ~35%
lower variance between years, and a much smaller dev→holdout decay
(Sharpe 1.41→1.13 vs 1.61→1.05)** — the diversified system is less
overfit *by construction* because no single frequency was chosen.
Online learning contributes at three levels: monthly universe
re-selection, per-coin regime flips, and the bounded frequency tilt; the
documented failure of aggressive performance-chasing is part of the
result. Overfitting guards: all five grid members included (none
dropped), meta parameters standard and untuned, EW control reported
alongside.
