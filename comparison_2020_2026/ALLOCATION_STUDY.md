# Allocation Study — Does a Portfolio of These Strategies Beat the Best Single One?

Addendum to `BACKTEST_COMPARISON_2020_2026.md`. Same daily return series, same common window 2020-02-01 → 2026-05-31. Anti-overfitting discipline: allocations are either **a-priori naive rules** (zero fitted parameters) or **causal adaptive rules** (weights from trailing data only, shift-1, monthly rebalance). A full-sample max-Sharpe portfolio is computed **only** as a labeled overfit ceiling. Allocation-level validation splits at 2024-01-01 (dev vs holdout).

> **Revision v2 (pack-test verified).** The first published version of this study understated combo drawdowns: its rebalance calendar seeded weights on calendar-month labels, which left every combo uninvested for February 2020 — the COVID crash month — so combo MDDs missed the crash. A full independent account simulation ("pack test", §5) caught the bug. All tables below are corrected; the singles' rows and all holdout-window numbers were never affected. Headline change: 50/50 blend MDD 9.2% → **13.0%**, Calmar 4.73 → **3.50**; Sharpe moves +0.03 to **2.32**. Every qualitative conclusion survives.

## 1. Why allocation should work here: the correlation structure

Daily-return correlations (monthly in parentheses):

| | PG-A2 | PG-B3 | PG-C1 | M10 | CT |
|---|---:|---:|---:|---:|---:|
| PG-A2 | 1 | .89 (.91) | .93 (.96) | .06 (.08) | .45 (.54) |
| PG-B3 | | 1 | .91 (.92) | .10 (.10) | .43 (.51) |
| PG-C1 | | | 1 | .08 (.05) | .47 (.57) |
| M10 | | | | 1 | .05 (−.09) |
| CT | | | | | 1 |

Three facts drive everything:
1. **The three PG systems are one family** (ρ ≈ 0.9): A2, B3, C1 share the same engine; holding more than one mostly re-buys the same book. B3 is the natural representative (it is the deployment-grade, risk-budgeted build).
2. **M10 is essentially uncorrelated with everything** (ρ ≈ 0.05–0.10). Different signal class (5m Elliott-Wave execution + market-neutral cross-sectional sleeves vs 4h channel breakout), different holding period (median 0.6d vs 2–4d).
3. **CT is a diluted version of the PG beta** (ρ ≈ 0.45) with a much weaker standalone record on this window.

Textbook arithmetic then predicts the headline: two ~equal-risk sleeves at ρ=0.10 give Sharpe ≈ (1.49+1.95)/√(2·1.10) ≈ **2.3** — and that is exactly what materializes.

## 2. Results — common window (monthly rebalance, drift between rebalances, frictionless; see §5 for costs)

| Allocation | CAGR | Vol | Sharpe | Sortino | MDD | Calmar | Worst mo | Worst yr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| *best single (Sharpe): M10* | *+64.5%* | *27.4%* | *1.95* | *3.18* | *23.1%* | *2.80* | *−9.6%* | *−6.5%* |
| *best single (MDD): PG-B3* | *+25.5%* | *16.1%* | *1.49* | *2.29* | *18.2%* | *1.40* | *−6.5%* | *+5.1%* |
| EW-5 (20% each) | +62.9% | 35.1% | 1.56 | 2.54 | 34.6% | 1.82 | −13.7% | +14.8% |
| IVW-5 (causal inverse-vol) | +46.3% | 22.5% | 1.80 | 2.95 | 16.8% | 2.76 | −7.4% | +8.6% |
| SharpeTilt-5 (causal, bounded) | +66.9% | 34.1% | 1.67 | 2.77 | 33.5% | 2.00 | −14.6% | +7.3% |
| MinVar-5 (causal) | +28.5% | 15.9% | 1.66 | 2.63 | 13.5% | 2.11 | −5.2% | +6.3% |
| EW-3fam (B3/M10/CT) | +35.2% | 17.0% | 1.86 | 3.29 | 14.5% | 2.43 | −5.8% | +4.9% |
| IVW-3fam (B3/M10/CT) | +33.7% | 15.3% | 1.98 | 3.37 | 13.3% | 2.54 | −5.1% | +4.0% |
| **50/50 B3+M10** | **+45.6%** | **16.8%** | **2.32** | **4.26** | **13.0%** | **3.50** | **−5.0%** | **−0.7%** |
| IVW B3+M10 (causal) | +40.7% | 15.0% | 2.35 | 4.19 | 13.0% | 3.14 | −4.9% | −0.5% |
| 45/45/10 B3/M10/CT | +42.5% | 16.2% | 2.27 | 4.28 | 13.5% | 3.16 | −4.7% | +1.0% |
| `[OVERFIT]` full-sample max-Sharpe (45% B3 + 50% M10 + 5% A2) | +50.4% | 18.3% | 2.32 | 4.30 | 13.3% | 3.79 | −5.5% | +0.7% |

**The naive 50/50 split of the two uncorrelated books equals the overfit ceiling** (Sharpe 2.32 both) — the in-sample optimizer independently lands on ≈50/50 B3/M10 and adds nothing. When the zero-parameter prior coincides with the in-sample optimum, weight selection carries no overfitting risk worth arguing about.

Naive-diversification pathologies are also visible: EW-5 is *worse* than M10 on every risk-adjusted measure because equal-weighting five systems triple-counts the correlated PG family and overweights the weakest system (CT). **Diversify across families, not across config files.**

## 3. Allocation-level dev/holdout validation (split 2024-01-01)

| | dev Sharpe | holdout Sharpe | dev MDD | holdout MDD |
|---|---:|---:|---:|---:|
| 50/50 B3+M10 | 2.40 | **2.17** | 13.0% | **9.2%** |
| IVW B3+M10 (causal) | 2.51 | 2.10 | 13.0% | 8.6% |
| 45/45/10 B3/M10/CT | 2.40 | 2.03 | 13.5% | 9.2% |
| dev-optimized max-Sharpe (60/40 B3/M10, frozen) | 2.45 | 2.10 | 11.0% | 8.6% |
| IVW-3fam | 2.30 | 1.51 | 13.3% | 8.9% |
| M10 alone | 1.95 | 1.95 | 23.1% | 17.2% |
| PG-B3 alone | 1.73 | 1.14 | 18.2% | 11.1% |
| CT alone | 0.87 | 0.01 | 20.5% | 33.4% |

The blend's edge is stable out-of-sample at the allocation layer: Sharpe 2.40→2.17, and it beats the best single system in the holdout (2.17 vs M10's 1.95) at roughly half the drawdown (9.2% vs 17.2%). Weights optimized on dev and frozen (60/40) do **not** beat the naive 50/50 out-of-sample — optimization adds nothing here, which is exactly what a non-overfit result looks like.

## 4. Answers

**Does % allocation across these strategies make sense?** Yes — but only across the *three families*, and the benefit comes almost entirely from one pairing: **PG-B3 + M10 (ρ≈0.10)**. Combining near-duplicates (A2+B3+C1) or adding a weak diluted book (CT at full weight) delivers naive-diversification, not real diversification.

**Is there a non-overfitting split that beats the best single system with lower drawdown?** Yes, and it is the most boring one possible: **50% PG-B3 / 50% M10, rebalanced monthly.**
- vs M10 alone: Sharpe 2.32 vs 1.95, Sortino 4.26 vs 3.18, Calmar 3.50 vs 2.80, **MDD 13.0% vs 23.1%**, worst month −5.0% vs −9.6% — at the cost of headline CAGR (+45.6%/yr vs +64.5%/yr, because the blend runs at 17% vol vs 27%).
- The CAGR gap is a *volatility* choice, not an efficiency loss: at matched risk (levering the blend ~1.6× on the futures account) the same Sharpe would imply materially higher return than M10 at M10's own drawdown depth — with financing costs and gap risk as the price of leverage.
- **Optional insurance sleeve:** 45/45/10 with CT costs ~0.05 Sharpe and buys regime convexity — CT was the only positive book in 2022's bear (+36.7%) and leads 2026 YTD (+20.3% while M10 is −3.7%). Justified by mechanism (only true short-trend book), not by its standalone stats.

**Practical rule, fully pre-registered:** hold the two uncorrelated families at equal risk (≈ equal capital here), rebalance monthly, and re-derive nothing from returns. If one must adapt, causal inverse-vol (IVW B3+M10) performed equivalently (holdout Sharpe 2.10) — there is no evidence that anything cleverer is needed.

## 5. Pack test — full account simulation of the 50/50 (verification layer)

Independent from-scratch simulator (`scripts/pack_test_5050.py`): $10,000 account, two sub-books compounding daily, explicit rebalance transfers charged per side on the moved notional, optional 1-day execution lag. This is the test that caught the v1 drawdown bug.

| Variant | CAGR | Sharpe | MDD | Calmar | Final $ |
|---|---:|---:|---:|---:|---:|
| monthly, frictionless, no lag | +45.6% | 2.32 | 13.0% | 3.51 | $107,795 |
| monthly, 10 bps/side, 1-day lag | +45.5% | 2.31 | 13.0% | 3.49 | $107,065 |
| monthly, 50 bps/side, 1-day lag | +45.2% | 2.30 | 13.1% | 3.47 | $105,935 |
| weekly, 10 bps, lag | +45.2% | 2.32 | 12.4% | 3.65 | $105,486 |
| quarterly, 10 bps, lag | +46.4% | 2.29 | 13.1% | 3.54 | $111,187 |
| never rebalance (drift) | +51.4% | 2.12 | 14.4% | 3.58 | $137,574 |
| M10 alone | +64.5% | 1.95 | 23.1% | 2.80 | $233,152 |
| PG-B3 alone | +25.5% | 1.49 | 18.2% | 1.40 | $41,996 |

Findings: rebalancing moves only ~**1.8% of NAV per month**, so costs are immaterial (50 bps/side costs 4 Sharpe-bps); the result is insensitive to rebalance frequency and execution lag; letting weights drift un-rebalanced slowly re-concentrates into M10 (higher CAGR, lower Sharpe, deeper MDD). Max-DD window: 2020-02-13 → 2020-06-22 (COVID) at −13.0%; next-worst episodes −9.2% (2026), −9.0% (2022), −8.7% (2024).

**Block-bootstrap significance** (stationary bootstrap, 5,000 draws, mean block 20 days) of the blend's edge over M10 alone:
- ΔSharpe (blend − M10): median **+0.37**, 90% CI [+0.08, +0.68] → **P(blend > M10) = 98.2%**
- ΔMDD (blend − M10): median **−9.6 pp**, 90% CI [−17.6, −4.3] → **P(blend shallower) = 99.8%**

## 6. Honest limits

1. **The allocation layer is non-overfit; the sleeves are not.** All five systems were *developed* on overlapping parts of this same history — PGPortfolio's own deflated-Sharpe audit puts its post-search edge probability near 0.5, and M10's holdout has been read five times across its milestones. A robust split of over-fit sleeves is still bounded by the sleeves' true forward edge. The blend inherits, and cannot launder, that uncertainty.
2. **B3's 18.2% MDD budget was calibrated full-sample by its authors** (the k=0.401 scaling), so its drawdown figure is partly by construction; the blend's ~13% MDD would scale accordingly if B3's live drawdown exceeds budget.
3. **Average correlation ≠ crash correlation.** ρ≈0.10 is a full-period average; the blend's −13% COVID drawdown is itself the proof that both books can fall together in a liquidation cascade. The 90% bootstrap CI on the MDD advantage ([−17.6, −4.3] pp) is the honest range.
4. Meta-level rebalancing costs are measured and immaterial (§5), but running both stacks requires operating two live infrastructures and split margin.

## Files

`data/allocation_common.csv`, `data/allocation_devhold.csv`, `data/allocation_daily.csv`, `data/pack_test_5050.csv`, `data/pack_test_5050_equity.csv`, `allocation_blend.png`, `scripts/allocation_study.py`, `scripts/pack_test_5050.py`, `scripts/make_alloc_chart.py`.
