# Allocation Study — Does a Portfolio of These Strategies Beat the Best Single One?

Addendum to `BACKTEST_COMPARISON_2020_2026.md`. Same daily return series, same common window 2020-02-01 → 2026-05-31. Anti-overfitting discipline: allocations are either **a-priori naive rules** (zero fitted parameters) or **causal adaptive rules** (weights from trailing data only, shift-1, monthly rebalance). A full-sample max-Sharpe portfolio is computed **only** as a labeled overfit ceiling. Allocation-level validation splits at 2024-01-01 (dev vs holdout).

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

## 2. Results — common window (monthly rebalance, drift between rebalances, no meta-level costs)

| Allocation | CAGR | Vol | Sharpe | Sortino | MDD | Calmar | Worst mo | Worst yr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| *best single (Sharpe): M10* | *+64.5%* | *27.4%* | *1.95* | *3.18* | *23.1%* | *2.80* | *−9.6%* | *−6.5%* |
| *best single (MDD): PG-B3* | *+25.5%* | *16.1%* | *1.49* | *2.29* | *18.2%* | *1.40* | *−6.5%* | *+5.1%* |
| EW-5 (20% each) | +60.0% | 35.0% | 1.51 | 2.44 | 34.6% | 1.74 | −13.7% | +14.8% |
| IVW-5 (causal inverse-vol) | +43.7% | 22.3% | 1.73 | 2.81 | 16.8% | 2.61 | −7.4% | +8.6% |
| SharpeTilt-5 (causal, bounded) | +63.9% | 34.0% | 1.62 | 2.67 | 33.5% | 1.91 | −14.6% | +7.3% |
| MinVar-5 (causal) | +26.3% | 15.6% | 1.57 | 2.45 | 11.3% | 2.33 | −5.2% | +6.3% |
| EW-3fam (B3/M10/CT) | +33.8% | 16.7% | 1.83 | 3.24 | 11.2% | 3.02 | −5.8% | +4.9% |
| IVW-3fam (B3/M10/CT) | +32.4% | 15.0% | 1.95 | 3.33 | 8.9% | 3.66 | −5.1% | +4.0% |
| **50/50 B3+M10** | **+43.6%** | **16.4%** | **2.29** | **4.31** | **9.2%** | **4.73** | **−5.0%** | **−0.7%** |
| 45/45/10 B3/M10/CT | +40.7% | 15.8% | 2.24 | 4.30 | 9.2% | 4.44 | −4.7% | +1.0% |
| `[OVERFIT]` full-sample max-Sharpe (45% B3 + 50% M10 + 5% A2) | +48.1% | 17.8% | 2.29 | 4.30 | 10.2% | 4.71 | −5.5% | +0.7% |

**The naive 50/50 split of the two uncorrelated books equals the overfit ceiling** (Sharpe 2.29 both) — the in-sample optimizer independently lands on ≈50/50 B3/M10 and adds nothing. When the zero-parameter prior coincides with the in-sample optimum, weight selection carries no overfitting risk worth arguing about.

Naive-diversification pathologies are also visible: EW-5 is *worse* than M10 on every risk-adjusted measure because equal-weighting five systems triple-counts the correlated PG family and overweights the weakest system (CT). **Diversify across families, not across config files.**

## 3. Allocation-level dev/holdout validation (split 2024-01-01)

| | dev Sharpe | holdout Sharpe | dev MDD | holdout MDD |
|---|---:|---:|---:|---:|
| 50/50 B3+M10 | 2.36 | **2.17** | 9.0% | **9.2%** |
| IVW B3+M10 (causal) | 2.50 | 2.10 | 9.2% | 8.6% |
| 45/45/10 B3/M10/CT | 2.36 | 2.03 | 8.5% | 9.2% |
| dev-optimized max-Sharpe (60/40 B3/M10, frozen) | 2.39 | 2.10 | 9.6% | 8.6% |
| IVW-3fam | 2.26 | 1.51 | 7.6% | 8.9% |
| M10 alone | 1.95 | 1.95 | 23.1% | 17.2% |
| PG-B3 alone | 1.73 | 1.14 | 18.2% | 11.1% |
| CT alone | 0.87 | 0.01 | 20.5% | 33.4% |

The blend's edge is stable out-of-sample at the allocation layer: Sharpe 2.36→2.17, MDD ~9% in both halves, and it beats the best single system in the holdout (2.17 vs M10's 1.95) with half the drawdown. Weights optimized on dev and frozen (60/40) do **not** beat the naive 50/50 out-of-sample — optimization adds nothing here, which is exactly what a non-overfit result looks like.

## 4. Answers

**Does % allocation across these strategies make sense?** Yes — but only across the *three families*, and the benefit comes almost entirely from one pairing: **PG-B3 + M10 (ρ≈0.10)**. Combining near-duplicates (A2+B3+C1) or adding a weak diluted book (CT at full weight) delivers naive-diversification, not real diversification.

**Is there a non-overfitting split that beats the best single system with lower drawdown?** Yes, and it is the most boring one possible: **50% PG-B3 / 50% M10, rebalanced monthly.**
- vs M10 alone: Sharpe 2.29 vs 1.95, Sortino 4.31 vs 3.18, Calmar 4.73 vs 2.80, **MDD 9.2% vs 23.1%**, worst month −5.0% vs −9.6% — at the cost of headline CAGR (+43.6%/yr vs +64.5%/yr, because the blend runs at 16% vol vs 27%).
- The CAGR gap is a *volatility* choice, not an efficiency loss: at matched risk (levering the blend ~1.7× on the futures account) the same Sharpe would imply materially higher return than M10 at M10's own drawdown depth — with financing costs and gap risk as the price of leverage.
- **Optional insurance sleeve:** 45/45/10 with CT costs ~0.05 Sharpe and buys regime convexity — CT was the only positive book in 2022's bear (+36.7%) and leads 2026 YTD (+20.3% while M10 is −3.7%). Justified by mechanism (only true short-trend book), not by its standalone stats.

**Practical rule, fully pre-registered:** hold the two uncorrelated families at equal risk (which ≈ equal capital here since both run ~16%/27% vol against their internal caps), rebalance monthly, and re-derive nothing from returns. If one must adapt, causal inverse-vol (IVW B3+M10) performed equivalently (holdout Sharpe 2.10) — there is no evidence that anything cleverer is needed.

## 5. Honest limits

1. **The allocation layer is non-overfit; the sleeves are not.** All five systems were *developed* on overlapping parts of this same history — PGPortfolio's own deflated-Sharpe audit puts its post-search edge probability near 0.5, and M10's holdout has been read five times across its milestones. A robust split of over-fit sleeves is still bounded by the sleeves' true forward edge. The blend inherits, and cannot launder, that uncertainty.
2. **B3's 18.2% MDD budget was calibrated full-sample by its authors** (the k=0.401 scaling), so its drawdown figure is partly by construction; the blend's ~9% MDD would scale accordingly if B3's live drawdown exceeds budget.
3. **Average correlation ≠ crash correlation.** ρ≈0.10 is a full-period average; in a single liquidation cascade all crypto books can gap together. The blend's worst joint month (−5.0%) spans May-2021, LUNA, FTX and the 2026 drawdown, which is encouraging but not a guarantee.
4. Meta-level rebalancing costs were ignored (monthly transfers between books, small vs the sleeves' internal turnover) and running both stacks requires operating two live infrastructures and split margin.

## Files

`data/allocation_common.csv`, `data/allocation_devhold.csv`, `data/allocation_daily.csv`, `allocation_blend.png`, `scripts/allocation_study.py`, `scripts/make_alloc_chart.py`.
