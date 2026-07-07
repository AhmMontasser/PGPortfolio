# Round 9 pre-registration (committed before any result is computed)

Baseline: round-8 final system (META-GT + R20c overlays), and its
constituents. Adoption rule: a research is adopted only if it improves
risk-adjusted performance in **at least 2 of 3 dev sub-folds**
(2020-01..2021-06, 2021-07..2022-09, 2022-10..2023-12) and does not
degrade the aggregate dev window. The holdout (2024-01..2026-06) is
evaluated once, for the final synthesis only. All parameters below are
fixed now; no post-hoc parameter changes will be reported as wins.

## A. Bias-audit block
1. Execution-delay sensitivity: execute decisions one bar late (4h);
   report degradation of the base system. (Implementation-bias bound.)
2. Sub-fold revalidation of the round-8 overlays (downside-vol, corr
   gate, vol-of-vol) under the 2-of-3 rule.
3. Bootstrap (stationary block, 1000 draws, 20-day blocks) 90% CI on the
   final system's holdout Sharpe.
4. Deflated Sharpe ratio of the final system given ~60 configurations
   tried across the project.

## B. New information / signals
5. Perp-spot basis (futures 4h klines vs spot): gate longs when basis
   > +1% (euphoria), shorts when < -1%.
6. Basis momentum book: weight ∝ -(basis - 5d mean basis)/0.5%, clip 3.
7. Funding *change* gate: block entries when |funding - its 3d mean|
   > 0.05%/8h (positioning shock).
8. Time-of-day: entries only allowed on bars ending 00/08/16 UTC
   (funding settlements); vs base.
9. Weekend gate: no new entries Sat/Sun (liquidity).
10. Cross-sectional dispersion scaling: gross x1.25 when 30d dispersion
    of universe daily returns is in its top expanding tercile, x0.75 in
    bottom tercile (computed on trailing data only).
11. New-high breadth: fraction of universe making 20d highs; long book
    scaled by breadth (0 highs -> 0.5x, all highs -> 1.25x).
12. Relative-strength regime: per-coin regime computed on coin/BTC price
    instead of coin/USDT (pure alpha trend), long book only.

## C. Risk / execution
13. Asymmetric stops: long 8% / short 5% trailing.
14. Staleness exit: close any position older than 30 days that is below
    +10% unrealized.
15. Partial profit: halve a position at +40% unrealized (rest trails).
16. Coin age filter: min_history_days 180 (skip fresh listings).
17. Liquidity-weighted sizing: position size multiplied by
    rank-normalized selection score (0.75..1.25).
18. Per-coin three-strikes: after 2 stop-outs in a coin within 30 days,
    skip that coin for 30 days.
19. Universe-transition smoothing: phase in new-coin positions over 3
    days at month boundaries.
20. Fee-tier scenario: 0.075% fee (BNB discount) on the base; accounting
    research, not selectable.

## D. Portfolio / meta
21. Member-specific drawdown brake (brake only the stream in drawdown,
    not the whole book).
22. Frequency-tilt by dispersion regime (fast members in high-vol/high-
    dispersion states, slow in quiet states; thresholds as in #10).
23. Overlay interaction map: R20c overlays applied one-at-a-time per
    sub-fold to attribute the stack's gain.
24. Meta of metas: 50/50 R20c-stack + round-5 gated 4h (system-level
    diversification with a non-frequency member).
25. Cash-yield accounting: 4% APR on idle USDT (money-market), report
    effect (accounting research).

## E. Parameter-stability maps (flatness = robustness evidence)
26. Donchian entry 10..40d x exit 5..20d heatmap on dev sub-folds
    (existence of a flat plateau around 20/10).
27. Regime length 50..200d heatmap for the short book.
28. Overlay threshold maps (corr gate 0.75..0.95, vol-of-vol 1.25..2.0).

## F. Synthesis
29. Combine all adopted researches; sub-fold check; single holdout look.
30. Final risk-budget menu (15/18/25% MDD) and per-year consistency
    table of the synthesized system.
