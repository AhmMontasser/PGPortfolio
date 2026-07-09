# Round 10 pre-registration — structure & level information

Committed before any result. Base: `r6_vol150` 4h system (research at 4h;
synthesis re-run across frequencies only for adopted items). Adoption
rule unchanged: ≥2 of 3 dev sub-folds (2020-01..2021-06,
2021-07..2022-09, 2022-10..2023-12) better than base AND aggregate dev
not degraded. One holdout look at the synthesis. All parameters fixed
below.

Structure gates act on **new entries only** (holds untouched), like the
round-7/9 entry filters. "dATR" = 14-day average true range at daily
scale (mean 4h high-low × 2.45).

1. **Swing S/R veto**: ZigZag pivots (10% reversals, trailing 180d).
   Block longs when the nearest overhead pivot high is < 1 dATR away;
   block shorts when nearest pivot low below is < 1 dATR away.
2. **Level-breakout size bonus**: entries that clear the nearest pivot
   (long above pivot high / short below pivot low by > 0.25 dATR) get
   1.25× weight (others 1.0×).
3. **Round-number veto**: block longs within 0.5 dATR *below* the
   nearest round level (1/2/5 × 10^n grid); shorts symmetric above.
4. **Volume-profile veto**: 90d volume-at-price, 30 log-price bins.
   Block longs whose path to +2 dATR crosses a bin with volume > 1.5×
   the mean bin (overhead supply); shorts symmetric.
5. **Fibonacci retrace veto**: over the trailing 180d swing (min→max),
   block new longs when price has retraced > 61.8% of the up-swing.
6. **Trend-line quality filter**: 60d log-price OLS; entries require
   |slope t-stat| > 2 in the trade direction.
7. **Wave-count caution (Elliott proxy)**: ZigZag legs (10%); block new
   longs after ≥5 consecutive completed up-legs without a 20% correction;
   shorts symmetric.
8. **52-week-high anchor**: longs within 5% of the 365d high get 1.25×;
   longs more than 30% below it are blocked (momentum-anchor evidence).
9. **Wick rejection veto**: block longs when the last completed bar has
   an upper wick > 60% of its range AND closed within 1 dATR of pivot
   resistance; shorts symmetric.
10. **Leader veto (leading information)**: block alt-coin entries when
    BTC's last bar moved > 1 dATR against the trade direction.
11. **Anchored VWAP gate**: month-anchored VWAP (volume feature); longs
    only above it, shorts only below.
12. **Prior-day-extremes control**: entries only on breaks of the prior
    day's high/low (redundancy control vs Donchian; expected ~neutral).
13. **Confluence sizing**: weight × (1 + 0.25 × [#2 bonus] + 0.25 ×
    [#8 proximity]) capped at 1.5×, using only adopted components.
14. **Synthesis**: adopted items combined; sub-fold check; single
    holdout look; risk-budget menu.
