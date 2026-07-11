# Universe-Width Study — A2 / C1 / B3 at 10 vs 20 vs 30 coins

Systems run exactly as committed with only `input.coin_number` changed
(14 new configs under experiments/: a2_n{20,30}, c1_n{20,30},
slv{30m,1h,2h,4h,8h}_n{20,30}); B3 rebuilt per its design at each width
(META-GT + R20c overlays + k re-solved to the 18.2% MDD budget:
k = 0.401 / 0.349 / 0.267 for N = 10 / 20 / 30). Window 2020-02..2026-05.

## Full-period (common window)

| | CAGR | Sharpe | MDD | Calmar | worst yr |
|---|---:|---:|---:|---:|---:|
| A2 n10 | +102.8% | 1.36 | 57.4% | 1.79 | +0.0% |
| A2 n20 | +129.6% | 1.35 | 75.0% | 1.73 | −41.7% |
| A2 n30 | +121.8% | 1.26 | 82.9% | 1.47 | −47.7% |
| C1 n10 | +87.2% | 1.24 | 65.3% | 1.33 | +10.2% |
| C1 n20 | +95.3% | 1.16 | 82.6% | 1.15 | −41.7% |
| C1 n30 | +105.9% | 1.18 | 82.6% | 1.28 | −29.3% |
| B3 n10 | +25.3% | 1.48 | 18.2% | 1.39 | +4.9% |
| B3 n20 | +34.2% | **1.58** | 18.2% | 1.88 | +3.1% |
| B3 n30 | +26.2% | 1.32 | 18.2% | 1.44 | −2.0% |

Mechanics: wider universes admit more concurrent breakout signals, so the
per-coin 0.35 cap binds less and realized gross/vol RISES (A2 gross 0.70x
→ 1.07x → 1.29x; vol 68% → 91% → 104%): at the raw-stack level width is
mostly a leverage increase, and worst-year flips deeply negative. Through
B3's fixed 18.2% budget, n=20 looks better on every risk-adjusted metric.

## The honest split (dev < 2024-01 / holdout ≥ 2024-01)

| B3 | dev CAGR | dev Sharpe | holdout CAGR | holdout Sharpe |
|---|---:|---:|---:|---:|
| n10 | +29.6% | 1.74 | +19.1% | **1.12** |
| n20 | +44.5% | 1.97 | +19.7% | 0.99 |
| n30 | +34.7% | 1.61 | +14.0% | 0.83 |

**n=20's entire advantage is dev-period; out-of-sample it is WORSE than
n=10, and n=30 is worst.** This replicates the repo's own round-4 verdict
("non-monotonic universe-width effect ... no OOS improvement, baseline
retained"). Selecting n=20 because the full-period table looks best would
be one more in-sample parameter choice.

## Additional width-specific caveats

* Gate-blind members grow with width: coins with NO perp funding history
  pass the funding gates with rate 0 (6 ever-members at n10 → 11 at n20 →
  14 at n30, incl. SHIB/PEPE-era and 2019-20 leveraged tokens).
* BULLUSDT/BEARUSDT (3x leveraged tokens) enter the n≥20 universes — the
  committed eligibility filter does not catch them; they never ranked
  top-10 so the gap was invisible before.
* B3's k is re-solved full-sample per width (by design), same caveat as n10.

**Verdict: keep N=10.** The width parameter was already explored by the
original authors and the out-of-sample evidence still favors the committed
configuration; n=20's full-period appeal does not survive the split.
