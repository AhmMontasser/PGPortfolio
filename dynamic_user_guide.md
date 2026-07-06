# Dynamic-Universe Backtesting

This extension replaces PGPortfolio's *static* coin selection (one top-N
snapshot before the experiment) with a **dynamic universe**: at the start of
every calendar month the top 10 coins by **volume and liquidity** are
re-selected, and the trading system is backtested walk-forward from 2020
onwards.

It lives in `pgportfolio/dynamic/` and is fully self-contained: it does not
use the legacy Poloniex/`pandas.Panel`/TF1 stack (which no longer runs on
modern Python), but it re-implements the same trading system:

* the **EIIE CNN** policy network of the paper (same topology as
  `pgportfolio/net_config.json`: ConvLayer → EIIE_Dense → EIIE_Output_WithW
  with a trainable cash bias), ported to TensorFlow 2;
* the same training scheme: portfolio-vector memory (PVM), geometrically
  sampled batches of consecutive periods, commission-aware
  `loss_function6 = -E[log(µ_t · ⟨w_t, y_t⟩)]`;
* the same iterative transaction-cost model
  (`calculate_pv_after_commission`);
* the classic on-line benchmarks shipped with the repo (OLMAR, PAMR, RMR)
  plus UBAH, UCRP and BTC buy-and-hold.

## Data

Historical candles come from Binance's public archive
(https://data.binance.vision), cached in `./database/dynamic_data.db`:

* **daily klines for every USDT pair that ever traded on Binance**
  (including delisted ones — no survivorship bias) drive universe selection;
* **30-minute klines** (the paper's trading period) for every coin that is
  ever selected drive training and trading.
* Prices are quoted in **USDT**, which is also the cash asset of the
  portfolio vector (the original used BTC on Poloniex).

Symbols that were re-used for a different token after a redenomination
(e.g. `LUNAUSDT`, which switched from Terra Classic to the new LUNA chain in
May 2022) are truncated at the discontinuity so a token swap is never
mistaken for a 1000x return.

## Universe selection

At every month start `t0`, using only data from before `t0`:

* eligibility: listed ≥ 45 days, traded on ≥ 28 of the last 30 days, not a
  stablecoin / fiat pair / wrapped token / leveraged token;
* score = `0.5 · rank(30-day quote volume)` +
  `0.25 · rank(avg trades per day)` +
  `0.25 · rank(Amihud illiquidity)` (lower Amihud = more liquid);
* the 10 best-scoring coins form the month's universe.

## Walk-forward protocol

1. The network is trained once on 2019 data (30-minute periods) of the first
   month's universe.
2. For each month since 2020-01: re-select the universe, fine-tune the
   network (`rolling_steps` batches) on the trailing
   `rolling_lookback_days` of the *new* universe ending at the month start,
   then trade the month out-of-sample, period by period.
3. At month boundaries, positions in coins that dropped out of the universe
   are sold and newcomers bought; that rebalance is charged with the same
   commission formula, evaluated over the union of both coin lists.

The EIIE topology shares all convolution weights across assets, so the same
trained network keeps working when the universe membership (or even its
size) changes — this is what makes a dynamic universe compatible with the
paper's design.

## Running

```bash
pip install -r requirements-dynamic.txt

python dynamic_main.py --mode download   # fetch/refresh market data (~1-2 GB)
python dynamic_main.py --mode universe   # print the monthly top-10 lists
python dynamic_main.py --mode backtest   # full walk-forward backtest
```

Configuration is in `dynamic_config.json`. Results (summary table, equity
curves CSV, monthly universes, plot) are written to `./dynamic_results/`.
The trading commission defaults to the repo's historical 0.25% per side;
set `trading.trading_consumption` to `0.001` for Binance's current spot fee.
