"""Dynamic (monthly) universe selection.

The original PGPortfolio selects one static set of coins - the top
``coin_number`` by 30-day volume just before the test range - and keeps it
for the whole experiment.  Here the universe is re-selected at the start of
every calendar month, using only information available strictly *before* the
month starts, so the backtest stays free of look-ahead bias:

* rank by 30-day quote volume (how much money changes hands), and
* rank by liquidity, measured by the average number of trades per day and
  the Amihud illiquidity ratio (average |daily return| per unit of quote
  volume traded - lower means more liquid),

then take the top N of the combined rank.  Stablecoins, fiat pairs, wrapped
tokens and Binance leveraged tokens are excluded, as are coins that have
been listed for less than ``min_history_days`` or that did not trade on
almost every day of the ranking window.
"""

from __future__ import absolute_import, division, print_function

import logging
import re

import numpy as np
import pandas as pd

DAY = 86400

# fiat, fiat-pegged and algorithmic stablecoins, and gold tokens: holding
# them is (or is meant to be) equivalent to holding the USDT cash asset,
# so they carry volume but no investable signal for the agent.
STABLE_OR_FIAT_BASES = {
    "USDC", "BUSD", "TUSD", "USDP", "PAX", "DAI", "SUSD", "UST", "USTC",
    "FDUSD", "PYUSD", "USDS", "USDSB", "GUSD", "HUSD", "USDE", "USD1",
    "XUSD", "AEUR", "EURI", "EUR", "GBP", "AUD", "BRL", "TRY", "RUB",
    "UAH", "NGN", "BIDR", "IDRT", "BVND", "VAI", "BKRW", "PAXG", "BFUSD",
}

# wrapped / staked twins of assets that are already tradeable directly
WRAPPED_BASES = {"WBTC", "WETH", "WBETH", "BETH", "BTCB", "BNSOL", "WSOL"}

# prefixes Binance used for leveraged tokens (BTCUP, ETHDOWN, EOSBULL, ...)
LEVERAGED_PREFIXES = {
    "BTC", "ETH", "BNB", "ADA", "LINK", "LTC", "EOS", "ETC", "TRX", "XRP",
    "DOT", "UNI", "FIL", "SUSHI", "YFI", "AAVE", "SXP", "1INCH", "XLM",
    "XTZ", "BCH", "DOGE",
}
_LEVERAGED_RE = re.compile(r"^(.+?)(UP|DOWN|BULL|BEAR)$")


def investable(symbol, quote="USDT"):
    """True if the pair is a plain crypto/quote market worth considering."""
    if not symbol.endswith(quote):
        return False
    base = symbol[:-len(quote)]
    if not base or base in STABLE_OR_FIAT_BASES or base in WRAPPED_BASES:
        return False
    leveraged = _LEVERAGED_RE.match(base)
    if leveraged and leveraged.group(1) in LEVERAGED_PREFIXES:
        return False
    return True


class UniverseSelector(object):
    def __init__(self, coin_number=10, volume_average_days=30,
                 min_history_days=45, min_active_days=28, quote="USDT",
                 exclude_symbols=()):
        self._exclude = set(exclude_symbols or ())
        self._coin_number = coin_number
        self._window_days = volume_average_days
        self._min_history_days = min_history_days
        self._min_active_days = min_active_days
        self._quote = quote

    def select(self, daily_frames, decision_ts):
        """Pick the universe for the month starting at ``decision_ts``.

        :param daily_frames: dict symbol -> daily candle DataFrame (indexed
            by open-time seconds), already token-swap sanitized.
        :param decision_ts: unix time of the month start; only candles with
            open-time strictly before it are used.
        :return: (list of selected symbols, diagnostics DataFrame)
        """
        window_start = decision_ts - self._window_days * DAY
        records = []
        for symbol, frame in daily_frames.items():
            if symbol in self._exclude or not investable(symbol, self._quote):
                continue
            history = frame[frame.index < decision_ts]
            if len(history) < self._min_history_days:
                continue
            window = history[history.index >= window_start]
            active = window[window["volume"] > 0]
            if len(active) < self._min_active_days:
                continue
            # still trading: candles present up to the decision day
            if history.index[-1] < decision_ts - 2 * DAY:
                continue
            quote_volume = window["quote_volume"].sum()
            if quote_volume <= 0:
                continue
            trades = window["trades"].mean()
            returns = window["close"].pct_change().abs().iloc[1:]
            daily_qv = window["quote_volume"].iloc[1:]
            amihud = float(np.mean(returns.to_numpy() /
                                   np.maximum(daily_qv.to_numpy(), 1e-12)))
            records.append((symbol, quote_volume, trades, amihud))
        if not records:
            raise ValueError("no eligible coins at %d" % decision_ts)
        table = pd.DataFrame(records, columns=["symbol", "quote_volume",
                                               "trades", "amihud"])
        table = table.set_index("symbol")
        # combined rank: volume counts half, liquidity (trade count and
        # Amihud illiquidity) the other half
        rank_volume = table["quote_volume"].rank(ascending=False)
        rank_trades = table["trades"].rank(ascending=False)
        rank_amihud = table["amihud"].rank(ascending=True)
        table["score"] = 0.5 * rank_volume + 0.25 * rank_trades + 0.25 * rank_amihud
        table = table.sort_values("score")
        selected = list(table.index[:self._coin_number])
        logging.info("universe @%s: %s",
                     pd.Timestamp(decision_ts, unit="s").date(), selected)
        return selected, table

    def monthly_schedule(self, start, end):
        """Month starts (unix seconds) from start to end inclusive."""
        months = pd.date_range(pd.Timestamp(start), pd.Timestamp(end),
                               freq="MS", tz="UTC")
        return [int(t.timestamp()) for t in months]
