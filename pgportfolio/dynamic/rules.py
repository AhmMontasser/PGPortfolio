"""Rule-based (non-learned) signed-weight agents for the margin backtester.

These implement the classic trend/momentum families as drop-in replacements
for the EIIE network (same ``decide_by_history(history, last_w) -> signed
weights`` interface, cash-free).  They need no training, so a full
walk-forward backtest runs in minutes - which is what makes a broad,
20-variant search feasible without touching the holdout window.

All lookbacks are specified in *days* and converted to periods using the
trading period, so the same config works at any bar size.  Position sizing
is inverse-volatility by default: each coin's signal is scaled by
1/sigma_i and the vector is normalized to a target gross exposure (the
backtester's leverage cap, vol targeting, drawdown budget, deadband and
stop-losses still apply on top).
"""

from __future__ import absolute_import, division, print_function

import numpy as np

DAY = 86400.0


def _periods(days, period_seconds):
    return max(2, int(round(days * DAY / period_seconds)))


class RuleAgentBase(object):
    def __init__(self, period_seconds, vol_days=30, target_gross=1.0):
        self._period = period_seconds
        self._vol_periods = _periods(vol_days, period_seconds)
        self._target_gross = target_gross

    def begin_month(self, coins):
        self._coins = list(coins)

    # -- helpers ---------------------------------------------------------

    def _inverse_vol_size(self, signal, close):
        """Scale +-1 signals by inverse volatility, normalize to gross."""
        window = close[:, -self._vol_periods:]
        rets = window[:, 1:] / window[:, :-1] - 1.0
        vol = rets.std(axis=1)
        vol = np.where(vol > 1e-8, vol, np.nan)
        weights = signal / vol
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        gross = np.abs(weights).sum()
        if gross > 1e-12:
            weights *= self._target_gross / gross
        return weights

    @staticmethod
    def _ma(close, periods):
        return close[:, -periods:].mean(axis=1)

    def decide_by_history(self, history, last_w):
        raise NotImplementedError()


class TSMomentum(RuleAgentBase):
    """Time-series momentum: long above trend, short below.

    ``mode="price_ma"``: sign(close - MA(slow_days)).
    ``mode="cross"``:    sign(MA(fast_days) - MA(slow_days)).
    ``trend_filter``: minimum |close - MA| / ATR to hold a position at all -
    below it the coin is left untraded (chop filter / no-trade regime).
    ``long_only``: shorts are replaced by cash (no position).
    """

    def __init__(self, period_seconds, slow_days=50, fast_days=10,
                 mode="price_ma", long_only=False, trend_filter=0.0,
                 vol_days=30, target_gross=1.0):
        RuleAgentBase.__init__(self, period_seconds, vol_days, target_gross)
        self._slow = _periods(slow_days, period_seconds)
        self._fast = _periods(fast_days, period_seconds)
        self._mode = mode
        self._long_only = long_only
        self._trend_filter = trend_filter

    def decide_by_history(self, history, last_w):
        close = history[0]
        slow_ma = self._ma(close, self._slow)
        if self._mode == "cross":
            reference = self._ma(close, self._fast)
        else:
            reference = close[:, -1]
        signal = np.sign(reference - slow_ma)
        if self._long_only:
            signal = np.maximum(signal, 0.0)
        if self._trend_filter > 0:
            high, low = history[1], history[2]
            atr = np.mean(high[:, -self._fast:] - low[:, -self._fast:], axis=1)
            strength = np.abs(reference - slow_ma) / np.maximum(atr, 1e-12)
            signal = np.where(strength >= self._trend_filter, signal, 0.0)
        return self._inverse_vol_size(signal, close)


class Donchian(RuleAgentBase):
    """Channel breakout (turtle-style) with hysteresis via ``last_w``.

    Enter long when the close breaks the ``entry_days`` high, exit when it
    breaks the ``exit_days`` low (shorts symmetric, disabled by
    ``long_only``).  Between entry and exit the previous position's sign is
    kept, so the agent holds through the trend instead of churning.
    """

    def __init__(self, period_seconds, entry_days=20, exit_days=10,
                 long_only=False, vol_days=30, target_gross=1.0):
        RuleAgentBase.__init__(self, period_seconds, vol_days, target_gross)
        self._entry = _periods(entry_days, period_seconds)
        self._exit = _periods(exit_days, period_seconds)
        self._long_only = long_only

    def decide_by_history(self, history, last_w):
        close = history[0]
        last = close[:, -1]
        entry_high = close[:, -self._entry - 1:-1].max(axis=1)
        entry_low = close[:, -self._entry - 1:-1].min(axis=1)
        exit_high = close[:, -self._exit - 1:-1].max(axis=1)
        exit_low = close[:, -self._exit - 1:-1].min(axis=1)

        held = np.sign(last_w)
        signal = np.zeros_like(last)
        signal[last >= entry_high] = 1.0
        if not self._long_only:
            signal[last <= entry_low] = -1.0
        keep_long = (held > 0) & (signal == 0) & (last > exit_low)
        keep_short = (held < 0) & (signal == 0) & (last < exit_high)
        signal[keep_long] = 1.0
        signal[keep_short] = -1.0
        return self._inverse_vol_size(signal, close)


class XSMomentum(RuleAgentBase):
    """Cross-sectional momentum: long the strongest, short the weakest.

    Ranks the coins by ``lookback_days`` total return; goes long the top
    ``k`` and (unless ``long_only``) short the bottom ``k``.  With
    ``btc_gate_days`` set, longs are only allowed while BTC is above its
    own gate MA and shorts only while below - a simple market-regime gate
    ("avoid trading against the tide").
    """

    def __init__(self, period_seconds, lookback_days=30, k=3,
                 long_only=False, btc_gate_days=0, vol_days=30,
                 target_gross=1.0):
        RuleAgentBase.__init__(self, period_seconds, vol_days, target_gross)
        self._lookback = _periods(lookback_days, period_seconds)
        self._k = k
        self._long_only = long_only
        self._btc_gate = (_periods(btc_gate_days, period_seconds)
                          if btc_gate_days else 0)
        self._btc_index = None

    def begin_month(self, coins):
        RuleAgentBase.begin_month(self, coins)
        self._btc_index = None
        for i, coin in enumerate(coins):
            if coin.startswith("BTC"):
                self._btc_index = i
                break

    def decide_by_history(self, history, last_w):
        close = history[0]
        returns = close[:, -1] / close[:, -self._lookback] - 1.0
        order = np.argsort(returns)
        signal = np.zeros_like(returns)
        signal[order[-self._k:]] = 1.0
        if not self._long_only:
            signal[order[:self._k]] = -1.0
        if self._btc_gate and self._btc_index is not None:
            btc = close[self._btc_index]
            btc_up = btc[-1] >= btc[-self._btc_gate:].mean()
            if btc_up:
                signal = np.maximum(signal, 0.0)
            else:
                signal = np.minimum(signal, 0.0)
        return self._inverse_vol_size(signal, close)


class RegimeTrend(RuleAgentBase):
    """Per-coin trend-regime detector with asymmetric long/short books.

    Every coin is classified each period into one of three regimes from its
    own moving averages:

    * **up**:      close > MA(regime_slow) and MA(regime_fast) > MA(regime_slow)
    * **down**:    close < MA(regime_slow) and MA(regime_fast) < MA(regime_slow)
    * **neutral**: anything else -> the coin is not traded at all

    (with ``slope_days`` set, the slow MA must additionally be rising /
    falling over that window - a stricter regime definition).

    Inside its regime a coin trades a Donchian channel on that side only:
    longs enter on the ``long_entry_days`` high and exit on the
    ``long_exit_days`` low; shorts enter on the ``short_entry_days`` low
    and exit on the ``short_exit_days`` high, scaled by ``short_scale``
    (bear-market rallies are violent, so half-sized, quick-exit shorts are
    a common choice).  ``market_gate_days``: shorts are additionally only
    allowed while BTC itself is below its own gate MA - bear years are
    market-wide, bull-market dips are not.
    """

    def __init__(self, period_seconds, regime_fast_days=20, regime_slow_days=50,
                 slope_days=0, long_entry_days=20, long_exit_days=10,
                 short_entry_days=20, short_exit_days=10, short_scale=1.0,
                 market_gate_days=0, vol_days=30, target_gross=1.0):
        RuleAgentBase.__init__(self, period_seconds, vol_days, target_gross)
        self._regime_fast = _periods(regime_fast_days, period_seconds)
        self._regime_slow = _periods(regime_slow_days, period_seconds)
        self._slope = _periods(slope_days, period_seconds) if slope_days else 0
        self._long_entry = _periods(long_entry_days, period_seconds)
        self._long_exit = _periods(long_exit_days, period_seconds)
        self._short_entry = _periods(short_entry_days, period_seconds)
        self._short_exit = _periods(short_exit_days, period_seconds)
        self._short_scale = short_scale
        self._market_gate = (_periods(market_gate_days, period_seconds)
                             if market_gate_days else 0)
        self._btc_index = None

    def begin_month(self, coins):
        RuleAgentBase.begin_month(self, coins)
        self._btc_index = None
        for i, coin in enumerate(coins):
            if coin.startswith("BTC"):
                self._btc_index = i
                break

    def _regimes(self, close):
        last = close[:, -1]
        slow = self._ma(close, self._regime_slow)
        fast = self._ma(close, self._regime_fast)
        up = (last > slow) & (fast > slow)
        down = (last < slow) & (fast < slow)
        if self._slope:
            past_slow = close[:, :-self._slope]
            old = past_slow[:, -self._regime_slow:].mean(axis=1)
            up &= slow > old
            down &= slow < old
        return up, down

    def decide_by_history(self, history, last_w):
        close = history[0]
        last = close[:, -1]
        up, down = self._regimes(close)
        held = np.sign(last_w)
        signal = np.zeros_like(last)

        long_break = last >= close[:, -self._long_entry - 1:-1].max(axis=1)
        long_hold = (held > 0) & (last > close[:, -self._long_exit - 1:-1].min(axis=1))
        signal[up & (long_break | long_hold)] = 1.0

        short_break = last <= close[:, -self._short_entry - 1:-1].min(axis=1)
        short_hold = (held < 0) & (last < close[:, -self._short_exit - 1:-1].max(axis=1))
        shorts = down & (short_break | short_hold)
        if self._market_gate and self._btc_index is not None:
            btc = close[self._btc_index]
            btc_bear = btc[-1] < btc[-self._market_gate:].mean()
            if not btc_bear:
                shorts &= False
        signal[shorts] = -self._short_scale
        return self._inverse_vol_size(signal, close)


class Ensemble(object):
    """Averages the signed weights of several rule agents (diversification
    across signal families)."""

    def __init__(self, period_seconds, members, mix=None):
        self._members = [build_rule_agent(member, period_seconds)
                         for member in members]
        self._mix = mix or [1.0 / len(self._members)] * len(self._members)

    def begin_month(self, coins):
        for member in self._members:
            if hasattr(member, "begin_month"):
                member.begin_month(coins)

    def decide_by_history(self, history, last_w):
        combined = np.zeros(history.shape[1])
        for weight, member in zip(self._mix, self._members):
            combined = combined + weight * np.asarray(
                member.decide_by_history(history, last_w))
        return combined


def build_rule_agent(rule_config, period_seconds):
    kind = rule_config["type"]
    params = {key: value for key, value in rule_config.items()
              if key not in ("type", "name")}
    if kind == "tsmom":
        return TSMomentum(period_seconds, **params)
    if kind == "donchian":
        return Donchian(period_seconds, **params)
    if kind == "xsmom":
        return XSMomentum(period_seconds, **params)
    if kind == "regime":
        return RegimeTrend(period_seconds, **params)
    if kind == "ensemble":
        return Ensemble(period_seconds, **params)
    raise ValueError("unknown rule agent type %r" % kind)
