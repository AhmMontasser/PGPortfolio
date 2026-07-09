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
                 market_gate_days=0, vol_days=30, target_gross=1.0,
                 rs_vs_btc=False):
        RuleAgentBase.__init__(self, period_seconds, vol_days, target_gross)
        self._rs_vs_btc = rs_vs_btc
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
        if self._rs_vs_btc and self._btc_index is not None:
            # regime on the coin/BTC ratio: pure relative strength, immune
            # to the market-wide tide (#12)
            close = close / np.maximum(close[self._btc_index], 1e-12)
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


class FundingCarry(RuleAgentBase):
    """Perp funding-rate carry: short crowded longs, long crowded shorts.

    Positive funding means longs pay shorts (crowded long positioning);
    persistently extreme funding has historically mean-reverted in the
    underlying. The signal is the last known 8-hour funding rate (from the
    panel's ``funding`` feature channel), scaled by ``threshold`` and
    clipped: coins with |funding| below the threshold are not traded.
    This is *information orthogonal to price trend* - the point of the
    book is diversification against the trend books.
    """

    def __init__(self, period_seconds, funding_channel=3, threshold=0.0003,
                 clip=3.0, vol_days=30, target_gross=1.0):
        RuleAgentBase.__init__(self, period_seconds, vol_days, target_gross)
        self._channel = funding_channel
        self._threshold = threshold
        self._clip = clip

    def decide_by_history(self, history, last_w):
        funding = history[self._channel, :, -1]
        signal = -np.clip(funding / self._threshold, -self._clip, self._clip)
        signal[np.abs(funding) < self._threshold] = 0.0
        return self._inverse_vol_size(signal, history[0])


class FundingGate(object):
    """Suppress positions that fight extreme positioning.

    Wraps any rule agent: longs are blocked while funding is above
    ``long_max`` (entering a crowded long), shorts while funding is below
    ``short_min`` (entering a crowded short, squeeze fuel).
    """

    def __init__(self, period_seconds, member, funding_channel=3,
                 long_max=0.00075, short_min=-0.00075):
        self._member = build_rule_agent(member, period_seconds)
        self._channel = funding_channel
        self._long_max = long_max
        self._short_min = short_min

    def begin_month(self, coins):
        if hasattr(self._member, "begin_month"):
            self._member.begin_month(coins)

    def decide_by_history(self, history, last_w):
        w = np.asarray(self._member.decide_by_history(history, last_w))
        funding = history[self._channel, :, -1]
        w = np.where((w > 0) & (funding > self._long_max), 0.0, w)
        w = np.where((w < 0) & (funding < self._short_min), 0.0, w)
        return w


class EntryFilterGate(object):
    """Blocks *new entries* (not held positions) that fail a quality check.

    Wraps any rule agent. A position change counts as an entry when the
    target sign differs from the currently held sign; entries failing the
    active checks are cancelled (weight reverts to the held value), while
    existing positions are never force-closed by a filter flicker - exits
    remain governed by the member's own logic and the backtester's stops.

    Checks (any subset):
    * ``mtf_confirm_days``: multi-timeframe confirmation - the trade
      direction must agree with the sign of the ``mtf_confirm_days``
      rate-of-change (a higher-timeframe trend screen);
    * ``volume_confirm``: the current bar's volume must exceed
      ``volume_confirm`` x the trailing 20-day average (breakouts without
      participation are suspect); needs a ``volume`` feature channel;
    * ``overextension_atr``: entry price must be within this many ATRs of
      the ``overextension_ma_days`` moving average (don't chase moves that
      already ran).
    """

    def __init__(self, period_seconds, member, mtf_confirm_days=0,
                 volume_confirm=0.0, volume_channel=4,
                 overextension_atr=0.0, overextension_ma_days=20,
                 atr_days=14, flow_confirm_days=0.0, flow_channel=4,
                 funding_shock=0.0, funding_channel=3,
                 basis_gate=0.0, basis_channel=4):
        self._member = build_rule_agent(member, period_seconds)
        self._period = period_seconds
        self._funding_shock = funding_shock
        self._funding_channel = funding_channel
        self._basis_gate = basis_gate
        self._basis_channel = basis_channel
        self._flow = _periods(flow_confirm_days, period_seconds) \
            if flow_confirm_days else 0
        self._flow_channel = flow_channel
        self._mtf = _periods(mtf_confirm_days, period_seconds) \
            if mtf_confirm_days else 0
        self._volume_confirm = volume_confirm
        self._volume_channel = volume_channel
        self._volume_avg = _periods(20, period_seconds)
        self._overext = overextension_atr
        self._overext_ma = _periods(overextension_ma_days, period_seconds)
        self._atr = _periods(atr_days, period_seconds)

    def begin_month(self, coins):
        if hasattr(self._member, "begin_month"):
            self._member.begin_month(coins)

    def decide_by_history(self, history, last_w):
        w = np.asarray(self._member.decide_by_history(history, last_w),
                       dtype=np.float64)
        close = history[0]
        entering = (np.sign(w) != np.sign(last_w)) & (np.sign(w) != 0)
        allowed = np.ones_like(w, dtype=bool)
        if self._mtf:
            roc = np.sign(close[:, -1] - close[:, -self._mtf])
            allowed &= np.sign(w) == roc
        if self._flow:
            # order-flow confirmation: recent taker imbalance must agree
            # with the trade direction (buyers in control for longs)
            imbalance = history[self._flow_channel, :, -self._flow:].mean(axis=1)
            allowed &= np.sign(w) == np.sign(imbalance)
        if self._funding_shock:
            # positioning shock: funding far from its 3-day mean = unstable
            funding = history[self._funding_channel]
            shock = np.abs(funding[:, -1] - funding[:, -18:].mean(axis=1))
            allowed &= shock <= self._funding_shock
        if self._basis_gate:
            # euphoria/panic gate: perp premium blocks longs, discount
            # blocks shorts
            basis = history[self._basis_channel, :, -1]
            allowed &= ~((w > 0) & (basis > self._basis_gate)) & \
                       ~((w < 0) & (basis < -self._basis_gate))
        if self._volume_confirm:
            volume = history[self._volume_channel]
            average = volume[:, -self._volume_avg:-1].mean(axis=1)
            allowed &= volume[:, -1] >= self._volume_confirm * \
                np.maximum(average, 1e-12)
        if self._overext:
            high, low = history[1], history[2]
            atr = np.mean(high[:, -self._atr:] - low[:, -self._atr:], axis=1)
            ma = close[:, -self._overext_ma:].mean(axis=1)
            distance = np.abs(close[:, -1] - ma) / np.maximum(atr, 1e-12)
            allowed &= distance <= self._overext
        blocked = entering & ~allowed
        return np.where(blocked, last_w, w)


class PullbackTrend(RuleAgentBase):
    """Buy weakness inside an up-regime (and sell strength in a down one).

    Entry-timing complement to breakout books: in an up regime (100d MA
    stack) a long is opened when price closes *below* the fast MA (a dip),
    and held while the regime lasts; symmetric for shorts unless
    ``long_only``. Uses the same inverse-vol sizing as the other books.
    """

    def __init__(self, period_seconds, regime_fast_days=20,
                 regime_slow_days=100, dip_ma_days=10, long_only=False,
                 vol_days=30, target_gross=1.0):
        RuleAgentBase.__init__(self, period_seconds, vol_days, target_gross)
        self._fast = _periods(regime_fast_days, period_seconds)
        self._slow = _periods(regime_slow_days, period_seconds)
        self._dip = _periods(dip_ma_days, period_seconds)
        self._long_only = long_only

    def decide_by_history(self, history, last_w):
        close = history[0]
        last = close[:, -1]
        slow = self._ma(close, self._slow)
        fast = self._ma(close, self._fast)
        dip = self._ma(close, self._dip)
        up = (last > slow) & (fast > slow)
        down = (last < slow) & (fast < slow)
        held = np.sign(last_w)
        signal = np.zeros_like(last)
        signal[up & ((last < dip) | (held > 0))] = 1.0
        if not self._long_only:
            signal[down & ((last > dip) | (held < 0))] = -1.0
        return self._inverse_vol_size(signal, close)


class BreadthEnsemble(object):
    """Regime-breadth-adaptive allocation between a long and a short book.

    Instead of a fixed mix, capital is split by *market breadth*: the
    fraction of universe coins whose own 100-day regime is up funds the
    long book, the fraction in a down regime funds the short book, and the
    neutral remainder stays in cash.  In a broad bull (2021) this is ~all
    long book; deep in a bear (mid-2022) ~all short book; in mixed tape it
    automatically de-grosses - the per-coin regime signal reused at the
    portfolio level.
    """

    def __init__(self, period_seconds, long_book, short_book,
                 regime_fast_days=20, regime_slow_days=100):
        self._long = build_rule_agent(long_book, period_seconds)
        self._short = build_rule_agent(short_book, period_seconds)
        self._fast = _periods(regime_fast_days, period_seconds)
        self._slow = _periods(regime_slow_days, period_seconds)

    def begin_month(self, coins):
        for member in (self._long, self._short):
            if hasattr(member, "begin_month"):
                member.begin_month(coins)

    def decide_by_history(self, history, last_w):
        close = history[0]
        last = close[:, -1]
        slow = close[:, -self._slow:].mean(axis=1)
        fast = close[:, -self._fast:].mean(axis=1)
        breadth_up = float(np.mean((last > slow) & (fast > slow)))
        breadth_down = float(np.mean((last < slow) & (fast < slow)))
        w = breadth_up * np.asarray(self._long.decide_by_history(history, last_w))
        if breadth_down > 0:
            w = w + breadth_down * np.asarray(
                self._short.decide_by_history(history, last_w))
        return w


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
    if kind == "breadth_ensemble":
        return BreadthEnsemble(period_seconds, **params)
    if kind == "funding_carry":
        return FundingCarry(period_seconds, **params)
    if kind == "funding_gate":
        return FundingGate(period_seconds, **params)
    if kind == "structure_gate":
        return StructureGate(period_seconds, **params)
    if kind == "entry_filter":
        return EntryFilterGate(period_seconds, **params)
    if kind == "pullback":
        return PullbackTrend(period_seconds, **params)
    if kind == "ensemble":
        return Ensemble(period_seconds, **params)
    raise ValueError("unknown rule agent type %r" % kind)


class StructureGate(object):
    """Price-structure / level-based entry gates (round-10 researches).

    Wraps a rule agent; like EntryFilterGate it only affects *new entries*.
    Structural level sets (ZigZag swing pivots, volume-at-price profile)
    are recomputed on the first bar of each month and cached - levels are
    slow-moving by construction.  All thresholds are in daily-ATR (dATR)
    units, dATR = mean(4h high-low, 14d) * 2.45.

    Checks (enable via constructor params; all default off):
      swing_sr        - veto entries into the nearest opposing pivot
      level_bonus     - 1.25x size for entries that cleared their pivot
      round_veto      - veto entries just below/above round-number levels
      profile_veto    - veto entries into overhead/underfoot high-volume
                        nodes (needs volume channel)
      fib_veto        - veto longs beyond a 61.8% retrace of the 180d swing
      tstat_filter    - require 60d OLS slope |t| > 2 in trade direction
      wave_caution    - veto longs after >=5 consecutive zigzag up-legs
      hi52_anchor     - 1.25x near the 365d high, veto >30% below it
      wick_veto       - veto after an opposing rejection wick at a pivot
      leader_veto     - veto alt entries against BTC's last bar (>1 dATR)
      avwap_gate      - longs only above / shorts only below the
                        month-anchored VWAP (needs volume channel)
      pd_extremes     - entries only on prior-day high/low breaks
    """

    def __init__(self, period_seconds, member, volume_channel=4, **checks):
        self._member = build_rule_agent(member, period_seconds)
        self._period = period_seconds
        self._per_day = int(round(DAY / period_seconds))
        self._checks = checks
        self._volume_channel = volume_channel
        self._coins = None

    def begin_month(self, coins):
        if hasattr(self._member, "begin_month"):
            self._member.begin_month(coins)
        self._coins = list(coins)
        self._btc = next((i for i, c in enumerate(coins)
                          if c.startswith("BTC")), None)
        self._cache = None
        self._bar = 0

    # ---------------------------------------------------------- structure

    @staticmethod
    def _zigzag(close, threshold=0.10):
        """Pivot list [(index, price, +1 high/-1 low), ...]."""
        pivots = []
        last_ext, last_i, direction = close[0], 0, 0
        for i in range(1, len(close)):
            p = close[i]
            if direction >= 0 and p > last_ext:
                last_ext, last_i = p, i
            elif direction <= 0 and p < last_ext:
                last_ext, last_i = p, i
            move = p / last_ext - 1.0
            if direction >= 0 and move < -threshold:
                pivots.append((last_i, last_ext, +1))
                direction, last_ext, last_i = -1, p, i
            elif direction <= 0 and move > threshold:
                pivots.append((last_i, last_ext, -1))
                direction, last_ext, last_i = +1, p, i
        return pivots

    def _build_cache(self, history):
        close = history[0]
        n_levels = 1080 if close.shape[1] >= 1080 else close.shape[1]
        cache = {"pivots": [], "profile": [], "uplegs": []}
        for i in range(close.shape[0]):
            series = close[i, -n_levels:]
            pivots = self._zigzag(series)
            cache["pivots"].append([(p, s) for _, p, s in pivots])
            signs = [s for _, _, s in pivots]
            up = 0
            for s in signs[::-1]:
                if s == +1:
                    up += 1
                elif up:
                    break
            cache["uplegs"].append(up)
            if self._checks.get("profile_veto") or self._checks.get("avwap_gate"):
                vol = history[self._volume_channel, i, -540:]
                pr = close[i, -540:]
                bins = np.exp(np.linspace(np.log(pr.min() + 1e-12),
                                          np.log(pr.max() + 1e-9), 31))
                hist, _ = np.histogram(pr, bins=bins, weights=vol)
                cache["profile"].append((bins, hist))
            else:
                cache["profile"].append(None)
        return cache

    # -------------------------------------------------------------- decide

    def decide_by_history(self, history, last_w):
        w = np.asarray(self._member.decide_by_history(history, last_w),
                       dtype=np.float64)
        if self._cache is None:
            self._cache = self._build_cache(history)
        self._bar += 1
        close = history[0]
        high, low = history[1], history[2]
        last = close[:, -1]
        atr_bars = 14 * self._per_day
        datr = np.mean(high[:, -atr_bars:] - low[:, -atr_bars:], axis=1) * \
            np.sqrt(self._per_day)
        entering = (np.sign(w) != np.sign(last_w)) & (np.sign(w) != 0)
        c = self._checks
        size = np.ones_like(w)
        allowed = np.ones_like(w, dtype=bool)

        for i in range(len(w)):
            if not entering[i]:
                continue
            sign = np.sign(w[i])
            pivots = self._cache["pivots"][i]
            above = [p for p, _ in pivots if p > last[i]]
            below = [p for p, _ in pivots if p < last[i]]
            res = min(above) if above else None
            sup = max(below) if below else None
            if c.get("swing_sr"):
                if sign > 0 and res and res - last[i] < datr[i]:
                    allowed[i] = False
                if sign < 0 and sup and last[i] - sup < datr[i]:
                    allowed[i] = False
            if c.get("level_bonus"):
                highs = [p for p, s in pivots if s > 0]
                lows = [p for p, s in pivots if s < 0]
                if sign > 0 and highs and last[i] > max(h for h in highs) + \
                        0.25 * datr[i]:
                    size[i] = 1.25
                if sign < 0 and lows and last[i] < min(l for l in lows) - \
                        0.25 * datr[i]:
                    size[i] = 1.25
            if c.get("round_veto"):
                exponent = np.floor(np.log10(max(last[i], 1e-12)))
                grid = np.array([1, 2, 5, 10]) * 10.0 ** exponent
                above_r = grid[grid >= last[i]]
                below_r = grid[grid <= last[i]]
                if sign > 0 and len(above_r) and \
                        above_r.min() - last[i] < 0.5 * datr[i]:
                    allowed[i] = False
                if sign < 0 and len(below_r) and \
                        last[i] - below_r.max() < 0.5 * datr[i]:
                    allowed[i] = False
            if c.get("profile_veto") and self._cache["profile"][i]:
                bins, hist = self._cache["profile"][i]
                lo_p, hi_p = sorted([last[i], last[i] + sign * 2 * datr[i]])
                mask = (bins[:-1] < hi_p) & (bins[1:] > lo_p)
                if mask.any() and hist[mask].max() > 1.5 * hist.mean():
                    allowed[i] = False
            if c.get("fib_veto") and sign > 0:
                swing = close[i, -1080:]
                lo_i = swing.argmin()
                hi_v = swing[lo_i:].max()
                lo_v = swing[lo_i]
                if hi_v > lo_v and (hi_v - last[i]) / (hi_v - lo_v) > 0.618:
                    allowed[i] = False
            if c.get("tstat_filter"):
                y = np.log(close[i, -60 * self._per_day:])
                x = np.arange(len(y), dtype=float)
                vx = x - x.mean()
                beta = (vx * (y - y.mean())).sum() / (vx ** 2).sum()
                resid = y - y.mean() - beta * vx
                se = np.sqrt((resid ** 2).sum() / (len(y) - 2) /
                             (vx ** 2).sum())
                t = beta / max(se, 1e-12)
                if sign * t < 2.0:
                    allowed[i] = False
            if c.get("wave_caution") and sign > 0 and \
                    self._cache["uplegs"][i] >= 5:
                allowed[i] = False
            if c.get("hi52_anchor") and sign > 0:
                hi52 = close[i, -365 * self._per_day:].max()
                if last[i] < 0.70 * hi52:
                    allowed[i] = False
                elif last[i] > 0.95 * hi52:
                    size[i] = 1.25
            if c.get("wick_veto"):
                rng = high[i, -1] - low[i, -1] + 1e-12
                if sign > 0:
                    wick = (high[i, -1] - max(close[i, -1], close[i, -2])) / rng
                    if wick > 0.6 and res and res - last[i] < datr[i]:
                        allowed[i] = False
                else:
                    wick = (min(close[i, -1], close[i, -2]) - low[i, -1]) / rng
                    if wick > 0.6 and sup and last[i] - sup < datr[i]:
                        allowed[i] = False
            if c.get("leader_veto") and self._btc is not None and \
                    i != self._btc:
                btc_move = close[self._btc, -1] - close[self._btc, -2]
                if sign * btc_move < -datr[self._btc] * \
                        (last[self._btc] / close[self._btc, -2]) * 0 - \
                        datr[self._btc]:
                    allowed[i] = False
            if c.get("avwap_gate") and self._cache["profile"][i] is not None:
                bars = min(self._bar, 540)
                vol = history[self._volume_channel, i, -bars:]
                vwap = (close[i, -bars:] * vol).sum() / max(vol.sum(), 1e-12)
                if (sign > 0 and last[i] < vwap) or \
                        (sign < 0 and last[i] > vwap):
                    allowed[i] = False
            if c.get("pd_extremes"):
                pd_hi = high[i, -2 * self._per_day:-self._per_day].max()
                pd_lo = low[i, -2 * self._per_day:-self._per_day].min()
                if sign > 0 and last[i] <= pd_hi:
                    allowed[i] = False
                if sign < 0 and last[i] >= pd_lo:
                    allowed[i] = False
        w = np.where(entering & ~allowed, last_w, w * size)
        return w
