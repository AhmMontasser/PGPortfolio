"""Walk-forward backtester for the long/short (margin) EIIE agent.

Reuses the dynamic-universe machinery of ``backtest.DynamicBacktester``
(monthly top-10 selection, data handling, panels) but with signed-weight
accounting in USDT:

* positions can be long or short, levered up to ``margin.max_leverage``;
* per-coin exposure is capped at ``margin.max_coin_weight`` of equity;
* commissions on turnover, financing on shorted coins and borrowed USDT;
* a liquidation guard stops the account out if equity hits zero.

Two optional *risk overlays* scale the agent's signed weights at trade
time (they are risk management, not part of the learned policy):

* **volatility targeting**: portfolio weights are scaled so the predicted
  annualized volatility (from the trailing covariance of the universe)
  does not exceed ``overlay.vol_target``;
* **drawdown deleveraging**: when the equity's trailing drawdown exceeds
  ``overlay.dd_soft``, exposure is linearly reduced, reaching
  ``overlay.dd_floor`` (e.g. 10% of normal size) at ``overlay.dd_hard``.
  This mechanically bounds the depth a losing streak can reach.
"""

from __future__ import absolute_import, division, print_function

import logging

import numpy as np
import pandas as pd

from pgportfolio.dynamic.backtest import DynamicBacktester, DAY, SECONDS_PER_YEAR
from pgportfolio.dynamic.margin import MarginEIIEAgent, margin_period_return


class MarginBacktester(DynamicBacktester):
    def __init__(self, config, archive=None):
        DynamicBacktester.__init__(self, config, archive=archive)
        self.margin_config = config["margin"]
        self.overlay = config.get("overlay", {})

    # ------------------------------------------------------------- agents

    def make_agents(self):
        margin = self.margin_config
        rule = self.config.get("rule")
        if rule:
            from pgportfolio.dynamic.rules import build_rule_agent
            self.eiie = build_rule_agent(rule, self.period)
            self.is_rule_agent = True
            self.agents = []
            return self.agents
        self.is_rule_agent = False
        network = self.config.get("network", {})
        self.eiie = MarginEIIEAgent(
            conv_filters=network.get("conv_filters", 3),
            dense_filters=network.get("dense_filters", 10),
            feature_number=self.features,
            window_size=self.window,
            commission_rate=self.commission,
            learning_rate=self.config["training"]["learning_rate"],
            max_leverage=margin["max_leverage"],
            max_coin_weight=margin["max_coin_weight"],
            short_borrow_apr=margin["short_borrow_apr"],
            usdt_borrow_apr=margin["usdt_borrow_apr"],
            trade_period=self.period,
            gross_penalty=margin.get("gross_penalty", 0.0),
            boundary_penalty=margin.get("boundary_penalty", 2e-5))
        self.agents = []
        return self.agents

    def initial_training(self):
        if getattr(self, "is_rule_agent", False) or \
                self.config["training"]["steps"] <= 0:
            return  # rule agents have nothing to train
        DynamicBacktester.initial_training(self)

    # ---------------------------------------------------------------- run

    def run(self):
        self.prepare_selection_data()
        self.build_universes()
        self.prepare_trading_data()
        self.make_agents()
        self.initial_training()

        train_config = self.config["training"]
        lookback = train_config["rolling_lookback_days"] * DAY
        state = {
            "pv": 1.0, "peak": 1.0, "pc": [], "times": [], "gross": [],
            "turnover": [], "coins": None, "omega": None, "bust": False,
            "equity": [],
        }
        periods_per_day = DAY / self.period
        borrow = (
            self.margin_config["short_borrow_apr"] / 365.0 / periods_per_day,
            self.margin_config["usdt_borrow_apr"] / 365.0 / periods_per_day)

        for month_index, month_ts in enumerate(self.month_starts):
            if state["bust"]:
                break
            coins = self.universes[month_ts]
            month_end = (self.month_starts[month_index + 1]
                         if month_index + 1 < len(self.month_starts)
                         else self.test_end)
            rolling_steps = train_config["rolling_steps"]
            if rolling_steps > 0 and not getattr(self, "is_rule_agent", False):
                train_panel = self.build_panel(coins, month_ts - lookback, month_ts)
                self.eiie.train_on_panel(
                    train_panel, rolling_steps,
                    batch_size=train_config["batch_size"],
                    sample_bias=train_config["buffer_biased"],
                    log_every=0, tag="rolling")
            if hasattr(self.eiie, "begin_month"):
                self.eiie.begin_month(coins)

            warmup_periods = max(self.window + 1,
                                 self.overlay.get("vol_window", 336) + 1)
            panel_end = min(month_end + self.period, self.test_end)
            panel = self.build_panel(
                coins, month_ts - warmup_periods * self.period, panel_end)
            grid = np.arange(month_ts - warmup_periods * self.period,
                             panel_end, self.period)
            self._trade_month_margin(state, coins, panel, grid,
                                     warmup_periods, borrow)
            logging.info("month %s done: pv %.4f dd %.3f",
                         pd.Timestamp(month_ts, unit="s").strftime("%Y-%m"),
                         state["pv"], 1 - state["pv"] / state["peak"])

        label = self.config.get("rule", {}).get("name") or \
            self.config.get("rule", {}).get("type") or "EIIE-LS"
        self.results = {
            label: {
                "pv": state["pv"],
                "pc_vector": np.array(state["pc"], dtype=np.float64),
                "times": np.array(state["times"], dtype=np.int64),
                "turnover": np.array(state["turnover"], dtype=np.float64),
                "gross": np.array(state["gross"], dtype=np.float64),
            }
        }
        return self.results

    # ------------------------------------------------------------ overlays

    def _overlay_scale(self, panel, t, w, state):
        scale = 1.0
        vol_target = self.overlay.get("vol_target", 0.0)
        if vol_target and np.abs(w).sum() > 1e-9:
            window = self.overlay.get("vol_window", 336)
            closes = panel[0, :, t - window:t + 1]
            rets = closes[:, 1:] / closes[:, :-1] - 1.0
            cov = np.cov(rets)
            variance = float(w @ cov @ w)
            periods_per_year = SECONDS_PER_YEAR / self.period
            vol = np.sqrt(max(variance, 1e-12) * periods_per_year)
            scale = min(scale, vol_target / max(vol, 1e-9))
        dd_soft = self.overlay.get("dd_soft", 0.0)
        if dd_soft:
            dd_hard = self.overlay.get("dd_hard", 0.15)
            dd_floor = self.overlay.get("dd_floor", 0.1)
            drawdown = 1.0 - state["pv"] / self._reference_peak(state)
            if drawdown > dd_soft:
                fraction = (dd_hard - drawdown) / (dd_hard - dd_soft)
                scale *= max(dd_floor, min(1.0, fraction))
        return scale

    def _reference_peak(self, state):
        """Peak used for the drawdown overlay.

        Three anchoring modes:
        * default: all-time peak (never releases after a deep loss);
        * ``overlay.peak_window_days``: trailing-window peak - after a
          controlled loss the old high rolls out of the window, so exposure
          recovers instead of staying pinned at the floor forever;
        * ``overlay.peak_anchor = "calendar_year"``: peak since Jan 1 - an
          annual loss budget, directly bounding the *per-year* max drawdown
          (episodes cannot chain within a year, and each new year re-arms
          the budget).
        """
        if self.overlay.get("peak_anchor") == "calendar_year":
            start = state.get("year_start_index", 0)
            recent = state["equity"][start:]
            return max(max(recent), state["pv"]) if recent else state["pv"]
        window_days = self.overlay.get("peak_window_days", 0)
        if not window_days:
            return state["peak"]
        periods = int(window_days * DAY / self.period)
        recent = state["equity"][-periods:] if state["equity"] else [state["pv"]]
        return max(max(recent), state["pv"])

    # ------------------------------------------------------------- trading

    def _trade_month_margin(self, state, coins, panel, grid, first_trade,
                            borrow):
        short_borrow, usdt_borrow = borrow
        old_coins, old_omega = state["coins"], state["omega"]
        omega = np.zeros(len(coins))
        boundary_turnover = 0.0
        if old_omega is not None:
            for j, coin in enumerate(old_coins):
                if coin in coins:
                    omega[coins.index(coin)] = old_omega[j]
                else:
                    # dropped coin: position closed at the boundary
                    boundary_turnover += abs(old_omega[j])

        cap = self.margin_config["max_coin_weight"]
        max_gross = self.margin_config["max_leverage"]
        deadband = self.margin_config.get("rebalance_threshold", 0.0)
        for t in range(first_trade, len(grid) - 1):
            history = panel[:, :, t - self.window + 1:t + 1]
            w = self.eiie.decide_by_history(history, omega.copy())
            w = self._apply_gates(panel, t, np.asarray(w, dtype=np.float64))
            w = w * self._overlay_scale(panel, t, w, state)
            w = np.clip(w, -cap, cap)
            gross = np.abs(w).sum()
            if gross > max_gross:
                w *= max_gross / gross
            w = self._apply_stops(state, coins, panel[0, :, t],
                                  int(grid[t]), w)
            # execution deadband: skip rebalances too small to pay for -
            # tiny decision noise otherwise churns commissions all day
            if t != first_trade and np.abs(w - omega).sum() < deadband:
                w = omega.copy()
            y = np.clip(panel[0, :, t + 1] / panel[0, :, t], 0.05, 20.0)

            turnover = np.abs(w - omega).sum()
            if t == first_trade:
                turnover += boundary_turnover
            long_exposure = np.maximum(w, 0).sum()
            short_exposure = np.maximum(-w, 0).sum()
            usdt_borrowed = max(0.0, long_exposure - 1.0)
            R = (1.0 + np.dot(w, y - 1.0)
                 - self.commission * turnover
                 - short_exposure * short_borrow
                 - usdt_borrowed * usdt_borrow)
            if R <= 0.0:
                logging.warning("liquidated at %s",
                                pd.Timestamp(int(grid[t + 1]), unit="s"))
                R = 1e-6
                state["bust"] = True
            state["pv"] *= R
            state["peak"] = max(state["peak"], state["pv"])
            year = pd.Timestamp(int(grid[t + 1]), unit="s").year
            if year != state.get("year", year):
                state["year_start_index"] = len(state["equity"])
            state["year"] = year
            state["equity"].append(state["pv"])
            state["pc"].append(R)
            state["times"].append(int(grid[t + 1]))
            state["turnover"].append(turnover)
            state["gross"].append(np.abs(w).sum())
            omega = np.zeros_like(w) if state["bust"] else w * y / R
            if state["bust"]:
                break

        state["coins"], state["omega"] = list(coins), omega

    # ----------------------------------------------------- gates and stops

    def _apply_gates(self, panel, t, w):
        """Per-coin no-trade filters applied to the raw decision.

        * ``overlay.trend_gate_days``: positions are only allowed in the
          direction of each coin's own moving-average trend (long above,
          short below) - "trade with the tide or not at all";
        * ``overlay.min_weight``: signals smaller than this fraction of
          equity are treated as noise and zeroed (no-trade region).
        """
        gate_days = self.overlay.get("trend_gate_days", 0)
        if gate_days:
            periods = int(gate_days * DAY / self.period)
            closes = panel[0, :, max(0, t - periods):t + 1]
            trend = np.sign(panel[0, :, t] - closes.mean(axis=1))
            w = np.where(np.sign(w) == trend, w, 0.0)
        min_weight = self.overlay.get("min_weight", 0.0)
        if min_weight:
            w = np.where(np.abs(w) >= min_weight, w, 0.0)
        return w

    def _apply_stops(self, state, coins, closes, now, w):
        """Trailing per-position stop-loss with cooldown.

        A long is stopped out when its price falls ``position_stop`` below
        the highest close seen since the position was opened (shorts
        symmetric from the lowest close).  A stopped coin cannot be
        re-entered for ``stop_cooldown_days``.
        """
        stop = self.margin_config.get("position_stop", 0.0)
        if not stop:
            return w
        cooldown = self.margin_config.get("stop_cooldown_days", 3) * DAY
        tracker = state.setdefault("stop_tracker", {})
        blocked = state.setdefault("stop_until", {})
        for i, coin in enumerate(coins):
            if blocked.get(coin, 0) > now:
                w[i] = 0.0
                continue
            sign = np.sign(w[i])
            held = tracker.get(coin)
            if sign == 0:
                tracker.pop(coin, None)
                continue
            if held is None or held["sign"] != sign:
                tracker[coin] = {"sign": sign, "extreme": closes[i]}
                continue
            if sign > 0:
                held["extreme"] = max(held["extreme"], closes[i])
                hit = closes[i] < held["extreme"] * (1.0 - stop)
            else:
                held["extreme"] = min(held["extreme"], closes[i])
                hit = closes[i] > held["extreme"] * (1.0 + stop)
            if hit:
                w[i] = 0.0
                blocked[coin] = now + cooldown
                tracker.pop(coin, None)
        return w

    # --------------------------------------------------------------- report

    def summary(self):
        frame = DynamicBacktester.summary(self)
        for name, result in self.results.items():
            if "gross" in result:
                frame.loc[name, "avg_gross_exposure"] = float(
                    np.mean(result["gross"]))
        return frame

    def yearly_table(self, name=None):
        if name is None:
            name = next(iter(self.results))
        result = self.results[name]
        series = pd.Series(result["pc_vector"],
                           index=pd.to_datetime(result["times"], unit="s"))
        rows = []
        for year, chunk in series.groupby(series.index.year):
            equity = chunk.cumprod()
            peaks = equity.cummax()
            rows.append({
                "year": year,
                "return": float(equity.iloc[-1]) - 1.0,
                "max_drawdown": float((1.0 - equity / peaks).max()),
            })
        return pd.DataFrame(rows).set_index("year")
