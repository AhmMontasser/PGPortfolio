"""Walk-forward backtester with a monthly re-selected (dynamic) universe.

For every calendar month of the test range:

1. the universe is re-selected (top N by volume + liquidity) using only
   data available before the month starts (``universe.UniverseSelector``);
2. the EIIE network is fine-tuned on a trailing window of data of the *new*
   universe, again ending before the month starts;
3. the month is traded out-of-sample, period by period.

When the universe changes at a month boundary, positions in coins that
dropped out have to be sold and positions in the newcomers bought.  That
handover is charged through the same iterative commission formula the
original project uses (``calculate_pv_after_commission``), evaluated on the
union of the outgoing and incoming coin lists.
"""

from __future__ import absolute_import, division, print_function

import json
import logging

import numpy as np
import pandas as pd

from pgportfolio.dynamic.binancedata import BinanceArchive, sanitize_token_swaps
from pgportfolio.dynamic.universe import UniverseSelector, investable, DAY
from pgportfolio.dynamic.eiie import EIIEAgent

SECONDS_PER_YEAR = 365.25 * DAY


def calculate_pv_after_commission(w1, w0, commission_rate):
    """Fraction of wealth left after rebalancing w0 -> w1 (cash first).

    Same iterative solution of the transaction-cost equation as
    ``pgportfolio.tools.trade.calculate_pv_after_commission`` (copied here so
    the dynamic package does not import the legacy Poloniex/pd.Panel stack).
    """
    mu0 = 1
    mu1 = 1 - 2 * commission_rate + commission_rate ** 2
    while abs(mu1 - mu0) > 1e-10:
        mu0 = mu1
        mu1 = (1 - commission_rate * w0[0] -
               (2 * commission_rate - commission_rate ** 2) *
               np.sum(np.maximum(w0[1:] - mu1 * w1[1:], 0))) / \
              (1 - commission_rate * w1[0])
    return mu1


# --------------------------------------------------------------------- agents

class MonthlyAgent(object):
    """Interface: an agent is told the new coin list at each month start and
    asked for a portfolio vector (cash first) every trading period."""

    name = "agent"

    def begin_month(self, coins, warmup_relative_prices):
        """:param warmup_relative_prices: [periods, coins+1] relative price
        vectors preceding the month, for algorithms that need history."""

    def decide(self, history, last_w):
        """:param history: [features, coins, window] price matrix
        :param last_w: [coins+1] current (drifted) portfolio vector
        :return: [coins+1] target portfolio vector"""
        raise NotImplementedError()


class TraditionalAgent(MonthlyAgent):
    """Wraps the numpy on-line portfolio selection algorithms shipped in
    ``pgportfolio.tdagent`` (OLMAR, PAMR, RMR, CRP...).  A fresh instance is
    created every month because the asset dimension changes with the
    universe; it is warmed up on the pre-month history."""

    def __init__(self, constructor, name):
        self._constructor = constructor
        self.name = name
        self._agent = None

    def begin_month(self, coins, warmup_relative_prices):
        self._agent = self._constructor()
        m = len(coins)
        b = np.ones(m + 1) / (m + 1)
        for rpv in warmup_relative_prices:
            b = np.ravel(self._agent.decide_by_history(rpv, b))
        self._first = True

    def decide(self, history, last_w):
        rpv = np.concatenate([[1.0], history[0, :, -1] / history[0, :, -2]])
        return np.ravel(self._agent.decide_by_history(rpv, last_w))


class UniformBuyAndHold(MonthlyAgent):
    """Buys the month's universe equally at the month start, then holds."""

    name = "UBAH"

    def begin_month(self, coins, warmup_relative_prices):
        self._target = np.concatenate([[0.0], np.ones(len(coins)) / len(coins)])
        self._first = True

    def decide(self, history, last_w):
        if self._first:
            self._first = False
            return self._target
        return last_w


class UniformConstantRebalanced(MonthlyAgent):
    """Rebalances to equal weights (over the coins) every period."""

    name = "UCRP"

    def begin_month(self, coins, warmup_relative_prices):
        self._target = np.concatenate([[0.0], np.ones(len(coins)) / len(coins)])

    def decide(self, history, last_w):
        return self._target


class BTCBuyAndHold(MonthlyAgent):
    """Holds BTC only - the natural benchmark of a USDT-quoted backtest."""

    name = "BTC-hold"

    def __init__(self, btc_symbol="BTCUSDT"):
        self._btc = btc_symbol

    def begin_month(self, coins, warmup_relative_prices):
        self._target = np.zeros(len(coins) + 1)
        if self._btc in coins:
            self._target[coins.index(self._btc) + 1] = 1.0
        else:
            self._target[0] = 1.0
        self._first = True

    def decide(self, history, last_w):
        if self._first:
            self._first = False
            return self._target
        return last_w


class EIIEMonthlyAgent(MonthlyAgent):
    """The deep RL agent; training is orchestrated by the backtester."""

    name = "EIIE"

    def __init__(self, agent):
        self._agent = agent

    def begin_month(self, coins, warmup_relative_prices):
        pass

    def decide(self, history, last_w):
        return self._agent.decide_by_history(history, last_w)


# ------------------------------------------------------------------- engine

class DynamicBacktester(object):
    def __init__(self, config, archive=None):
        self.config = config
        self.archive = archive or BinanceArchive(
            db_path=config["data"]["database"],
            max_workers=config["data"].get("download_workers", 16))
        input_config = config["input"]
        self.quote = input_config["quote"]
        self.period = input_config["trade_period"]
        self.window = input_config["window_size"]
        self.features = input_config["feature_number"]
        self.commission = config["trading"]["trading_consumption"]
        self.selector = UniverseSelector(
            coin_number=input_config["coin_number"],
            volume_average_days=input_config["volume_average_days"],
            min_history_days=input_config["min_history_days"],
            quote=self.quote)
        self.test_start = int(pd.Timestamp(input_config["start_date"], tz="UTC").timestamp())
        self.test_end = int(pd.Timestamp(input_config["end_date"], tz="UTC").timestamp())
        self.train_start = int(pd.Timestamp(input_config["train_start_date"], tz="UTC").timestamp())
        self.results = {}

    # ------------------------------------------------------------- data prep

    def prepare_selection_data(self):
        logging.info("listing archive symbols...")
        symbols = self.archive.list_symbols(quote=self.quote)
        candidates = [s for s in symbols if investable(s, self.quote)]
        logging.info("%d %s pairs, %d investable candidates",
                     len(symbols), self.quote, len(candidates))
        selection_start = self.test_start - 90 * DAY
        self.archive.ensure_data(candidates, "1d", pd.Timestamp(selection_start, unit="s"),
                                 pd.Timestamp(self.test_end - 1, unit="s"))
        self.daily = self.archive.read_daily_panel(
            candidates, selection_start, self.test_end)
        return candidates

    def build_universes(self):
        months = self.selector.monthly_schedule(
            pd.Timestamp(self.test_start, unit="s"),
            pd.Timestamp(self.test_end - 1, unit="s"))
        self.month_starts = months
        self.universes = {}
        for month_ts in months:
            coins, _ = self.selector.select(self.daily, month_ts)
            self.universes[month_ts] = coins
        union = sorted({c for coins in self.universes.values() for c in coins})
        self.union_coins = union
        return self.universes

    def prepare_trading_data(self):
        self.archive.ensure_data(self.union_coins, "30m",
                                 pd.Timestamp(self.train_start, unit="s"),
                                 pd.Timestamp(self.test_end - 1, unit="s"))
        logging.info("loading 30m candles for %d coins...", len(self.union_coins))
        self._frames = {}
        for coin in self.union_coins:
            frame = self.archive.read_frame(coin, "30m", self.train_start - 60 * DAY,
                                            self.test_end)
            self._frames[coin] = sanitize_token_swaps(frame, gap_days=3, jump=5.0)

    def build_panel(self, coins, start_ts, end_ts):
        """[features, coins, time] close/high/low array on the 30m grid.

        Gaps are forward- then back-filled per feature, like the original
        ``panel_fillna(panel, "both")``.
        """
        grid = np.arange(int(start_ts), int(end_ts), self.period)
        index = pd.Index(grid)
        panel = np.empty((self.features, len(coins), len(grid)), dtype=np.float32)
        feature_names = ["close", "high", "low"][:self.features]
        for i, coin in enumerate(coins):
            frame = self._frames[coin]
            frame = frame[(frame.index >= grid[0]) & (frame.index <= grid[-1])]
            for f, feature in enumerate(feature_names):
                series = frame[feature].reindex(index)
                series = series.ffill().bfill()
                panel[f, i, :] = series.to_numpy(dtype=np.float32)
        if np.isnan(panel).any():
            raise ValueError("NaNs left in panel for coins %s" % coins)
        return panel

    # -------------------------------------------------------------- backtest

    def make_agents(self):
        from pgportfolio.tdagent.algorithms.olmar import OLMAR
        from pgportfolio.tdagent.algorithms.pamr import PAMR
        from pgportfolio.tdagent.algorithms.rmr import RMR
        eiie = EIIEAgent(feature_number=self.features,
                         window_size=self.window,
                         commission_rate=self.commission,
                         learning_rate=self.config["training"]["learning_rate"])
        self.eiie = eiie
        self.agents = [
            EIIEMonthlyAgent(eiie),
            TraditionalAgent(OLMAR, "OLMAR"),
            TraditionalAgent(PAMR, "PAMR"),
            TraditionalAgent(RMR, "RMR"),
            UniformBuyAndHold(),
            UniformConstantRebalanced(),
            BTCBuyAndHold(btc_symbol="BTC" + self.quote),
        ]
        return self.agents

    def initial_training(self):
        first_universe = self.universes[self.month_starts[0]]
        panel = self.build_panel(first_universe, self.train_start, self.test_start)
        steps = self.config["training"]["steps"]
        logging.info("initial EIIE training: %d steps on %d periods of %s",
                     steps, panel.shape[2], first_universe)
        self.eiie.train_on_panel(
            panel, steps,
            batch_size=self.config["training"]["batch_size"],
            sample_bias=self.config["training"]["buffer_biased"],
            tag="initial")

    def run(self):
        self.prepare_selection_data()
        self.build_universes()
        self.prepare_trading_data()
        self.make_agents()
        self.initial_training()

        train_config = self.config["training"]
        lookback = train_config["rolling_lookback_days"] * DAY
        state = {agent.name: {"pv": 1.0, "pc": [], "turnover": [],
                              "coins": None, "omega": None, "times": []}
                 for agent in self.agents}

        for month_index, month_ts in enumerate(self.month_starts):
            coins = self.universes[month_ts]
            month_end = (self.month_starts[month_index + 1]
                         if month_index + 1 < len(self.month_starts)
                         else self.test_end)
            # fine-tune the network on the trailing window of the new universe
            rolling_steps = train_config["rolling_steps"]
            if rolling_steps > 0:
                train_panel = self.build_panel(coins, month_ts - lookback, month_ts)
                self.eiie.train_on_panel(
                    train_panel, rolling_steps,
                    batch_size=train_config["batch_size"],
                    sample_bias=train_config["buffer_biased"],
                    log_every=0, tag="rolling")

            # trading panel includes enough history for the input window and
            # traditional-agent warmup; it extends one period past the month
            # end so the month's last period is realized too (the following
            # month picks up trading exactly at month_end).
            warmup_periods = max(self.window + 1, 40)
            panel_end = min(month_end + self.period, self.test_end)
            panel = self.build_panel(
                coins, month_ts - warmup_periods * self.period, panel_end)
            grid = np.arange(month_ts - warmup_periods * self.period,
                             panel_end, self.period)
            first_trade = warmup_periods  # index on the grid of the month start
            warmup_close = panel[0, :, :first_trade]
            warmup_rpv = [
                np.concatenate([[1.0], warmup_close[:, t] / warmup_close[:, t - 1]])
                for t in range(1, first_trade)]

            for agent in self.agents:
                agent.begin_month(coins, warmup_rpv)
                self._trade_month(agent, state[agent.name], coins, panel,
                                  grid, first_trade)
            logging.info("month %s done: %s",
                         pd.Timestamp(month_ts, unit="s").strftime("%Y-%m"),
                         {a.name: round(state[a.name]["pv"], 4) for a in self.agents})

        self.results = {
            name: {
                "pv": s["pv"],
                "pc_vector": np.array(s["pc"], dtype=np.float64),
                "times": np.array(s["times"], dtype=np.int64),
                "turnover": np.array(s["turnover"], dtype=np.float64),
            }
            for name, s in state.items()
        }
        return self.results

    def _trade_month(self, agent, state, coins, panel, grid, first_trade):
        """Trade one month; ``state`` carries wealth and positions across
        months (including the universe-handover rebalancing cost)."""
        old_coins, old_omega = state["coins"], state["omega"]
        # project the carried portfolio onto the new universe: positions in
        # coins that dropped out are treated as sold at the boundary; the
        # cost of that sale is charged through the union-vector commission
        # computation at the first decision below.
        omega = np.zeros(len(coins) + 1)
        if old_omega is None:
            omega[0] = 1.0
        else:
            omega[0] = old_omega[0]
            for j, coin in enumerate(old_coins):
                weight = old_omega[j + 1]
                if coin in coins:
                    omega[coins.index(coin) + 1] = weight
                else:
                    omega[0] += weight

        # decisions at grid[t] use closes up to t; realized over t -> t+1
        for t in range(first_trade, len(grid) - 1):
            history = panel[:, :, t - self.window + 1:t + 1]
            target = np.ravel(agent.decide(history, omega.copy()))
            target = np.clip(target, 0, None)
            target /= target.sum()
            future_price = np.concatenate(
                [[1.0], np.clip(panel[0, :, t + 1] / panel[0, :, t], 0.05, 20.0)])

            if t == first_trade and old_omega is not None and old_coins != coins:
                mu = self._handover_mu(old_coins, old_omega, coins, target)
            else:
                mu = calculate_pv_after_commission(target, omega, self.commission)
            portfolio_change = mu * np.dot(target, future_price)
            state["pv"] *= portfolio_change
            state["pc"].append(portfolio_change)
            state["times"].append(int(grid[t + 1]))
            state["turnover"].append(np.abs(target[1:] - omega[1:]).sum())
            omega = mu * target * future_price / portfolio_change

        state["coins"], state["omega"] = list(coins), omega

    def _handover_mu(self, old_coins, old_omega, new_coins, new_target):
        """Commission factor for the month-boundary rebalance, evaluated on
        the union of the outgoing and incoming universes so that both the
        forced sales (dropped coins) and the new purchases are charged."""
        union = sorted(set(old_coins) | set(new_coins))
        w0 = np.zeros(len(union) + 1)
        w1 = np.zeros(len(union) + 1)
        w0[0] = old_omega[0]
        for j, coin in enumerate(old_coins):
            w0[union.index(coin) + 1] = old_omega[j + 1]
        w1[0] = new_target[0]
        for j, coin in enumerate(new_coins):
            w1[union.index(coin) + 1] = new_target[j + 1]
        return calculate_pv_after_commission(w1, w0, self.commission)

    # --------------------------------------------------------------- metrics

    def summary(self):
        rows = []
        periods_per_year = SECONDS_PER_YEAR / self.period
        for name, result in self.results.items():
            pc = result["pc_vector"]
            pv = result["pv"]
            years = len(pc) / periods_per_year
            annualized = pv ** (1.0 / years) - 1.0
            sharpe = (np.mean(pc - 1.0) / np.std(pc - 1.0)) * np.sqrt(periods_per_year)
            rows.append({
                "agent": name,
                "fAPV": pv,
                "annualized_return": annualized,
                "sharpe_annualized": sharpe,
                "max_drawdown": max_drawdown(pc),
                "avg_daily_turnover": float(np.mean(result["turnover"])) *
                                      (DAY / self.period),
            })
        return pd.DataFrame(rows).set_index("agent").sort_values(
            "fAPV", ascending=False)


def max_drawdown(pc_array):
    values = np.cumprod(pc_array)
    peaks = np.maximum.accumulate(values)
    return float(np.max(1.0 - values / peaks))
