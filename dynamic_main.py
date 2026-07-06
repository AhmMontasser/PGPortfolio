"""Entry point for the dynamic-universe backtest.

Usage:
    python dynamic_main.py --mode download   # fetch/refresh market data only
    python dynamic_main.py --mode universe   # print the monthly universes
    python dynamic_main.py --mode backtest   # full walk-forward backtest

All parameters live in ``dynamic_config.json``.
"""

from __future__ import absolute_import, division, print_function

import json
import logging
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd


def build_parser():
    parser = ArgumentParser()
    parser.add_argument("--mode", dest="mode", default="backtest",
                        choices=["download", "universe", "backtest"])
    parser.add_argument("--config", dest="config", default="dynamic_config.json")
    parser.add_argument("--seed", dest="seed", type=int, default=100)
    return parser


def save_results(backtester, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    summary = backtester.summary()
    summary.to_csv(os.path.join(output_dir, "summary.csv"))
    print(summary.to_string(float_format=lambda v: "%.4f" % v))

    curves = {}
    for name, result in backtester.results.items():
        curves[name] = pd.Series(
            np.cumprod(result["pc_vector"]),
            index=pd.to_datetime(result["times"], unit="s"))
    curve_frame = pd.DataFrame(curves)
    curve_frame.to_csv(os.path.join(output_dir, "portfolio_values.csv"))

    universes = pd.DataFrame(
        {pd.Timestamp(ts, unit="s").strftime("%Y-%m"): pd.Series(coins)
         for ts, coins in backtester.universes.items()}).T
    universes.to_csv(os.path.join(output_dir, "universes.csv"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(12, 6))
        for name in curve_frame.columns:
            axis.plot(curve_frame.index, curve_frame[name], label=name,
                      linewidth=1.2)
        axis.set_yscale("log")
        axis.set_ylabel("portfolio value (initial = 1, USDT)")
        axis.set_title("Dynamic top-10 universe backtest (monthly re-selection)")
        axis.legend()
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(os.path.join(output_dir, "backtest.png"), dpi=150)
    except Exception as e:  # plotting must never kill a finished backtest
        logging.warning("could not plot: %s", e)


def main():
    options = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    with open(options.config) as config_file:
        config = json.load(config_file)
    np.random.seed(options.seed)

    os.makedirs("./database", exist_ok=True)
    from pgportfolio.dynamic.backtest import DynamicBacktester
    backtester = DynamicBacktester(config)

    if options.mode == "download":
        backtester.prepare_selection_data()
        backtester.build_universes()
        backtester.prepare_trading_data()
        return
    if options.mode == "universe":
        backtester.prepare_selection_data()
        universes = backtester.build_universes()
        for ts in sorted(universes):
            print(pd.Timestamp(ts, unit="s").strftime("%Y-%m"),
                  " ".join(universes[ts]))
        return

    backtester.run()
    save_results(backtester, config["output"]["directory"])


if __name__ == "__main__":
    main()
