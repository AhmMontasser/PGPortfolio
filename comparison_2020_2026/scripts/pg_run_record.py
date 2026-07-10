"""Run a PGPortfolio dynamic margin backtest with position recording.

Runs the repo's MarginBacktester unmodified except for a subclass hook on
_apply_exit_policies (called once per traded period, last decision-transform
before the execution deadband) that records the decision weight vector.
Standard outputs are written by the repo's own save_results(); the extra
position record goes to a parquet next to the run directory.

Usage: pg_run_record.py <config_path> [seed]
"""
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

os.chdir("/home/user/PGPortfolio")
sys.path.insert(0, "/home/user/PGPortfolio")

from pgportfolio.dynamic.margin_backtest import MarginBacktester
from dynamic_main import save_results


class RecordingBacktester(MarginBacktester):
    def __init__(self, config, archive=None):
        super().__init__(config, archive)
        self.position_records = []

    def _apply_exit_policies(self, state, coins, closes, now, w):
        w = super()._apply_exit_policies(state, coins, closes, now, w)
        self.position_records.append(
            (int(now), list(coins), np.asarray(closes, dtype=float).copy(),
             np.asarray(w, dtype=float).copy()))
        return w


def main():
    config_path = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    with open(config_path) as fh:
        config = json.load(fh)
    np.random.seed(seed)
    os.makedirs("./database", exist_ok=True)

    bt = RecordingBacktester(config)
    bt.run()
    out_dir = config["output"]["directory"]
    save_results(bt, out_dir)
    if hasattr(bt, "yearly_table"):
        table = bt.yearly_table()
        table.to_csv(os.path.join(out_dir, "yearly.csv"))
        print(table.to_string(float_format=lambda v: "%.4f" % v))

    # flatten position records to long format
    rows = []
    for now, coins, closes, w in bt.position_records:
        for i, c in enumerate(coins):
            if w[i] != 0.0 or True:  # keep zeros too; needed for episode ends
                rows.append((now, c, closes[i], w[i]))
    rec = pd.DataFrame(rows, columns=["time", "coin", "close", "weight"])
    name = os.path.basename(out_dir.rstrip("/"))
    rec_path = os.path.join(
        "/tmp/claude-0/-home-user/626257e9-422c-51bb-a61c-220ecd9df43d/scratchpad",
        "pg_positions", f"{name}.parquet")
    os.makedirs(os.path.dirname(rec_path), exist_ok=True)
    rec.to_parquet(rec_path)
    print(f"positions recorded: {len(rec)} rows -> {rec_path}")


if __name__ == "__main__":
    main()
