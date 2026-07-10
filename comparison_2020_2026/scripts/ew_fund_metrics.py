"""Glue driver: download funding for all coins the M10 pipeline needs.

Uses EW_trader's own download_funding() and membership builders — no repo
code is modified. Coins = union of futures ever-members (to 2026-05-31)
and the spot-panel membership union (research_xs_momentum.load_panels).
"""
import os, sys

ROOT = "/home/user/EW_trader"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "opt8"))

from ewtrader import universe as U
from m8_data import build_futures_membership
import research_xs_momentum as R
from futures_backtest_2020_2024 import download_funding

# futures ever-members through the holdout end
mem_f, coins_f = build_futures_membership("2026-05-31")

# spot membership union over the full window (panels 2019-09 .. 2026-05-31)
params = U.UniverseParams(min_age_days=90, min_liquidity_usd=3e6, max_coins=50)
mem_s = U.build_membership("2019-09-01", "2026-05-31", params, freq="MS")
coins_s = sorted(mem_s.coins())

coins = sorted(set(coins_f) | set(coins_s))
print(f"funding download for {len(coins)} coins "
      f"({len(coins_f)} futures-members, {len(coins_s)} spot-members)")
download_funding(coins, "2019-06-01", "2026-06-30")
print("funding downloads complete")
