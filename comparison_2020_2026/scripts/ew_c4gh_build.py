"""Regenerate data/cache_opt/m9_stop04_C4gh.parquet.

Exact reproduction of the committed pattern (scripts/opt9/breadth_m9.py
lines 16-19, applied to the top-50 caches): disable take-profit targets,
resim the stop04 futures trade book with the C4 exit knobs, gap-honest.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = "/home/user/EW_trader"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
for sub in ("opt", "opt7", "opt8", "opt9"):
    sys.path.insert(0, os.path.join(ROOT, "scripts", sub))

import m9lib          # noqa: E402
import optlib as L    # noqa: E402

tr = pd.read_parquet(os.path.join(L.CACHE, "trades_stop04_fut.parquet")).copy()
tr["target_price"] = np.where(tr.direction > 0, 1e12, 1e-12)
trX = m9lib.resim_fut(tr, **m9lib.C4)   # default end=2025-01-01, gap_fill=True
out = os.path.join(L.CACHE, "m9_stop04_C4gh.parquet")
trX.to_parquet(out)
print(f"wrote {out}: {len(trX)} trades")
