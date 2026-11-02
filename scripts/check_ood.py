"""How much of each test case lies outside the feature range seen in training?

usage: python scripts/check_ood.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ccml.data import load_cases, stack  # noqa: E402

X, _, _ = stack(load_cases(ROOT / "cache", "train"))
lo, hi = X.min(axis=0), X.max(axis=0)
p_lo, p_hi = np.percentile(X, 0.5, axis=0), np.percentile(X, 99.5, axis=0)

print(f"{'test case':22s} {'outside min/max':>16s} {'outside 0.5-99.5%':>18s}")
for c in load_cases(ROOT / "cache", "test"):
    strict = ((c["X"] < lo) | (c["X"] > hi)).any(axis=1).mean()
    loose = ((c["X"] < p_lo) | (c["X"] > p_hi)).any(axis=1).mean()
    print(f"{c['name']:22s} {strict*100:15.1f}% {loose*100:17.1f}%")
