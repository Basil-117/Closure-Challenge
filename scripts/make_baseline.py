"""Write the 'no ML' submission: the k-omega SST RANS velocity as-is.
Every ML model should be compared against this.

usage: python scripts/make_baseline.py --points ../ml-turbulence-benchmark/data/evaluation_points
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ccml.data import load_cases, write_submission  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="cache")
ap.add_argument("--points", required=True)
args = ap.parse_args()

test = load_cases(args.cache, "test")
out = write_submission(test, {c["name"]: c["U_rans"] for c in test}, args.points,
                       ROOT / "submissions/baseline_rans")
print("wrote", out)
