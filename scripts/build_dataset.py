"""Read every train / val / test case from the benchmark repo, compute
features + targets, and cache them as .npz files.

usage: python scripts/build_dataset.py --data ../ml-turbulence-benchmark/data
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ccml.cases import case_path, split  # noqa: E402
from ccml.features import build_case  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="path to ml-turbulence-benchmark/data")
ap.add_argument("--out", default="cache")
args = ap.parse_args()

out = Path(args.out)
out.mkdir(exist_ok=True)
for part, names in split(args.data).items():
    for name in names:
        f = out / f"{name}.npz"
        if f.exists():
            continue
        t0 = time.time()
        c = build_case(case_path(args.data, name), name)
        np.savez_compressed(f, split=part, **c)
        print(f"[{part:5s}] {name:22s} cells={len(c['X']):6d}  {time.time()-t0:5.1f}s", flush=True)
