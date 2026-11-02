"""Model 1: gradient-boosted trees (LightGBM) that predict the RANS velocity
error in the local flow frame from invariant RANS features.

usage:
  python scripts/train_lightgbm.py --points ml-turbulence-benchmark/data/evaluation_points
  python scripts/train_lightgbm.py --points ml-turbulence-benchmark/data/evaluation_points ^
         --fallback --out submissions/lightgbm_ood

--fallback: in cells where any input feature lies outside the range seen in
training, the model is not trusted and the plain k-omega SST velocity is kept.
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ccml.data import load_cases, scaled_mae, stack, write_submission  # noqa: E402
from ccml.features import FEATURE_NAMES, reconstruct  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="cache")
ap.add_argument("--points", required=True, help="ml-turbulence-benchmark/data/evaluation_points")
ap.add_argument("--out", default="submissions/lightgbm")
ap.add_argument("--fallback", action="store_true",
                help="keep the SST velocity where features are outside the training range")
args = ap.parse_args()

train, val, test = (load_cases(args.cache, p) for p in ("train", "val", "test"))
Xtr, ytr, wtr = stack(train)
Xva, yva, wva = stack(val)
print(f"train rows {len(Xtr):,}  val rows {len(Xva):,}")

params = dict(
    objective="huber", alpha=0.5, learning_rate=0.03, num_leaves=31, min_data_in_leaf=200,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    verbose=-1, num_threads=2, seed=0,
)
models = []
for j, comp in enumerate(["streamwise", "wall_normal", "cross"]):
    dtr = lgb.Dataset(Xtr, np.ascontiguousarray(ytr[:, j]), weight=wtr, feature_name=FEATURE_NAMES)
    dva = lgb.Dataset(Xva, np.ascontiguousarray(yva[:, j]), weight=wva, reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    print(f"  {comp:11s}: best iteration {m.best_iteration}")
    models.append(m)

lo, hi = Xtr.min(axis=0), Xtr.max(axis=0)   # full range of every feature seen in training


def predict(case):
    y = np.stack([m.predict(case["X"], num_iteration=m.best_iteration) for m in models], 1)
    if args.fallback:
        outside = ((case["X"] < lo) | (case["X"] > hi)).any(axis=1)  # would need extrapolation
        y[outside] = 0.0                                            # no correction -> keep SST
    return reconstruct(case, y)


val_scores = {c["name"]: {"rans": scaled_mae(c["U_rans"], c["U_les"]),
                          "model": scaled_mae(predict(c), c["U_les"])} for c in val}
for k, v in val_scores.items():
    print(f"  val {k:22s} RANS {v['rans']:.4f} -> LightGBM {v['model']:.4f}")

out = ROOT / args.out
write_submission(test, {c["name"]: predict(c) for c in test}, args.points, out)
(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "results").mkdir(exist_ok=True)
for j, m in enumerate(models):
    m.save_model(str(ROOT / f"models/lightgbm_comp{j}.txt"), num_iteration=m.best_iteration)
imp = np.sum([m.feature_importance("gain") for m in models], 0)
json.dump({"fallback": args.fallback, "val": val_scores,
           "feature_importance_gain": dict(zip(FEATURE_NAMES, imp.tolist()))},
          open(ROOT / f"results/{out.name}_val.json", "w"), indent=1)
print("wrote", out)
