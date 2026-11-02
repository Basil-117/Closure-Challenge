"""Model 2: a fully-connected neural network (PyTorch MLP) with the same
inputs and targets as the LightGBM model, so the two are directly comparable.

usage:
  python scripts/train_mlp.py --points ml-turbulence-benchmark/data/evaluation_points
  python scripts/train_mlp.py --points ml-turbulence-benchmark/data/evaluation_points ^
         --fallback --out submissions/mlp_ood

--fallback: in cells where any input feature lies outside the range seen in
training, the network is not trusted and the plain k-omega SST velocity is kept.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ccml.data import load_cases, scaled_mae, stack, write_submission  # noqa: E402
from ccml.features import reconstruct  # noqa: E402
from ccml.models import make_mlp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cache", default="cache")
ap.add_argument("--points", required=True)
ap.add_argument("--out", default="submissions/mlp")
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--fallback", action="store_true",
                help="keep the SST velocity where features are outside the training range")
args = ap.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.set_num_threads(2)

# ---- 1. data: identical to Model 1 --------------------------------------
train, val, test = (load_cases(args.cache, p) for p in ("train", "val", "test"))
Xtr, ytr, wtr = stack(train)
Xva, yva, wva = stack(val)
print(f"train rows {len(Xtr):,}  val rows {len(Xva):,}")

# ---- 2. scaling: neural nets need inputs/outputs of similar size ---------
mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6       # per-feature mean and spread (training data only!)
ysd = ytr.std(0) + 1e-6                        # balance the three output components
T = lambda a: torch.as_tensor(a, dtype=torch.float32)  # noqa: E731
norm = lambda X: T(np.clip((X - mu) / sd, -10, 10))   # noqa: E731  clip guards extreme inputs

# ---- 3. the network: 18 -> 128 -> 128 -> 128 -> 64 -> 3 (src/ccml/models.py)
net = make_mlp(Xtr.shape[1])
print(f"network has {sum(p.numel() for p in net.parameters()):,} trainable weights")
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)  # big steps -> small steps
loss_fn = nn.HuberLoss(reduction="none", delta=1.0)

# ---- 4. training loop ----------------------------------------------------
xt, yt, wt = norm(Xtr), T(ytr / ysd), T(wtr)
xv, yv, wv = norm(Xva), T(yva / ysd), T(wva)
mon = torch.as_tensor(np.random.default_rng(0).choice(len(xt), 200000, replace=False))  # random rows to monitor train loss
best, best_state, best_ep, history = np.inf, None, -1, []
for ep in range(args.epochs):
    net.train()
    perm = torch.randperm(len(xt))                          # shuffle every epoch
    for i in range(0, len(xt), 1024):                       # one batch of 1,024 rows
        b = perm[i:i + 1024]
        loss = (loss_fn(net(xt[b]), yt[b]).sum(1) * wt[b]).mean()   # predict + measure error
        opt.zero_grad()
        loss.backward()                                     # backpropagation
        opt.step()                                          # nudge all weights
    sched.step()
    net.eval()
    with torch.no_grad():                                   # score validation cases
        vl = float((loss_fn(net(xv), yv).sum(1) * wv).mean())
        tl = float((loss_fn(net(xt[mon]), yt[mon]).sum(1) * wt[mon]).mean())
    history.append((tl, vl))
    if vl < best:                                           # keep the best epoch
        best, best_ep = vl, ep
        best_state = {k: v.clone() for k, v in net.state_dict().items()}
    if ep % 5 == 0 or ep == args.epochs - 1:
        print(f"epoch {ep:3d}  train loss {tl:.4f}  val loss {vl:.4f}", flush=True)
net.load_state_dict(best_state)
print(f"best validation loss at epoch {best_ep}")

# ---- 5. predict, evaluate on validation, write test submission -----------
lo, hi = Xtr.min(axis=0), Xtr.max(axis=0)


def predict(case):
    with torch.no_grad():
        y = net(norm(case["X"])).numpy() * ysd
    if args.fallback:
        outside = ((case["X"] < lo) | (case["X"] > hi)).any(axis=1)
        y[outside] = 0.0                                    # no correction -> keep SST
    return reconstruct(case, y)


val_scores = {c["name"]: {"rans": scaled_mae(c["U_rans"], c["U_les"]),
                          "model": scaled_mae(predict(c), c["U_les"])} for c in val}
for k, v in val_scores.items():
    print(f"  val {k:22s} RANS {v['rans']:.4f} -> MLP {v['model']:.4f}")

out = ROOT / args.out
write_submission(test, {c["name"]: predict(c) for c in test}, args.points, out)
(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "results").mkdir(exist_ok=True)
torch.save({"state_dict": net.state_dict(), "mu": mu, "sd": sd, "ysd": ysd}, ROOT / "models/mlp.pt")
json.dump({"fallback": args.fallback, "best_epoch": best_ep, "val": val_scores, "history": history},
          open(ROOT / f"results/{out.name}_val.json", "w"), indent=1)
print("wrote", out)
