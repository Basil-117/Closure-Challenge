"""Model definitions + loaders shared by training, evaluation and plotting."""
from pathlib import Path

import numpy as np


def make_mlp(n_in: int):
    from torch import nn
    return nn.Sequential(
        nn.Linear(n_in, 128), nn.SiLU(),
        nn.Linear(128, 128), nn.SiLU(),
        nn.Linear(128, 128), nn.SiLU(),
        nn.Linear(128, 64), nn.SiLU(),
        nn.Linear(64, 3),
    )


def load_lightgbm(model_dir: Path):
    import lightgbm as lgb
    ms = [lgb.Booster(model_file=str(Path(model_dir) / f"lightgbm_comp{j}.txt")) for j in range(3)]
    return lambda X: np.stack([m.predict(X) for m in ms], 1)


def load_mlp(model_dir: Path):
    import torch
    ck = torch.load(Path(model_dir) / "mlp.pt", weights_only=False)
    net = make_mlp(len(ck["mu"]))
    net.load_state_dict(ck["state_dict"])
    net.eval()

    def predict(X):
        x = torch.as_tensor(np.clip((X - ck["mu"]) / ck["sd"], -10, 10), dtype=torch.float32)
        with torch.no_grad():
            return net(x).numpy() * ck["ysd"]
    return predict
