"""Loading the cached dataset, the benchmark metric, and writing submissions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .cases import TEST


def load_cases(cache: str | Path, part: str) -> list[dict]:
    out = []
    for f in sorted(Path(cache).glob("*.npz")):
        z = np.load(f, allow_pickle=False)
        if str(z["split"]) == part:
            out.append({k: z[k] for k in z.files})
            out[-1]["name"] = str(z["name"])
    return out


def stack(cases: list[dict]):
    """Concatenate cases; per-sample weights give every case equal total weight."""
    X = np.concatenate([c["X"] for c in cases])
    y = np.concatenate([c["y"] for c in cases])
    w = np.concatenate([np.full(len(c["X"]), 1.0 / len(c["X"])) for c in cases])
    return X, y, (w / w.mean()).astype(np.float32)


def scaled_mae(U_pred: np.ndarray, U_true: np.ndarray) -> float:
    """The Closure Challenge metric: mean |U_pred - U_true| / mean |U_true|."""
    return float(np.mean(np.linalg.norm(U_pred - U_true, axis=-1))
                 / np.mean(np.linalg.norm(U_true, axis=-1)))


def to_eval_points(case: dict, U_cells: np.ndarray, points_dir: str | Path) -> np.ndarray:
    """Pick the cell values at the official evaluation points (they are cell centres)."""
    pts = np.loadtxt(Path(points_dir) / f"{case['name']}_points.csv", delimiter=",")
    dist, idx = cKDTree(case["C"]).query(pts)
    scale = np.ptp(case["C"], axis=0).max()
    assert dist.max() < 1e-6 * scale, f"evaluation points are not cell centres ({dist.max()})"
    return U_cells[idx]


def write_submission(test_cases: list[dict], U_by_case: dict, points_dir, out_dir) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in test_cases:
        assert c["name"] in TEST
        np.savetxt(out_dir / f"{c['name']}.csv", to_eval_points(c, U_by_case[c["name"]], points_dir),
                   delimiter=",")
    return out_dir
