"""Turn a benchmark case into (features, local frame, targets).

Inputs  : invariant, non-dimensional features of the k-omega SST RANS solution
          (no LES information).
Target  : the velocity error  dU = U_LES - U_RANS, expressed in a local,
          flow-aligned frame and divided by a local velocity scale. That keeps
          the learning problem independent of the case's coordinate system and
          units, which is what lets a model trained on hills + ducts be applied
          to the NASA hump.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .cases import TEST
from .foam_io import latest_time_dir, read_internal_field, read_nu
from .mesh import load_mesh, ls_gradient, wall_distance

FEATURE_NAMES = [
    "tr_S2", "tr_S3", "tr_R2", "tr_R2S", "tr_R2S2",      # Pope-type invariants
    "tr_Ak2", "tr_Ap2", "tr_Ak_S_Ap", "tr_Ap2_S",       # invariants with grad k, grad p
    "Q_crit", "turb_intensity", "Re_d", "Re_d_far",
    "streamwise_dp", "strain_timescale", "visc_ratio", "dp_vs_convection",
    "k_convection_vs_production",
]
EPS = 1e-30


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + EPS)


def _skew(v: np.ndarray) -> np.ndarray:
    """Antisymmetric tensor A_ij = -eps_ijk v_k."""
    z = np.zeros(len(v))
    return -np.stack([
        np.stack([z, v[:, 2], -v[:, 1]], -1),
        np.stack([-v[:, 2], z, v[:, 0]], -1),
        np.stack([v[:, 1], -v[:, 0], z], -1),
    ], 1)


def _tr(*mats):
    out = mats[0]
    for m in mats[1:]:
        out = out @ m
    return np.trace(out, axis1=1, axis2=2)


def build_case(case_dir: Path, name: str) -> dict:
    case_dir = Path(case_dir)
    mesh = load_mesh(case_dir)
    n = mesh.n
    t = latest_time_dir(case_dir)
    U = read_internal_field(t / "U", n)
    k = np.maximum(read_internal_field(t / "k", n), 1e-12)
    om = np.maximum(read_internal_field(t / "omega", n), 1e-12)
    p = read_internal_field(t / "p", n)
    nut = read_internal_field(t / "nut", n)
    nu = read_nu(case_dir)
    d = wall_distance(mesh)

    J = ls_gradient(mesh, U, wall_value=0.0)          # dU_i/dx_j
    gk = ls_gradient(mesh, k, wall_value=0.0)
    gp = ls_gradient(mesh, p)
    gd = ls_gradient(mesh, d, wall_value=0.0)

    S = 0.5 * (J + np.transpose(J, (0, 2, 1)))
    R = 0.5 * (J - np.transpose(J, (0, 2, 1)))
    nS = np.linalg.norm(S, axis=(1, 2))
    nR = np.linalg.norm(R, axis=(1, 2))
    Sh = S / (nS + om)[:, None, None]
    Rh = R / (nR + om)[:, None, None]
    # gradients of k and p, non-dimensionalised and bounded
    Ak = _skew(gk / (np.linalg.norm(gk, axis=1) + om * np.sqrt(k) + EPS)[:, None])
    conv = np.einsum("nij,nj->ni", J, U)              # (U.grad)U
    gp_scale = np.linalg.norm(gp, axis=1) + np.linalg.norm(conv, axis=1) + EPS
    Ap = _skew(gp / gp_scale[:, None])

    Umag = np.linalg.norm(U, axis=1)
    U_dot_gp = (U * gp).sum(1)
    U_dot_gk = (U * gk).sum(1)
    prod = 2 * nut * nS**2
    sqk = np.sqrt(k)

    X = np.stack([
        _tr(Sh, Sh), _tr(Sh, Sh, Sh), _tr(Rh, Rh), _tr(Rh, Rh, Sh), _tr(Rh, Rh, Sh, Sh),
        _tr(Ak, Ak), _tr(Ap, Ap), _tr(Ak, Sh, Ap), _tr(Ap, Ap, Sh),
        (nR**2 - nS**2) / (nR**2 + nS**2 + EPS),
        k / (0.5 * Umag**2 + k),
        np.minimum(sqk * d / (50 * nu), 2.0),
        np.tanh(sqk * d / (500 * nu)),
        U_dot_gp / (np.sqrt((gp**2).sum(1) * Umag**2) + np.abs(U_dot_gp) + EPS),
        nS / (nS + om),
        nut / (nut + 100 * nu),
        np.linalg.norm(gp, axis=1) / gp_scale,
        U_dot_gk / (np.abs(U_dot_gk) + prod + EPS),
    ], 1).astype(np.float32)

    # ---- local orthonormal frame ---------------------------------------
    e1 = _unit(U)
    e2 = gd - (gd * e1).sum(1, keepdims=True) * e1          # wall-normal
    e2 = _unit(e2)
    gUm = np.einsum("nij,ni->nj", J, e1)                     # grad |U|
    e3 = gUm - (gUm * e1).sum(1, keepdims=True) * e1 - (gUm * e2).sum(1, keepdims=True) * e2
    weak = np.linalg.norm(e3, axis=1) < 1e-6 * (np.linalg.norm(gUm, axis=1) + EPS)
    e3 = _unit(e3)
    e3[weak] = _unit(np.cross(e1[weak], e2[weak]))
    frame = np.stack([e1, e2, e3], 1)                        # (n, 3, 3), rows = basis
    u_scale = np.sqrt(Umag**2 + 2 * k)

    out = dict(name=name, X=X, U_rans=U, frame=frame, u_scale=u_scale, C=mesh.C)
    # ---- target (never for test cases) ----------------------------------
    if name not in TEST:
        U_les = read_internal_field(case_dir / "0/U_LES", n)
        dU_local = np.einsum("nij,nj->ni", frame, U_les - U) / u_scale[:, None]
        out["U_les"] = U_les
        out["y"] = np.clip(dU_local, -3, 3).astype(np.float32)
    return out


def reconstruct(case: dict, y_pred: np.ndarray) -> np.ndarray:
    """Map local, scaled predictions back to a global velocity field."""
    dU = np.einsum("nji,nj->ni", case["frame"], y_pred) * case["u_scale"][:, None]
    return case["U_rans"] + dU
