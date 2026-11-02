"""Mesh utilities: cell centres, least-squares gradients and wall distance.

The benchmark gives RANS velocity, k, omega, p ... but no gradients of the
RANS fields, so we compute them ourselves from the OpenFOAM polyMesh.
All benchmark meshes are effectively 2-D (one cell thick for the hills /
hump / step, one cell long in x for the duct cross-section), so gradients
are solved only in the two active directions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .foam_io import read_boundary, read_faces, read_internal_field, read_plain_list


@dataclass
class Mesh:
    C: np.ndarray            # (n, 3) cell centres
    owner: np.ndarray        # (n_faces,)
    neighbour: np.ndarray    # (n_internal,)
    wall_owner: np.ndarray   # cells adjacent to wall faces
    wall_Cf: np.ndarray      # (n_wall, 3) wall face centres
    wall_pts: np.ndarray     # wall vertices (denser sampling of the wall)
    active: np.ndarray       # indices of the 2 (or 3) active directions

    @property
    def n(self) -> int:
        return len(self.C)


def load_mesh(case: Path) -> Mesh:
    pm = case / "constant/polyMesh"
    cfile = case / "0/C" if (case / "0/C").exists() else case / "constant/C"
    C = read_internal_field(cfile)
    owner = read_plain_list(pm / "owner", 1).astype(int)
    neighbour = read_plain_list(pm / "neighbour", 1).astype(int)
    points = read_plain_list(pm / "points", 3)
    faces = read_faces(pm / "faces")

    w_owner, w_cf, w_pts = [], [], []
    for patch in read_boundary(pm / "boundary").values():
        if patch["type"] != "wall":
            continue
        s, nf = patch["startFace"], patch["nFaces"]
        for f in range(s, s + nf):
            w_owner.append(owner[f])
            w_cf.append(points[faces[f]].mean(0))
            w_pts.append(points[faces[f]])
    wall_pts = np.unique(np.concatenate(w_pts), axis=0)

    span = C.max(0) - C.min(0)
    active = np.where(span > 1e-6 * span.max())[0]
    return Mesh(C, owner, neighbour, np.array(w_owner), np.array(w_cf), wall_pts, active)


def ls_gradient(mesh: Mesh, phi: np.ndarray, wall_value: float | np.ndarray | None = None) -> np.ndarray:
    """Weighted least-squares cell gradient.

    phi: (n,) or (n, m). Returns (n, m, 3) with grad[i, a, b] = d phi_a / d x_b.
    If ``wall_value`` is given, wall faces act as extra neighbours with that
    value (e.g. 0 for velocity / k with no-slip walls).
    """
    scalar = phi.ndim == 1
    phi = phi[:, None] if scalar else phi
    n, m = phi.shape
    act = mesh.active
    d = len(act)
    C = mesh.C[:, act]

    own = mesh.owner[: len(mesh.neighbour)]
    nei = mesh.neighbour
    src = [own, nei]
    dx = [C[nei] - C[own], C[own] - C[nei]]
    dphi = [phi[nei] - phi[own], phi[own] - phi[nei]]
    if wall_value is not None and len(mesh.wall_owner):
        wo = mesh.wall_owner
        src.append(wo)
        dx.append(mesh.wall_Cf[:, act] - C[wo])
        dphi.append(np.broadcast_to(wall_value, (len(wo), m)) - phi[wo])
    src = np.concatenate(src)
    dx = np.concatenate(dx)
    dphi = np.concatenate(dphi)

    w = 1.0 / np.maximum((dx**2).sum(1), 1e-300)
    A = np.zeros((n, d, d))
    B = np.zeros((n, d, m))
    np.add.at(A, src, w[:, None, None] * dx[:, :, None] * dx[:, None, :])
    np.add.at(B, src, w[:, None, None] * dx[:, :, None] * dphi[:, None, :])
    A += 1e-12 * np.trace(A, axis1=1, axis2=2)[:, None, None] * np.eye(d)
    g = np.linalg.solve(A, B)  # (n, d, m)

    grad = np.zeros((n, m, 3))
    grad[:, :, act] = np.transpose(g, (0, 2, 1))
    return grad[:, 0, :] if scalar else grad


def wall_distance(mesh: Mesh) -> np.ndarray:
    """Distance from each cell centre to the nearest wall (face centres + vertices)."""
    pts = np.vstack([mesh.wall_Cf, mesh.wall_pts])
    act = mesh.active
    tree = cKDTree(pts[:, act])
    d, _ = tree.query(mesh.C[:, act])
    return d
