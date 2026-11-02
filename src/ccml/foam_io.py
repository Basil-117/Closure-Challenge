"""Minimal, dependency-light readers for the OpenFOAM ASCII files in the
Closure Challenge benchmark repository.

The benchmark mixes a few file flavours:
  * normal OpenFOAM fields with a FoamFile header      (periodic hills, hump)
  * header-less dumps such as ``U_LES nonuniform ...``  (square duct)
  * fields whose internalField is ``#include``-d       (curved backward step)
This module hides those differences behind :func:`read_internal_field`.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_NUM_RE = re.compile(_NUM)


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _parse_list(text: str, start: int, ncomp: int) -> np.ndarray:
    """Parse ``N ( ... )`` beginning at ``start`` (index of the count)."""
    m = re.compile(r"(\d+)\s*\(").match(text, start)
    n = int(m.group(1))
    body_start = m.end()
    # Find the matching closing parenthesis of the outer list.
    depth, i = 1, body_start
    while depth:
        c = text[i]
        depth += c == "("
        depth -= c == ")"
        i += 1
    vals = np.array(_NUM_RE.findall(text[body_start : i - 1]), dtype=float)
    if vals.size != n * ncomp:
        raise ValueError(f"expected {n}x{ncomp} values, parsed {vals.size}")
    return vals.reshape(n, ncomp) if ncomp > 1 else vals


def _ncomp(text: str) -> int:
    if re.search(r"List<\s*symmTensor\s*>|volSymmTensorField", text):
        return 6
    if re.search(r"List<\s*tensor\s*>|volTensorField", text):
        return 9
    if re.search(r"List<\s*vector\s*>|volVectorField", text):
        return 3
    return 1


def read_internal_field(path: str | Path, n_cells: int | None = None) -> np.ndarray:
    """Return the internal (cell-centred) values of an OpenFOAM field file."""
    path = Path(path)
    raw = path.read_text()
    text = _strip_comments(raw)
    ncomp = _ncomp(text)

    # Case 1: internalField pulled in through an #include + $variable.
    inc = re.search(r'#include\s+"([^"]+)"', text)
    if inc and re.search(r"internalField\s+\$", text):
        inc_text = _strip_comments((path.parent / inc.group(1)).read_text())
        ncomp = max(ncomp, _ncomp(inc_text))
        m = re.search(r"nonuniform\s+List<\s*\w+\s*>\s*", inc_text)
        return _parse_list(inc_text, m.end(), ncomp)

    # Case 2: header-less dump: "<name> nonuniform List<...>\nN\n(...)".
    m = re.search(r"(?:internalField|^\w+)\s+nonuniform\s+List<\s*\w+\s*>\s*", text, re.M)
    if m:
        return _parse_list(text, m.end(), ncomp)

    # Case 3: uniform value.
    m = re.search(r"internalField\s+uniform\s+(\([^)]*\)|" + _NUM + ")", text)
    if m:
        v = np.array(_NUM_RE.findall(m.group(1)), dtype=float)
        if n_cells is None:
            raise ValueError(f"{path} is uniform; pass n_cells")
        return np.tile(v, (n_cells, 1)) if v.size > 1 else np.full(n_cells, v[0])
    raise ValueError(f"could not parse internalField in {path}")


def read_plain_list(path: str | Path, ncomp: int) -> np.ndarray:
    """Read a polyMesh list file (points / owner / neighbour)."""
    text = _strip_comments(Path(path).read_text())
    text = re.sub(r"FoamFile\s*\{.*?\}", "", text, flags=re.S)
    m = re.search(r"(\d+)\s*\(", text)
    return _parse_list(text, m.start(), ncomp)


def read_faces(path: str | Path) -> list[np.ndarray]:
    text = _strip_comments(Path(path).read_text())
    text = re.sub(r"FoamFile\s*\{.*?\}", "", text, flags=re.S)
    m = re.search(r"(\d+)\s*\(", text)
    n = int(m.group(1))
    faces = re.findall(r"\d+\s*\(([\d\s]+)\)", text[m.end():])
    assert len(faces) == n, (len(faces), n)
    return [np.array(f.split(), dtype=int) for f in faces]


def read_boundary(path: str | Path) -> dict[str, dict]:
    text = _strip_comments(Path(path).read_text())
    text = re.sub(r"FoamFile\s*\{.*?\}", "", text, flags=re.S)
    out = {}
    for name, body in re.findall(r"(\w+)\s*\{([^}]*)\}", text):
        d = dict(re.findall(r"(\w+)\s+([^;]+);", body))
        out[name] = {
            "type": d["type"].strip(),
            "nFaces": int(d["nFaces"]),
            "startFace": int(d["startFace"]),
        }
    return out


def read_nu(case: Path) -> float:
    """Kinematic viscosity from transportProperties (or the duct caseDef)."""
    tp = _strip_comments((case / "constant/transportProperties").read_text())
    m = re.search(r"^\s*nu\s+(?:nu\s+)?\[[^\]]*\]\s+(" + _NUM + ")", tp, re.M)
    if m:
        return float(m.group(1))
    cd = _strip_comments((case / "caseDef").read_text())
    return float(re.search(r"^\s*nu\s+(" + _NUM + ")", cd, re.M).group(1))


def latest_time_dir(case: Path) -> Path:
    times = [p for p in case.iterdir() if p.is_dir() and re.fullmatch(r"\d+(\.\d+)?", p.name)]
    times = [p for p in times if float(p.name) > 0]
    return max(times, key=lambda p: float(p.name))
