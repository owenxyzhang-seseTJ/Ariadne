"""Small-molecule crystallographic data with explicit units and conventions.

Lengths are angstroms; angles are degrees; U tensors are Cartesian angstrom².
Reciprocal vectors omit 2*pi.  Atomic occupancies are physical site fractions,
including atoms on special positions.  Observations retain negative intensities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, floor
from typing import Any

import gemmi
import numpy as np


def _vectors(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0 or arr.shape[-1] != 3 or not np.isfinite(arr).all():
        raise ValueError(f"{name} must have a finite final axis of length 3")
    return arr


@dataclass(frozen=True)
class Cell:
    """A right-handed direct basis, with basis vectors as matrix columns."""

    a: float
    b: float
    c: float
    alpha: float = 90.0
    beta: float = 90.0
    gamma: float = 90.0

    def __post_init__(self) -> None:
        for key in ("a", "b", "c", "alpha", "beta", "gamma"):
            value = float(getattr(self, key))
            if not np.isfinite(value):
                raise ValueError(f"cell {key} is not finite")
            object.__setattr__(self, key, value)
        if min(self.a, self.b, self.c) <= 0:
            raise ValueError("cell lengths must be strictly positive")
        if not all(0 < x < 180 for x in (self.alpha, self.beta, self.gamma)):
            raise ValueError("cell angles must lie strictly between 0 and 180 degrees")
        ca, cb, cg = np.cos(np.deg2rad([self.alpha, self.beta, self.gamma]))
        determinant = 1 + 2 * ca * cb * cg - ca * ca - cb * cb - cg * cg
        if determinant <= 1e-14:
            raise ValueError("cell metric is singular or not positive definite")

    @property
    def matrix(self) -> np.ndarray:
        """Orthogonalization matrix A such that r_cart = A @ x_frac."""
        ca, cb, cg = np.cos(np.deg2rad([self.alpha, self.beta, self.gamma]))
        sg = np.sin(np.deg2rad(self.gamma))
        cy = (ca - cb * cg) / sg
        return np.array(
            [[self.a, self.b * cg, self.c * cb],
             [0.0, self.b * sg, self.c * cy],
             [0.0, 0.0, self.c * np.sqrt(1.0 - cb * cb - cy * cy)]],
            dtype=float,
        )

    @property
    def reciprocal(self) -> np.ndarray:
        """Reciprocal basis A^{-T}, in inverse angstroms, without 2*pi."""
        return np.linalg.inv(self.matrix).T

    @property
    def metric(self) -> np.ndarray:
        a = self.matrix
        return a.T @ a

    @property
    def volume(self) -> float:
        return float(np.linalg.det(self.matrix))

    def to_cartesian(self, frac: Any) -> np.ndarray:
        return _vectors(frac, "fractional coordinates") @ self.matrix.T

    def to_fractional(self, cart: Any) -> np.ndarray:
        return _vectors(cart, "Cartesian coordinates") @ np.linalg.inv(self.matrix).T

    def stol_sq(self, hkl: Any) -> np.ndarray:
        q = _vectors(hkl, "Miller indices") @ self.reciprocal.T
        return np.sum(q * q, axis=-1) / 4.0

    def d_spacing(self, hkl: Any) -> np.ndarray:
        q2 = 4 * self.stol_sq(hkl)
        with np.errstate(divide="ignore"):
            return 1.0 / np.sqrt(q2)

    def minimum_image(self, delta_frac: Any) -> np.ndarray:
        """Solve the three-dimensional closest lattice-vector problem exactly.

        A branch-and-bound sphere decoder uses the triangular direct basis.
        The initial rounded image supplies a finite radius.  Each recursion
        bounds the remaining integer interval by the unused squared distance,
        so skew cells do not rely on an unjustified 27-neighbour search.
        Ties on Voronoi boundaries retain the deterministic initial/first image.
        """
        delta = _vectors(delta_frac, "fractional displacement")
        shape = delta.shape
        result = np.empty_like(delta).reshape(-1, 3)
        basis = self.matrix
        for row, vec in zip(result, delta.reshape(-1, 3)):
            # Remove a coarse translation, which also improves precision.
            vec = vec - np.rint(vec)
            best_n = np.zeros(3, dtype=np.int64)
            n = best_n.copy()
            best2 = float(np.sum((basis @ vec) ** 2))

            def visit(k: int, partial: float) -> None:
                nonlocal best_n, best2
                if k < 0:
                    if partial < best2 - 1e-13 * max(best2, 1.0):
                        best2 = partial
                        best_n = n.copy()
                    return
                rest = float(basis[k, k + 1:] @ (vec[k + 1:] - n[k + 1:]))
                center = float(vec[k] + rest / basis[k, k])
                radius = np.sqrt(max(best2 - partial, 0.0)) / abs(basis[k, k])
                low, high = ceil(center - radius - 1e-12), floor(center + radius + 1e-12)
                candidates = sorted(range(low, high + 1), key=lambda j: (abs(j - center), j))
                for candidate in candidates:
                    residual = basis[k, k] * (center - candidate)
                    new_partial = partial + residual * residual
                    if new_partial <= best2 + 1e-12 * max(best2, 1.0):
                        n[k] = candidate
                        visit(k - 1, new_partial)

            visit(2, 0.0)
            row[:] = vec - best_n
        return result.reshape(shape)

    def to_dict(self) -> dict[str, float]:
        return {name: getattr(self, name) for name in ("a", "b", "c", "alpha", "beta", "gamma")}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Cell":
        return cls(**data)


@dataclass
class Atom:
    label: str
    element: str
    frac: np.ndarray
    occupancy: float = 1.0
    u_iso: float = 0.02
    u_aniso: np.ndarray | None = None
    f_prime: float = 0.0
    f_double_prime: float = 0.0
    refine_xyz: bool = True
    refine_u: bool = True
    refine_occupancy: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("atom label must be a nonempty string")
        element = gemmi.Element(self.element)
        if (not isinstance(self.element, str) or self.element.casefold() != element.name.casefold()
                or element.atomic_number == 0 or element.it92 is None):
            raise ValueError(f"no neutral IT92 form factor for element {self.element!r}")
        self.element = element.name
        self.frac = _vectors(self.frac, "atom fractional coordinates").copy()
        if self.frac.shape != (3,):
            raise ValueError("one atom requires exactly three fractional coordinates")
        for key in ("occupancy", "u_iso", "f_prime", "f_double_prime"):
            value = float(getattr(self, key))
            if not np.isfinite(value):
                raise ValueError(f"atom {self.label}: {key} is not finite")
            setattr(self, key, value)
        if not 0 <= self.occupancy <= 1:
            raise ValueError("physical site occupancy must be between 0 and 1")
        if self.u_iso < 0:
            raise ValueError("isotropic U must be nonnegative")
        if self.u_aniso is not None:
            u = np.asarray(self.u_aniso, dtype=float)
            if u.shape != (3, 3) or not np.isfinite(u).all():
                raise ValueError("anisotropic U must be a finite Cartesian 3x3 matrix")
            if not np.allclose(u, u.T, atol=1e-12, rtol=1e-10):
                raise ValueError("anisotropic U must be symmetric")
            if np.linalg.eigvalsh(u).min() < -1e-12:
                raise ValueError("anisotropic U must be positive semidefinite")
            self.u_aniso = (u + u.T) / 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label, "element": self.element, "frac": self.frac.tolist(),
            "occupancy": float(self.occupancy), "u_iso": float(self.u_iso),
            "u_aniso": None if self.u_aniso is None else np.asarray(self.u_aniso).tolist(),
            "f_prime": float(self.f_prime), "f_double_prime": float(self.f_double_prime),
            "refine_xyz": bool(self.refine_xyz), "refine_u": bool(self.refine_u),
            "refine_occupancy": bool(self.refine_occupancy),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Atom":
        return cls(**data)


@dataclass
class Structure:
    cell: Cell
    atoms: list[Atom]
    spacegroup: str = "P 1"
    wavelength: float = 0.71073
    sym_ops: list[str] | None = None
    _operation_cache: tuple[Any, list[gemmi.Op]] | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.cell, Cell):
            raise TypeError("structure cell must be a Cell")
        self.atoms = list(self.atoms)
        if not all(isinstance(atom, Atom) for atom in self.atoms):
            raise TypeError("structure atoms must be Atom objects")
        labels = [atom.label.casefold() for atom in self.atoms]
        if len(set(labels)) != len(labels):
            raise ValueError("atom labels must be unique, including letter case")
        self.wavelength = float(self.wavelength)
        if not np.isfinite(self.wavelength) or self.wavelength <= 0:
            raise ValueError("X-ray wavelength must be finite and strictly positive")
        if self.sym_ops is not None:
            self.sym_ops = list(self.sym_ops)
        _ = self.operations

    @property
    def operations(self) -> list[gemmi.Op]:
        from .symmetry import get_operations

        key = (self.spacegroup, None if self.sym_ops is None else tuple(self.sym_ops), self.cell)
        if self._operation_cache is None or self._operation_cache[0] != key:
            ops = get_operations(self.spacegroup, self.sym_ops, self.cell)
            self._operation_cache = (key, ops)
        return self._operation_cache[1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1, "cell": self.cell.to_dict(),
            "atoms": [atom.to_dict() for atom in self.atoms],
            "spacegroup": self.spacegroup, "wavelength": self.wavelength,
            "sym_ops": self.sym_ops,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Structure":
        version = data.get("schema_version", 1)
        if version != 1:
            raise ValueError(f"unsupported structure schema version {version!r}")
        allowed = {"schema_version", "cell", "atoms", "spacegroup", "wavelength", "sym_ops"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown structure fields: {sorted(unknown)}")
        return cls(
            cell=Cell.from_dict(data["cell"]),
            atoms=[Atom.from_dict(atom) for atom in data["atoms"]],
            spacegroup=data.get("spacegroup", "P 1"),
            wavelength=data.get("wavelength", 0.71073),
            sym_ops=data.get("sym_ops"),
        )


@dataclass
class Reflections:
    hkl: np.ndarray
    intensity: np.ndarray
    sigma: np.ndarray
    batch: np.ndarray | None = None
    free_flags: np.ndarray | None = None

    def __post_init__(self) -> None:
        raw_hkl = np.asarray(self.hkl)
        if raw_hkl.ndim != 2 or raw_hkl.shape[1] != 3 or len(raw_hkl) == 0:
            raise ValueError("reflection hkl must be a nonempty N x 3 array")
        numeric_hkl = np.asarray(raw_hkl, dtype=float)
        if not np.isfinite(numeric_hkl).all() or not np.equal(numeric_hkl, np.rint(numeric_hkl)).all():
            raise ValueError("Miller indices must be finite integers")
        if np.any(np.abs(numeric_hkl) > np.iinfo(np.int32).max):
            raise ValueError("Miller indices exceed supported integer range")
        self.hkl = numeric_hkl.astype(np.int64)
        n = len(self.hkl)
        for key in ("intensity", "sigma"):
            value = np.asarray(getattr(self, key), dtype=float)
            if value.shape != (n,) or not np.isfinite(value).all():
                raise ValueError(f"{key} must be a finite length-N array")
            setattr(self, key, value.copy())
        if np.any(self.sigma <= 0):
            raise ValueError("all reflection standard uncertainties must be positive")
        if self.batch is not None:
            batch = np.asarray(self.batch, dtype=float)
            if batch.shape != (n,) or not np.isfinite(batch).all() or not np.equal(batch, np.rint(batch)).all():
                raise ValueError("reflection batch identifiers must be finite integers")
            self.batch = batch.astype(np.int64)
        if self.free_flags is not None:
            flags = np.asarray(self.free_flags)
            if flags.shape != (n,) or not np.isin(flags, [0, 1, False, True]).all():
                raise ValueError("free_flags must be a length-N boolean/0/1 array")
            self.free_flags = flags.astype(bool)

    def __len__(self) -> int:
        return len(self.hkl)

    def subset(self, mask: Any) -> "Reflections":
        idx = np.atleast_1d(np.arange(len(self))[mask])
        return Reflections(
            self.hkl[idx], self.intensity[idx], self.sigma[idx],
            None if self.batch is None else self.batch[idx],
            None if self.free_flags is None else self.free_flags[idx],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hkl": self.hkl.tolist(), "intensity": self.intensity.tolist(),
            "sigma": self.sigma.tolist(),
            "batch": None if self.batch is None else self.batch.tolist(),
            "free_flags": None if self.free_flags is None else self.free_flags.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Reflections":
        return cls(**data)
