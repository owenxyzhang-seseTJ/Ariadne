"""Direct kinematic X-ray structure factors and incoherent domain mixtures.

The implementation is independent Python/NumPy; Gemmi supplies published IT92
neutral-atom coefficients and symmetry tables.  No empirical absorption,
extinction, neutron, electron-diffraction, or solvent correction is implicit.
"""

from __future__ import annotations

from typing import Any

import gemmi
import numpy as np

from .model import Structure
from .symmetry import unique_sites


def _indices(hkl: Any) -> tuple[np.ndarray, bool]:
    raw = np.asarray(hkl, dtype=float)
    one = raw.ndim == 1
    values = np.atleast_2d(raw)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Miller indices must be a finite N x 3 array")
    if not np.equal(values, np.rint(values)).all():
        raise ValueError("structure factors require integral Miller indices")
    if np.any(np.abs(values) > np.iinfo(np.int32).max):
        raise ValueError("Miller indices exceed supported integer range")
    return values.astype(np.int64), one


def atomic_form_factor(element: str, stol_sq: Any) -> np.ndarray:
    """Neutral IT92 f0(s) = sum a_i exp(-b_i s²) + c, in electrons.

    Tabulated coefficients are used without renormalizing f0(0) to Z.
    The optional anomalous corrections are added by structure_factors.
    """
    s2 = np.asarray(stol_sq, dtype=float)
    if not np.isfinite(s2).all() or np.any(s2 < 0):
        raise ValueError("squared sin(theta)/lambda must be finite and nonnegative")
    elem = gemmi.Element(element)
    coef = elem.it92
    if (not isinstance(element, str) or element.casefold() != elem.name.casefold()
            or elem.atomic_number == 0 or coef is None):
        raise ValueError(f"no IT92 coefficients for {element!r}")
    a, b = np.asarray(coef.a, dtype=float), np.asarray(coef.b, dtype=float)
    return np.sum(a * np.exp(-s2[..., None] * b), axis=-1) + float(coef.c)


def structure_factors(structure: Structure, hkl: Any) -> np.ndarray:
    """Compute F(h)=sum_j,o occ_j f_j(s) DW_j,o(h) exp(2*pi*i*h*x_j,o).

    Symmetry-equivalent sites are summed once each.  Cartesian ADPs rotate
    with each symmetry operation; stabilizer averaging is applied at special
    positions.  f' and f'' are explicit atom fields and default to zero.
    """
    indices, one = _indices(hkl)
    q = indices @ structure.cell.reciprocal.T
    s2 = np.sum(q * q, axis=1) / 4
    form_factors: dict[str, np.ndarray] = {}
    result = np.zeros(len(indices), dtype=complex)
    for atom in structure.atoms:
        # Atoms are mutable during optimization: check physical values here too.
        if not np.isfinite(atom.occupancy) or not 0 <= atom.occupancy <= 1:
            raise ValueError(f"invalid physical occupancy for {atom.label}")
        if atom.u_aniso is None:
            if not np.isfinite(atom.u_iso) or atom.u_iso < 0:
                raise ValueError(f"invalid isotropic displacement for {atom.label}")
        else:
            u = np.asarray(atom.u_aniso)
            if u.shape != (3, 3) or not np.isfinite(u).all() or not np.allclose(u, u.T, atol=1e-11):
                raise ValueError(f"invalid anisotropic displacement for {atom.label}")
            if np.linalg.eigvalsh(u).min() < -1e-10:
                raise ValueError(f"non-positive-semidefinite displacement tensor for {atom.label}")
        if not np.isfinite(atom.frac).all() or not np.isfinite([atom.f_prime, atom.f_double_prime]).all():
            raise ValueError(f"nonfinite atomic parameter for {atom.label}")
        if atom.element not in form_factors:
            form_factors[atom.element] = atomic_form_factor(atom.element, s2)
        scattering = form_factors[atom.element] + atom.f_prime + 1j * atom.f_double_prime
        atom_sum = np.zeros(len(indices), dtype=complex)
        for position, u_cart in unique_sites(structure, atom):
            exponent = -2 * np.pi**2 * np.einsum("ni,ij,nj->n", q, u_cart, q)
            atom_sum += np.exp(exponent + 2j * np.pi * (indices @ position))
        result += atom.occupancy * scattering * atom_sum
    return result[0] if one else result


def validate_twin_laws(twin_laws: Any, twin_fractions: Any,
                       cell: Any | None = None) -> tuple[list[np.ndarray], np.ndarray]:
    """Validate merohedral law convention: implicit identity + nonidentity laws.

    T acts on reciprocal column indices: h' = T h (rows use h @ T.T).
    Laws must preserve reciprocal metric when a cell is supplied.  This model
    describes fully overlapped reflections, not general non-merohedral HKLF5.
    """
    nonidentity = [] if twin_laws is None else list(twin_laws)
    laws = [np.eye(3, dtype=np.int64)]
    for raw in nonidentity:
        law = np.asarray(raw, dtype=float)
        if law.shape != (3, 3) or not np.isfinite(law).all() or not np.equal(law, np.rint(law)).all():
            raise ValueError("twin law must be an integer 3x3 reciprocal-index matrix")
        law = law.astype(np.int64)
        if not np.isclose(abs(np.linalg.det(law)), 1.0, atol=1e-10):
            raise ValueError("twin law must be unimodular")
        if any(np.array_equal(law, previous) for previous in laws):
            raise ValueError("twin_laws must contain distinct nonidentity laws")
        if cell is not None:
            reciprocal_metric = cell.reciprocal.T @ cell.reciprocal
            if not np.allclose(law.T @ reciprocal_metric @ law, reciprocal_metric, rtol=2e-6,
                               atol=2e-6 * max(np.max(np.abs(reciprocal_metric)), 1e-12)):
                raise ValueError("native twin law must preserve the reciprocal metric (merohedral model)")
        laws.append(law)
    if twin_fractions is None:
        if len(laws) != 1:
            raise ValueError("twin fractions, including identity, are required for twin laws")
        fractions = np.array([1.0])
    else:
        fractions = np.asarray(twin_fractions, dtype=float)
        if fractions.shape != (len(laws),) or not np.isfinite(fractions).all() or np.any(fractions < 0):
            raise ValueError("twin fractions must be finite, nonnegative, and include identity")
        if not np.isclose(fractions.sum(), 1.0, atol=1e-10, rtol=0):
            raise ValueError("twin fractions must sum to one; no implicit renormalization")
    return laws, fractions


def calculated_intensities(structure: Structure, hkl: Any, scale: float = 1.0,
                           twin_laws: Any = None, twin_fractions: Any = None) -> np.ndarray:
    """Intensity-scale times the incoherent sum of squared domain amplitudes."""
    scale = float(scale)
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("intensity scale must be finite and nonnegative")
    indices, one = _indices(hkl)
    laws, fractions = validate_twin_laws(twin_laws, twin_fractions, structure.cell)
    result = np.zeros(len(indices), dtype=float)
    for law, fraction in zip(laws, fractions):
        if fraction:
            result += fraction * np.abs(structure_factors(structure, indices @ law.T)) ** 2
    result *= scale
    return result[0] if one else result


def intensity_derivative(f_calc: Any, df_dp: Any) -> np.ndarray:
    """d|F|²/dp = 2 Re(conj(F) dF/dp), useful for analytical adapters."""
    return 2 * np.real(np.conjugate(f_calc) * df_dp)
