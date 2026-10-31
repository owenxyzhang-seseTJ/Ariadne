"""Fourier maps with an explicit crystallographic sign and volume convention.

F(h)=integral rho(x) exp(+2 pi i h.x) dV, so rho=fftn(F_grid)/V.
Unmeasured coefficients remain zero in a displayed truncated Fourier map.
This is different from latent, updated coefficients in iterative phasing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import gemmi
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.fft import next_fast_len

from .model import Cell, Reflections, Structure
from .scattering import structure_factors


def _integral_hkl(hkl):
    raw = np.asarray(hkl, dtype=float)
    if (raw.ndim != 2 or raw.shape[1] != 3 or not len(raw)
            or not np.isfinite(raw).all() or not np.equal(raw, np.rint(raw)).all()):
        raise ValueError("Miller indices must be a nonempty finite integer N by 3 array")
    if np.max(np.abs(raw)) > 100_000:
        raise ValueError("Miller indices exceed the map resource budget")
    return raw.astype(int)


@dataclass
class DensityMap:
    values: np.ndarray
    cell: Cell
    kind: str = "electron_density"
    units: str = "e/angstrom^3"
    metadata: dict | None = None

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim != 3 or not np.isfinite(self.values).all():
            raise ValueError("Density map must be a finite three-dimensional array")
        self.metadata = dict(self.metadata or {})

    def peaks(self, n=30, min_distance=0.8, positive=True):
        """Periodic local maxima with metric-aware nonmaximum suppression."""
        if n < 1 or min_distance <= 0:
            raise ValueError("n and minimum distance must be positive")
        values = self.values if positive else -self.values
        candidates = np.argwhere((values == maximum_filter(values, size=3, mode="wrap")) & (values > 0))
        order = np.argsort(values[tuple(candidates.T)])[::-1]
        selected = []
        shape = np.asarray(values.shape)
        for j in order:
            idx = candidates[j]
            frac = idx / shape
            if any(np.linalg.norm(self.cell.to_cartesian(self.cell.minimum_image(frac - p["frac"])))
                   < min_distance for p in selected):
                continue
            selected.append({"frac": frac.tolist(), "value": float(self.values[tuple(idx)]),
                             "grid_index": idx.tolist()})
            if len(selected) == n:
                break
        return selected

    def save_npz(self, path):
        np.savez_compressed(path, values=self.values,
                            cell=[self.cell.a, self.cell.b, self.cell.c,
                                  self.cell.alpha, self.cell.beta, self.cell.gamma],
                            kind=self.kind, units=self.units,
                            metadata=json.dumps(self.metadata, allow_nan=False))

    def save_ccp4(self, path):
        """Write a complete P1 grid; all requested symmetry is already expanded."""
        grid = gemmi.FloatGrid(*self.values.shape)
        grid.set_unit_cell(gemmi.UnitCell(self.cell.a, self.cell.b, self.cell.c,
                                        self.cell.alpha, self.cell.beta, self.cell.gamma))
        grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
        np.asarray(grid)[:] = self.values.astype(np.float32)
        ccp4 = gemmi.Ccp4Map()
        ccp4.grid = grid
        ccp4.update_ccp4_header()
        ccp4.write_ccp4_map(str(path))


def grid_shape(hkl, oversampling=1.5, minimum=16, maximum_voxels=16_777_216):
    hkl = _integral_hkl(hkl)
    if hkl.ndim != 2 or hkl.shape[1] != 3 or not len(hkl):
        raise ValueError("hkl must be a nonempty N by 3 integer array")
    if not np.isfinite(oversampling) or oversampling < 1:
        raise ValueError("oversampling must be at least one")
    # Strictly more than twice the index, preventing +/- Nyquist collisions.
    shape = tuple(next_fast_len(max(minimum, int(np.ceil((2*m+1)*oversampling))))
                  for m in np.max(np.abs(hkl), axis=0))
    if np.prod(shape) > maximum_voxels:
        raise ValueError(f"Requested Fourier grid {shape} exceeds the voxel budget")
    return shape


def expand_coefficients(hkl, coefficients, structure=None, force_real=True):
    """Symmetry expansion, averaging repeated coefficients rather than summing.

    F(R^T h)=exp(-2pi i h.t) F(h). Scalar real maps additionally average
    Friedel conjugates; this does not change the observed refinement table.
    """
    hkl = _integral_hkl(hkl)
    coefficients = np.asarray(coefficients, dtype=complex)
    if coefficients.shape != (len(hkl),) or not np.isfinite(coefficients).all():
        raise ValueError("Nonfinite or mismatched map coefficients")
    buckets = {}
    operations = structure.operations if structure is not None else [gemmi.Op("x,y,z")]
    for h, c in zip(hkl, coefficients):
        for op in operations:
            hp = tuple(op.apply_to_hkl([int(i) for i in h]))
            cp = c * np.exp(1j * op.phase_shift([int(i) for i in h]))
            buckets.setdefault(hp, []).append(cp)
            if force_real:
                buckets.setdefault(tuple(-i for i in hp), []).append(cp.conjugate())
    keys = sorted(buckets)
    return np.array(keys, dtype=int), np.array([np.mean(buckets[k]) for k in keys])


def fourier_map(cell, hkl, coefficients, *, structure=None, shape=None,
                kind="electron_density", units="e/angstrom^3", f000=None):
    hk, values = expand_coefficients(hkl, coefficients, structure, force_real=True)
    shape = grid_shape(hk) if shape is None else tuple(int(s) for s in shape)
    if len(shape) != 3 or min(shape) < 2 or np.prod(shape) > 16_777_216:
        raise ValueError("Invalid or excessive Fourier grid")
    if np.any(2*np.max(np.abs(hk), axis=0) >= np.asarray(shape)):
        raise ValueError("Fourier grid is too small; signed Miller indices would alias")
    reciprocal = np.zeros(shape, dtype=complex)
    reciprocal[tuple((hk % shape).T)] = values
    if f000 is not None:
        if not np.isfinite(f000):
            raise ValueError("F000 must be finite")
        reciprocal[0, 0, 0] = f000
    rho = np.fft.fftn(reciprocal) / cell.volume
    imaginary = float(np.max(np.abs(rho.imag)))
    return DensityMap(rho.real, cell, kind, units,
                      {"coefficient_count": len(hk), "f000": f000,
                       "friedel_symmetrized_for_real_map": True,
                       "max_imaginary_roundoff": imaginary,
                       "fourier_sign": -1, "missing_coefficients": "zero",
                       "grid_shape": list(shape)})


def model_map(structure: Structure, hkl, shape=None):
    f000 = float(structure_factors(structure, np.array([[0, 0, 0]]))[0].real)
    return fourier_map(structure.cell, hkl, structure_factors(structure, hkl),
                       structure=structure, shape=shape, f000=f000,
                       kind="calculated_density")


def difference_map(structure: Structure, reflections: Reflections, scale=1.0,
                   kind="fo-fc", shape=None, include_holdout=False):
    """Conventional, unweighted model-phased diagnostic map.

    sqrt(max(I/k,0)) is used only for this map, never for F² refinement.
    No sigma-A weights, twin deconvolution, or density-modification claims.
    """
    if scale <= 0 or not np.isfinite(scale):
        raise ValueError("Intensity scale must be positive")
    if kind not in {"fo-fc", "2fo-fc"}:
        raise ValueError("kind must be fo-fc or 2fo-fc")
    use = np.ones(len(reflections), dtype=bool)
    if reflections.free_flags is not None and not include_holdout:
        use &= ~reflections.free_flags
    hk = reflections.hkl[use]
    fc = structure_factors(structure, hk)
    fo = np.sqrt(np.maximum(reflections.intensity[use]/scale, 0))
    phase = np.divide(fc, np.abs(fc), out=np.ones_like(fc), where=np.abs(fc) > 1e-14)
    coefficient = ((2 if kind == "2fo-fc" else 1)*fo - np.abs(fc)) * phase
    result = fourier_map(structure.cell, hk, coefficient, structure=structure,
                         shape=shape, kind=kind)
    result.metadata.update({"intensity_scale": scale, "holdout_included": include_holdout,
                            "weighting": "unweighted", "twinned_data_supported": False,
                            "negative_I_for_display": "amplitude zero; original intensity unchanged"})
    return result


def patterson_map(cell: Cell, reflections: Reflections, scale=1.0, structure=None, shape=None):
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be finite and positive")
    use = np.ones(len(reflections), dtype=bool)
    if reflections.free_flags is not None:
        use &= ~reflections.free_flags
    # Intensities have rotational symmetry without the translational phase
    # factors of complex F coefficients (essential for screws/glides).
    buckets = {}
    operations = structure.operations if structure is not None else [gemmi.Op("x,y,z")]
    for h, intensity in zip(reflections.hkl[use], reflections.intensity[use]/scale):
        for op in operations:
            hp = tuple(op.apply_to_hkl([int(i) for i in h]))
            buckets.setdefault(hp, []).append(float(intensity))
            buckets.setdefault(tuple(-i for i in hp), []).append(float(intensity))
    hk = np.array(sorted(buckets), dtype=int)
    intensity = np.array([np.mean(buckets[tuple(h)]) for h in hk])
    return fourier_map(cell, hk, intensity,
                       shape=shape, kind="patterson",
                       units="e^2/angstrom^3")


def load_map(path):
    with np.load(Path(path), allow_pickle=False) as z:
        return DensityMap(z["values"], Cell(*z["cell"]), str(z["kind"]),
                          str(z["units"]), json.loads(str(z["metadata"])))
