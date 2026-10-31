"""Independent P1 charge-flipping phase retrieval and atomic peak extraction.

This is not a reimplementation of the integrated SHELXT space-group search.
The known unit-cell composition is required. It is intended for complete,
high-resolution, non-twinned X-ray data with small anomalous differences.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

import gemmi
import numpy as np

from .model import Atom, Cell, Reflections, Structure
from .maps import DensityMap, grid_shape


@dataclass
class PhasingOptions:
    trials: int = 8
    iterations: int = 800
    seed: int = 1729
    threshold_sigma: float = 1.1
    oversampling: float = 1.2
    minimum_peak_distance: float = 0.7
    sharpen: bool = True
    keep: int = 3
    algorithm: str = "charge_flip"

    def __post_init__(self):
        if any(isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), Integral)
               for name in ("trials", "iterations", "keep", "seed")):
            raise ValueError("Phasing budgets, keep and seed must be integers")
        if not 1 <= self.trials <= 256 or not 1 <= self.iterations <= 100_000:
            raise ValueError("Phasing trial/iteration budget is out of range")
        if (not np.isfinite(self.threshold_sigma) or not np.isfinite(self.minimum_peak_distance)
                or self.threshold_sigma <= 0 or self.minimum_peak_distance <= 0):
            raise ValueError("Threshold and peak distance must be positive")
        if not np.isfinite(self.oversampling) or self.oversampling < 1:
            raise ValueError("Phasing oversampling must be finite and at least one")
        if self.keep < 1 or self.keep > self.trials:
            raise ValueError("keep must be between one and trials")
        if self.algorithm not in {"charge_flip", "atomic_projection"}:
            raise ValueError("Unknown phase-retrieval algorithm")


@dataclass
class PhasingResult:
    candidates: list[dict]
    history: list[dict]
    seed: int
    assumptions: list[str] = field(default_factory=lambda: [
        "P1 unit-cell model; composition is supplied, not inferred",
        "Real-density phasing averages Friedel intensities",
        "No twin deconvolution, automatic space-group assignment or absolute-structure claim",
        "Scores are phasing diagnostics, not calibrated solution probabilities",
    ])

    def to_dict(self):
        return {"schema_version": 1, "seed": self.seed, "assumptions": self.assumptions,
                "candidates": [{k: (v.to_dict() if isinstance(v, Structure) else v)
                                for k, v in c.items()} for c in self.candidates],
                "history": self.history}


def _amplitudes(reflections, scale):
    """Merge repeated/Friedel intensities, retaining negatives until sqrt for phasing."""
    buckets = {}
    assigned_flags = {}
    for n, (h, y, sigma) in enumerate(zip(reflections.hkl, reflections.intensity, reflections.sigma)):
        key = min(tuple(h), tuple(-h))
        if reflections.free_flags is not None:
            flag = bool(reflections.free_flags[n])
            if key in assigned_flags and assigned_flags[key] != flag:
                raise ValueError("Holdout leakage: Friedel-related observations have different flags")
            assigned_flags[key] = flag
        if reflections.free_flags is not None and reflections.free_flags[n]:
            continue
        if not np.any(h):
            continue
        bucket = buckets.setdefault(key, [0., 0.])
        weight = 1/sigma**2
        bucket[0] += y*weight
        bucket[1] += weight
    if len(buckets) < 10:
        raise ValueError("P1 phase retrieval requires at least ten independent Friedel pairs")
    hkl, amp = [], []
    for h, (sy, sw) in sorted(buckets.items()):
        a = np.sqrt(max(sy/sw/scale, 0))
        hkl.extend([h, tuple(-v for v in h)])
        amp.extend([a, a])
    return np.array(hkl, dtype=int), np.array(amp)


def _subgrid_peaks(density, elements, min_distance):
    raw = density.peaks(len(elements), min_distance)
    if len(raw) < len(elements):
        return None
    values = density.values
    shape = np.array(values.shape)
    elements = sorted(elements, key=lambda x: gemmi.Element(x).atomic_number, reverse=True)
    atoms = []
    counts = {}
    for p, element in zip(raw, elements):
        idx = np.array(p["grid_index"])
        shift = np.zeros(3)
        for axis in range(3):
            prev, nxt = idx.copy(), idx.copy()
            prev[axis] = (prev[axis]-1) % shape[axis]
            nxt[axis] = (nxt[axis]+1) % shape[axis]
            a, b, c = values[tuple(prev)], values[tuple(idx)], values[tuple(nxt)]
            denom = a-2*b+c
            if denom < -1e-14:
                shift[axis] = np.clip(0.5*(a-c)/denom, -.5, .5)
        counts[element] = counts.get(element, 0)+1
        atoms.append(Atom(f"{element}{counts[element]}", element,
                          ((idx+shift)/shape) % 1, u_iso=0.02))
    return Structure(density.cell, atoms, spacegroup="P 1")


def solve_p1(cell: Cell, reflections: Reflections, elements: list[str],
             options: PhasingOptions | None = None, scale=1.0):
    options = options or PhasingOptions()
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be finite and positive")
    if not elements or len(elements) > 256:
        raise ValueError("Provide 1..256 unit-cell atom elements")
    for element in elements:
        if gemmi.Element(element).atomic_number == 0 or gemmi.Element(element).it92 is None:
            raise ValueError(f"Unknown or unsupported scattering element: {element}")
    hkl, amplitude = _amplitudes(reflections, scale)
    shape = grid_shape(hkl, oversampling=options.oversampling)
    if np.prod(shape) > 2_097_152:
        raise ValueError("Native P1 solver budget is 2,097,152 voxels")
    idx = tuple((hkl % shape).T)
    measured = np.zeros(shape, dtype=bool)
    measured[idx] = True
    spectrum = np.zeros(shape)
    s2 = cell.stol_sq(hkl)
    factors = np.array([[gemmi.Element(e).it92.calculate_sf(float(s)) for s in s2]
                        for e in elements])
    # Equalise average atomic fall-off without altering any observed data table.
    envelope = np.sqrt(np.sum(factors*factors, axis=0))
    envelope /= envelope.max()
    sharpen = np.maximum(envelope, 0.08) if options.sharpen else np.ones(len(hkl))
    target = amplitude/sharpen
    spectrum[idx] = target
    index_axes = [np.rint(np.fft.fftfreq(n)*n).astype(int) for n in shape]
    all_hkl = np.stack(np.meshgrid(*index_axes, indexing="ij"), axis=-1)
    band = cell.stol_sq(all_hkl.reshape(-1, 3)).reshape(shape) <= s2.max()*(1+1e-9)
    # The zero-order term is unmeasured and is updated by density modification.
    band[0, 0, 0] = True
    rng = np.random.default_rng(options.seed)
    candidates, history = [], []
    denom = max(float(target.sum()), 1e-14)
    for trial in range(options.trials):
        initial = rng.normal(size=shape)
        f = np.fft.ifftn(initial)
        phase = np.divide(f[idx], np.abs(f[idx]), out=np.ones(len(hkl), complex), where=np.abs(f[idx]) > 0)
        f[idx] = target*phase
        f[~band] = 0
        f[0, 0, 0] = 0
        best_score, best_f = np.inf, f.copy()
        for iteration in range(options.iterations):
            rho = np.fft.fftn(f).real
            if options.algorithm == "charge_flip":
                delta = options.threshold_sigma * rho.std()
                modified = np.where(rho < delta, -rho, rho)
            else:
                # Sparse positive voxel projection, a separately labelled alternative.
                count = min(rho.size, max(len(elements)*8, 1))
                threshold = np.partition(rho.ravel(), -count)[-count]
                modified = np.where(rho >= max(threshold, 0), rho, 0)
            g = np.fft.ifftn(modified)
            magnitude = np.abs(g[idx])
            score = float(np.sum(np.abs(magnitude-target))/denom)
            phase = np.divide(g[idx], magnitude, out=np.ones(len(hkl), complex), where=magnitude > 1e-15)
            g[idx] = target*phase
            g[~band] = 0
            f = g
            if score < best_score:
                best_score, best_f = score, f.copy()
            if iteration % 25 == 0 or iteration == options.iterations-1:
                history.append({"trial": trial, "iteration": iteration+1,
                                "amplitude_projection_residual": score,
                                "best_residual": best_score})
        # Restore the physical observed fall-off before atom extraction.
        best_f[idx] *= sharpen
        density = DensityMap(np.fft.fftn(best_f).real/cell.volume, cell, "phase_retrieval")
        model = _subgrid_peaks(density, elements, options.minimum_peak_distance)
        if model is not None:
            candidates.append({"trial": trial, "projection_residual": best_score,
                               "structure": model, "grid_shape": list(shape),
                               "algorithm": options.algorithm})
    candidates.sort(key=lambda c: c["projection_residual"])
    return PhasingResult(candidates[:options.keep], history, options.seed)
