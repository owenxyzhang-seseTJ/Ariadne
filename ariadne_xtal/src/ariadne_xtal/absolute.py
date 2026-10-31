"""Explicit anomalous factors and a conditional Friedel-quotient diagnostic."""
from __future__ import annotations
import copy
import gemmi
import numpy as np

from .scattering import calculated_intensities


def with_anomalous_factors(structure, wavelength=None):
    """Return a copy with isolated-atom Cromer-Liberman f', f'' from Gemmi.

    This is opt-in. It is not an experimental near-edge spectral calibration.
    Gemmi's tabulated applicability limits still apply.
    """
    result = copy.deepcopy(structure)
    if wavelength is not None:
        result.wavelength = float(wavelength)
    if not np.isfinite(result.wavelength) or not 0.16 <= result.wavelength <= 2.67:
        raise ValueError("This adapter restricts wavelength to 0.16..2.67 angstrom")
    for atom in result.atoms:
        z = gemmi.Element(atom.element).atomic_number
        if not 3 <= z <= 92:
            raise ValueError("Cromer-Liberman adapter supports atomic numbers 3..92 only")
        atom.f_prime, atom.f_double_prime = gemmi.cromer_liberman(z, gemmi.hc/result.wavelength)
    return result


def friedel_quotient(structure, reflections, minimum_sum_significance=5.0):
    """Fit Qobs=(1-2*x)Qcalc with fixed structure and independent pair errors.

    Returned x is not clipped to [0,1]. The s.u. is conditional on the input
    model and intensity error model. It does not include structural covariance,
    common scale/absorption errors, or a Bayesian absolute-structure test.
    """
    if minimum_sum_significance < 0:
        raise ValueError("Minimum pair-sum significance must be nonnegative")
    buckets = {}
    for i, (h, y, sig) in enumerate(zip(reflections.hkl, reflections.intensity, reflections.sigma)):
        if reflections.free_flags is not None and reflections.free_flags[i]:
            continue
        key = tuple(map(int, h))
        b = buckets.setdefault(key, [0., 0.])
        b[0] += y/sig**2
        b[1] += 1/sig**2
    q_obs, q_calc, variance, pairs = [], [], [], []
    for h, (weighted_i, precision) in sorted(buckets.items()):
        hm = tuple(-v for v in h)
        if hm not in buckets or h <= hm:
            continue
        ip, sp2 = weighted_i/precision, 1/precision
        im, sm2 = buckets[hm][0]/buckets[hm][1], 1/buckets[hm][1]
        total = ip+im
        if total <= minimum_sum_significance*np.sqrt(sp2+sm2) or total <= 0:
            continue
        jp, jm = calculated_intensities(structure, np.array([h, hm]))
        if jp+jm <= 1e-14:
            continue
        v = 4*(im*im*sp2+ip*ip*sm2)/total**4
        if v <= 0:
            continue
        q_obs.append((ip-im)/total)
        q_calc.append((jp-jm)/(jp+jm))
        variance.append(v)
        pairs.append(list(h))
    if len(q_obs) < 3:
        raise ValueError("Fewer than three usable Friedel pairs")
    q_obs, q_calc, variance = map(np.asarray, (q_obs, q_calc, variance))
    information = float(np.sum(q_calc*q_calc/variance))
    if information <= 1e-12:
        raise ValueError("Model provides no identifiable anomalous-difference signal")
    slope = float(np.sum(q_calc*q_obs/variance)/information)
    return {"x": (1-slope)/2, "standard_uncertainty_conditional": 0.5/np.sqrt(information),
            "n_friedel_pairs": len(pairs), "chi_square": float(np.sum((q_obs-slope*q_calc)**2/variance)),
            "degrees_of_freedom": len(pairs)-1,
            "pair_hkl": pairs, "q_observed": q_obs.tolist(), "q_calculated": q_calc.tolist(),
            "minimum_sum_significance": minimum_sum_significance,
            "method": "weighted quotient slope, fixed input structure",
            "caveat": "conditional uncertainty; shared systematic and model errors are not included"}
