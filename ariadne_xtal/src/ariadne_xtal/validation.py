"""Symmetry-orbit holdouts and data diagnostics; never delete observations."""
from __future__ import annotations

import hashlib
import json
import numpy as np

from .model import Reflections, Structure
from .scattering import calculated_intensities


def reflection_orbit(structure: Structure, hkl, twin_laws=None, max_size=4096):
    generators = [np.asarray(op.rot, dtype=float).T / op.DEN for op in structure.operations]
    for law in twin_laws or []:
        t = np.asarray(law, dtype=float)
        if t.shape != (3, 3) or not np.allclose(t, np.rint(t)):
            raise ValueError("Twin law must be an integer 3x3 reciprocal transformation")
        if not np.isclose(abs(np.linalg.det(t)), 1):
            raise ValueError("Twin law must be unimodular")
        gstar = structure.cell.reciprocal.T @ structure.cell.reciprocal
        if not np.allclose(t.T@gstar@t, gstar, rtol=1e-6, atol=1e-9):
            raise ValueError("Holdout twin law does not preserve the reciprocal metric")
        generators.extend([t, np.linalg.inv(t)])
    # Group Friedel pairs even with anomalous scattering: the pair is one split unit.
    generators.append(-np.eye(3))
    todo = [tuple(int(v) for v in hkl)]
    seen = set(todo)
    while todo:
        h = np.asarray(todo.pop())
        for g in generators:
            hp = tuple(np.rint(g@h).astype(int))
            if hp not in seen:
                seen.add(hp)
                todo.append(hp)
                if len(seen) > max_size:
                    raise ValueError("Unbounded/excessive reflection orbit; check twin laws")
    return tuple(sorted(seen))


def make_holdout(structure, reflections, fraction=0.1, seed=1729, twin_laws=None):
    if not 0 < fraction < 0.5:
        raise ValueError("Holdout fraction must be in (0, 0.5)")
    flags = np.zeros(len(reflections), bool)
    cache = {}
    groups = {}
    for n, h in enumerate(reflections.hkl):
        key = tuple(h)
        if key not in cache:
            orbit = reflection_orbit(structure, h, twin_laws)
            canonical = orbit[0]
            for member in orbit:
                cache[member] = canonical
        canonical = cache[key]
        if canonical not in groups:
            payload = json.dumps([int(seed), list(map(int, canonical))]).encode()
            u = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
            groups[canonical] = u < fraction
        flags[n] = groups[canonical]
    if not flags.any() or flags.all():
        raise ValueError("This split has an empty train/test set; provide more independent groups or a new seed")
    return Reflections(reflections.hkl.copy(), reflections.intensity.copy(), reflections.sigma.copy(),
                       None if reflections.batch is None else reflections.batch.copy(), flags)


def data_diagnostics(structure, reflections, scale=1.0, twin_laws=None, twin_fractions=None):
    pred = calculated_intensities(structure, reflections.hkl, scale, twin_laws, twin_fractions)
    z = (reflections.intensity-pred)/reflections.sigma
    indices = np.argsort(np.abs(z))[::-1][:min(25, len(z))]
    return {"n_reflections": len(reflections),
            "negative_intensities": int(np.sum(reflections.intensity < 0)),
            "held_out": int(np.sum(reflections.free_flags)) if reflections.free_flags is not None else 0,
            "residual_rms_sigma": float(np.sqrt(np.mean(z*z))),
            "outlier_candidates": [{"row": int(i), "hkl": reflections.hkl[i].tolist(),
                                     "standardized_residual": float(z[i]),
                                     "observed_intensity": float(reflections.intensity[i]),
                                     "calculated_intensity": float(pred[i])} for i in indices],
            "policy": "Diagnostics only. No reflection is removed or replaced."}
