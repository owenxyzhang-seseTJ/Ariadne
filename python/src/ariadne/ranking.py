"""Conditional holdout evidence for single-lattice candidate ranking.

Candidate generation/index assignment still see all input spots. This is a
refinement holdout, not independent end-to-end cross validation. Scores are
not calibrated domain probabilities or space-group evidence.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
import uuid

import numpy as np
from scipy.special import logsumexp
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scitbx.array_family import flex


def holdout_selection(reflections, params):
    """每次候选搜索共用同一 scan-block 留出掩膜。"""
    generator = np.random.default_rng(params.seed)
    if 'imageset_id' not in reflections:
        return None  # Candidate-dependent assigned ids must not define the validation fold.
    scan = np.asarray(reflections['imageset_id'], dtype=int)
    frame = np.asarray(reflections['xyzobs.px.value'], dtype=float)[:, 2]
    block = np.floor(frame / params.holdout_block_frames).astype(int)
    keys, inverse = np.unique(np.column_stack((scan, block)), axis=0, return_inverse=True)
    if len(keys) < 2:
        return None
    chosen = generator.permutation(len(keys))[:max(1, int(len(keys) * params.holdout_fraction))]
    return np.isin(inverse, chosen)


def refinement_partition(reflections, params, validation=None):
    if validation is None:
        validation = holdout_selection(reflections, params)
    if validation is None:
        return None
    indexed = np.asarray(reflections["id"]) >= 0
    train = indexed & ~validation
    if np.count_nonzero(train) < params.min_training or np.count_nonzero(validation) < params.min_validation:
        return None
    return reflections.select(flex.bool(train)), reflections.select(flex.bool(validation))


def periodic_gaussian(A, fractional, sigma, radii, chunk_size, tail_sigma):
    """按项数选取实空间或 Fourier 表示，并独立求最近晶格点。"""
    offset = np.asarray(list(itertools.product(*[range(-r, r + 1) for r in radii])))
    lattice = offset @ A.T
    points = fractional @ A.T
    frequency_radius = np.ceil(
        max(8., tail_sigma) * np.linalg.norm(A, axis=0) / (2 * np.pi * sigma)
    ).astype(int)
    frequencies = None
    if int(np.prod(2 * frequency_radius + 1)) < len(offset):
        k = np.asarray(list(itertools.product(*[
            range(-r, r + 1) for r in frequency_radius])))
        # ±k 共轭配对，常数项为 1。
        first_nonzero = np.argmax(k != 0, axis=1)
        k = k[k[np.arange(len(k)), first_nonzero] > 0]
        reciprocal = np.linalg.solve(A.T, k.T).T
        coefficients = 2 * np.exp(-2 * np.pi**2 * sigma**2 * np.sum(reciprocal**2, axis=1))
        # 此条件同时给出密度的正下界，Fourier 求和无需截断负数。
        if coefficients.sum() < .5:
            frequencies = k
    log_ratio = np.empty(len(points))
    residual = np.empty(len(points))
    nearest_offset = np.empty((len(points), 3))
    terms = len(offset) if frequencies is None else len(frequencies)
    chunk = min(chunk_size, max(1, 64 * 1024**2 // (max(1, terms) * 8)))
    normalizer = np.log(abs(np.linalg.det(A))) - 1.5 * np.log(2 * np.pi * sigma**2)
    tree = cKDTree(lattice) if frequencies is not None else None
    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        if frequencies is None:
            r2 = cdist(points[start:stop], lattice, metric='sqeuclidean')
            indices = np.argmin(r2, axis=1)
            residual[start:stop] = np.sqrt(r2[np.arange(stop - start), indices])
            nearest_offset[start:stop] = offset[indices]
            r2 *= -.5 / sigma**2
            log_ratio[start:stop] = normalizer + logsumexp(r2, axis=1)
        else:
            distance, indices = tree.query(points[start:stop])
            residual[start:stop] = distance
            nearest_offset[start:stop] = offset[indices]
            phase = 2 * np.pi * (fractional[start:stop] @ frequencies.T)
            log_ratio[start:stop] = np.log1p(np.cos(phase) @ coefficients)
    return log_ratio, residual, nearest_offset, {
        'periodic_representation': 'real' if frequencies is None else 'fourier',
        'periodic_terms': terms, 'effective_score_chunk': chunk,
        'periodic_images': len(offset),
    }


def evidence(crystal, validation, params):
    q = np.asarray(validation["rlp"], dtype=float).reshape(-1, 3)
    if not np.all(np.isfinite(q)):
        raise ValueError("Nonfinite reciprocal coordinates in candidate evidence")
    A = np.asarray(crystal.get_A(), dtype=float).reshape(3, 3)
    f = np.linalg.solve(A, q.T).T
    nearest = np.rint(f)
    residual_fractional = f - nearest
    sigma = params.q_sigma
    fractional_sigma = sigma * np.linalg.norm(np.linalg.inv(A), axis=1)
    # The wrapped Gaussian sums enough integer images for the configured tail
    # radius in each fractional coordinate. Large cells get a larger stencil;
    # memory remains bounded by chunk size, and excessive work is explicit.
    radii = np.maximum(1, np.ceil(.5 + params.tail_sigma * fractional_sigma).astype(int))
    if np.max(radii) > params.max_periodic_radius:
        return {"usable": False, "reason": "periodic_sum_budget_exceeded",
                "required_periodic_radius_xyz": radii.tolist(),
                "next": "improve reduced basis/noise calibration or increase explicit sum budget"}
    log_ratio, residual, offset, summation = periodic_gaussian(
        A, residual_fractional, sigma, radii, params.score_chunk, params.tail_sigma
    )
    nearest += offset
    eta = params.lattice_fraction
    log_mixture = np.logaddexp(np.log(eta) + log_ratio, np.log1p(-eta))
    matched = (residual <= params.match_sigma * sigma) & np.any(nearest != 0, axis=1)
    unique = np.unique(nearest[matched], axis=0)
    rank = int(np.linalg.matrix_rank(unique - unique.mean(axis=0))) if len(unique) > 1 else 0
    usable = (int(np.count_nonzero(matched)) >= params.min_validation_indexed and rank == 3
              and float(np.mean(log_mixture)) > params.min_log_density_ratio)
    return {"usable": usable, "validation_rows": len(q), "validation_indexed": int(np.count_nonzero(matched)),
            "validation_fraction": float(np.mean(matched)), "unique_hkl": len(unique), "hkl_rank": rank,
            "mean_log_density_ratio": float(np.mean(log_mixture)),
            "residual_p95_inv_angstrom": float(np.percentile(residual, 95)),
            "q_sigma_inv_angstrom": sigma, "noise_scale_source": "configured assumption; not measured covariance",
            "A": A.tolist(), "cell": list(crystal.get_unit_cell().parameters()),
            "refinement_instrument_geometry": "beam/detector/goniometer fixed",
            "validation_semantics": "scan-block conditional refinement holdout; proposals and HKL assignments use full input",
            "probability_calibration": "none", "score_model": "bounded adaptive periodic Gaussian/uniform mixture",
            "periodic_radius_xyz": radii.tolist(),
            "tail_sigma": params.tail_sigma, **summation}


class EvidenceRank:
    def __init__(self):
        self.all_solutions = []
        self.usable_solutions = []

    def append(self, item):
        self.all_solutions.append(item)
        if item.evidence and item.evidence.get("usable"):
            self.usable_solutions.append(item)

    def extend(self, items):
        for item in items:
            self.append(item)

    def __len__(self):
        return len(self.usable_solutions)

    def best_model(self):
        return max(self.usable_solutions, key=lambda s: s.evidence["mean_log_density_ratio"])

    def __str__(self):
        return json.dumps([s.evidence for s in self.all_solutions], indent=2, allow_nan=False)


def write_trace(directory, results, attempted):
    if not directory:
        return
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    path = root / ("candidates-" + uuid.uuid4().hex + ".json")
    value = {"schema": "ariadne-candidate-evidence/1", "status": "experimental_untested",
             "attempted": attempted, "candidates": [r.evidence if r is not None else
             {"usable": False, "reason": "native_gate_refinement_failure_or_insufficient_holdout"} for r in results]}
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
