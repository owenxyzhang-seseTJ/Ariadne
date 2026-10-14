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
from scitbx.array_family import flex


def refinement_partition(reflections, params):
    n = len(reflections)
    # Candidate tables in one ranking call keep common original row order.
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
    validation = np.isin(inverse, chosen)
    indexed = np.asarray(reflections["id"]) >= 0
    train = indexed & ~validation
    if np.count_nonzero(train) < params.min_training or np.count_nonzero(validation) < params.min_validation:
        return None
    return reflections.select(flex.bool(train)), reflections.select(flex.bool(validation))


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
    offset = np.asarray(list(itertools.product(*[range(-r, r + 1) for r in radii])))
    residual = np.empty(len(q))
    log_density = np.empty(len(q))
    chunk = min(params.score_chunk, max(1, 64 * 1024**2 // (len(offset) * 48)))
    for start in range(0, len(q), chunk):
        stop = min(start + chunk, len(q))
        delta = (residual_fractional[start:stop, None, :] - offset[None, :, :]) @ A.T
        r2 = np.sum(delta**2, axis=2)
        nearest[start:stop] += offset[np.argmin(r2, axis=1)]
        residual[start:stop] = np.sqrt(np.min(r2, axis=1))
        log_density[start:stop] = logsumexp(-.5 * r2 / sigma**2, axis=1)
    det = abs(np.linalg.det(A))
    # Density ratio on the lattice fundamental cell. Cell-volume normalization
    # penalizes chance fits from an excessively dense candidate reciprocal lattice.
    log_ratio = np.log(det) - 1.5 * np.log(2 * np.pi * sigma**2) + log_density
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
            "periodic_radius_xyz": radii.tolist(), "periodic_images": len(offset),
            "tail_sigma": params.tail_sigma, "effective_score_chunk": chunk}


class EvidenceRank:
    def __init__(self):
        self.all_solutions = []
        self.usable_solutions = []

    def append(self, item):
        self.all_solutions.append(item)
        if item.evidence and item.evidence.get("usable"):
            self.usable_solutions.append(item)

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
