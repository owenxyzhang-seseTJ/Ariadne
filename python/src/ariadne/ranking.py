"""Conditional holdout evidence for single-lattice candidate ranking.

Candidate generation/index assignment still see all input spots. This is a
refinement holdout, not independent end-to-end cross validation. Scores are
not calibrated domain probabilities or space-group evidence.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import uuid

import numpy as np
from .lattice_math import (
    PeriodicBudgetError, affine_integer_rank, periodic_gaussian, reduced_basis,
)


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
    from scitbx.array_family import flex
    return reflections.select(flex.bool(train)), reflections.select(flex.bool(validation))


def evidence(crystal, validation, params):
    q = np.asarray(validation["rlp"], dtype=float).reshape(-1, 3)
    if not len(q):
        return {"usable": False, "reason": "empty_validation_set"}
    if not np.all(np.isfinite(q)):
        raise ValueError("Nonfinite reciprocal coordinates in candidate evidence")
    A = np.asarray(crystal.get_A(), dtype=float).reshape(3, 3)
    scoring_basis, basis_transform = reduced_basis(A)
    f = np.linalg.solve(scoring_basis, q.T).T
    nearest = np.rint(f)
    if np.any(np.abs(nearest) > 2**52):
        raise ValueError('Miller indices exceed exact floating integer range')
    residual_fractional = f - nearest
    sigma = params.q_sigma
    if (not np.isfinite(sigma) or sigma <= 0
            or not np.isfinite(params.tail_sigma) or params.tail_sigma <= 0):
        raise ValueError("q_sigma and tail_sigma must be finite and positive")
    fractional_sigma = sigma * np.linalg.norm(np.linalg.inv(scoring_basis), axis=1)
    original_radii = [max(1, int(np.ceil(.5 + params.tail_sigma * width)))
                      for width in sigma * np.linalg.norm(np.linalg.inv(A), axis=1)]
    # The wrapped Gaussian sums enough integer images for the configured tail
    # radius in each fractional coordinate. Large cells get a larger stencil;
    # memory remains bounded by chunk size, and excessive work is explicit.
    radii = np.maximum(1, np.ceil(.5 + params.tail_sigma * fractional_sigma).astype(int))
    try:
        log_ratio, residual, offset, summation = periodic_gaussian(
            scoring_basis, residual_fractional, sigma, radii,
            params.score_chunk, params.tail_sigma, max_radius=params.max_periodic_radius
        )
    except PeriodicBudgetError as exc:
        return {"usable": False, "reason": "periodic_sum_budget_exceeded",
                "required_periodic_radius_xyz": radii.tolist(),
                "detail": str(exc),
                "next": "review noise calibration or increase explicit sum budget"}
    nearest += offset
    eta = params.lattice_fraction
    if not np.isfinite(eta) or not 0 < eta < 1:
        raise ValueError("lattice_fraction must lie strictly between zero and one")
    log_mixture = np.logaddexp(np.log(eta) + log_ratio, np.log1p(-eta))
    matched = (residual <= params.match_sigma * sigma) & np.any(nearest != 0, axis=1)
    unique = np.unique(nearest[matched], axis=0)
    rank = affine_integer_rank(unique)
    # Exact integer rank answers algebraic support.  Conditioning in physical
    # reciprocal units is a separate diagnostic, independent of HKL basis.
    if len(unique) > 1:
        support = (unique - unique[0]) @ scoring_basis.T
        spread = np.linalg.svd(support - support.mean(axis=0), compute_uv=False)
        spread = np.pad(spread, (0, 3 - len(spread)))
    else:
        spread = np.zeros(3)
    usable = (int(np.count_nonzero(matched)) >= params.min_validation_indexed and rank == 3
              and float(np.mean(log_mixture)) > params.min_log_density_ratio)
    return {"usable": usable, "validation_rows": len(q), "validation_indexed": int(np.count_nonzero(matched)),
            "validation_fraction": float(np.mean(matched)), "unique_hkl": len(unique), "hkl_rank": rank,
            "mean_log_density_ratio": float(np.mean(log_mixture)),
            "residual_p95_inv_angstrom": float(np.percentile(residual, 95)),
            "q_sigma_inv_angstrom": sigma, "noise_scale_source": "configured assumption; not measured covariance",
            "A": A.tolist(), "cell": list(crystal.get_unit_cell().parameters()),
            "scoring_basis_A": scoring_basis.tolist(),
            "scoring_basis_transform": basis_transform.tolist(),
            "hkl_rank_method": "exact integer affine rank in reduced basis",
            "support_geometry_singular_values_inv_angstrom": spread.tolist(),
            "support_geometry_axis_ratio": float(spread[-1] / spread[0]) if spread[0] > 0 else None,
            "refinement_instrument_geometry": "beam/detector/goniometer fixed",
            "validation_semantics": "scan-block conditional refinement holdout; proposals and HKL assignments use full input",
            "probability_calibration": "none", "score_model": "bounded adaptive periodic Gaussian/uniform mixture",
            "tail_sigma": params.tail_sigma, **summation,
            # Preserve legacy diagnostic coordinates while explicitly recording
            # the reduced-coordinate stencil actually evaluated.
            "periodic_radius_xyz": original_radii,
            "periodic_images": math.prod(2*r + 1 for r in original_radii),
            "scoring_periodic_radius_xyz": summation['periodic_radius_xyz'],
            "scoring_periodic_images": summation['periodic_images']}


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
