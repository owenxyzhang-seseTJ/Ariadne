"""Exclusive single-source centroid EM and scan-block predictive comparison.

All Gaussian components have fixed equal prior mass within an observed scan.
The initial *complete observable* DIALS prediction identities are frozen; their
positions are recalculated at every objective evaluation. There is no nearest
match assignment, q gate, responsibility cutoff, or local-density renormalizing.
This is a local, static-crystal model, NOT a model of composite-spot centroids.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.special import logsumexp
from scipy.stats import t as student_t


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Budget:
    max_calls: int = 500
    max_seconds: float = 300.
    calls: int = 0
    started: float = 0.

    def __post_init__(self):
        self.started = time.monotonic()

    def check(self, prediction=False):
        if time.monotonic() - self.started > self.max_seconds:
            raise BudgetExceeded('wall-time budget exceeded')
        if prediction:
            if self.calls >= self.max_calls:
                raise BudgetExceeded('prediction-call budget exceeded')
            self.calls += 1


def scan_partition(observed_ids, frames, source_rows, block_frames=20,
                   holdout_fraction=.2, seed=42):
    """Partition raw rows before fitting, without using centroids or matches."""
    if block_frames < 1 or not 0 < holdout_fraction < .5:
        raise ValueError('Invalid scan-block partition settings')
    source_rows = np.asarray(source_rows, dtype=int)
    if len(np.unique(source_rows)) != len(source_rows):
        raise ValueError('Raw source row identifiers must be unique')
    keys = [(int(e), int(np.floor(f / block_frames)))
            for e, f in zip(observed_ids, frames)]
    blocks = sorted(set(keys))
    if len(blocks) < 2:
        raise ValueError('Insufficient scan blocks; leave the model unresolved')
    order = np.random.default_rng(seed).permutation(len(blocks))
    heldout = {blocks[i] for i in order[:max(1, int(len(blocks) * holdout_fraction))]}
    test = np.array([k in heldout for k in keys])
    return {'training_source_rows': source_rows[~test].tolist(),
            'validation_source_rows': source_rows[test].tolist(),
            'heldout_blocks': [list(b) for b in sorted(heldout)],
            'block_frames': block_frames, 'holdout_fraction': holdout_fraction,
            'seed': seed}, test, keys


def covariance_factors(covariance):
    covariance = np.asarray(covariance, dtype=float)
    if (covariance.ndim != 3 or covariance.shape[1:] != (3, 3)
            or not np.all(np.isfinite(covariance))
            or not np.allclose(covariance, covariance.transpose(0, 2, 1))):
        raise ValueError('Require finite symmetric full centroid covariance')
    try:
        chol = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Centroid plus model covariance must be positive definite') from exc
    whitening = np.linalg.inv(chol)
    log_normalizer = -1.5 * np.log(2 * np.pi) - np.log(
        np.diagonal(chol, axis1=1, axis2=2)).sum(axis=1)
    return whitening, log_normalizer


def expectation(y, means, whitening, log_normalizer, allowed, counts,
                outlier_density, outlier_fraction):
    """Return normalized responsibilities and complete row log densities.

    `counts` includes ALL components in the row's scan, even those on other
    panels. Panel is a discrete measured coordinate. Gaussian support is R^3
    in native mm/mm/unwrapped-rad, and the outlier is uniform over the declared
    detector/scan volume. These are densities, not best-match similarities.
    """
    if not 0 < outlier_fraction < 1:
        raise ValueError('Outlier fraction must be strictly between zero and one')
    counts = np.asarray(counts, dtype=float)
    outlier_density = np.asarray(outlier_density, dtype=float)
    if np.any(counts <= 0) or np.any(outlier_density <= 0):
        raise ValueError('Need nonempty complete prediction sets and positive outlier density')
    residual = np.einsum('iab,ijb->ija', whitening, y[:, None, :] - means[None, :, :])
    log_components = (log_normalizer[:, None] - .5 * np.sum(residual**2, axis=2)
                      + np.log1p(-outlier_fraction) - np.log(counts)[:, None])
    log_components = np.where(allowed, log_components, -np.inf)
    log_outlier = np.log(outlier_fraction) + np.log(outlier_density)
    log_density = np.logaddexp(logsumexp(log_components, axis=1), log_outlier)
    responsibilities = np.exp(log_components - log_density[:, None])
    outlier = np.exp(log_outlier - log_density)
    return responsibilities, outlier, log_density


def weighted_targets(y, whitening, responsibilities):
    """Exact sufficient statistics for the M-step Gaussian quadratic.

    Sum_i r_ij (y_i - mu_j)' C_i^-1 (y_i - mu_j) differs from
    (mu_j - target_j)' precision_j (mu_j - target_j) by a constant.
    This avoids a huge (observations * predictions * parameters) Jacobian.
    Only exactly zero posterior mass is omitted, with no probability threshold.
    """
    precision = np.einsum('iab,iac->ibc', whitening, whitening)
    active = np.flatnonzero(responsibilities.sum(axis=0) > 0)
    r = responsibilities[:, active]
    combined = np.einsum('ij,iab->jab', r, precision)
    rhs = np.einsum('ij,iab,ib->ja', r, precision, y)
    targets = np.linalg.solve(combined, rhs[..., None])[..., 0]
    factors = np.linalg.cholesky(combined).transpose(0, 2, 1)
    return active, factors, targets


def fit_em(adapter, y, covariance, allowed, counts, outlier_density, *,
           outlier_fraction=.05, em_iterations=8, max_nfev=20,
           min_support=25, tolerance=1e-6):
    """Generalized EM: accept only nondecreasing observed training likelihood.

    One row contributes a *single expected complete-data objective*. Its latent
    alternatives are weighted by responsibilities summing to at most one. The
    full covariance is whitened outside DIALS, not passed as off-diagonal native
    DIALS refinement weights. Priors, noise and component identities stay fixed.
    """
    whitening, lognorm = covariance_factors(covariance)
    x = adapter.x0.copy()
    means = adapter.predict(x)
    resp, outlier, ll = expectation(y, means, whitening, lognorm, allowed,
                                   counts, outlier_density, outlier_fraction)
    history = [float(ll.sum())]
    status, reason, jacobian_rank = 'unresolved', 'iteration_limit', None
    final_jac = None
    optimizer_history = []
    for _ in range(em_iterations):
        old_x, old_means = x.copy(), means.copy()
        try:
            active, factors, targets = weighted_targets(y, whitening, resp)
            if not len(active):
                reason = 'no effective Gaussian support'
                break

            def objective(values):
                pred = adapter.predict(values)[active]
                return np.einsum('jab,jb->ja', factors, pred - targets).ravel()

            result = least_squares(objective, x, bounds=adapter.bounds,
                                   x_scale='jac', max_nfev=max_nfev,
                                   ftol=tolerance, xtol=tolerance, gtol=tolerance)
            trial_means = adapter.predict(result.x)
            trial = expectation(y, trial_means, whitening, lognorm, allowed,
                                counts, outlier_density, outlier_fraction)
        except (BudgetExceeded, RuntimeError, ValueError) as exc:
            adapter.set_values(old_x)
            reason = str(exc)
            break
        optimizer_ok = bool(result.success and np.isfinite(result.optimality))
        optimizer_history.append({'success': bool(result.success),
                                  'status': int(result.status),
                                  'optimality': float(result.optimality) if np.isfinite(result.optimality) else None,
                                  'nfev': int(result.nfev), 'message': str(result.message)})
        gain = float(trial[2].sum() - history[-1])
        if gain < 0:
            adapter.set_values(old_x)
            means = old_means
            reason = 'M step rejected: training likelihood decreased'
            break
        x, means = result.x.copy(), trial_means
        resp, outlier, ll = trial
        history.append(float(ll.sum()))
        final_jac = result.jac
        if not optimizer_ok:
            status, reason = 'unresolved', 'M step optimizer limit or failure: ' + str(result.message)
            break
        if gain <= tolerance * (1 + abs(history[-2])):
            status, reason = 'converged', 'training likelihood tolerance'
            break
    adapter.set_values(x)
    support = np.bincount(adapter.component_crystals, weights=resp.sum(axis=0),
                          minlength=adapter.n_crystals)
    if final_jac is not None:
        jacobian_rank = int(np.linalg.matrix_rank(final_jac))
    supported = bool(np.all(support >= min_support))
    identifiable = jacobian_rank == len(x)
    at_bound = bool(np.any(np.minimum(x - adapter.bounds[0], adapter.bounds[1] - x)
                                  <= 1e-5 * (adapter.bounds[1] - adapter.bounds[0])))
    if at_bound:
        status, reason = 'unresolved', 'local parameter bound reached; prediction support requires review'
    if not supported or not identifiable:
        status = 'unresolved'
        reason = 'insufficient effective crystal support or rank-deficient geometry'
    return {'values': x, 'means': means, 'responsibilities': resp, 'outlier': outlier,
            'log_density': ll, 'status': status, 'reason': reason,
            'training_log_likelihood_history': history,
            'optimizer_history': optimizer_history,
            'effective_crystal_support': support.tolist(),
            'jacobian_rank': jacobian_rank, 'parameter_count': len(x),
            'supported': supported, 'identifiable': identifiable, 'at_parameter_bound': at_bound}


def block_comparison(baseline, alternative, block_keys):
    """Paired, row-weighted gain with delete-scan-block jackknife uncertainty."""
    delta = np.asarray(alternative) - np.asarray(baseline)
    if not len(delta) or not np.all(np.isfinite(delta)):
        raise ValueError('Need identical nonempty finite heldout row scores')
    blocks = sorted(set(block_keys))
    totals = np.array([sum(delta[i] for i, key in enumerate(block_keys) if key == b)
                       for b in blocks])
    sizes = np.array([sum(key == b for key in block_keys) for b in blocks])
    result = {'heldout_rows': len(delta), 'heldout_blocks': len(blocks),
              'log_likelihood_gain': float(delta.sum()),
              'mean_log_likelihood_gain': float(delta.mean()),
              'block_gains': [{'block': list(b), 'rows': int(n), 'gain': float(s)}
                              for b, n, s in zip(blocks, sizes, totals)],
              'block_jackknife_se': None, 'mean_gain_interval_95': None,
              'decision': 'unresolved'}
    if len(blocks) >= 3:
        deleted = (delta.sum() - totals) / (len(delta) - sizes)
        se = float(np.sqrt((len(blocks) - 1) / len(blocks)
                           * np.sum((deleted - deleted.mean())**2)))
        half_width = float(student_t.ppf(.975, len(blocks) - 1) * se)
        interval = [float(delta.mean() - half_width), float(delta.mean() + half_width)]
        result.update(block_jackknife_se=se, mean_gain_interval_95=interval,
                      decision='positive_predictive_gain' if interval[0] > 0 else
                      'negative_predictive_gain' if interval[1] < 0 else 'unresolved')
    return result


class DialsCrystalAdapter:
    """DIALS symmetry-preserving U/B parameters and native rotation prediction.

    Tight local bounds keep the frozen observable-support approximation explicit.
    The reference phi is a *prediction*, used only to retain branch/turn identity;
    no measured centroid is supplied to the predictor. Instrument models never
    enter the optimized parameter vector. Shared crystal objects share parameters.
    """
    def __init__(self, experiments, budget, max_predictions=20000,
                 orientation_bound_mrad=5., cell_relative_bound=.02):
        from dials.array_family import flex
        from dials.algorithms.refinement.parameterisation.crystal_parameters import (
            CrystalOrientationParameterisation, CrystalUnitCellParameterisation)
        from dials.algorithms.refinement.prediction.managed_predictors import (
            ExperimentsPredictorFactory)
        self.models = copy.deepcopy(experiments)
        self.budget = budget
        self.flex = flex
        self.parameterisations = []
        crystal_ids = {}
        self.experiment_crystals = []
        lower, upper, values = [], [], []
        for eid, e in enumerate(self.models):
            if (not e.crystal or not e.scan or not e.goniometer
                    or e.crystal.num_scan_points or e.beam.num_scan_points
                    or e.goniometer.num_scan_points or e.scan.get_oscillation()[1] <= 0):
                raise ValueError('Require static crystals and instrument, positive rotation scans')
            key = id(e.crystal)
            if key not in crystal_ids:
                crystal_ids[key] = len(crystal_ids)
                orientation = CrystalOrientationParameterisation(e.crystal)
                cell = CrystalUnitCellParameterisation(e.crystal)
                for parameterisation, radius in (
                        (orientation, np.full(3, orientation_bound_mrad)),
                        (cell, np.maximum(np.abs(cell.get_param_vals()), 1.) * cell_relative_bound)):
                    v = np.asarray(parameterisation.get_param_vals())
                    values.extend(v)
                    lower.extend(v - radius)
                    upper.extend(v + radius)
                    self.parameterisations.append(parameterisation)
            self.experiment_crystals.append(crystal_ids[key])
        self.n_crystals = len(crystal_ids)
        self.x0 = np.asarray(values)
        self.bounds = (np.asarray(lower), np.asarray(upper))
        budget.check(prediction=True)
        tables, total = [], 0
        for eid, e in enumerate(self.models):
            # Do not use observed strong-spot predictions or a q-neighbour list.
            # Disable profile hooks to use static geometrical detector coverage.
            e.profile = None
            table = flex.reflection_table.from_predictions(e, margin=0)
            table['id'] = flex.int(len(table), eid)
            total += len(table)
            budget.check()
            if total > max_predictions:
                raise BudgetExceeded('complete prediction count exceeds budget; no truncation')
            tables.append(table)
        self.table = flex.reflection_table()
        for table in tables:
            self.table.extend(table)
        if not len(self.table):
            raise ValueError('No observable predictions')
        self.ids = np.asarray(self.table['id'], dtype=int)
        self.panels = np.asarray(self.table['panel'], dtype=int)
        self.component_crystals = np.asarray(self.experiment_crystals)[self.ids]
        self.reference = np.asarray(self.table['xyzcal.mm']).copy()
        self.table['xyzobs.mm.value'] = flex.vec3_double([tuple(p) for p in self.reference])
        self.predictor = ExperimentsPredictorFactory.from_experiments(self.models)

    def set_values(self, values):
        start = 0
        for parameterisation in self.parameterisations:
            stop = start + len(parameterisation.get_param_vals())
            parameterisation.set_param_vals(list(values[start:stop]))
            start = stop

    def predict(self, values):
        self.budget.check(prediction=True)
        self.set_values(values)
        # Clear stale prediction flags before each native prediction.
        self.table.unset_flags(self.flex.bool(len(self.table), True), self.table.flags.predicted)
        self.predictor(self.table)
        prediction = np.asarray(self.table['xyzcal.mm']).copy()
        valid = np.asarray(self.table.get_flags(self.table.flags.predicted), dtype=bool)
        if not np.all(valid) or not np.all(np.isfinite(prediction)):
            raise ValueError('Frozen candidate became physically unpredictable; reduce local bounds')
        return prediction


def run(args):
    """CLI adapter; holdout centroids are evaluated only after all fits finish.

    Selection JSON declares `single_source_source_rows`, `composite_source_rows`
    and `provenance`; all other raw rows stay unresolved and are excluded. Never
    infer single-source status from the number of geometric explanations.
    Optional `candidate_generation_source_rows` and `calibration_source_rows`
    document leakage checks against the raw partition, for ALL compared models.
    """
    import json
    from pathlib import Path
    from dials.array_family import flex
    from dxtbx.model.experiment_list import ExperimentListFactory
    from soft_multi_indexer import raw_key, signature

    evidence = json.loads(args.compatibility.read_text())
    if evidence.get('schema') != 'ariadne-soft-compatibility/1':
        raise ValueError('Require compatibility evidence for coordinate/calibration provenance')
    if (Path(evidence['models_expt']).resolve() != args.models_expt.resolve()
            or Path(evidence['source_spots']).resolve() != args.spots.resolve()):
        raise ValueError('Compatibility provenance does not match requested models/spots')
    if args.output.exists():
        raise ValueError('Output directory already exists')
    if args.centroid_selection is None:
        raise ValueError('--method em requires --centroid-selection: independent single-source/overlap evidence')
    selection = json.loads(args.centroid_selection.read_text())
    if not selection.get('provenance'):
        raise ValueError('Single-source classification needs provenance')
    observed = ExperimentListFactory.from_json_file(evidence['observed_expt'], check_format=False)
    table = flex.reflection_table.from_file(str(args.spots))
    if 'imageset_id' not in table and len(observed) == 1 and table['id'].all_eq(-1):
        table['imageset_id'] = flex.int(len(table), 0)
    if 'xyzobs.mm.value' not in table or 'xyzobs.mm.variance' not in table:
        if 'xyzobs.px.variance' not in table:
            raise ValueError('Missing measured centroid variance')
        table.centroid_px_to_mm(observed)
    source = np.asarray(table['scxrd_source_row'], dtype=int) if 'scxrd_source_row' in table else np.arange(len(table))
    ids = np.asarray(table['imageset_id'] if 'imageset_id' in table else table['id'], dtype=int)
    panels = np.asarray(table['panel'], dtype=int)
    frames = np.asarray(table['xyzobs.px.value'])[:, 2]
    if not np.all(np.isfinite(frames)) or np.any(ids < 0) or np.any(ids >= len(observed)):
        raise ValueError('Invalid raw scan coordinates')
    partition, test, keys = scan_partition(ids, frames, source, args.block_frames,
                                          args.holdout_fraction, args.seed)
    known = set(map(int, source))
    single = set(selection['single_source_source_rows'])
    composite = set(selection['composite_source_rows'])
    if single & composite or not (single | composite).issubset(known):
        raise ValueError('Selection has conflicting/unknown raw row identifiers')
    eligible = np.array([int(row) in single for row in source])
    train = eligible & ~test
    heldout = eligible & test
    if train.sum() < args.min_anchors or not np.any(heldout):
        raise ValueError('Insufficient eligible single-source training/heldout rows; unresolved')
    # Provenance cannot prove upstream independence, but known leakage is fatal.
    upstream_independent = True
    for name in ('candidate_generation_source_rows', 'calibration_source_rows'):
        if name not in selection:
            upstream_independent = False
            continue
        used = set(selection[name])
        if not used.issubset(set(partition['training_source_rows'])):
            raise ValueError(name + ' includes holdout/unknown rows; regenerate using training only')
    y = np.asarray(table['xyzobs.mm.value'], dtype=float)
    variance = np.asarray(table['xyzobs.mm.variance'], dtype=float)
    centroid_cov = (np.asarray(table['xyzobs.mm.covariance'], dtype=float).reshape(-1, 3, 3)
                    if 'xyzobs.mm.covariance' in table else
                    np.array([np.diag(v) for v in variance]))
    model_cov = np.asarray(evidence['calibration']['model_covariance_mm_rad'], dtype=float)
    if (model_cov.shape != (3, 3) or not np.all(np.isfinite(model_cov))
            or not np.allclose(model_cov, model_cov.T)
            or np.min(np.linalg.eigvalsh(model_cov)) < 0):
        raise ValueError('Model covariance must be finite symmetric PSD')
    covariance = centroid_cov + model_cov
    covariance_factors(covariance)
    if not np.all(np.isfinite(y)) or np.any(variance <= 0):
        raise ValueError('Invalid measured centroid/variance')
    raw_keys = [raw_key(e) for e in observed]
    density = np.empty(len(table))
    for eid, e in enumerate(observed):
        start, stop = e.scan.get_oscillation_range(deg=False)
        volume = sum(np.prod(np.asarray(p.get_image_size()) * p.get_pixel_size())
                     for p in e.detector) * (stop - start)
        if volume <= 0:
            raise ValueError('Nonpositive detector/scan outlier volume')
        chosen = np.flatnonzero(ids == eid)
        for i in chosen:
            if not 0 <= panels[i] < len(e.detector):
                raise ValueError('Unknown observed panel')
            size = np.asarray(e.detector[int(panels[i])].get_image_size()) * e.detector[int(panels[i])].get_pixel_size()
            if (np.any(y[i, :2] < 0) or np.any(y[i, :2] > size)
                    or not start <= y[i, 2] <= stop):
                raise ValueError('Observed centroid lies outside declared outlier support')
        density[chosen] = 1. / volume
    paths = [args.models_expt] + list(args.compare_models)
    if len(paths) > args.max_hypotheses or len({str(p.resolve()) for p in paths}) != len(paths):
        raise ValueError('Invalid/repeated hypothesis set or hypothesis budget exceeded')
    budget = Budget(args.max_prediction_calls, args.max_seconds)
    fitted = []
    # No holdout likelihood is computed inside this fitting loop.
    for path in paths:
        models = ExperimentListFactory.from_json_file(str(path), check_format=False)
        model_keys = [raw_key(e) for e in models]
        if not set(model_keys).issubset(set(raw_keys)):
            raise ValueError('Hypothesis predicts an unknown raw scan')
        for eid, e in enumerate(models):
            obs = observed[raw_keys.index(model_keys[eid])]
            if len(e.detector) != len(obs.detector) or any(
                    not np.allclose(signature(e, p), signature(obs, p), rtol=1e-9, atol=1e-8)
                    for p in range(len(e.detector))):
                raise ValueError('All hypotheses must use the same fixed instrument')
        adapter = DialsCrystalAdapter(models, budget, args.max_predictions,
                                      args.orientation_bound_mrad, args.cell_relative_bound)
        if len(table) * len(adapter.table) > args.max_pairs:
            raise BudgetExceeded('Full observation/prediction pair budget exceeded; no truncation')
        allowed_scan = np.array([[raw_keys[int(eid)] == model_keys[int(pid)]
                                  for pid in adapter.ids] for eid in ids])
        counts = allowed_scan.sum(axis=1)
        allowed = allowed_scan & (panels[:, None] == adapter.panels[None, :])
        if np.any(counts == 0):
            raise ValueError('A hypothesis omits an observed raw scan')
        initial = adapter.reference.copy()
        fit = fit_em(adapter, y[train], covariance[train], allowed[train], counts[train],
                     density[train], outlier_fraction=args.outlier_fraction,
                     em_iterations=args.em_iterations, max_nfev=args.max_iterations,
                     min_support=args.min_anchors, tolerance=args.em_tolerance)
        fitted.append((path, adapter, initial, fit, allowed, counts))
    train_indices = np.flatnonzero(train)
    test_indices = np.flatnonzero(heldout)
    evaluations, final_scores = [], []
    white, lognorm = covariance_factors(covariance[heldout])
    for path, adapter, initial, fit, allowed, counts in fitted:
        _, _, initial_ll = expectation(y[heldout], initial, white, lognorm,
                                       allowed[heldout], counts[heldout], density[heldout], args.outlier_fraction)
        _, _, final_ll = expectation(y[heldout], fit['means'], white, lognorm,
                                     allowed[heldout], counts[heldout], density[heldout], args.outlier_fraction)
        final_scores.append(final_ll)
        evaluation = {k: fit[k] for k in ('status', 'reason', 'training_log_likelihood_history',
                      'effective_crystal_support', 'jacobian_rank', 'parameter_count', 'at_parameter_bound', 'optimizer_history')}
        evaluation.update(models=str(path.resolve()), predictions=len(adapter.table),
                          heldout_log_likelihood=float(final_ll.sum()),
                          initial_heldout_log_likelihood=float(initial_ll.sum()),
                          refinement_gain=block_comparison(initial_ll, final_ll, [keys[i] for i in test_indices]),
                          heldout_row_log_density=final_ll.tolist())
        if fit['status'] != 'converged':
            evaluation['refinement_gain']['decision'] = 'unresolved'
        evaluations.append(evaluation)
    for i, evaluation in enumerate(evaluations):
        comparison = block_comparison(final_scores[0], final_scores[i], [keys[j] for j in test_indices])
        if not upstream_independent or any(e['status'] == 'unresolved' for e in (evaluations[0], evaluation)):
            comparison['decision'] = 'unresolved'
        evaluation['against_first_hypothesis'] = comparison
    _, adapter, _, fit, _, _ = fitted[0]
    training = table.select(flex.size_t(train_indices.tolist()))
    # Preserve one row per raw observation. MAP columns are diagnostics only;
    # native weighted refinement never consumes this table as repeated data.
    resp = fit['responsibilities']
    best = np.argmax(resp, axis=1)
    accepted = resp[np.arange(len(best)), best] > fit['outlier']
    training['id'] = flex.int([int(adapter.ids[j]) if ok else -1 for j, ok in zip(best, accepted)])
    training['miller_index'] = flex.miller_index([
        tuple(adapter.table['miller_index'][int(j)]) if ok else (0, 0, 0)
        for j, ok in zip(best, accepted)])
    training['em.outlier_probability'] = flex.double(fit['outlier'].tolist())
    training['em.map_probability'] = flex.double(resp[np.arange(len(best)), best].tolist())
    training.unset_flags(flex.bool(len(training), True), training.flags.indexed | training.flags.used_in_refinement)
    for key in list(training.experiment_identifiers().keys()):
        del training.experiment_identifiers()[key]
    for eid, e in enumerate(adapter.models):
        if e.identifier:
            training.experiment_identifiers()[eid] = e.identifier
    expected_square = np.einsum('ij,ija->a', resp,
                                (y[train, None, :] - fit['means'][None, :, :])**2)
    rmsds = np.sqrt(expected_square / max(float(resp.sum()), np.finfo(float).tiny))
    report = {'schema': 'ariadne-joint-geometry/1', **partition,
              'status': 'proposal_requires_holdout_review' if fit['status'] == 'converged' else 'unresolved',
              'rmsds_mm_mm_rad': rmsds.tolist(),
              'method': 'exclusive single-source centroid EM; full covariance whitening; native DIALS U/B and prediction; fixed instrument',
              'training_source_rows': source[train].tolist(),
              'validation_source_rows': source[heldout].tolist(),
              'raw_partition': partition,
              'composite_source_rows_excluded': sorted(composite),
              'unresolved_source_rows_excluded': sorted(known - single - composite),
              'ambiguous_training_rows_not_duplicated': sorted(composite & set(partition['training_source_rows'])),
              'centroid_selection': str(args.centroid_selection.resolve()),
              'selection_provenance': selection['provenance'],
              'parent_models': str(args.models_expt.resolve()),
              'compatibility': str(args.compatibility.resolve()),
              'covariance_approximation': 'full Cholesky whitening; diagonal measured variance unless xyzobs.mm.covariance (mat3) supplied; fixed full model covariance',
              'candidate_policy': 'fixed initial complete observable prediction identities per hypothesis, positions updated at every objective call; equal component prior mass per raw scan, no gating/truncation',
              'support_scope': 'local frozen prediction support; not scan-varying, intensity, visibility, or multi-source centroid inference',
              'outlier_fraction': args.outlier_fraction,
              'outlier_density_by_row': density[heldout].tolist(),
              'outlier_support': 'uniform joint panel/mm/mm/rad density over full detector area times scan range; identical across hypotheses',
              'heldout_scored_source_rows': source[heldout].tolist(),
              'evaluations': evaluations,
              'upstream_training_provenance_declared': upstream_independent,
              'holdout_scope': 'final comparison only; no holdout-based tuning, refit, automatic K choice, or formal Bayesian selection; missing upstream provenance leaves model selection unresolved',
              'uncertainty': 'paired delete-scan-block jackknife, Student-t 95% interval; fewer than 3 scored blocks unresolved; assumes blocks sufficiently independent',
              'budgets': {'prediction_calls': budget.calls, 'max_prediction_calls': args.max_prediction_calls,
                          'elapsed_seconds': time.monotonic() - budget.started, 'max_seconds': args.max_seconds,
                          'max_pairs': args.max_pairs, 'max_predictions': args.max_predictions,
                          'max_hypotheses': args.max_hypotheses, 'em_iterations': args.em_iterations,
                          'm_step_max_nfev': args.max_iterations,
                          'orientation_bound_mrad': args.orientation_bound_mrad,
                          'cell_relative_bound': args.cell_relative_bound}}
    # Write only after all training and final comparisons have completed.
    args.output.mkdir(parents=True, exist_ok=False)
    adapter.models.as_file(str(args.output / 'refined.expt'))
    training.as_file(str(args.output / 'training.refl'))
    for i, (_, candidate, _, _, _, _) in enumerate(fitted[1:], 1):
        candidate.models.as_file(str(args.output / f'hypothesis_{i}_refined.expt'))
    (args.output / 'joint_refinement.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
