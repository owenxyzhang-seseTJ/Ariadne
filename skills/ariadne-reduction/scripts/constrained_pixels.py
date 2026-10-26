"""Nonnegative fits and conditional profile-likelihood intervals in calibrated ADU.

Background is estimated on background-only pixels by WLS, then FROZEN for both
MLE and intervals. Its estimation uncertainty is excluded, as are uncertainties
in profiles, gain/readout calibration and geometry. LR cutoffs use chi-square(1)
asymptotics, NOT exact coverage (including at a nonnegative boundary).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq, lsq_linear, minimize
from scipy.special import gammaln, xlogy
from scipy.stats import chi2

from poisson_gaussian import LikelihoodBudgetError, PoissonGaussian


_SCOPE = ('conditional on frozen WLS-estimated background, supplied profiles, '
          'geometry and noise calibration; background-estimation uncertainty excluded')


def _noise_parameters(noise, method):
    if not isinstance(noise, dict) or not noise.get('source'):
        raise ValueError('Likelihood requires calibrated noise source/provenance')
    # Existing source is the backwards-compatible calibration provenance field.
    if 'provenance' in noise and not noise['provenance']:
        raise ValueError('Noise calibration provenance must not be empty')
    if noise.get('offset_removed') is not True:
        raise ValueError('Likelihood requires offset_removed=True for corrected ADU')
    try:
        gain, readout = float(noise['gain']), float(noise['readout_variance'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Require calibrated gain and readout_variance in ADU units') from exc
    if not np.isfinite(gain) or gain <= 0 or not np.isfinite(readout) or readout < 0:
        raise ValueError('Gain must be finite positive ADU/count; readout_variance finite nonnegative ADU^2')
    if method == 'poisson' and (readout != 0 or noise.get('poisson_counts_verified') is not True):
        raise ValueError('Poisson requires explicit calibrated genuine-count evidence and zero readout noise')
    return gain, readout


def _projected_error(x, gradient):
    return float(np.max(np.abs(np.where((x <= 1e-9) & (gradient > 0), 0., gradient)))) if len(x) else 0.


def _minimize_nonnegative(fun, initial):
    result = minimize(fun, initial, jac=True, method='L-BFGS-B',
                      bounds=[(0., None)] * len(initial),
                      options={'maxiter': 2000, 'maxls': 50, 'ftol': 1e-14, 'gtol': 1e-9})
    value, grad = fun(result.x)
    error = _projected_error(result.x, grad)
    return result.x, value, bool(result.success and np.isfinite(value) and error <= 1e-6), error, str(result.message)


def _profile_intervals(fun, mle, units, covariance, nullspace, scales,
                       confidence_level, converged, gaussian=None):
    """Reoptimize all other intensities at fixed component or total ADU."""
    k = len(mle)
    threshold = float(chi2.ppf(confidence_level, 1))
    baseline = fun(mle)[0]
    estimates = mle * units
    evaluations = 0
    failures = []

    def interval(contrast):
        nonlocal evaluations
        scaled = contrast / scales
        fraction = float(np.linalg.norm(nullspace @ scaled) / np.linalg.norm(scaled))
        identifiable = fraction <= max(1e-10, 10 * np.finfo(float).eps * k)
        estimate = float(contrast @ estimates)
        out = {'identifiable': identifiable, 'nullspace_fraction': fraction,
               'estimate': estimate if identifiable else None, 'lower': None, 'upper': None,
               'confidence_level': confidence_level, 'likelihood_ratio_threshold': threshold,
               'calibration': 'chi-square(1) asymptotic; not exact, including at boundary',
               'unit': 'calibrated_adu', 'scope': _SCOPE}
        if not identifiable:
            return dict(out, status='unidentifiable')
        if not converged:
            return dict(out, status='fit_not_converged')
        indices = np.flatnonzero(contrast)
        is_component = len(indices) == 1
        fixed = int(indices[0]) if is_component else None
        free = np.delete(np.arange(k), fixed) if is_component else None
        cache = {estimate: (0., mle.copy())}

        def profile(t):
            nonlocal evaluations
            if t in cache:
                return cache[t][0]
            evaluations += 1
            if is_component:
                x = mle.copy()
                x[fixed] = t / units[fixed]
                # Positive nuisance seed distinguishes an impossible fixed
                # component (Poisson support) from a recoverable zero seed.
                trial = x.copy()
                trial[free] = np.maximum(trial[free], 1e-3)
                if np.isposinf(fun(trial)[0]):
                    cache[t] = (np.inf, trial)
                    return np.inf
                if len(free):
                    if gaussian is not None:
                        A, target = gaussian
                        fit = lsq_linear(A[:, free], target - A[:, fixed] * x[fixed],
                                         bounds=(0., np.inf), tol=1e-12, max_iter=1000)
                        x[free] = fit.x
                        value, gradient = fun(x)
                        ok = fit.success and _projected_error(x[free], gradient[free]) <= 1e-6
                    else:
                        def nuisance(z):
                            x[free] = z
                            value, gradient = fun(x)
                            return value, gradient[free]
                        # Positive seed keeps a Poisson model in its support even
                        # when fixing a previously positive component to zero.
                        seed = np.maximum(mle[free], 1e-3)
                        z, value, ok, _, _ = _minimize_nonnegative(nuisance, seed)
                        x[free] = z
                    if not ok:
                        raise RuntimeError('component nuisance optimization did not satisfy KKT')
                else:
                    value = fun(x)[0]
            elif t == 0:
                x = np.zeros(k)
                value = fun(x)[0]
            else:
                c = units / np.linalg.norm(units)
                target = t / np.linalg.norm(units)
                seed = mle * (t / estimate) if estimate > 0 else np.full(k, t / units.sum())
                fit = minimize(fun, seed, jac=True, method='SLSQP', bounds=[(0., None)] * k,
                               constraints={'type': 'eq', 'fun': lambda z: c @ z - target,
                                            'jac': lambda z: c},
                               options={'maxiter': 1500, 'ftol': 1e-12})
                x, value = fit.x, fun(fit.x)[0]
                gradient = fun(x)[1]
                active = x > 1e-8
                multiplier = -float(c[active] @ gradient[active]) / float(c[active] @ c[active]) if np.any(active) else 0.
                reduced = gradient + multiplier * c
                error = _projected_error(x, reduced)
                if (not fit.success or error > 2e-6
                        or abs(c @ x - target) > 1e-8 * max(1., target)):
                    raise RuntimeError('total nuisance optimization did not satisfy equality/KKT')
            delta = 2 * (value - baseline)
            if np.isnan(delta) or delta < -1e-6:
                raise RuntimeError('profile objective below fitted minimum or nonfinite')
            cache[t] = (max(0., float(delta)), x.copy())
            return cache[t][0]

        try:
            at_zero = profile(0.)
            lower = 0. if estimate == 0 or at_zero <= threshold else float(brentq(
                lambda t: profile(t) - threshold, 0., estimate, xtol=1e-10, rtol=1e-10))
            step = max(float(np.sqrt(max(contrast @ covariance @ contrast, 0.))),
                       .1 * max(estimate, 1.), 1e-3)
            upper = estimate + step
            for _ in range(60):
                if profile(upper) >= threshold:
                    break
                step *= 2
                upper = estimate + step
            else:
                raise RuntimeError('upper profile root not bracketed within 60 expansions')
            upper = float(brentq(lambda t: profile(t) - threshold, estimate, upper,
                                 xtol=1e-10, rtol=1e-10))
            return dict(out, status='ok', lower=lower, upper=upper,
                        lower_at_nonnegative_boundary=lower == 0.,
                        lower_likelihood_ratio=at_zero if lower == 0 else profile(lower),
                        upper_likelihood_ratio=profile(upper))
        except LikelihoodBudgetError:
            raise
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            failures.append(str(exc))
            return dict(out, status='profile_failed', reason=str(exc))

    components = [interval(np.eye(k)[i]) for i in range(k)]
    total = interval(np.ones(k))
    return components, total, {'profile_evaluations': evaluations, 'profile_failures': failures,
                               'confidence_level': confidence_level, 'lr_threshold': threshold,
                               'threshold_distribution': 'chi-square(1) asymptotic, not exact coverage',
                               'nuisance': 'all other nonnegative intensities reoptimized; background frozen'}


def fit_constrained(y, variance, profiles, valid, basis, noise, method, rcond,
                    confidence_level=.95):
    """Fit nnls / poisson / poisson-gaussian; amplitudes and CIs are in ADU.

    Noise gain is ADU/count and readout_variance is ADU^2; source supplies
    calibration provenance (an explicitly supplied provenance must be nonempty).
    PG admits negative ADU with positive readout variance. Zero readout variance
    uses the exact lattice Poisson limit. variance is used for the frozen WLS
    background and numerical scaling, not as a PG likelihood approximation.
    """
    if method not in ('nnls', 'poisson', 'poisson-gaussian'):
        raise ValueError('Unknown constrained method: ' + str(method))
    if not np.isfinite(rcond) or not 0 < rcond < 1 or not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError('Require 0 < rcond, confidence_level < 1')
    y, variance = np.asarray(y, float), np.asarray(variance, float)
    profiles, basis = np.asarray(profiles, float), np.asarray(basis, float)
    valid = np.asarray(valid, bool)
    if (y.ndim != 1 or variance.shape != y.shape or valid.shape != y.shape
            or profiles.ndim != 2 or profiles.shape[0] != len(y) or profiles.shape[1] == 0
            or basis.ndim != 2 or basis.shape[0] != len(y)
            or not np.all(np.isfinite(profiles)) or np.any(profiles < 0)
            or not np.all(np.isfinite(basis))):
        raise ValueError('Require vector data/mask and finite nonnegative pixel-by-component profiles')
    P, B, data = profiles[valid], basis[valid], y[valid]
    if not len(data) or not np.all(np.isfinite(data)) or not np.all(np.isfinite(variance[valid])) or np.any(variance[valid] <= 0):
        raise ValueError('Valid data must be finite with positive finite variance')
    gain, readout = _noise_parameters(noise, method) if method != 'nnls' else (1., 0.)
    counts = data / gain
    discrete = method == 'poisson' or (method == 'poisson-gaussian' and readout == 0)
    if discrete and (np.any(counts < 0) or not np.allclose(counts, np.rint(counts), atol=1e-6, rtol=0)):
        raise ValueError('Zero-readout likelihood requires nonnegative integer photon counts')
    if discrete:
        counts = np.rint(counts)
    significant = np.any(P > .01 * np.maximum(P.max(axis=0), 1e-300), axis=1)
    selected = ~significant
    if selected.sum() < B.shape[1] + 5:
        raise ValueError('Insufficient independent background pixels for constrained fit')
    root_weight = 1 / np.sqrt(variance[valid])
    beta, _, rank_b, _ = np.linalg.lstsq(B[selected] * root_weight[selected, None],
                                       data[selected] * root_weight[selected], rcond=None)
    if rank_b != B.shape[1]:
        raise ValueError('Background-only design is rank deficient')
    background = B @ beta
    if np.any(background < 0):
        raise ValueError('Frozen background predicts negative expectation; revise background model')
    weighted = P * root_weight[:, None]
    norms = np.linalg.norm(weighted, axis=0)
    if np.any(norms <= 1e-15):
        raise ValueError('Masked or uninformative component')
    units = 1 / norms
    A = P * units
    gaussian = (weighted * units, (data - background) * root_weight)
    initial = lsq_linear(*gaussian, bounds=(0., np.inf), tol=1e-12, max_iter=1000)
    pg = PoissonGaussian(data, gain, readout) if method == 'poisson-gaussian' and readout > 0 else None

    def objective_and_gradient(x):
        expected_adu = background + A @ x
        if method == 'nnls':
            residual = (expected_adu - data) * root_weight
            return .5 * float(residual @ residual), (A * root_weight[:, None]).T @ residual
        mean = expected_adu / gain
        if pg is not None:
            logs, score, _ = pg.evaluate(mean)
            return -float(logs.sum()), -(A / gain).T @ score
        # xlogy(0, 0)=0; zero mean with positive counts has infinite NLL.
        value = float(np.sum(mean - xlogy(counts, mean) + gammaln(counts + 1)))
        ratio = np.divide(counts, mean, out=np.zeros_like(mean), where=mean > 0)
        ratio[(mean == 0) & (counts > 0)] = np.inf
        terms = np.zeros_like(A)
        np.multiply(A / gain, (1 - ratio)[:, None], out=terms, where=A > 0)
        gradient = terms.sum(axis=0)
        return value, gradient

    if discrete and np.any((counts > 0) & (background == 0) & ~np.any(P > 0, axis=1)):
        raise ValueError('Positive counts outside the support of the frozen-background signal model')
    if method == 'nnls':
        x, minimum = initial.x, objective_and_gradient(initial.x)[0]
        kkt = _projected_error(x, objective_and_gradient(x)[1])
        converged, message = bool(initial.success and kkt <= 1e-6), str(initial.message)
    else:
        x, minimum, converged, kkt, message = _minimize_nonnegative(objective_and_gradient,
                                                                  np.maximum(initial.x, 1e-3))
    intensity = units * x
    expected_adu = background + P @ intensity
    if method == 'nnls':
        fit_weight = root_weight
        objective = 2 * minimum
        objective_kind = 'Gaussian chi-square with supplied fixed variances'
    elif discrete:
        mean = expected_adu / gain
        ratio = np.divide(counts, mean, out=np.ones_like(mean), where=mean > 0)
        objective = float(2 * np.sum(xlogy(counts, ratio) - counts + mean))
        # At zero mean Fisher weights are singular. Keep a finite diagnostic
        # covariance using supplied variances there; intervals do not use it.
        var_proxy = gain * expected_adu
        fit_weight = 1 / np.sqrt(np.where(var_proxy > 0, var_proxy, variance[valid]))
        objective_kind = 'Poisson deviance'
    else:
        fit_weight = 1 / np.sqrt(gain * expected_adu + readout)
        objective = 2 * minimum
        objective_kind = 'minus twice normalized PG log density; NOT chi-square or deviance'
    design = P * fit_weight[:, None]
    scales = np.linalg.norm(design, axis=0)
    _, singular, Vh = np.linalg.svd(design / scales, full_matrices=design.shape[0] < design.shape[1])
    rank = int(np.sum(singular > singular[0] * rcond))
    retained = Vh[:rank].T
    covariance = ((retained / singular[:rank]) @ (retained / singular[:rank]).T) / scales[:, None] / scales[None, :]
    nullspace = Vh[rank:]
    intervals, total, diagnostics = _profile_intervals(
        objective_and_gradient, x, units, covariance, nullspace, scales,
        confidence_level, converged, gaussian if method == 'nnls' else None)
    diagnostics.update({'method': method, 'scope': _SCOPE, 'negative_adu_allowed': pg is not None,
                        'objective_kind': objective_kind, 'minimum_negative_log_likelihood': minimum,
                        'optimizer_message': message, 'projected_gradient_inf_norm': kkt,
                        'optimizer_kkt_tolerance': 1e-6, 'rank_rcond': rcond,
                        'gain_unit': 'ADU/count', 'readout_variance_unit': 'ADU^2',
                        'noise_provenance': noise.get('provenance', noise.get('source')) if noise else None,
                        'covariance_approximation': 'moment-weighted Gaussian proxy for PG; Fisher for Poisson except zero-mean rows use supplied variance'})
    if pg is not None:
        diagnostics['convolution'] = pg.diagnostics()
        diagnostics['scaled_gradient_truncation_bound'] = (pg.max_score_error
            * float(np.max(np.sum(np.abs(A / gain), axis=0))))
    elif discrete:
        diagnostics['convolution'] = {'evaluation': 'exact discrete Poisson (zero readout variance)'}
    boundary = np.flatnonzero(x <= 1e-9).tolist()
    dof = len(data) - rank
    return {'intensities_internal': intensity, 'covariance_identifiable_subspace': covariance,
            'nullspace': nullspace, 'column_scales': scales, 'rank': rank,
            'singular_values': singular, 'background_coefficients': beta,
            'fitted': profiles @ intensity + basis @ beta,
            'chi2': objective, 'dof': dof,
            'reduced_chi2': objective / dof if dof > 0 and pg is None else None,
            'converged': converged, 'boundary_components': boundary,
            'covariance_scope': 'interior approximation only; invalid as boundary confidence intervals; ' + _SCOPE,
            'profile_intervals': intervals, 'total_interval': total,
            'likelihood_diagnostics': diagnostics}
