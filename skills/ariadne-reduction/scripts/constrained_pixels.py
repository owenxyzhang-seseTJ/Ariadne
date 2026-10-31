"""Nonnegative fits and conditional profile-likelihood intervals in calibrated ADU.

Background is estimated on background-only pixels by WLS, then FROZEN for both
MLE and intervals. Its estimation uncertainty is excluded, as are uncertainties
in profiles, gain/readout calibration and geometry. LR cutoffs use chi-square(1)
asymptotics, NOT exact coverage (including at a nonnegative boundary).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import qr
from scipy.optimize import brentq, lsq_linear, minimize
from scipy.stats import chi2

from poisson_gaussian import LikelihoodBudgetError, PoissonGaussian, _poisson_deviance, _poisson_logpmf


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


def _compress_gaussian(design, target):
    """Preserve ||A x - b|| for every x in at most K+1 Householder QR rows.

    The augmented target column retains the parameter-independent residual, so
    this also preserves absolute objectives and MLE-centered likelihood ratios.
    No rank cutoff is used: weak and null directions keep their original role
    in the nonnegative fits and equality-constrained total profiles.
    """
    columns = design.shape[1] + 1
    if len(target) <= max(128, 4 * columns):
        return design, target
    augmented = np.empty((len(target), columns), order='F')
    augmented[:, :-1], augmented[:, -1] = design, target
    factor = qr(augmented, mode='r', overwrite_a=True, check_finite=False)[0][:columns].copy()
    return factor[:, :-1], factor[:, -1]


def _poisson_predictor_reduction(counts, background, design, profiles):
    """Exactly aggregate equal-contributor Poisson rays at zero background.

    If every positive entry of a physical-profile row is the same p_j, its
    mean is p_j/g times the sum of the indicated physical amplitudes. Counts
    with the same contributor set have a Poisson sufficient statistic: their
    sum, with the exposures summed as well. This includes isolated profiles
    and exact complete overlaps. No tolerance groups near-equal profiles.
    Other affine predictors are retained as individual observations.
    """
    present = profiles > 0
    peak = profiles.max(axis=1)
    eligible = (background == 0) & np.all(~present | (profiles == peak[:, None]), axis=1)
    selected = np.flatnonzero(eligible)
    if len(selected) < 2:
        return counts, background, design, 0
    # Summation must not invent fractional photon counts above float64's exact
    # integer range. Such data keep their original per-pixel likelihood.
    if float(counts[selected].sum()) >= 2**52:
        return counts, background, design, 0
    keys = np.packbits(present[selected], axis=1)
    _, groups = np.unique(keys, axis=0, return_inverse=True)
    number = int(groups.max()) + 1
    if number == len(selected):
        return counts, background, design, 0
    aggregate_design = np.zeros((number, design.shape[1]))
    np.add.at(aggregate_design, groups, design[selected])
    aggregate_counts = np.bincount(groups, weights=counts[selected], minlength=number)
    keep = ~eligible
    return (np.r_[counts[keep], aggregate_counts],
            np.r_[background[keep], np.zeros(number)],
            np.vstack((design[keep], aggregate_design)), len(selected)-number)


def _poisson_likelihood_difference(counts, reference, difference, mean=None):
    """NLL(mu0 + dmu) - NLL(mu0), retaining close-mean likelihood ratios.

    D(mu0,mu) + (mu0-n)*log(mu/mu0) avoids cancellation in dmu-n*log1p(dmu/mu0).
    The supplied difference comes from the linear predictor, not subtraction
    of two large complete means. Poisson support endpoints remain exact.
    """
    mean = reference + difference if mean is None else mean
    if np.any(mean < 0):
        raise ValueError('Poisson likelihood-ratio means must be nonnegative')
    out = difference.copy()
    positive = counts > 0
    supported = positive & (mean > 0) & (reference > 0)
    out[positive & (mean == 0)] = np.inf
    if np.any(positive & (reference == 0)):
        raise ValueError('Likelihood-ratio reference is outside Poisson support')
    if np.any(supported):
        base, shift, n = reference[supported], difference[supported], counts[supported]
        relative = shift / base
        # D(base, base+shift) loses the supplied shift if rounded means coincide.
        small = np.abs(relative) <= .125
        log_ratio = np.empty_like(relative)
        log_ratio[small] = np.log1p(relative[small])
        log_ratio[~small] = np.log(mean[supported][~small]) - np.log(base[~small])
        remainder = np.empty_like(relative)
        u = relative[small]
        # u-log1p(u) = u^2 * (1/2-u/3+u^2/4-...).
        polynomial = np.full_like(u, 1./24)
        for power in range(23, 1, -1):
            polynomial = ((-1.) ** power) / power + u * polynomial
        remainder[small] = base[small] * u * u * polynomial
        remainder[~small] = shift[~small] - base[~small] * log_ratio[~small]
        out[supported] = remainder + (base-n) * log_ratio
    return float(out.sum())


def _independent_poisson_profile(counts, background, design, units, mle):
    """Return an exact nuisance solver for independent measured amplitude sums.

    Each predictor must depend on a disjoint contributor set with exactly equal
    physical coefficients and zero background. For group totals T_g the model
    is N_g ~ Poisson(e_g*T_g). A fixed grand total is solved by one scalar dual
    variable: T_g=N_g/(e_g+lambda), including zero-count boundary groups.
    No regularizer or approximate profile matching is introduced.
    """
    present = design > 0
    if np.any(background != 0) or np.any(present.sum(axis=0) != 1):
        return None
    members = [np.flatnonzero(row) for row in present]
    physical = design / units
    exposure = np.empty(len(counts))
    for group, columns in enumerate(members):
        if not len(columns):
            return None
        coefficients = physical[group, columns]
        if not np.all(coefficients == coefficients[0]):
            return None
        exposure[group] = coefficients[0]
    unconstrained = counts / exposure
    owner = np.argmax(present, axis=0)
    reference = mle * units

    def allocate(totals):
        amplitudes = np.zeros_like(reference)
        for group, columns in enumerate(members):
            previous = reference[columns].sum()
            fractions = reference[columns]/previous if previous > 0 else np.full(len(columns),1./len(columns))
            amplitudes[columns] = totals[group]*fractions
        return amplitudes/units

    def solve(contrast, target):
        if np.count_nonzero(contrast) == 1:
            component = int(np.flatnonzero(contrast)[0])
            group = owner[component]
            if len(members[group]) != 1:
                raise ValueError('Individual member of a summed Poisson group is unidentifiable')
            totals = unconstrained.copy()
            totals[group] = target
            return allocate(totals)
        if target == 0:
            return np.zeros_like(mle)
        count_sum = float(counts.sum())
        cheapest = int(np.argmin(exposure))
        if count_sum == 0:
            totals = np.zeros_like(counts)
            totals[cheapest] = target
            return allocate(totals)
        positive = counts > 0
        # z = (lambda + min exposure) * target / total_count puts the
        # monotone dual root in [0,1], independent of ADU units and exposure.
        offsets = (exposure[positive]-exposure[cheapest]) * (target/count_sum)
        fractions = counts[positive]/count_sum
        if np.all(offsets > 0):
            at_zero = float(np.sum(fractions/offsets))
            if at_zero <= 1:
                totals = np.zeros_like(counts)
                totals[positive] = target*fractions/offsets
                totals[cheapest] = max(target-totals.sum(),0.)
                return allocate(totals)
        def mass(z):
            denominator = offsets+z
            return float(np.sum(np.divide(fractions,denominator,
                         out=np.full_like(fractions,np.inf),where=denominator>0)))-1.
        z = brentq(mass,0.,1.,xtol=np.nextafter(0.,1.),rtol=4*np.finfo(float).eps,maxiter=200)
        totals = np.zeros_like(counts)
        totals[positive] = target*fractions/(offsets+z)
        return allocate(totals)

    return solve


def _profile_intervals(fun, mle, units, covariance, nullspace, scales,
                       confidence_level, converged, gaussian=None, centered=None,
                       exact_poisson=None, poisson_support=False):
    """Reoptimize all other intensities at fixed component or total ADU."""
    k = len(mle)
    threshold = float(chi2.ppf(confidence_level, 1))
    if centered is not None:
        fun, baseline = centered, 0.
    else:
        baseline = fun(mle)[0]
    estimates = mle * units
    evaluations = 0
    exact_solutions = 0
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
            nonlocal evaluations, exact_solutions
            if t in cache:
                return cache[t][0]
            evaluations += 1
            # Convex profile solutions vary continuously between active-set
            # changes. Reuse the nearest feasible solved state on either side;
            # never change the objective, stopping criteria or physical bounds.
            nearby = min((key for key, value in cache.items() if np.isfinite(value[0])),
                         key=lambda key: abs(key-t))
            warm = cache[nearby][1].copy()
            if exact_poisson is not None:
                x = exact_poisson(contrast,t)
                value, gradient = fun(x)
                if np.isfinite(value):
                    if is_component:
                        error = _projected_error(x[free],gradient[free])
                    elif t > 0:
                        active = np.flatnonzero(x > 0)
                        multiplier = -float(gradient[active[0]]/units[active[0]]) if len(active) else 0.
                        error = _projected_error(x,gradient+multiplier*units) if len(active) else np.inf
                    else:
                        error = 0.
                    if error > 2e-6 or abs(float((contrast*units)@x)-t) > 1e-8*max(1.,t):
                        raise RuntimeError('analytic Poisson nuisance solution did not satisfy equality/KKT')
                exact_solutions += 1
            elif is_component:
                x = warm
                x[fixed] = t / units[fixed]
                # Positive nuisance seed distinguishes an impossible fixed
                # component (Poisson support) from a recoverable zero seed.
                if poisson_support:
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
                        seed = x[free].copy()
                        if poisson_support and not np.isfinite(fun(x)[0]):
                            seed = np.maximum(seed, 1e-3)
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
                warm_total = float(units @ warm)
                seed = warm * (t / warm_total) if warm_total > 0 else np.full(k, t / units.sum())
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
            # Root tolerances live in dimensionless bracket coordinates, not
            # arbitrary detector ADU. This preserves precision on unit changes.
            lower = 0. if estimate == 0 or at_zero <= threshold else float(estimate*brentq(
                lambda u: profile(estimate*u)-threshold,0.,1.,xtol=1e-12,rtol=1e-12))
            natural_scale = float(np.hypot.reduce(contrast*units))
            step = max(float(np.sqrt(max(contrast @ covariance @ contrast, 0.))),
                       .1 * max(estimate,natural_scale),1e-3*natural_scale)
            upper = estimate + step
            for _ in range(60):
                if profile(upper) >= threshold:
                    break
                step *= 2
                upper = estimate + step
            else:
                raise RuntimeError('upper profile root not bracketed within 60 expansions')
            span = upper-estimate
            upper = float(estimate+span*brentq(lambda u:profile(estimate+span*u)-threshold,
                                             0.,1.,xtol=1e-12,rtol=1e-12))
            lower_lr = at_zero if lower == 0 else profile(lower)
            upper_lr = profile(upper)
            residual_tolerance = 1e-6*max(1.,threshold)
            if abs(upper_lr-threshold)>residual_tolerance or (lower>0 and abs(lower_lr-threshold)>residual_tolerance):
                raise RuntimeError('profile root did not resolve the requested likelihood-ratio threshold')
            return dict(out, status='ok', lower=lower, upper=upper,
                        lower_at_nonnegative_boundary=lower == 0.,
                        lower_likelihood_ratio=lower_lr,upper_likelihood_ratio=upper_lr)
        except LikelihoodBudgetError:
            raise
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            failures.append(str(exc))
            return dict(out, status='profile_failed', reason=str(exc))

    components = [interval(np.eye(k)[i]) for i in range(k)]
    # The one-component total is precisely the same likelihood contrast.
    total = components[0].copy() if k == 1 else interval(np.ones(k))
    return components, total, {'profile_evaluations': evaluations, 'profile_failures': failures,
                               'confidence_level': confidence_level, 'lr_threshold': threshold,
                               'threshold_distribution': 'chi-square(1) asymptotic, not exact coverage',
                               'nuisance': 'all other nonnegative intensities reoptimized; background frozen',
                               'initialization': 'nearest solved feasible profile state; exact objective and KKT rechecked',
                               'analytic_poisson_nuisance_solutions': exact_solutions,
                               'root_coordinates': 'dimensionless bracket coordinate; endpoint likelihood-ratio residual verified',
                               'likelihood_ratio_arithmetic': 'MLE-centered per-observation differences' if centered is not None else 'difference of objective sums'}


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
    signal_rows = np.any(P > 0, axis=1)
    signal_profiles = P[signal_rows]
    fit_data, fit_background = data[signal_rows], background[signal_rows]
    fit_root_weight = root_weight[signal_rows]
    weighted = signal_profiles * fit_root_weight[:, None]
    norms = np.linalg.norm(weighted, axis=0)
    if np.any(norms <= 1e-15):
        raise ValueError('Masked or uninformative component')
    units = 1 / norms
    weighted *= units
    gaussian = (weighted, (fit_data - fit_background) * fit_root_weight)
    if method == 'nnls':
        gaussian = _compress_gaussian(*gaussian)
    photon_design = signal_profiles * (units/gain)
    model_background = fit_background / gain
    model_counts = counts[signal_rows]
    reduced_rows = 0
    if discrete:
        model_counts, model_background, photon_design, reduced_rows = _poisson_predictor_reduction(
            model_counts, model_background, photon_design, signal_profiles)
    initial = lsq_linear(*gaussian, bounds=(0., np.inf), tol=1e-12, max_iter=1000)
    pg = PoissonGaussian(fit_data, gain, readout) if method == 'poisson-gaussian' and readout > 0 else None

    def objective_and_gradient(x):
        if method == 'nnls':
            residual = gaussian[0] @ x - gaussian[1]
            return .5 * float(residual @ residual), gaussian[0].T @ residual
        mean = model_background + photon_design @ x
        if pg is not None:
            logs, score, _ = pg.evaluate(mean)
            return -float(logs.sum()), -photon_design.T @ score
        # Parameter-independent factorial terms are unnecessary in optimization.
        value = float(_poisson_deviance(model_counts, mean).sum())
        derivative = np.divide(mean-model_counts, mean, out=np.ones_like(mean), where=mean > 0)
        derivative[(mean == 0) & (model_counts > 0)] = -np.inf
        if np.all(np.isfinite(derivative)):
            gradient = photon_design.T @ derivative
        else:
            terms = np.zeros_like(photon_design)
            np.multiply(photon_design, derivative[:, None], out=terms, where=photon_design > 0)
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
    reference = x.copy()
    reference_mean = model_background + photon_design @ reference
    reference_residual = gaussian[0] @ reference - gaussian[1]
    reference_logs = pg.evaluate(reference_mean)[0] if pg is not None else None

    def centered_objective(trial):
        """Exact same LR with constants removed before summing observations."""
        difference = trial-reference
        if method == 'nnls':
            change = gaussian[0] @ difference
            value = float(change @ reference_residual + .5 * (change @ change))
            return value, gaussian[0].T @ (reference_residual+change)
        mean = model_background + photon_design @ trial
        if pg is not None:
            logs, score, _ = pg.evaluate(mean)
            return -float((logs-reference_logs).sum()), -photon_design.T @ score
        value = _poisson_likelihood_difference(model_counts, reference_mean,
                                               photon_design @ difference, mean)
        return value, objective_and_gradient(trial)[1]

    covariance_status, covariance_reason = 'resolved', None
    information_diagnostics = None
    constant_convolution = None
    if method == 'nnls':
        fit_weight = fit_root_weight
        residual = (expected_adu-data) * root_weight
        minimum = .5 * float(residual @ residual)
        objective = 2 * minimum
        objective_kind = 'Gaussian chi-square with supplied fixed variances'
    elif discrete:
        mean = expected_adu / gain
        minimum = -float(_poisson_logpmf(counts, mean).sum())
        objective = float(2 * _poisson_deviance(counts, mean).sum())
        # At zero mean Fisher weights are singular. Keep a finite diagnostic
        # covariance using supplied variances there; intervals do not use it.
        var_proxy = gain * expected_adu
        fit_weight = 1 / np.sqrt(np.where(var_proxy > 0, var_proxy, variance[valid]))[signal_rows]
        objective_kind = 'Poisson deviance'
    else:
        fit_weight = 1 / np.sqrt(gain * expected_adu[signal_rows] + readout)
        # Frozen zero-profile pixels contribute once to the absolute diagnostic,
        # never to derivatives, profile likelihood ratios or nonlinear iterations.
        minimum = -float(reference_logs.sum())
        if np.any(~signal_rows):
            constant_pg = PoissonGaussian(data[~signal_rows], gain, readout)
            logs, _, _ = constant_pg.evaluate(background[~signal_rows] / gain)
            minimum -= float(logs.sum())
            constant_convolution = constant_pg.diagnostics()
            del constant_pg, logs
        objective = 2 * minimum
        objective_kind = 'minus twice normalized PG log density; NOT chi-square or deviance'
        # The curvature of the fitted likelihood, rather than only its first
        # two moments, supplies the interior uncertainty. A proxy remains an
        # internal numerical scale when curvature cannot be resolved; callers
        # must not publish that proxy as an observed-information standard error.
        information, information_quality = pg.observed_information(expected_adu[signal_rows] / gain)
        unresolved = ~information_quality['reliable_positive']
        information_diagnostics = {
            'method': 'exact PG observed information (f1/f0)^2-f2/f0',
            'unresolved_signal_pixels': int(np.count_nonzero(unresolved)),
            'minimum_signal_information_per_count2': float(information.min()),
            'maximum_absolute_truncation_bound': float(information_quality['absolute_truncation_bound'].max()),
            'maximum_roundoff_resolution': float(information_quality['roundoff_resolution'].max()),
        }
        if np.any(unresolved):
            covariance_status = 'unresolved'
            covariance_reason = 'PG local curvature is nonpositive or below truncation/roundoff resolution'
        else:
            fit_weight = np.sqrt(np.maximum(information, 0.)) / gain
    design = signal_profiles * fit_weight[:, None]
    scales = np.linalg.norm(design, axis=0)
    _, singular, Vh = np.linalg.svd(design / scales, full_matrices=design.shape[0] < design.shape[1])
    rank = int(np.sum(singular > singular[0] * rcond))
    retained = Vh[:rank].T
    covariance = ((retained / singular[:rank]) @ (retained / singular[:rank]).T) / scales[:, None] / scales[None, :]
    nullspace = Vh[rank:]
    exact_poisson = (_independent_poisson_profile(model_counts,model_background,photon_design,units,x)
                     if discrete else None)
    intervals, total, diagnostics = _profile_intervals(
        objective_and_gradient, x, units, covariance, nullspace, scales,
        confidence_level, converged, gaussian if method == 'nnls' else None,
        centered=centered_objective,exact_poisson=exact_poisson,poisson_support=discrete)
    diagnostics.update({'method': method, 'scope': _SCOPE, 'negative_adu_allowed': pg is not None,
                        'objective_kind': objective_kind, 'minimum_negative_log_likelihood': minimum,
                        'optimizer_message': message, 'projected_gradient_inf_norm': kkt,
                        'optimizer_kkt_tolerance': 1e-6, 'rank_rcond': rcond,
                        'likelihood_factorization': {
                            'valid_pixels': len(data),
                            'parameter_dependent_pixels': int(signal_rows.sum()),
                            'parameter_independent_pixels': int((~signal_rows).sum()),
                            'optimization_observations': len(gaussian[1]) if method == 'nnls' else len(model_background),
                            'gaussian_qr_rows_removed': int(signal_rows.sum())-len(gaussian[1]) if method == 'nnls' else 0,
                            'poisson_sufficient_statistics_rows_removed': reduced_rows,
                            'scope': 'exact zero-profile factor separation; Gaussian augmented Householder QR; exact equal-contributor zero-background Poisson sums; no weak-pixel threshold'},
                        'gain_unit': 'ADU/count', 'readout_variance_unit': 'ADU^2',
                        'noise_provenance': noise.get('provenance', noise.get('source')) if noise else None,
                        'covariance_information_status': covariance_status,
                        'covariance_information_reason': covariance_reason,
                        'covariance_estimate_usable': covariance_status == 'resolved',
                        'covariance_approximation': (
                            'exact PG observed-information interior approximation' if pg is not None and covariance_status == 'resolved'
                            else 'INTERNAL moment-weighted PG proxy; observed-information covariance unresolved, do not publish sigma' if pg is not None
                            else 'fixed-variance Gaussian' if method == 'nnls'
                            else 'Poisson Fisher except zero-mean rows use supplied variance; exact likelihood intervals remain authoritative')})
    if pg is not None:
        diagnostics['observed_information'] = information_diagnostics
        diagnostics['convolution'] = pg.diagnostics()
        diagnostics['constant_pixel_convolution'] = constant_convolution
        diagnostics['scaled_gradient_truncation_bound'] = (pg.max_score_error
            * float(np.max(np.sum(photon_design, axis=0))))
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
