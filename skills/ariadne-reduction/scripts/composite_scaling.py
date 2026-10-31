"""Small dense global scale/absorption fit to independent composite observations.

Only NumPy/SciPy are required; this module performs no data-file I/O. Blocks are
independent, but rows *within* a block may be correlated. A contribution record
is one column of a block's design, not a separately measured observation.
Callers must encode (domain, hkl equivalence) in each opaque equivalence ID;
the kernel neither merges domains nor fits domain multipliers. Repeated IDs
across or within blocks tie signed intensities; their design contributions add.

Features must be centered over all contribution records,
with no nonzero constant column. Known factors are fixed and positive. beta
starts at zero. The local nonlinear solve really alternates exact signed WLS
and fixed-intensity least_squares, with an optional joint least_squares polish.
There is no positivity constraint or regularization. 'global' refers to the
shared fit across blocks, not a guarantee of the global nonlinear minimum.

Fixed block designs and data are whitened once. Linear intensity solves and
final rank/covariance analysis use equilibrated columns: changing an intensity
unit must not manufacture a missing reciprocal-space constraint.

Covariances are local Gauss--Newton covariances for KNOWN absolute measurement
covariance, without reduced-chi-square rescaling. For deficient rank the returned
pseudoinverse describes only the identifiable subspace; zero variance along a
null direction is NOT a claim of certainty. Such a result is never marked converged.
Time limits are cooperative, checked around dense operations and in solver
callbacks; a running BLAS operation cannot be preempted.
"""
from __future__ import annotations

from time import monotonic

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares


class _StopFit(Exception):
    pass


def _result(reason):
    return dict(status='unresolved', reason=reason, converged=False, optimizer_converged=False, I=None,
                equivalence_ids=[], beta=None, fitted=None, residual=None,
                whitened_residual=None, parameter_order=[], joint_covariance=None,
                covariance_scope='local Gauss-Newton; known absolute covariance; equilibrated generalized inverse when rank deficient',
                covariance_identifiable_subspace_only=None,
                block_parameter_data_covariance=None, parameter_response=None,
                factors=None, factor_beta_jacobian=None, joint_jacobians=None, rank=None,
                singular_values=None, nullspace=None, identifiable_parameters=None,
                chi2=None, dof=None, reduced_chi2=None, gradient=None,
                gradient_norm=None, relative_chi2_change=None,
                gauge_diagnostics=None, trace=[])


def _svd(matrix, rcond):
    # full V is necessary for an underdetermined system, but full U for a tall
    # system would introduce an unnecessary measurements-squared allocation.
    _, singular, vt = np.linalg.svd(matrix, full_matrices=matrix.shape[0] < matrix.shape[1])
    cutoff = rcond * singular[0] if singular.size else 0.
    rank = int(np.count_nonzero(singular > cutoff))
    return rank, singular, vt


def _column_scales(matrix):
    """Nonzero column norms without squaring the original physical units."""
    peak = np.max(np.abs(matrix), axis=0)
    safe = np.where(peak > 0, peak, 1.)
    lengths = np.linalg.norm(matrix / safe, axis=0)
    return safe * np.where(lengths > 0, lengths, 1.)


def _equilibrated_lstsq(matrix, target, rcond):
    """Solve signed WLS in dimensionless amplitude coordinates."""
    scales = _column_scales(matrix)
    return np.linalg.lstsq(matrix / scales, target, rcond=rcond)[0] / scales


def _amplitude_arithmetic_resolution(matrix, target, intensity, rcond):
    """Estimate floating-point cancellation in each signed linear amplitude.

    This is an arithmetic audit, not a statistical standard error or SNR cut.
    A local |A^+| action propagates rounding-sized perturbations in both the
    data and predicted data. Disconnected exact support components are solved
    separately so an independent bright measurement cannot set the numerical
    floor of a genuinely tiny channel. The SVD itself is not an interval
    computation: these are resolution estimates, not certified error bounds.
    """
    support = matrix != 0
    adjacent = support.T @ support
    remaining = set(range(matrix.shape[1]))
    resolution = np.zeros(matrix.shape[1])
    groups = []
    epsilon = np.finfo(float).eps
    while remaining:
        connected = {min(remaining)}
        frontier = list(connected)
        while frontier:
            column = frontier.pop()
            found = set(np.flatnonzero(adjacent[column])) & remaining - connected
            connected.update(found)
            frontier.extend(found)
        remaining -= connected
        columns = np.asarray(sorted(connected), dtype=int)
        rows = np.flatnonzero(np.any(support[:, columns], axis=1))
        if not len(rows):
            continue
        local = matrix[np.ix_(rows, columns)]
        scales = _column_scales(local)
        u, singular, vt = np.linalg.svd(local / scales, full_matrices=False)
        rank = int(np.count_nonzero(singular > rcond * singular[0]))
        inverse = ((vt[:rank].T / singular[:rank]) @ u[:, :rank].T) / scales[:, None]
        # gamma_l is the standard accumulated-roundoff factor for l operations;
        # the local operation count includes both dot-product actions. This does
        # not replace a rigorous componentwise backward bound for the SVD.
        rounding = epsilon * (len(rows) + 2*len(columns) + 1)
        gamma = rounding / (1-rounding)
        magnitude = np.abs(target[rows]) + np.abs(local) @ np.abs(intensity[columns])
        resolution[columns] = gamma * (np.abs(inverse) @ magnitude + np.abs(intensity[columns]))
        groups.append(dict(columns=columns.tolist(), data_rows=len(rows), gamma=gamma))
    return resolution, groups


def _linearized_statistics(jacobian, rcond):
    """SVD response and covariance without forming a normal matrix.

    J = U diag(s) V.T diag(scales).  The linearized response to whitened
    data is diag(1/scales) V diag(1/s) U.T.  Null vectors are returned in
    physical parameter coordinates; identifiability is tested before that
    coordinate conversion so it cannot depend on unit magnitudes.
    """
    scales = _column_scales(jacobian)
    normalized = jacobian / scales
    u, singular, vt = np.linalg.svd(
        normalized, full_matrices=normalized.shape[0] < normalized.shape[1])
    rank = int(np.count_nonzero(singular > rcond * singular[0])) if singular.size else 0
    root = (vt[:rank].T / singular[:rank]) / scales[:, None]
    response = root @ u[:, :rank].T
    covariance = root @ root.T
    scaled_nullspace = vt[rank:]
    nullspace = scaled_nullspace / scales
    # An orthonormal basis in the declared physical parameter coordinates.
    if len(nullspace):
        nullspace = np.linalg.qr(nullspace.T, mode='reduced')[0].T
    return dict(rank=rank, singular_values=singular, nullspace=nullspace,
                identifiable_parameters=np.linalg.norm(scaled_nullspace, axis=0) <= 1e-8,
                parameter_scales=scales,
                singular_value_coordinates='column-equilibrated whitened Jacobian',
                covariance=covariance, whitened_response=response)


def _stationary_curvature(jw, residual, intensity, prepared, whitened_designs, gains):
    """Check the exact local objective curvature, in equilibrated coordinates.

    Gauss--Newton covariance uses J.T J, but local minimality also depends on
    sum_i r_i Hessian(mu_i). A negative direction of this exact Hessian proves
    that a stationary full-rank fit is not a local minimum. Nonnegative
    curvature is not a guarantee against higher-order or global alternatives.
    """
    n = len(intensity)
    scales = _column_scales(jw)
    normalized = jw / scales
    hessian = normalized.T @ normalized
    cross = np.zeros((n, jw.shape[1]-n))
    beta_curvature = np.zeros((jw.shape[1]-n, jw.shape[1]-n))
    start = 0
    for block, design, gain in zip(prepared, whitened_designs, gains):
        size = len(block[0])
        x, indices = block[3], block[5]
        weighted_residual = gain * (design.T @ residual[start:start+size])
        xs = x / scales[n:]
        np.add.at(cross, indices, (weighted_residual/scales[indices])[:, None] * xs)
        beta_curvature += xs.T @ ((weighted_residual*intensity[indices])[:, None] * xs)
        start += size
    hessian[:n, n:] += cross
    hessian[n:, :n] += cross.T
    hessian[n:, n:] += beta_curvature
    eigenvalues = np.linalg.eigvalsh((hessian+hessian.T)*.5)
    rounding_scale = (len(residual)+4*jw.shape[1])*np.finfo(float).eps*max(1., np.linalg.norm(hessian, ord=np.inf))
    return dict(min_eigenvalue=float(eigenvalues[0]), rounding_scale=float(rounding_scale),
                negative_curvature=bool(eigenvalues[0] < -rounding_scale),
                coordinates='column-equilibrated physical model Jacobian',
                scope='exact local Hessian of chi-square/2; detects negative curvature, not a global or higher-order minimum certificate')


def fit_global(blocks, *, max_iterations=100, tolerance=1e-8, rcond=1e-10,
               max_parameters=256, max_seconds=30., max_measurements=4096,
               optimizer='alternating'):
    """Fit mu_b = design_b @ (known_b * exp(features_b @ beta) * I[id_b]).

    optimizer='profiled' eliminates linear signed intensities at every nonlinear
    scale trial, using the full residual-dependent variable-projection Jacobian.
    A rank-deficient linear design falls back to the original alternating path.
    The default remains 'alternating'; both use the same final physical-model
    rank, curvature, covariance and export acceptance conditions.

    Each block is a dict with y (M,), covariance (M,M) symmetric SPD,
    equivalence_ids (list of K nonempty strings), design (M,K), features (K,P),
    and known_factors (K,). P is shared by all blocks and may be zero. Every
    supplied row is valid: invalid shapes, NaNs, infinities, repeated block
    objects, and non-SPD covariance raise ValueError.
    Repeated *values* in independent blocks are legitimate observations.

    Finite limits: max_iterations counts outer alternating cycles; each inner
    least_squares has at most 100 residual evaluations. max_parameters bounds
    len(I)+P; max_measurements bounds sum(M), and contribution records are capped
    at 4*max_measurements. Dimension/time/iteration/solver failures return
    status='unresolved', with unavailable outputs set to None. max_iterations=0
    returns the initial WLS fit without claiming convergence.

    Output I follows first appearance in equivalence_ids; parameter_order is
    [('I', id), ..., ('beta', index), ...]. fitted/residual/whitened_residual
    are lists in block order; residual is prediction minus y. trace contains
    accepted states (including a retained state when a proposal is rejected),
    with monotonically nonincreasing chi2. dof = M-rank.
    gradient = Jw.T @ rw (gradient of chi2/2); gradient_norm is its maximum
    column-normalized magnitude divided by max(1, sqrt(chi2)). Success requires
    gradient_norm <= tolerance AND relative chi2 change <= tolerance, valid
    centering, full joint rank, and no detected negative local curvature.
    Iteration/solver failure never yields status='converged'.

    joint_covariance is the full inverse information when identifiable; otherwise
    a generalized covariance for estimable contrasts, obtained by column-scaled
    SVD without regularization or variance rescaling. singular_values refer to
    the column-equilibrated whitened Jacobian and parameter_scales undo that
    normalization. nullspace rows follow physical parameter_order. parameter_response has
    shape (len(I)+P, sum(M)), with columns in concatenated block/row order:
    Cov(theta) @ Jw.T @ blockdiag(L_b^-1). This is the requested linearized
    response, not a residual-Hessian correction to a nonlinear estimator.
    If floating-point cancellation makes a fitted amplitude numerically zero,
    its induced scale derivatives cannot establish rank. arithmetic_resolution
    records this non-statistical audit; information_jacobians is supplied when
    rank/covariance use that audited linearization. Model derivatives and I are
    retained unchanged. Loss of a scale direction prevents convergence status.
    block_parameter_data_covariance[b] = Cov(theta) @ J_raw_b.T; equivalently
    parameter_response[:, block_rows] @ covariance_b. factors[b] = 1/g_b,
    factor_beta_jacobian[b] = -f_b[:,None]*features_b. These allow the caller
    to propagate corrections and correlations with the same observations.
    """
    if optimizer not in ('alternating', 'profiled'):
        raise ValueError('optimizer must be alternating or profiled')
    for name, value, minimum in [('max_iterations', max_iterations, 0),
                                  ('max_parameters', max_parameters, 1),
                                  ('max_measurements', max_measurements, 1)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    for name, value in [('tolerance', tolerance), ('rcond', rcond), ('max_seconds', max_seconds)]:
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if tolerance >= 1 or not np.finfo(float).eps <= rcond < 1:
        raise ValueError('Require tolerance < 1 and machine epsilon <= rcond < 1')
    if not isinstance(blocks, list) or not blocks:
        raise ValueError('blocks must be a nonempty list of dicts')
    if len({id(b) for b in blocks}) != len(blocks):
        raise ValueError('Duplicate block objects would count the same observation twice')
    out = _result('not_started')
    deadline = monotonic() + max_seconds

    def check_time():
        if monotonic() >= deadline:
            raise _StopFit('time_budget_exceeded')

    prepared, order, lookup = [], [], {}
    measurements = contributions = 0
    p = None
    try:
        for b in blocks:
            check_time()
            if not isinstance(b, dict):
                raise ValueError('Each block must be a dict')
            required = ('y', 'covariance', 'equivalence_ids', 'design', 'features', 'known_factors')
            if any(key not in b for key in required):
                raise ValueError('Missing block field; require ' + ', '.join(required))
            ids = b['equivalence_ids']
            if not isinstance(ids, list) or not ids or any(not isinstance(s, str) or not s for s in ids):
                raise ValueError('equivalence_ids must be a nonempty list of nonempty strings')
            arrays = []
            for key in ('y', 'covariance', 'design', 'features', 'known_factors'):
                if np.iscomplexobj(b[key]):
                    raise ValueError(f'{key} must be real')
                try:
                    a = np.asarray(b[key], dtype=float)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f'{key} must be a real numeric array') from exc
                if not np.all(np.isfinite(a)):
                    raise ValueError(f'{key} must be finite')
                arrays.append(a)
            y, cov, design, features, known = arrays
            k = len(ids)
            if y.ndim != 1 or y.size == 0:
                raise ValueError('y must have shape (M,) with M > 0')
            m = len(y)
            if cov.shape != (m, m) or design.shape != (m, k) or known.shape != (k,):
                raise ValueError('Require covariance (M,M), design (M,K), known_factors (K,)')
            if features.ndim != 2 or features.shape[0] != k:
                raise ValueError('features must have shape (K,P)')
            if p is None:
                p = features.shape[1]
            if features.shape[1] != p:
                raise ValueError('All blocks must share the same feature count P')
            if np.any(known <= 0):
                raise ValueError('known_factors must be strictly positive')
            if not np.allclose(cov, cov.T, rtol=1e-12, atol=0.):
                raise ValueError('covariance must be symmetric SPD')
            measurements += m
            contributions += k
            if measurements > max_measurements:
                raise _StopFit('measurement_budget_exceeded')
            if contributions > 4 * max_measurements:
                raise _StopFit('contribution_budget_exceeded')
            indices = []
            for label in ids:
                if label not in lookup:
                    lookup[label] = len(order)
                    order.append(label)
                indices.append(lookup[label])
            if len(order) + p > max_parameters:
                raise _StopFit('parameter_budget_exceeded')
            try:
                lower = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError as exc:
                raise ValueError('covariance must be symmetric SPD') from exc
            check_time()
            prepared.append((y, lower, design, features, known, np.array(indices)))

        n = len(order)
        q = n + p
        out.update(equivalence_ids=order, beta=np.zeros(p),
                   parameter_order=[('I', s) for s in order] + [('beta', j) for j in range(p)])
        # Centering is over contribution records, not measurements or design
        # weights. A nonzero mean is never silently removed by this kernel.
        all_x = np.vstack([b[3] for b in prepared])
        all_ids = np.concatenate([b[5] for b in prepared])
        means = np.zeros((n, p))
        np.add.at(means, all_ids, all_x)
        means /= np.bincount(all_ids, minlength=n)[:, None]
        centered = all_x - means[all_ids]
        if p:
            feature_scales = _column_scales(all_x)
            feature_rank, feature_singular, _ = _svd(all_x / feature_scales, rcond)
            _, centered_singular, centered_vt = _svd(centered / feature_scales, rcond)
            # Relative to the original normalized features, not a tiny remainder
            # of subtracting identical group means. Numerical cancellation cannot
            # create a within-equivalence scale constraint.
            centered_rank = int(np.count_nonzero(centered_singular > rcond * max(1., centered_singular[0])))
            centered_ok = bool(np.max(np.abs(all_x.mean(axis=0)) / feature_scales) <= 1e-10)
            beta_gauge_nullspace = centered_vt[centered_rank:] / feature_scales
            if len(beta_gauge_nullspace):
                beta_gauge_nullspace = np.linalg.qr(beta_gauge_nullspace.T, mode='reduced')[0].T
        else:
            feature_rank = centered_rank = 0
            feature_singular = centered_singular = np.empty(0)
            beta_gauge_nullspace = np.empty((0, 0))
            centered_ok = True
        out['gauge_diagnostics'] = dict(centered=centered_ok, global_feature_mean=all_x.mean(axis=0),
                                       group_feature_means=means,
                                       feature_rank=feature_rank, feature_singular_values=feature_singular,
                                       centered_feature_rank=centered_rank,
                                       centered_feature_singular_values=centered_singular,
                                       beta_gauge_nullspace=beta_gauge_nullspace)
        check_time()
        # L is fixed: L^-1 D diag(g) = (L^-1 D) diag(g).
        # Repeated triangular solves inside each nonlinear callback are unnecessary.
        whitened_designs = [solve_triangular(b[1], b[2], lower=True, check_finite=False)
                            for b in prepared]
        whitened_data = [solve_triangular(b[1], b[0], lower=True, check_finite=False)
                         for b in prepared]
        yw = np.concatenate(whitened_data)

        cached_parameters = cached_model = None

        def model(intensity, beta):
            nonlocal cached_parameters, cached_model
            check_time()
            # 残差与雅可比共享最近一次精确参数结果；副本防止求解器原地修改参数。
            if (cached_parameters is not None
                    and np.array_equal(intensity, cached_parameters[0])
                    and np.array_equal(beta, cached_parameters[1])):
                return cached_model
            matrices, derivatives, gains = [], [], []
            with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
                for (_, _, _, x, known, indices), design in zip(prepared, whitened_designs):
                    gain = known * np.exp(x @ beta)
                    if np.any(gain <= 0) or not np.all(np.isfinite(gain)):
                        raise _StopFit('nonfinite_model')
                    a = np.zeros((len(design), n))
                    for column, index in enumerate(indices):
                        a[:, index] += design[:, column] * gain[column]
                    jb = design @ ((gain * intensity[indices])[:, None] * x)
                    matrices.append(a)
                    derivatives.append(jb)
                    gains.append(gain)
                jw = np.column_stack((np.vstack(matrices), np.vstack(derivatives)))
                residual = jw[:, :n] @ intensity - yw
            if not np.all(np.isfinite(jw)) or not np.all(np.isfinite(residual)):
                raise _StopFit('nonfinite_model')
            check_time()
            cached_parameters = intensity.copy(), beta.copy()
            cached_model = residual, jw, gains
            return cached_model

        intensity = np.zeros(n)
        beta = np.zeros(p)
        rw, jw, gains = model(intensity, beta)
        intensity = _equilibrated_lstsq(jw[:, :n], yw, rcond)
        rw, jw, gains = model(intensity, beta)
        chi2 = float(rw @ rw)
        if not np.isfinite(chi2):
            raise _StopFit('nonfinite_objective')
        trace = out['trace']
        trace.append(dict(iteration=0, phase='initial_wls', chi2=chi2))
        relative_change = None
        converged = False
        reason = 'iteration_budget_exceeded'

        def accept(candidate_i, candidate_beta, iteration, phase, solver=None, increment=None):
            nonlocal intensity, beta, rw, jw, gains, chi2, cached_parameters, cached_model
            new_r, new_j, new_gains = model(candidate_i, candidate_beta)
            if increment is None:
                new_chi2 = float(new_r @ new_r)
            else:
                # The objective difference can be much smaller than either
                # objective. Evaluate it before cancellation of two large sums.
                new_chi2 = chi2 + float(increment @ (2*rw + increment))
                new_r = rw + increment
            accepted = np.isfinite(new_chi2) and new_chi2 <= chi2
            if accepted:
                intensity, beta = candidate_i.copy(), candidate_beta.copy()
                rw, jw, gains, chi2 = new_r, new_j, new_gains, max(new_chi2, 0.)
                if increment is not None:
                    cached_parameters = intensity.copy(), beta.copy()
                    cached_model = rw, jw, gains
            entry = dict(iteration=iteration, phase=phase, chi2=chi2, accepted=bool(accepted))
            if solver is not None:
                entry.update(solver_status=int(solver.status), nfev=int(solver.nfev))
            trace.append(entry)
            return bool(accepted)

        def gradient_norm():
            gradient = jw.T @ rw
            norms = _column_scales(jw)
            scaled = np.divide(np.abs(gradient), norms, out=np.zeros(q), where=norms > 0)
            return gradient, float(np.max(scaled) / max(1., np.sqrt(chi2)))

        # Solver tolerances cannot be below machine precision. Readiness is
        # checked separately against the caller's actual gradient tolerance.
        solver_tol = max(1e-14, min(tolerance * .001, 1e-12))
        profiled = None
        if optimizer == 'profiled' and p:
            from variable_projection import ProfiledScaleModel
            profiled = ProfiledScaleModel(prepared, whitened_designs, yw, n, rcond, check_time,
                                         include_joint_jacobian=False)
        out['optimization_method'] = ('full-rank variable projection with exact residual Jacobian'
                                      if profiled else 'alternating signed WLS and scale least squares')
        try:
            for iteration in range(1, max_iterations + 1):
                check_time()
                before = chi2
                if profiled is not None:
                    from variable_projection import RankDeficientProfile
                    try:
                        fit = least_squares(lambda value: profiled.state(value)['residual'], beta,
                            jac=lambda value: profiled.state(value)['jacobian'], x_scale='jac',
                            max_nfev=100, ftol=solver_tol, xtol=solver_tol, gtol=solver_tol)
                        state = profiled.state(fit.x)
                        accept(state['intensity'], fit.x, iteration, 'variable_projection', fit)
                        if not fit.success:
                            raise _StopFit('profiled_solver_failed')
                    except RankDeficientProfile:
                        # A changing/rank-deficient linear subproblem lacks the
                        # derivative assumed here. Retain the legacy route and
                        # its full rank diagnostics; never add a ridge to proceed.
                        out['profiled_fallback'] = 'rank-deficient intensity design'
                        profiled = None
                        out['optimization_method'] = 'alternating after rank-deficient variable projection'
                if p and profiled is None:
                    frozen_i = intensity.copy()

                    def beta_fun(value):
                        return model(frozen_i, value)[0]

                    def beta_jac(value):
                        return model(frozen_i, value)[1][:, n:]

                    fit = least_squares(beta_fun, beta, jac=beta_jac, x_scale='jac', max_nfev=100,
                                        ftol=solver_tol, xtol=solver_tol, gtol=solver_tol)
                    accept(frozen_i, fit.x, iteration, 'fixed_intensity_beta', fit)
                    if not fit.success:
                        raise _StopFit('beta_solver_failed')
                if profiled is None:
                    wls_i = _equilibrated_lstsq(jw[:, :n], yw, rcond)
                    accept(wls_i, beta, iteration, 'fixed_beta_wls')
                # Accelerate strongly coupled intensity/scale directions only
                # after actually executing both alternating steps.
                if p and profiled is None and gradient_norm()[1] > tolerance:
                    def joint_fun(theta):
                        return model(theta[:n], theta[n:])[0]

                    def joint_jac(theta):
                        return model(theta[:n], theta[n:])[1]

                    fit = least_squares(joint_fun, np.r_[intensity, beta], jac=joint_jac,
                                        x_scale='jac', max_nfev=100,
                                        ftol=solver_tol, xtol=solver_tol, gtol=solver_tol)
                    accept(fit.x[:n], fit.x[n:], iteration, 'joint_polish', fit)
                    if not fit.success:
                        raise _StopFit('joint_solver_failed')
                    wls_i = _equilibrated_lstsq(jw[:, :n], yw, rcond)
                    accept(wls_i, beta, iteration, 'fixed_beta_wls_after_polish')
                # Close to a stationary noisy fit, subtraction of two nearly
                # equal predictions/chi-squares can reject a valid tiny step and
                # repeat the entire alternating cycle indefinitely. Refine with
                # an equilibrated Gauss--Newton step and evaluate its *exact*
                # exponential prediction increment using expm1.
                if p and gradient_norm()[1] > tolerance and before - chi2 <= tolerance * max(1., before):
                    direction = _equilibrated_lstsq(jw, -rw, rcond)
                    for step in (1., .5, .25, .125):
                        candidate_i = intensity + step * direction[:n]
                        candidate_beta = beta + step * direction[n:]
                        if profiled is not None:
                            candidate_i = profiled.state(candidate_beta)['intensity']
                        delta_beta = candidate_beta - beta
                        increments = []
                        for (_, _, _, x, _, indices), design, gain in zip(prepared, whitened_designs, gains):
                            delta_i = candidate_i[indices] - intensity[indices]
                            increments.append(design @ (gain * (
                                delta_i + candidate_i[indices] * np.expm1(x @ delta_beta))))
                        increment = np.concatenate(increments)
                        if accept(candidate_i, candidate_beta, iteration, 'compensated_stationarity', increment=increment):
                            break
                relative_change = (before - chi2) / max(1., before)
                grad, grad_norm = gradient_norm()
                trace[-1].update(gradient_norm=grad_norm, relative_chi2_change=relative_change)
                if grad_norm <= tolerance and relative_change <= tolerance:
                    converged, reason = True, 'converged'
                    break
        except _StopFit as exc:
            reason = str(exc)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError) as exc:
            reason = 'numerical_solver_failure'
            out['solver_error'] = str(exc)

        # Preserve the last accepted fit even when diagnostics hit the budget.
        out.update(I=intensity, beta=beta, chi2=chi2, optimizer_converged=converged,
                   reason=reason, relative_chi2_change=relative_change)
        if profiled is not None:
            out['profiled_evaluations'] = profiled.evaluations
        out['gradient'], out['gradient_norm'] = gradient_norm()
        offsets = np.cumsum([0] + [len(b[0]) for b in prepared])
        # Raw derivatives are needed for reporting once, not at every solver step.
        raw_blocks = []
        for (_, _, design, x, _, indices), gain in zip(prepared, gains):
            a = np.zeros((len(design), n))
            for column, index in enumerate(indices):
                a[:, index] += design[:, column] * gain[column]
            raw_blocks.append(np.column_stack((a, design @ ((gain * intensity[indices])[:, None] * x))))
        raw_j = np.vstack(raw_blocks)
        out['joint_jacobians'] = raw_blocks
        out['fitted'] = [raw_j[lo:hi, :n] @ intensity for lo, hi in zip(offsets[:-1], offsets[1:])]
        out['residual'] = [pred - b[0] for pred, b in zip(out['fitted'], prepared)]
        out['whitened_residual'] = [rw[lo:hi].copy() for lo, hi in zip(offsets[:-1], offsets[1:])]
        with np.errstate(over='raise', divide='raise', invalid='raise'):
            out['factors'] = [1. / g for g in gains]
            out['factor_beta_jacobian'] = [-f[:, None] * b[3] for f, b in zip(out['factors'], prepared)]
        check_time()
        information_jw = jw
        nominal_rank = None
        if p:
            with np.errstate(over='raise', divide='raise', invalid='raise'):
                resolution, resolution_groups = _amplitude_arithmetic_resolution(jw[:, :n], yw, intensity, rcond)
            numerically_supported = np.abs(intensity) > resolution
            audited_intensity = np.where(numerically_supported, intensity, 0.)
            out['arithmetic_resolution'] = dict(
                amplitude_resolution_estimate=resolution,
                numerically_supported_amplitudes=numerically_supported,
                support_components=resolution_groups,
                scope='floating-point cancellation audit only; not sigma, SNR, a detection threshold, or a certified interval')
            if not np.all(numerically_supported):
                audited_beta = np.vstack([
                    design @ ((gain * audited_intensity[b[5]])[:, None] * b[3])
                    for b, design, gain in zip(prepared, whitened_designs, gains)])
                if not np.array_equal(audited_beta, jw[:, n:]):
                    nominal_rank = _svd(jw / _column_scales(jw), rcond)[0]
                    information_jw = np.column_stack((jw[:, :n], audited_beta))
                    out['arithmetic_resolution']['nominal_rank_before_audit'] = nominal_rank
                    # Public model derivatives remain the derivatives at the
                    # reported I. Distinguish the audited information model used
                    # for effective rank/covariance instead of changing them silently.
                    out['information_jacobians'] = [b[1] @ information_jw[lo:hi]
                        for b, lo, hi in zip(prepared, offsets[:-1], offsets[1:])]
                    out['covariance_scope'] = (
                        'local Gauss-Newton of arithmetic-audited linearization (information_jacobians); '
                        'known absolute covariance; estimable-subspace generalized inverse; '
                        'model joint_jacobians and fitted intensities remain unaltered')
        with np.errstate(over='raise', divide='raise', invalid='raise'):
            statistics = _linearized_statistics(information_jw, rcond)
        rank = statistics['rank']
        covariance = statistics.pop('covariance')
        whitened_response = statistics.pop('whitened_response')
        out.update(statistics, dof=measurements - rank,
                   reduced_chi2=chi2 / (measurements - rank) if measurements > rank else None,
                   covariance_identifiable_subspace_only=rank < q)
        check_time()
        with np.errstate(over='raise', divide='raise', invalid='raise'):
            # Use J^+ directly. C @ J.T would multiply two factors with opposite
            # large scales and unnecessarily square the condition number.
            cross = [whitened_response[:, lo:hi] @ b[1].T
                     for b, lo, hi in zip(prepared, offsets[:-1], offsets[1:])]
            response = np.hstack([solve_triangular(b[1].T, whitened_response[:, lo:hi].T,
                                                  lower=False, check_finite=False).T
                                  for b, lo, hi in zip(prepared, offsets[:-1], offsets[1:])])
        check_time()
        out.update(joint_covariance=covariance, block_parameter_data_covariance=cross,
                   parameter_response=response)
        if converged:
            if nominal_rank is not None and rank < nominal_rank:
                out['reason'] = 'scale_information_below_arithmetic_resolution'
            elif rank < q:
                out['reason'] = 'rank_deficient'
            elif not centered_ok:
                out['reason'] = 'uncentered_features'
            else:
                if p:
                    with np.errstate(over='raise', divide='raise', invalid='raise'):
                        curvature = _stationary_curvature(jw, rw, intensity, prepared, whitened_designs, gains)
                    out['stationary_curvature'] = curvature
                    check_time()
                    if curvature['negative_curvature']:
                        out['reason'] = 'negative_curvature_stationary_point'
                    else:
                        out.update(status='converged', reason='converged', converged=True)
                else:
                    out.update(status='converged', reason='converged', converged=True)
    except _StopFit as exc:
        out.update(status='unresolved', reason=str(exc), converged=False)
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        out.update(status='unresolved', reason='numerical_failure', solver_error=str(exc), converged=False)
    return out
