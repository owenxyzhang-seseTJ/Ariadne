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
                covariance_scope='local Gauss-Newton; known absolute covariance',
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


def fit_global(blocks, *, max_iterations=100, tolerance=1e-8, rcond=1e-10,
               max_parameters=256, max_seconds=30., max_measurements=4096):
    """Fit mu_b = design_b @ (known_b * exp(features_b @ beta) * I[id_b]).

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
    centering, and full joint rank. Iteration/solver failure never yields status='converged'.

    joint_covariance = (Jw.T Jw)^+ via SVD, without regularization or variance
    rescaling. nullspace rows follow parameter_order. parameter_response has
    shape (len(I)+P, sum(M)), with columns in concatenated block/row order:
    Cov(theta) @ Jw.T @ blockdiag(L_b^-1). This is the requested linearized
    response, not a residual-Hessian correction to a nonlinear estimator.
    block_parameter_data_covariance[b] = Cov(theta) @ J_raw_b.T; equivalently
    parameter_response[:, block_rows] @ covariance_b. factors[b] = 1/g_b,
    factor_beta_jacobian[b] = -f_b[:,None]*features_b. These allow the caller
    to propagate corrections and correlations with the same observations.
    """
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
            feature_rank, feature_singular, _ = _svd(all_x, rcond)
            centered_rank, centered_singular, centered_vt = _svd(centered, rcond)
            scale = max(1., float(np.max(np.abs(all_x))))
            centered_ok = bool(np.max(np.abs(all_x.mean(axis=0))) <= 1e-10 * scale)
            beta_gauge_nullspace = centered_vt[centered_rank:]
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
        yw = np.concatenate([solve_triangular(b[1], b[0], lower=True, check_finite=False)
                             for b in prepared])

        cached_parameters = cached_model = None

        def model(intensity, beta):
            nonlocal cached_parameters, cached_model
            check_time()
            # 残差与雅可比共享最近一次精确参数结果；副本防止求解器原地修改参数。
            if (cached_parameters is not None
                    and np.array_equal(intensity, cached_parameters[0])
                    and np.array_equal(beta, cached_parameters[1])):
                return cached_model
            raw_a, raw_b, gains = [], [], []
            with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
                for _, lower, design, x, known, indices in prepared:
                    gain = known * np.exp(x @ beta)
                    if np.any(gain <= 0) or not np.all(np.isfinite(gain)):
                        raise _StopFit('nonfinite_model')
                    a = np.zeros((len(design), n))
                    for column, index in enumerate(indices):
                        a[:, index] += design[:, column] * gain[column]
                    jb = design @ ((gain * intensity[indices])[:, None] * x)
                    raw_a.append(a)
                    raw_b.append(jb)
                    gains.append(gain)
                raw_j = np.vstack([np.column_stack((a, jb)) for a, jb in zip(raw_a, raw_b)])
                jw = np.vstack([solve_triangular(b[1], np.column_stack((a, jb)), lower=True,
                                                check_finite=False)
                                for b, a, jb in zip(prepared, raw_a, raw_b)])
                residual = np.concatenate([solve_triangular(
                    b[1], a @ intensity - b[0], lower=True, check_finite=False)
                    for b, a in zip(prepared, raw_a)])
            if not np.all(np.isfinite(jw)) or not np.all(np.isfinite(residual)):
                raise _StopFit('nonfinite_model')
            check_time()
            cached_parameters = intensity.copy(), beta.copy()
            cached_model = residual, jw, raw_j, gains
            return cached_model

        intensity = np.zeros(n)
        beta = np.zeros(p)
        rw, jw, raw_j, gains = model(intensity, beta)
        intensity = np.linalg.lstsq(jw[:, :n], yw, rcond=rcond)[0]
        rw, jw, raw_j, gains = model(intensity, beta)
        chi2 = float(rw @ rw)
        if not np.isfinite(chi2):
            raise _StopFit('nonfinite_objective')
        trace = out['trace']
        trace.append(dict(iteration=0, phase='initial_wls', chi2=chi2))
        relative_change = np.inf
        converged = False
        reason = 'iteration_budget_exceeded'

        def accept(candidate_i, candidate_beta, iteration, phase, solver=None):
            nonlocal intensity, beta, rw, jw, raw_j, gains, chi2
            new_r, new_j, new_raw, new_gains = model(candidate_i, candidate_beta)
            new_chi2 = float(new_r @ new_r)
            accepted = np.isfinite(new_chi2) and new_chi2 <= chi2
            if accepted:
                intensity, beta = candidate_i.copy(), candidate_beta.copy()
                rw, jw, raw_j, gains, chi2 = new_r, new_j, new_raw, new_gains, new_chi2
            entry = dict(iteration=iteration, phase=phase, chi2=chi2, accepted=bool(accepted))
            if solver is not None:
                entry.update(solver_status=int(solver.status), nfev=int(solver.nfev))
            trace.append(entry)

        def gradient_norm():
            gradient = jw.T @ rw
            norms = np.linalg.norm(jw, axis=0)
            scaled = np.divide(np.abs(gradient), norms, out=np.zeros(q), where=norms > 0)
            return gradient, float(np.max(scaled) / max(1., np.sqrt(chi2)))

        # Solver tolerances cannot be below machine precision. Readiness is
        # checked separately against the caller's actual gradient tolerance.
        solver_tol = max(1e-14, min(tolerance * .001, 1e-12))
        try:
            for iteration in range(1, max_iterations + 1):
                check_time()
                before = chi2
                if p:
                    frozen_i = intensity.copy()

                    def beta_fun(value):
                        return model(frozen_i, value)[0]

                    def beta_jac(value):
                        return model(frozen_i, value)[1][:, n:]

                    fit = least_squares(beta_fun, beta, jac=beta_jac, max_nfev=100,
                                        ftol=solver_tol, xtol=solver_tol, gtol=solver_tol)
                    accept(frozen_i, fit.x, iteration, 'fixed_intensity_beta', fit)
                    if not fit.success:
                        raise _StopFit('beta_solver_failed')
                wls_i = np.linalg.lstsq(jw[:, :n], yw, rcond=rcond)[0]
                accept(wls_i, beta, iteration, 'fixed_beta_wls')
                # Accelerate strongly coupled intensity/scale directions only
                # after actually executing both alternating steps.
                if p and gradient_norm()[1] > tolerance:
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
                    wls_i = np.linalg.lstsq(jw[:, :n], yw, rcond=rcond)[0]
                    accept(wls_i, beta, iteration, 'fixed_beta_wls_after_polish')
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
        out['gradient'], out['gradient_norm'] = gradient_norm()
        offsets = np.cumsum([0] + [len(b[0]) for b in prepared])
        out['joint_jacobians'] = [raw_j[lo:hi].copy() for lo, hi in zip(offsets[:-1], offsets[1:])]
        out['fitted'] = [raw_j[lo:hi, :n] @ intensity for lo, hi in zip(offsets[:-1], offsets[1:])]
        out['residual'] = [pred - b[0] for pred, b in zip(out['fitted'], prepared)]
        out['whitened_residual'] = [rw[lo:hi].copy() for lo, hi in zip(offsets[:-1], offsets[1:])]
        with np.errstate(over='raise', divide='raise', invalid='raise'):
            out['factors'] = [1. / g for g in gains]
            out['factor_beta_jacobian'] = [-f[:, None] * b[3] for f, b in zip(out['factors'], prepared)]
        check_time()
        rank, singular, vt = _svd(jw, rcond)
        out.update(rank=rank, singular_values=singular, nullspace=vt[rank:],
                   identifiable_parameters=np.linalg.norm(vt[rank:], axis=0) <= 1e-8,
                   dof=measurements - rank,
                   reduced_chi2=chi2 / (measurements - rank) if measurements > rank else None,
                   covariance_identifiable_subspace_only=rank < q)
        check_time()
        with np.errstate(over='raise', divide='raise', invalid='raise'):
            inverse_root = vt[:rank].T / singular[:rank]
            covariance = inverse_root @ inverse_root.T
            cross = [covariance @ raw_j[lo:hi].T for lo, hi in zip(offsets[:-1], offsets[1:])]
            response = np.hstack([solve_triangular(b[1].T, (covariance @ jw[lo:hi].T).T,
                                                  lower=False, check_finite=False).T
                                  for b, lo, hi in zip(prepared, offsets[:-1], offsets[1:])])
        check_time()
        out.update(joint_covariance=covariance, block_parameter_data_covariance=cross,
                   parameter_response=response)
        if converged:
            if rank < q:
                out['reason'] = 'rank_deficient'
            elif not centered_ok:
                out['reason'] = 'uncentered_features'
            else:
                out.update(status='converged', reason='converged', converged=True)
    except _StopFit as exc:
        out.update(status='unresolved', reason=str(exc), converged=False)
    except (FloatingPointError, np.linalg.LinAlgError) as exc:
        out.update(status='unresolved', reason='numerical_failure', solver_error=str(exc), converged=False)
    return out
