"""Structure-assisted inference in measured, signed intensity space.

One latent variable represents each unique intensity, regardless of how often it
is observed. The prior is diagonal plus low rank; measurement blocks are merged
only where they share latent variables. No dense global covariance is formed.
This module does not alter raw measurements or export posterior means as HKLF4.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import block_diag, cholesky, solve_triangular
from scipy.optimize import nnls


FloatArray = NDArray[np.float64]


def finite_array(value: ArrayLike, ndim: int, name: str) -> FloatArray:
    out = np.asarray(value, dtype=float)
    if out.ndim != ndim or not np.all(np.isfinite(out)):
        raise ValueError(f'{name}: require finite {ndim}-dimensional array')
    return out


def spd_factor(matrix: FloatArray, name: str) -> FloatArray:
    if matrix.shape[0] != matrix.shape[1] or not np.allclose(
        matrix, matrix.T, rtol=1e-12, atol=1e-14 * np.max(np.abs(matrix), initial=0.)
    ):
        raise ValueError(f'{name}: require symmetric square covariance')
    try:
        return cholesky(matrix, lower=True, check_finite=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f'{name}: covariance must be positive definite; no hidden ridge is added') from exc


def chol_solve(L: FloatArray, rhs: FloatArray) -> FloatArray:
    return solve_triangular(L.T, solve_triangular(L, rhs, lower=True, check_finite=False),
                            lower=False, check_finite=False)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


@dataclass(frozen=True)
class Observation:
    key: str
    columns: tuple[int, ...]
    design: FloatArray
    values: FloatArray
    covariance: FloatArray
    raw_ids: tuple[str, ...] = ()
    equivalence_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, obj: dict[str, Any], n: int) -> 'Observation':
        raw_columns = obj['columns']
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in raw_columns):
            raise ValueError('columns must be integer latent-variable indices')
        cols = tuple(int(v) for v in raw_columns)
        if not cols or len(set(cols)) != len(cols) or min(cols) < 0 or max(cols) >= n:
            raise ValueError('columns must be nonempty, unique, and in range')
        H = finite_array(obj['design'], 2, 'design')
        y = finite_array(obj['values'], 1, 'values')
        R = finite_array(obj['covariance'], 2, 'covariance')
        if H.shape != (len(y), len(cols)) or R.shape != (len(y), len(y)) or not len(y):
            raise ValueError('observation dimensions disagree')
        spd_factor(R, 'observation')
        key = str(obj['id'])
        if not key:
            raise ValueError('observation id must be nonempty')
        return cls(key, cols, H, y, R, tuple(map(str, obj.get('raw_ids', []))),
                   tuple(map(str, obj.get('equivalence_ids', []))))


def observations(items: list[dict[str, Any]], n: int) -> list[Observation]:
    blocks = [Observation.from_dict(item, n) for item in items]
    if len({b.key for b in blocks}) != len(blocks):
        raise ValueError('duplicate observation id')
    used: set[str] = set()
    for b in blocks:
        if len(set(b.raw_ids)) != len(b.raw_ids) or used.intersection(b.raw_ids):
            raise ValueError('raw pixels are counted more than once; merge into one correlated block')
        used.update(b.raw_ids)
    return blocks


def closed_split(items: list[dict[str, Any]], n: int, seed: str = '0',
                 fractions: tuple[float, float, float] = (.6, .2, .2)) -> dict[str, str]:
    """Split closed groups, never individual rows. Seed must be fixed before fitting.

    Shared latent intensities and declared equivalence/twin-orbit IDs join groups.
    Duplicate raw pixels are rejected before splitting, not rendered independent
    by assigning them to the same fold.
    """
    blocks = observations(items, n)
    if len(fractions) != 3 or min(fractions) <= 0 or not np.isclose(sum(fractions), 1):
        raise ValueError('three positive fractions must sum to one')
    uf = UnionFind(len(blocks))
    owners: dict[tuple[str, Any], int] = {}
    for i, b in enumerate(blocks):
        tokens = [('I', j) for j in b.columns] + [('E', j) for j in b.equivalence_ids]
        for token in tokens:
            if token in owners:
                uf.union(i, owners[token])
            else:
                owners[token] = i
    groups: dict[int, list[str]] = {}
    for i, b in enumerate(blocks):
        groups.setdefault(uf.find(i), []).append(b.key)
    if len(groups) < 3:
        raise ValueError('fewer than three independent closed groups: train/validation/test unavailable')
    ordered = sorted(groups.values(), key=lambda g: hashlib.sha256(
        (str(seed) + '\0' + '\0'.join(sorted(g))).encode()).hexdigest())
    nv = max(1, int(round(len(ordered) * fractions[1])))
    nt = max(1, int(round(len(ordered) * fractions[2])))
    while nv + nt >= len(ordered):
        if nv >= nt and nv > 1:
            nv -= 1
        elif nt > 1:
            nt -= 1
        else:
            break
    ntrain = len(ordered) - nv - nt
    return {key: ('train' if i < ntrain else 'validation' if i < ntrain + nv else 'test')
            for i, group in enumerate(ordered) for key in group}


def _groups(blocks: list[Observation], n: int) -> list[tuple[list[int], list[Observation]]]:
    uf = UnionFind(n)
    for block in blocks:
        for j in block.columns[1:]:
            uf.union(block.columns[0], j)
    cols: dict[int, list[int]] = {}
    by_block: dict[int, list[Observation]] = {}
    for j in range(n):
        cols.setdefault(uf.find(j), []).append(j)
    for b in blocks:
        by_block.setdefault(uf.find(b.columns[0]), []).append(b)
    return [(indices, by_block.get(key, [])) for key, indices in cols.items()]


def condition(mean: ArrayLike, independent_variance: ArrayLike, factors: ArrayLike,
              items: list[dict[str, Any]], *, max_local_elements: int = 4_000_000) -> dict[str, Any]:
    """Exact Gaussian conditioning with block-local and global low-rank factors.

    A prior supplied here is applied once per latent intensity, not per exposure.
    Return cov(I|y) as block factors L_g L_g.T plus a shared loading V V.T.
    Diagnostics explicitly separate measurement rank from posterior precision.
    """
    m = finite_array(mean, 1, 'mean')
    d = finite_array(independent_variance, 1, 'independent_variance')
    U = finite_array(factors, 2, 'factors')
    n, rank = len(m), U.shape[1]
    if n == 0 or d.shape != m.shape or U.shape[0] != n or np.any(d <= 0):
        raise ValueError('require positive independent variances and matching prior dimensions')
    if max_local_elements < 1:
        raise ValueError('max_local_elements must be positive')
    blocks = observations(items, n)
    A = np.eye(rank)
    rhs = np.zeros(rank)
    local: list[dict[str, Any]] = []
    logdet = 0.
    num_observed = 0
    data_rank = 0
    for columns, group in _groups(blocks, n):
        idx = np.asarray(columns)
        k = len(idx)
        p = sum(len(b.values) for b in group)
        if max(k*k, p*p, p*k, k*rank, p*rank) > max_local_elements:
            raise MemoryError('connected overlap group exceeds explicit local-matrix budget')
        if not group:
            local.append(dict(columns=columns, mean=m[idx].copy(), base=np.diag(np.sqrt(d[idx])),
                              loading=U[idx].copy(), W=np.zeros((0, rank)), z=np.zeros(0),
                              data_rank=0, identifiable=[False]*k))
            continue
        lookup = {j: i for i, j in enumerate(columns)}
        H = np.zeros((p, k))
        cursor = 0
        for b in group:
            H[cursor:cursor+len(b.values), [lookup[j] for j in b.columns]] = b.design
            cursor += len(b.values)
        y = np.concatenate([b.values for b in group])
        R = block_diag(*[b.covariance for b in group])
        LR = spd_factor(R, 'noise')
        s = np.sqrt(d[idx])
        T = solve_triangular(LR, H*s, lower=True, check_finite=False)
        t = solve_triangular(LR, y-H@m[idx], lower=True, check_finite=False)
        # QR of [I; R^-1/2 H D^1/2] preserves the prior precision square root.
        _, upper = np.linalg.qr(np.vstack((np.eye(k), T)), mode='reduced')
        inv_upper = solve_triangular(upper, np.eye(k), lower=False, check_finite=False)
        base = s[:, None]*inv_upper
        precision_solve = inv_upper @ inv_upper.T
        conditional_mean = m[idx] + s*(precision_solve @ (T.T@t))
        loading = s[:, None]*(precision_solve @ (U[idx]/s[:, None]))
        # Marginalizing the independent prior produces only local observation blocks.
        Q = R + (H*d[idx])@H.T
        LQ = spd_factor(Q, 'marginal noise')
        z = solve_triangular(LQ, y-H@m[idx], lower=True, check_finite=False)
        W = solve_triangular(LQ, H@U[idx], lower=True, check_finite=False)
        A += W.T@W
        rhs += W.T@z
        logdet += 2*np.log(np.diag(LQ)).sum()
        num_observed += p
        # Column equilibration prevents measurement units deciding numerical rank.
        Hw = solve_triangular(LR, H, lower=True, check_finite=False)
        scales = np.linalg.norm(Hw, axis=0)
        scaled = Hw/np.where(scales > 0, scales, 1.)
        _, singular, vt = np.linalg.svd(scaled, full_matrices=True)
        threshold = max(scaled.shape)*np.finfo(float).eps*(singular[0] if len(singular) else 0.)
        r = int(np.count_nonzero(singular > threshold))
        identifiable = (np.sum(vt[r:]**2, axis=0) < 1e-12).tolist()
        data_rank += r
        local.append(dict(columns=columns, mean=conditional_mean, base=base, loading=loading,
                          z=z, W=W, data_rank=r, identifiable=identifiable))
    LA = cholesky(A, lower=True)
    fmean = chol_solve(LA, rhs)
    froot = solve_triangular(LA.T, np.eye(rank), lower=False)
    logdet += 2*np.log(np.diag(LA)).sum()
    # Stable profiled quadratic: no subtraction of two large quadratic forms.
    quadratic = float(fmean@fmean)
    posterior = np.empty(n)
    variance = np.empty(n)
    identifiable_global = np.zeros(n, dtype=bool)
    outputs = []
    for g in local:
        idx = g['columns']
        factor = g['loading']@froot
        posterior[idx] = g['mean'] + g['loading']@fmean
        variance[idx] = np.sum(g['base']**2, axis=1) + np.sum(factor**2, axis=1)
        identifiable_global[idx] = g['identifiable']
        residual = g['z'] - g['W']@fmean
        quadratic += float(residual@residual)
        outputs.append(dict(columns=idx, local_factor=g['base'].tolist(),
                            global_loading=factor.tolist(), data_rank=g['data_rank']))
    return dict(schema='ariadne-conditional-intensities/1', mean=posterior.tolist(),
                variance=variance.tolist(), covariance_blocks=outputs,
                predictive_nll=float(.5*(quadratic + logdet + num_observed*np.log(2*np.pi))),
                num_observed=num_observed, data_rank=data_rank, num_latents=n,
                data_identifiable=identifiable_global.tolist(),
                semantics='structure_assisted_signed_gaussian_posterior_not_new_measurements',
                scientific_acceptance='not_established')


def dense_covariance(result: dict[str, Any], *, max_size: int = 4096) -> FloatArray:
    """Reference/testing convenience; never called by the production writer."""
    n = result['num_latents']
    if n > max_size:
        raise MemoryError('explicit dense covariance budget exceeded')
    K = np.zeros((n, n))
    rank = len(result['covariance_blocks'][0]['global_loading'][0]) if n else 0
    V = np.zeros((n, rank))
    for block in result['covariance_blocks']:
        idx = block['columns']
        L = np.asarray(block['local_factor'])
        K[np.ix_(idx, idx)] = L@L.T
        V[idx] = np.asarray(block['global_loading'])
    return K+V@V.T


def ensemble_prior(predictions: ArrayLike, weights: ArrayLike | None = None,
                   absolute_sd: float = 1., relative_sd: float = .2) -> dict[str, Any]:
    J = finite_array(predictions, 2, 'predictions')
    if min(J.shape) == 0 or np.any(J < 0):
        raise ValueError('require nonnegative candidate intensities')
    if not np.isfinite(absolute_sd) or absolute_sd <= 0 or not np.isfinite(relative_sd) or relative_sd < 0:
        raise ValueError('require positive absolute and nonnegative relative model-error scales')
    w = np.ones(J.shape[1]) if weights is None else finite_array(weights, 1, 'weights')
    if w.shape != (J.shape[1],) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError('invalid ensemble weights')
    w = w/w.sum()
    m = J@w
    U = (J-m[:, None])*np.sqrt(w)
    d = absolute_sd**2 + relative_sd**2*(J**2@w)
    return dict(mean=m.tolist(), independent_variance=d.tolist(), factors=U.tolist(),
                weights=w.tolist(), semantics='epistemic_model_ensemble_not_physical_disorder_average')


def calibrate(prior: dict[str, Any], domains: list[int], train: list[dict[str, Any]]) -> dict[str, Any]:
    """Conditional NNLS calibration on training data only.

    Coefficient uncertainty uses an interior, fixed-structure Gaussian approximation.
    Boundary/rank-deficient solutions are rejected instead of reporting fictitious
    finite uncertainty. Structural-model/calibration cross-covariance is not inferred.
    """
    m = finite_array(prior['mean'], 1, 'mean')
    d = finite_array(prior['independent_variance'], 1, 'variance')
    U = finite_array(prior['factors'], 2, 'factors')
    if d.shape != m.shape or U.shape[0] != len(m) or np.any(d <= 0):
        raise ValueError('invalid calibration prior dimensions or variances')
    dom = np.asarray(domains)
    if dom.shape != m.shape or not np.issubdtype(dom.dtype, np.integer) or np.any(dom < 0):
        raise ValueError('invalid domain indices')
    labels = sorted(set(dom.tolist()))
    # R2/HKLF5 use domains numbered from one; other callers use zero.
    # Preserve those physical labels and use a separate dense coefficient index.
    positions = np.searchsorted(labels, dom)
    design_rows, values = [], []
    for b in observations(train, len(m)):
        J = np.zeros((len(b.columns), len(labels)))
        for i, j in enumerate(b.columns):
            J[i, positions[j]] = m[j]
        L = spd_factor(b.covariance, 'calibration')
        design_rows.append(solve_triangular(L, b.design@J, lower=True))
        values.append(solve_triangular(L, b.values, lower=True))
    if not design_rows:
        raise ValueError('calibration requires training observations')
    X, y = np.vstack(design_rows), np.concatenate(values)
    scales = np.linalg.norm(X, axis=0)
    if np.any(scales == 0):
        raise ValueError('a domain has no calibration information')
    Xs = X/scales
    _, singular, vt = np.linalg.svd(Xs, full_matrices=False)
    if len(singular) < len(labels) or singular[-1] <= max(Xs.shape)*np.finfo(float).eps*singular[0]:
        raise ValueError('calibration domains are not separately identifiable')
    cs, _ = nnls(Xs, y)
    c = cs/scales
    if np.any(cs <= 0):
        raise ValueError('calibration reached a nonnegative boundary; interval-aware calibration is required')
    root_c = (vt.T/singular)/scales[:, None]
    S = c[positions]
    Jall = np.zeros((len(m), len(labels)))
    Jall[np.arange(len(m)), positions] = m
    Ucal = np.hstack((S[:, None]*U, Jall@root_c))
    return dict(mean=(S*m).tolist(), independent_variance=(S**2*d).tolist(), factors=Ucal.tolist(),
                coefficients=c.tolist(), domain_labels=labels,
                common_scale=float(c.sum()), fractions=(c/c.sum()).tolist(),
                coefficient_covariance=(root_c@root_c.T).tolist(),
                coefficient_signal_to_noise=(c/np.sqrt(np.sum(root_c**2, axis=1))).tolist(),
                calibration_model='training_only_fixed_structure_interior_gaussian_approximation',
                cross_covariance_structure_calibration='not_estimated')


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()
