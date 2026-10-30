"""Exact variable projection for signed intensities and nonlinear scales.

For r = A(beta) I_hat(beta) - y, at constant full column rank,
dr/dbeta = (1-AA^+) (dA/dbeta) I_hat - A^{+T} (dA/dbeta)^T r.
The residual-dependent second term is essential away from an exact fit.

Whitened contributions are assembled once.  The fixed row/intensity support
graph partitions only the linear solve; scale parameters remain global.  SVD
acts on equilibrated components, batched by shape, without normal equations.
All components use one global singular-value cutoff, exactly as a full SVD.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.sparse import block_diag, bmat, coo_matrix
from scipy.sparse.csgraph import connected_components


class RankDeficientProfile(ValueError):
    """The differentiable full-column-rank profile is not available."""


@dataclass(frozen=True)
class _ComponentBatch:
    """Disjoint linear components with the same (rows, columns) shape."""

    rows: np.ndarray
    columns: np.ndarray
    slots: np.ndarray
    contributions: np.ndarray
    weights: np.ndarray
    target: np.ndarray

    def matrix(self, gains):
        count, rows = self.rows.shape
        columns = self.columns.shape[1]
        return np.bincount(
            self.slots, weights=self.weights * gains[self.contributions],
            minlength=count * rows * columns,
        ).reshape(count, rows, columns)


class ProfiledScaleModel:
    def __init__(self, prepared, designs, target, intensity_count, rcond, check_time,
                 *, include_joint_jacobian=True):
        self.target = np.asarray(target)
        self.n, self.rcond, self.check_time = intensity_count, rcond, check_time
        self.include_joint_jacobian = include_joint_jacobian
        self._features = np.concatenate([block[3] for block in prepared])
        self._known = np.concatenate([block[4] for block in prepared])
        self._indices = np.concatenate([block[5] for block in prepared])
        self._gain_offsets = np.cumsum([len(block[4]) for block in prepared])[:-1]
        # C maps individual contributions to whitened observations.  Different
        # contributors may have the same intensity ID but different features.
        self._contribution = block_diag((coo_matrix(designs[0]), *designs[1:]), format='csr')
        self._contribution.eliminate_zeros()
        self._transpose = self._contribution.T
        contribution_count = len(self._known)
        self._incidence_transpose = coo_matrix(
            (np.ones(contribution_count), (self._indices, np.arange(contribution_count))),
            shape=(self.n, contribution_count),
        ).tocsr()
        self._batches, self._structural_rank_failure = self._partition()
        self._beta = self._state = None
        self.evaluations = 0

    def _partition(self):
        c = self._contribution
        measurement_count = len(self.target)
        edge_rows = np.repeat(np.arange(measurement_count), np.diff(c.indptr))
        edge_columns = self._indices[c.indices]
        # Boolean support cannot cancel when repeated contributors have opposite
        # signs.  Partitioning depends on possible support, never on trial beta.
        support = coo_matrix((np.ones(len(edge_rows), dtype=bool),
                              (edge_rows, edge_columns)),
                             shape=(measurement_count, self.n)).tocsr()
        _, labels = connected_components(bmat([[None, support], [support.T, None]],
                                               format='csr'), directed=False)
        row_labels, column_labels = labels[:measurement_count], labels[measurement_count:]
        shapes = defaultdict(list)
        rank_failure = False
        for label in np.unique(column_labels):
            rows = np.flatnonzero(row_labels == label)
            columns = np.flatnonzero(column_labels == label)
            if len(rows) < len(columns):
                rank_failure = True
            shapes[len(rows), len(columns)].append((label, rows, columns))
        batches = []
        for (row_count, column_count), components in shapes.items():
            if row_count < column_count:
                continue
            rows = np.array([item[1] for item in components])
            columns = np.array([item[2] for item in components])
            component_number = np.full(len(labels), -1, dtype=int)
            component_number[[item[0] for item in components]] = np.arange(len(components))
            selected = component_number[row_labels[edge_rows]] >= 0
            local_rows = np.empty(measurement_count, dtype=int)
            local_columns = np.empty(self.n, dtype=int)
            local_rows[rows] = np.arange(row_count)
            local_columns[columns] = np.arange(column_count)
            selected_rows, selected_columns = edge_rows[selected], edge_columns[selected]
            slots = ((component_number[row_labels[selected_rows]] * row_count
                      + local_rows[selected_rows]) * column_count
                     + local_columns[selected_columns])
            batches.append(_ComponentBatch(rows, columns, slots, c.indices[selected],
                                            c.data[selected], self.target[rows]))
        return batches, rank_failure

    def state(self, beta):
        self.check_time()
        if self._beta is not None and np.array_equal(beta, self._beta):
            return self._state
        if self._structural_rank_failure:
            raise RankDeficientProfile('Intensity design is structurally rank deficient')
        gain = self._known * np.exp(self._features @ beta)
        if np.any(gain <= 0) or not np.all(np.isfinite(gain)):
            raise ValueError('Nonfinite variable-projection scale')
        decompositions = []
        largest_singular = 0.
        for batch in self._batches:
            matrix = batch.matrix(gain)
            peak = np.max(np.abs(matrix), axis=1)
            safe = np.where(peak > 0, peak, 1.)
            norms = np.linalg.norm(matrix / safe[:, None, :], axis=1)
            scales = safe * np.where(norms > 0, norms, 1.)
            equilibrated = matrix / scales[:, None, :]
            if matrix.shape[2] == 1:
                # A one-column component has a closed-form thin SVD.  Its norm
                # is computed after equilibration to avoid squaring large units.
                singular = np.linalg.norm(equilibrated, axis=1)
                u = equilibrated / np.where(singular > 0, singular, 1.)[:, None, :]
                vt = np.ones((len(matrix), 1, 1))
            else:
                u, singular, vt = np.linalg.svd(equilibrated, full_matrices=False)
            largest_singular = max(largest_singular, float(np.max(singular)))
            decompositions.append((batch, matrix, scales, u, singular, vt))
        # A block-diagonal operator's singular spectrum is the union of its
        # component spectra.  A common cutoff preserves the full-SVD rank rule.
        cutoff = self.rcond * largest_singular
        if any(np.any(singular <= cutoff) for _, _, _, _, singular, _ in decompositions):
            raise RankDeficientProfile('Intensity design is rank deficient for variable projection')
        intensity = np.empty(self.n)
        residual = -self.target.copy()
        for batch, matrix, scales, u, singular, vt in decompositions:
            coefficients = (u.swapaxes(1, 2) @ batch.target[..., None])[..., 0] / singular
            local_i = (vt.swapaxes(1, 2) @ coefficients[..., None])[..., 0] / scales
            intensity[batch.columns] = local_i
            residual[batch.rows] = (matrix @ local_i[..., None])[..., 0] - batch.target
        jb = self._contribution @ ((gain * intensity[self._indices])[:, None] * self._features)
        contribution_adjoint = gain * (self._transpose @ residual)
        adjoint = self._incidence_transpose @ (contribution_adjoint[:, None] * self._features)
        jacobian = jb.copy()
        joint_jacobian = None
        if self.include_joint_jacobian:
            joint_jacobian = np.zeros((len(residual), self.n + len(beta)))
            joint_jacobian[:, self.n:] = jb
        for batch, matrix, scales, u, singular, vt in decompositions:
            # A^{+T} adjoint = U Sigma^-1 V.T S^-1 adjoint.  Differentiate
            # the original A, not the equilibrated matrix or its column norms.
            correction = (vt @ (adjoint[batch.columns] / scales[..., None])) / singular[..., None]
            jacobian[batch.rows] -= u @ (u.swapaxes(1, 2) @ jb[batch.rows] + correction)
            if joint_jacobian is not None:
                joint_jacobian[batch.rows[:, :, None], batch.columns[:, None, :]] = matrix
        self.check_time()
        self._beta = np.asarray(beta).copy()
        self._state = {'intensity': intensity, 'residual': residual, 'jacobian': jacobian,
                       'joint_jacobian': joint_jacobian,
                       'gains': np.split(gain, self._gain_offsets)}
        self.evaluations += 1
        return self._state
