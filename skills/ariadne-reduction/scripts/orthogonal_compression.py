"""Bounded-workspace square-root compression of a weighted linear model.

The augmented matrix [background, signal, data] is reduced by successive
Householder QR factorizations. Its small triangular factor preserves every
least-squares residual norm, including the parameter-independent residual.
No normal equations, ridge, or intensity threshold are introduced.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import qr


def augmented_qr(data, variance, signal, background, valid, chunk_rows=4096):
    """Return R for W^.5 [B P y], without full weighted matrix or Q storage.

    Inputs and requested fitted values remain caller-owned O(NK) / O(N)
    arrays. Additional QR workspace is O(chunk_rows * (K+B) + (K+B)^2).
    Invalid pixels never enter a factor. Raw row chunks keep boolean selection
    bounded as well. Empty and underdetermined inputs are handled by the caller.
    """
    if isinstance(chunk_rows, (bool, np.bool_)) or not isinstance(chunk_rows, (int, np.integer)) or chunk_rows < 1:
        raise ValueError('chunk_rows must be a positive integer')
    columns = background.shape[1] + signal.shape[1] + 1
    factor = np.empty((0, columns))
    count = 0
    for start in range(0, len(data), chunk_rows):
        stop = min(start + chunk_rows, len(data))
        selected = valid[start:stop]
        size = int(np.count_nonzero(selected))
        if not size:
            continue
        weights = 1 / np.sqrt(variance[start:stop][selected])
        work = np.empty((len(factor) + size, columns), order='F')
        work[:len(factor)] = factor
        rows = work[len(factor):]
        nuisance = background.shape[1]
        rows[:, :nuisance] = background[start:stop][selected] * weights[:, None]
        rows[:, nuisance:-1] = signal[start:stop][selected] * weights[:, None]
        rows[:, -1] = data[start:stop][selected] * weights
        triangular = qr(work, mode='r', overwrite_a=True, check_finite=False)[0]
        # scipy's r-only interface can return a tall zero-padded R. Copy only
        # its nonzero rows so the previous work chunk is not kept alive.
        factor = triangular[:min(columns, len(triangular))].copy()
        count += size
    return factor, count
