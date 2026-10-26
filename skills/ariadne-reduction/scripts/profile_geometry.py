"""体素积分轮廓及位置扰动；坐标顺序为 XYZ，数组顺序为 ZYX。"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def gaussian_voxels(covariance, centre, bounds):
    covariance = np.asarray(covariance, dtype=float)
    centre = np.asarray(centre, dtype=float)
    if (covariance.shape != (3, 3) or not np.all(np.isfinite(covariance))
            or not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14)
            or np.any(np.linalg.eigvalsh(covariance) <= 0)
            or centre.shape != (3,) or not np.all(np.isfinite(centre))):
        raise ValueError('Require finite centre and symmetric positive-definite profile covariance')
    x0, x1, y0, y1, z0, z1 = bounds
    if np.all(covariance == np.diag(np.diag(covariance))):
        probabilities = []
        for axis, (lo, hi) in enumerate(((x0, x1), (y0, y1), (z0, z1))):
            edges = (np.arange(lo, hi + 1) - centre[axis]) / np.sqrt(covariance[axis, axis])
            # 正尾使用生存函数，避免两个接近 1 的 CDF 相减。
            left, right = edges[:-1], edges[1:]
            probabilities.append(np.where(left > 0, ndtr(-left) - ndtr(-right), ndtr(right) - ndtr(left)))
        return probabilities[2][:, None, None] * probabilities[1][None, :, None] * probabilities[0][None, None, :]
    z, y, x = np.mgrid[z0:z1, y0:y1, x0:x1]
    offsets = np.stack((x + .5 - centre[0], y + .5 - centre[1], z + .5 - centre[2]), axis=-1)
    inverse = np.linalg.inv(covariance)
    # 条件宽度可反映旋转窄峰；过窄且强相关的轮廓不静默使用欠采样近似。
    counts = np.maximum(np.ceil(2 * np.sqrt(np.diag(inverse))), 2).astype(int)
    if np.any(counts > 16):
        raise ValueError('Correlated profile needs more than 16 subvoxel nodes per axis; revise calibration/support')
    nodes = [(np.arange(n) + .5) / n - .5 for n in counts]
    values = np.zeros(z.shape)
    for dz in nodes[2]:
        for dy in nodes[1]:
            for dx in nodes[0]:
                delta = offsets + (dx, dy, dz)
                values += np.exp(-.5 * np.einsum('...i,ij,...j->...', delta, inverse, delta))
    return values / (np.prod(counts) * np.sqrt((2 * np.pi) ** 3 * np.linalg.det(covariance)))
