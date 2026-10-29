"""体素积分轮廓及位置扰动；坐标顺序为 XYZ，数组顺序为 ZYX。"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.special import ndtr, roots_legendre


@lru_cache(maxsize=64)
def _voxel_quadrature(order):
    nodes, weights = roots_legendre(order)
    return nodes / 2, weights / 2


def gaussian_voxels(covariance, centre, bounds):
    """返回 ZYX 顺序的 Gaussian 体素概率。

    相关轮廓解析积分条件最窄的轴，其余两轴使用
    max(6, ceil(3/sigma_j)+2) 阶 Gauss-Legendre 求积，
    sigma_j = 1/sqrt(precision[j,j]) 表示联合密度的条件宽度。
    节点规则同时解析边际密度与条件 CDF 的变化；精度由独立求积验证。
    两个数值轴分别限 64 节点，解析轴不受此上限限制。
    """
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
    inverse = np.linalg.inv(covariance)
    exact_axis = int(np.argmax(np.diag(inverse)))
    axes = [axis for axis in range(3) if axis != exact_axis]
    # f(x_a, x_B) = f_B(x_B) f(x_a | x_B)。由精度矩阵直接得到条件
    # 均值系数和方差，避免对强相关协方差做易相消的 Schur 补减法。
    conditional_scale = np.sqrt(inverse[exact_axis, exact_axis])
    regression = -inverse[exact_axis, axes] / inverse[exact_axis, exact_axis]
    marginal = covariance[np.ix_(axes, axes)]
    marginal_inverse = np.linalg.inv(marginal)
    normalizer = 2 * np.pi * np.sqrt(np.linalg.det(marginal))

    # 两个数值方向保留原联合密度的条件宽度作为保守分辨率尺度。
    orders = np.maximum(6, np.ceil(3 * np.sqrt(np.diag(inverse)[axes])) + 2)
    if np.any(orders > 64):
        raise ValueError('Correlated profile needs more than 64 quadrature nodes per numerical axis; revise calibration/support')
    quadratures = [_voxel_quadrature(int(order)) for order in orders]
    limits = ((x0, x1), (y0, y1), (z0, z1))
    shape = (z1 - z0, y1 - y0, x1 - x0)
    coordinates = []
    for axis, (lo, hi) in enumerate(limits):
        axis_shape = [1, 1, 1]
        axis_shape[2 - axis] = hi - lo
        coordinates.append((np.arange(lo, hi) + .5 - centre[axis]).reshape(axis_shape))
    half_width = .5 * conditional_scale
    values = np.zeros(shape)
    for node_b, weight_b in zip(*quadratures[0]):
        delta_b = coordinates[axes[0]] + node_b
        for node_c, weight_c in zip(*quadratures[1]):
            delta_c = coordinates[axes[1]] + node_c
            exponent = (marginal_inverse[0, 0] * delta_b ** 2
                        + 2 * marginal_inverse[0, 1] * delta_b * delta_c
                        + marginal_inverse[1, 1] * delta_c ** 2)
            conditional_mean = regression[0] * delta_b + regression[1] * delta_c
            # 正区间关于零反射到负尾，只需两次 CDF；同时避免近 1 的
            # 两个 CDF 相减，保留很小但非零的体素概率。
            midpoint = -np.abs((coordinates[exact_axis] - conditional_mean) * conditional_scale)
            mass = ndtr(midpoint + half_width) - ndtr(midpoint - half_width)
            values += weight_b * weight_c * np.exp(-.5 * exponent) * mass
    return values / normalizer
