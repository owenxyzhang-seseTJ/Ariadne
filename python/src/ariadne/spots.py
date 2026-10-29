"""Mask-aware multi-scale count excess proposals; not a Bragg classifier.

Box sums use separable running filters: O(scales * pixels), not window-area
loops. Poisson deviance is conditional on the estimated background; threshold
scores are not calibrated p-values after adaptive estimation/multiple scales.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.special import xlogy


def box_sum(array, width):
    return uniform_filter(array, size=width, mode="constant", cval=0.) * width**2


def propose(image, mask, *, gain, offset, readout_variance, noise_model,
            background_width, exclusion_width, scales, score_threshold,
            support_sigma, min_valid_fraction, background_iterations):
    if gain is None or not np.isfinite(gain) or gain <= 0:
        raise ValueError("Supply calibrated gain_adu_per_photon; gain is never inferred")
    if noise_model not in ("poisson", "poisson_gaussian"):
        raise ValueError("Unsupported pixel noise model")
    if readout_variance < 0 or (noise_model == "poisson" and readout_variance != 0):
        raise ValueError("Readout noise requires poisson_gaussian approximation")
    if not scales or any(w < 1 or w % 2 != 1 for w in scales):
        raise ValueError("Use positive odd-pixel scales")
    if (background_width % 2 != 1 or exclusion_width % 2 != 1
            or background_width <= exclusion_width or exclusion_width < max(scales)):
        raise ValueError("Background annulus must exclude all candidate kernel supports")
    if not 0 < min_valid_fraction <= 1 or background_iterations < 0 or score_threshold <= 0 or support_sigma < 0:
        raise ValueError("Invalid threshold settings")
    data = (np.asarray(image, dtype=float) - offset) / gain
    valid = np.asarray(mask, dtype=bool) & np.isfinite(data)
    if data.ndim != 2 or valid.shape != data.shape:
        raise ValueError("A two-dimensional panel and matching mask are required")
    if noise_model == "poisson" and np.any(data[valid] < 0):
        raise ValueError("Negative corrected pixels contradict the selected Poisson model")
    readout = readout_variance / gain**2
    retained = valid.copy()
    annulus_area = background_width**2 - exclusion_width**2

    def background(selection):
        weights = selection.astype(float)
        counts = box_sum(weights, background_width) - box_sum(weights, exclusion_width)
        selected_data = np.where(selection, data, 0.)
        sums = box_sum(selected_data, background_width) - box_sum(selected_data, exclusion_width)
        usable = counts >= min_valid_fraction * annulus_area
        mean = np.maximum(np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0), 0.)
        return mean, counts, usable

    for _ in range(background_iterations):
        mean, count, usable = background(retained)
        # Remove positive contaminants only; do not winsorise the measured pixels.
        clipping_variance = mean + readout + (mean + readout) / np.maximum(count, 1.)
        retained &= ~(usable & (data > mean + 4. * np.sqrt(clipping_variance)))
    mean, count, usable = background(retained)
    support = data > mean + support_sigma * np.sqrt(mean + readout)
    if noise_model == "poisson_gaussian":
        background_variance = (mean + readout) / np.maximum(count, 1.)
        np.copyto(background_variance, 0., where=~valid)
    del count, retained
    # data 与 mean 为本次调用拥有的数组，完成原值计算后直接掩膜。
    np.copyto(data, 0., where=~valid)
    np.copyto(mean, 0., where=~valid)
    best = np.zeros_like(data)
    valid_weights = valid.astype(float)
    for width in scales:
        n_valid = box_sum(valid_weights, width)
        observed = box_sum(data, width)
        expected = box_sum(mean, width)
        enough = n_valid >= width**2 * min_valid_fraction
        if noise_model == "poisson":
            # A finite expected-count floor avoids infinity for zero sampled background.
            # This regularization is an algorithm parameter, not a measured variance.
            expected = np.maximum(expected, .5)
            excess = np.maximum(observed, expected)
            statistic = np.sqrt(np.maximum(2. * (xlogy(excess, excess / expected) - excess + expected), 0.))
        else:
            # Quasi-Gaussian excess with annulus background-estimation uncertainty.
            variance = expected + readout * n_valid
            variance += box_sum(background_variance, width) * width**2
            statistic = np.maximum(observed - expected, 0.) / np.sqrt(np.maximum(variance, .5))
        np.maximum(best, statistic, out=best, where=enough & usable)
    return valid & usable & support & (best >= score_threshold)
