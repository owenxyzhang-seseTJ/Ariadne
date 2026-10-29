"""Exact bounded Poisson*Gaussian convolution in offset-corrected ADU.

Y = gain*N + Normal(0, readout_variance), N ~ Poisson(mean).
Geometric bounds on both omitted tails follow from the decreasing adjacent
term ratio. They bound truncation, not floating-point rounding. No Gaussian
high-count replacement is made. Budget exhaustion raises ValueError.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp


class LikelihoodBudgetError(ValueError):
    """The requested truncation accuracy could not be met within the budget."""


class PoissonGaussian:
    def __init__(self, data, gain, readout_variance, relative_tolerance=1e-12,
                 max_terms_per_pixel=100000, max_terms_per_evaluation=2000000):
        self.data = np.asarray(data, dtype=float)
        self.gain, self.variance = float(gain), float(readout_variance)
        if (self.data.ndim != 1 or not np.all(np.isfinite(self.data))
                or not np.isfinite(self.gain) or self.gain <= 0
                or not np.isfinite(self.variance) or self.variance <= 0):
            raise ValueError('Poisson-Gaussian requires finite ADU, positive gain and readout variance')
        if not 0 < relative_tolerance < 1 or min(max_terms_per_pixel, max_terms_per_evaluation) < 1:
            raise ValueError('Invalid convolution accuracy or term budget')
        self.tolerance = float(relative_tolerance)
        self.pixel_budget = int(max_terms_per_pixel)
        self.evaluation_budget = int(max_terms_per_evaluation)
        self.calls = self.terms = self.max_terms = 0
        self.max_tail = self.max_score_error = 0.

    def _sum(self, y, mean, shift, remaining):
        """Return log density and a bound on omitted density / retained density."""
        if remaining < 1:
            raise LikelihoodBudgetError('Poisson-Gaussian evaluation term budget exceeded')
        if mean == 0:
            return -.5 * (np.log(2 * np.pi * self.variance)
                           + (y - self.gain * shift) ** 2 / self.variance), 0., 1
        log_mean = np.log(mean)

        def ratio(n):
            # log(t[n+1]/t[n]); strictly decreasing for n >= 0.
            return (log_mean - np.log(n + 1.)
                    + self.gain * (y - self.gain * (n + shift + .5)) / self.variance)

        extent = max(mean, y / self.gain - shift)
        if not np.isfinite(extent) or extent >= 2**52:
            raise ValueError('Poisson-Gaussian photon index exceeds reliable float64 integer range')
        left, right = 0, max(1, int(np.ceil(extent)))
        while left < right:
            mid = (left + right) // 2
            if ratio(mid) > 0:
                left = mid + 1
            else:
                right = mid
        mode, width, used = left, 8, 0
        while True:
            lo, hi = max(0, mode - width), mode + width
            size = hi - lo + 1
            if used + size > min(self.pixel_budget, remaining):
                raise LikelihoodBudgetError('Poisson-Gaussian exact-sum term budget exceeded; no approximation used')
            n = np.arange(lo, hi + 1, dtype=float)
            log_terms = (n * log_mean - mean - gammaln(n + 1)
                         - .5 * (np.log(2 * np.pi * self.variance)
                                 + (y - self.gain * (n + shift)) ** 2 / self.variance))
            log_sum = float(logsumexp(log_terms))
            used += size
            tails = []
            if lo:
                log_r = -ratio(lo - 1)
                tails.append(float(log_terms[0] + log_r - np.log(-np.expm1(log_r))))
            log_r = ratio(hi)
            tails.append(float(log_terms[-1] + log_r - np.log(-np.expm1(log_r))))
            log_bound = float(logsumexp(tails)) - log_sum
            if log_bound <= np.log(self.tolerance):
                return log_sum, float(np.exp(log_bound)), used
            width *= 2

    def evaluate(self, mean):
        """Return per-pixel log density, d(log density)/d(mean), diagnostics.

        The score is f_shift1/f - 1, avoiding E[N]/mean at mean=0.
        Each shifted sum has the same tail guarantee. The score error bound
        follows from the interval for the ratio of these positive sums.
        """
        mean = np.asarray(mean, dtype=float)
        if mean.shape != self.data.shape or not np.all(np.isfinite(mean)) or np.any(mean < 0):
            raise ValueError('Poisson means must be finite and nonnegative')
        logs, scores = np.empty_like(mean), np.empty_like(mean)
        used, worst_tail, worst_score = 0, 0., 0.
        for i, (y, mu) in enumerate(zip(self.data, mean)):
            log_f, err_f, count = self._sum(y, mu, 0, self.evaluation_budget - used)
            used += count
            log_h, err_h, count = self._sum(y, mu, 1, self.evaluation_budget - used)
            used += count
            if used > self.evaluation_budget:
                raise LikelihoodBudgetError('Poisson-Gaussian evaluation term budget exceeded')
            with np.errstate(over='ignore', invalid='ignore'):
                ratio = float(np.exp(log_h - log_f))
            if not np.isfinite(log_f) or not np.isfinite(ratio):
                raise ValueError('Poisson-Gaussian density/score exceeds floating-point range')
            logs[i], scores[i] = log_f, np.expm1(log_h - log_f)
            worst_tail = max(worst_tail, err_f, err_h)
            worst_score = max(worst_score, ratio * max(err_h, err_f / (1 + err_f)))
        self.calls += 1
        self.terms += used
        self.max_terms = max(self.max_terms, used)
        self.max_tail = max(self.max_tail, worst_tail)
        self.max_score_error = max(self.max_score_error, worst_score)
        return logs, scores, {'max_relative_density_truncation_bound': worst_tail,
                              'max_absolute_score_truncation_bound': worst_score,
                              'negative_log_likelihood_truncation_bound': len(mean) * np.log1p(worst_tail),
                              'terms_evaluated': used}

    def diagnostics(self):
        return {'evaluation': 'exact logsumexp Poisson*Gaussian bounded sum; no high-count approximation',
                'relative_truncation_tolerance': self.tolerance,
                'max_relative_density_truncation_bound': self.max_tail,
                'max_absolute_score_truncation_bound': self.max_score_error,
                'negative_log_likelihood_truncation_bound_per_evaluation': len(self.data) * np.log1p(self.max_tail),
                'error_scope': 'omitted tails only; excludes floating-point rounding',
                'max_terms_per_shifted_pixel_sum': self.pixel_budget,
                'max_terms_per_evaluation': self.evaluation_budget,
                'budget_exhaustion': 'raises LikelihoodBudgetError; no approximate fallback',
                'evaluations': self.calls, 'terms_evaluated': self.terms,
                'largest_evaluation_terms': self.max_terms}
