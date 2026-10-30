"""Bounded Poisson--Gaussian convolution in offset-corrected ADU.

Y = gain*N + Normal(0, readout_variance), N ~ Poisson(mean).
The same exact discrete model is evaluated in mode-centred batches; no Gaussian
high-count replacement is made. Geometric bounds cover both omitted tails,
not floating-point rounding. Unchanged pixels reuse their certified sums.
"""
from __future__ import annotations

import operator

import numpy as np
from scipy.special import gammaln, logsumexp


class LikelihoodBudgetError(ValueError):
    """The requested truncation accuracy could not be met within the budget."""


def _poisson_deviance(n, mean):
    """Return n*log(n/mean)-n+mean without cancellation near n=mean."""
    n, mean = np.broadcast_arrays(np.asarray(n, float), np.asarray(mean, float))
    out = mean.copy()
    positive = n > 0
    k, mu = n[positive], mean[positive]
    if not len(k):
        return out
    delta = k - mu
    close = np.abs(delta) <= .125 * mu
    deviance = np.empty_like(k)
    d = delta[close] / mu[close]
    # ((1+d)*log(1+d)-d)/d**2 = sum (-d)**j/((j+1)*(j+2)).
    polynomial = np.full_like(d, 1. / (24 * 23))
    for power in range(23, 1, -1):
        polynomial = ((-1.) ** power) / (power * (power - 1)) + d * polynomial
    deviance[close] = (mu[close] * d) * d * polynomial
    with np.errstate(divide='ignore'):
        deviance[~close] = k[~close] * (np.log(k[~close]) - np.log(mu[~close])) - delta[~close]
    out[positive] = deviance
    return out


def _poisson_logpmf(n, mean):
    """Stable log P(N=n), including n ~= mean at very high count.

    Direct n*log(mean)-mean-gammaln(n+1) loses small likelihood differences.
    Here log p = -D(n,mean) - log(2*pi*n)/2 - Stirling remainder. A convergent
    series evaluates D near its zero; the Stirling tail at n>=16 is below
    float64 roundoff. This changes arithmetic, not the probability model.
    """
    n, mean = np.broadcast_arrays(np.asarray(n, float), np.asarray(mean, float))
    out = -_poisson_deviance(n, mean)
    positive = n > 0
    k = n[positive]
    if not len(k):
        return out
    remainder = np.empty_like(k)
    small = k < 16
    remainder[small] = (gammaln(k[small] + 1) - (k[small] + .5) * np.log(k[small])
                        + k[small] - .5 * np.log(2 * np.pi))
    inv = 1 / k[~small]
    inv2 = inv * inv
    remainder[~small] = inv * (1/12 + inv2 * (-1/360 + inv2 * (1/1260
        + inv2 * (-1/1680 + inv2 * (1/1188 + inv2 * (-691/360360 + inv2 / 156))))))
    out[positive] -= .5 * (np.log(2 * np.pi) + np.log(k)) + remainder
    return out


class PoissonGaussian:
    def __init__(self, data, gain, readout_variance, relative_tolerance=1e-12,
                 max_terms_per_pixel=100000, max_terms_per_evaluation=2000000):
        self.data = np.array(data, dtype=float, copy=True)
        self.data.setflags(write=False)
        self.gain, self.variance = float(gain), float(readout_variance)
        if (self.data.ndim != 1 or not np.all(np.isfinite(self.data))
                or not np.isfinite(self.gain) or self.gain <= 0
                or not np.isfinite(self.variance) or self.variance <= 0):
            raise ValueError('Poisson-Gaussian requires finite ADU, positive gain and readout variance')
        try:
            self.pixel_budget = operator.index(max_terms_per_pixel)
            self.evaluation_budget = operator.index(max_terms_per_evaluation)
        except TypeError as exc:
            raise ValueError('Convolution term budgets must be integers') from exc
        if not 0 < relative_tolerance < 1 or min(self.pixel_budget, self.evaluation_budget) < 1:
            raise ValueError('Invalid convolution accuracy or term budget')
        self.tolerance = float(relative_tolerance)
        self.calls = self.terms = self.max_terms = self.cached_pixels = 0
        self.max_tail = self.max_score_error = 0.
        self._normalizer = -.5 * (np.log(2 * np.pi) + np.log(self.variance))
        self._last_mean = np.full_like(self.data, np.nan)
        self._logs = np.empty_like(self.data)
        self._shifted_logs = np.empty_like(self.data)
        self._tails = np.empty((2, len(self.data)))

    def _bounded_sums(self, y, mean, remaining):
        """Batch shifted densities; each series has its own mode and tail test."""
        y, mean = np.asarray(y, float), np.asarray(mean, float)
        count = len(y)
        logs, errors = np.empty(count), np.zeros(count)
        zero = mean == 0
        used = int(np.count_nonzero(zero))
        if used > remaining:
            raise LikelihoodBudgetError('Poisson-Gaussian evaluation term budget exceeded')
        logs[zero] = self._normalizer - .5 * (y[zero] / np.sqrt(self.variance)) ** 2
        active = np.flatnonzero(~zero)
        if not len(active):
            return logs, errors, used
        observed = y[active]
        mu = mean[active]
        log_mu = np.log(mu)
        extent = np.maximum(mu, observed / self.gain)
        if not np.all(np.isfinite(extent)) or np.any(extent >= 2**52):
            raise ValueError('Poisson-Gaussian photon index exceeds reliable float64 integer range')
        left = np.zeros(len(active), dtype=np.int64)
        right = np.maximum(1, np.ceil(extent)).astype(np.int64)

        def ratio(n, indices):
            return (log_mu[indices] - np.log(n + 1.)
                    + self.gain * (observed[indices] - self.gain * (n + .5)) / self.variance)

        searching = np.flatnonzero(left < right)
        while len(searching):
            middle = left[searching] + (right[searching] - left[searching]) // 2
            rising = ratio(middle, searching) > 0
            left[searching[rising]] = middle[rising] + 1
            right[searching[~rising]] = middle[~rising]
            searching = searching[left[searching] < right[searching]]
        mode, spent = left, np.zeros(len(active), dtype=np.int64)
        pending, width = np.arange(len(active)), 8
        while len(pending):
            # Bound temporary arrays independently of the scientific term budget.
            batch = max(1, 262144 // (2 * width + 1))
            unfinished = []
            offsets = np.arange(-width, width + 1, dtype=np.int64)
            for start in range(0, len(pending), batch):
                ids = pending[start:start + batch]
                lo = np.maximum(0, mode[ids] - width)
                hi = mode[ids] + width
                sizes = hi - lo + 1
                cost = int(sizes.sum())
                if np.any(spent[ids] + sizes > self.pixel_budget) or used + cost > remaining:
                    raise LikelihoodBudgetError('Poisson-Gaussian exact-sum term budget exceeded; no approximation used')
                n = mode[ids, None] + offsets
                admissible = n >= 0
                n = np.maximum(n, 0)
                terms = (_poisson_logpmf(n, mu[ids, None]) + self._normalizer
                         - .5 * ((observed[ids, None] - self.gain * n) / np.sqrt(self.variance)) ** 2)
                terms[~admissible] = -np.inf
                total = logsumexp(terms, axis=1)
                left_term = terms[np.arange(len(ids)), np.maximum(0, width - mode[ids])]
                lower = np.full(len(ids), -np.inf)
                has_left = lo > 0
                lr = -ratio(lo[has_left] - 1, ids[has_left])
                lower[has_left] = left_term[has_left] + lr - np.log(-np.expm1(lr))
                lr = ratio(hi, ids)
                upper = terms[:, -1] + lr - np.log(-np.expm1(lr))
                tail = np.logaddexp(lower, upper) - total
                done = tail <= np.log(self.tolerance)
                target = active[ids[done]]
                logs[target], errors[target] = total[done], np.exp(tail[done])
                unfinished.append(ids[~done])
                spent[ids] += sizes
                used += cost
            pending = np.concatenate(unfinished)
            width *= 2
        return logs, errors, used

    def _sum(self, y, mean, shift, remaining):
        """Compatibility wrapper for one bounded shifted sum."""
        logs, errors, used = self._bounded_sums([y - self.gain * shift], [mean], remaining)
        return float(logs[0]), float(errors[0]), used

    def evaluate(self, mean):
        """Per-pixel log density, d(log density)/d(mean), and tail bounds.

        The exact identity score = f(y-gain; mean)/f(y; mean)-1 is regular
        at mean=0. Cached entries are valid only for bitwise-equal means.
        """
        mean = np.asarray(mean, dtype=float)
        if mean.shape != self.data.shape or not np.all(np.isfinite(mean)) or np.any(mean < 0):
            raise ValueError('Poisson means must be finite and nonnegative')
        changed = mean != self._last_mean
        ids = np.flatnonzero(changed)
        used = 0
        if len(ids):
            y = np.concatenate((self.data[ids], self.data[ids] - self.gain))
            mu = np.tile(mean[ids], 2)
            logs, errors, used = self._bounded_sums(y, mu, self.evaluation_budget)
            log_f, log_h = np.split(logs, 2)
            if not np.all(np.isfinite(logs)):
                raise ValueError('Poisson-Gaussian density/score exceeds floating-point range')
            self._logs[ids], self._shifted_logs[ids] = log_f, log_h
            self._tails[:, ids] = errors.reshape(2, -1)
            self._last_mean[ids] = mean[ids]
        difference = self._shifted_logs - self._logs
        with np.errstate(over='ignore', invalid='ignore'):
            ratio = np.exp(difference)
            scores = np.expm1(difference)
        if not np.all(np.isfinite(ratio)):
            raise ValueError('Poisson-Gaussian density/score exceeds floating-point range')
        score_error = ratio * np.maximum(self._tails[1], self._tails[0] / (1 + self._tails[0]))
        worst_tail = float(self._tails.max(initial=0.))
        worst_score = float(score_error.max(initial=0.))
        self.calls += 1
        self.terms += used
        self.max_terms = max(self.max_terms, used)
        self.cached_pixels += len(mean) - len(ids)
        self.max_tail = max(self.max_tail, worst_tail)
        self.max_score_error = max(self.max_score_error, worst_score)
        return self._logs.copy(), scores, {'max_relative_density_truncation_bound': worst_tail,
            'max_absolute_score_truncation_bound': worst_score,
            'negative_log_likelihood_truncation_bound': float(np.log1p(self._tails[0]).sum()),
            'terms_evaluated': used, 'unchanged_pixels_reused': len(mean) - len(ids)}

    def observed_information(self, mean):
        """Return -d² log f/d mean² and an absolute truncation-error bound.

        f_s denotes the density at y-s*gain. The information is
        (f_1/f_0)^2 - f_2/f_0, with expm1 limiting cancellation. A nonpositive
        value, or one below its error/roundoff scale, is not invented information.
        The caller decides how to report an unreliable local curvature.
        """
        _, _, density_diagnostics = self.evaluate(mean)
        logs2, tails2, used = self._bounded_sums(self.data - 2 * self.gain,
            np.asarray(mean, float), self.evaluation_budget - density_diagnostics['terms_evaluated'])
        a = self._shifted_logs - self._logs
        b = logs2 - self._logs
        with np.errstate(over='ignore', invalid='ignore'):
            r1sq, r2 = np.exp(2 * a), np.exp(b)
            information = -r1sq * np.expm1(b - 2 * a)
        if not np.all(np.isfinite(information)):
            raise ValueError('Poisson-Gaussian observed information exceeds floating-point range')
        e0, e1 = self._tails
        error = (r1sq * np.maximum(2 * e1 + e1 * e1, -np.expm1(-2 * np.log1p(e0)))
                 + r2 * np.maximum(tails2, e0 / (1 + e0)))
        log_scale = np.maximum.reduce((np.abs(self._logs), np.abs(self._shifted_logs),
                                       np.abs(logs2), np.ones_like(mean)))
        roundoff = 64 * np.finfo(float).eps * log_scale * np.maximum(r1sq + r2, 1.)
        self.terms += used
        self.max_terms = max(self.max_terms, used + density_diagnostics['terms_evaluated'])
        self.max_tail = max(self.max_tail, float(tails2.max(initial=0.)))
        return information, {'absolute_truncation_bound': error,
                             'roundoff_resolution': roundoff,
                             'reliable_positive': information > error + roundoff,
                             'terms_evaluated': used}

    def diagnostics(self):
        return {'evaluation': 'exact logsumexp Poisson*Gaussian bounded sum; no high-count approximation',
                'arithmetic': 'batched mode-centred sum; stable Poisson deviance and Stirling remainder',
                'relative_truncation_tolerance': self.tolerance,
                'max_relative_density_truncation_bound': self.max_tail,
                'max_absolute_score_truncation_bound': self.max_score_error,
                'negative_log_likelihood_truncation_bound_per_evaluation': len(self.data) * np.log1p(self.max_tail),
                'error_scope': 'omitted tails only; excludes floating-point rounding',
                'max_terms_per_shifted_pixel_sum': self.pixel_budget,
                'max_terms_per_evaluation': self.evaluation_budget,
                'budget_exhaustion': 'raises LikelihoodBudgetError; no approximate fallback',
                'evaluations': self.calls, 'terms_evaluated': self.terms,
                'largest_evaluation_terms': self.max_terms,
                'unchanged_pixels_reused': self.cached_pixels}
