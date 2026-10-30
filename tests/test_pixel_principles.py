"""Independent likelihood, curvature, and weak-domain numerical checks."""
from decimal import Decimal, localcontext
from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp

SCRIPTS = Path(__file__).resolve().parents[1] / 'skills' / 'ariadne-reduction' / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from constrained_pixels import fit_constrained
from poisson_gaussian import LikelihoodBudgetError, PoissonGaussian, _poisson_deviance


def direct_log_density(y, mean, gain, variance, stop=600):
    """Independent, deliberately over-wide finite sum (not mode search)."""
    y, mean = np.asarray(y), np.asarray(mean)
    n = np.arange(stop)
    poisson = (n[None, :] * np.log(np.maximum(mean[:, None], np.finfo(float).tiny))
               - mean[:, None] - gammaln(n + 1))
    poisson[mean == 0, 1:] = -np.inf
    return logsumexp(poisson - .5 * (np.log(2*np.pi*variance)
        + (y[:, None] - gain * n)**2 / variance), axis=1)


def decimal_high_count_density(y, mean, gain=1., variance=.2):
    """80-digit scalar sum; direct logarithms retain high-count cancellation.

    Stirling's alternating remainder bound for n~1e9 makes the first omitted
    factorial term <1e-100. The finite Gaussian support here is +/-50 counts.
    """
    with localcontext() as context:
        context.prec = 80
        y, mu, g, var = map(Decimal.from_float, map(float, (y, mean, gain, variance)))
        pi = Decimal('3.14159265358979323846264338327950288419716939937510582097494459230781640628620899')
        center = int(y/g)
        terms = []
        for k in range(center-50, center+51):
            n = Decimal(k)
            correction = (1/(12*n) - 1/(360*n**3) + 1/(1260*n**5)
                          - 1/(1680*n**7) + 1/(1188*n**9) - Decimal(691)/(360360*n**11))
            log_factorial = (n+Decimal('.5')) * n.ln() - n + (2*pi).ln()/2 + correction
            value = n*mu.ln() - mu - log_factorial - ((2*pi*var).ln() + (y-g*n)**2/var)/2
            terms.append(value)
        maximum = max(terms)
        return float(maximum + sum((v-maximum).exp() for v in terms).ln())


def calibration(gain=1., variance=1.):
    return {'source': 'synthetic test calibration', 'offset_removed': True,
            'gain': gain, 'readout_variance': variance, 'poisson_counts_verified': True}


def fit_pixels(signal, profiles, background=.2, gain=1., readout=1., rcond=1e-8):
    y = np.r_[signal, np.full(8, background)]
    P = np.vstack([profiles, np.zeros((8, profiles.shape[1]))])
    return fit_constrained(y, np.ones(len(y)), P, np.ones(len(y), bool),
                           np.ones((len(y), 1)), calibration(gain, readout),
                           'poisson-gaussian', rcond)


class PixelPrinciplesTests(unittest.TestCase):
    def test_batched_log_density_and_score_against_independent_sum(self):
        rng = np.random.default_rng(20261029)
        means = np.r_[0., 1e-10, np.exp(rng.uniform(-6, 4, 73))]
        gain, readout = 1.3, 3.7
        y = gain * rng.poisson(means) + rng.normal(0, np.sqrt(readout), len(means))
        pg = PoissonGaussian(y, gain, readout)
        logs, score, diag = pg.evaluate(means)
        expected = direct_log_density(y, means, gain, readout)
        shifted = direct_log_density(y-gain, means, gain, readout)
        assert_allclose(logs, expected, atol=8e-13, rtol=0)
        assert_allclose(score, np.expm1(shifted-expected), atol=2e-12, rtol=2e-12)
        self.assertLessEqual(diag['max_relative_density_truncation_bound'], 1e-12)

    def test_high_count_likelihood_against_80_digit_reference(self):
        for mean in (1e8, 1e9, 1e12):
            y = mean + .3
            expected = decimal_high_count_density(y, mean)
            actual = PoissonGaussian([y], 1., .2).evaluate([mean])[0][0]
            assert_allclose(actual, expected, atol=4e-14, rtol=0)

    def test_high_count_poisson_deviance_preserves_small_likelihood_changes(self):
        for shift in (0.01, 1., 1000., 1e6):
            n, mu = 1e12, 1e12 + shift
            with localcontext() as context:
                context.prec = 70
                count, mean = Decimal.from_float(n), Decimal.from_float(mu)
                expected = float(count * (count/mean).ln() - count + mean)
            assert_allclose(_poisson_deviance([n], [mu])[0], expected, rtol=2e-15, atol=1e-28)

    def test_score_and_observed_information_match_finite_differences(self):
        y = np.array([-2.4, .7, 3.2, 32.1])
        mu = np.array([.07, .9, 4., 30.])
        pg = PoissonGaussian(y, 1.2, 1.7)
        information, diagnostics = pg.observed_information(mu)
        step = mu * 1e-4
        up, down = pg.evaluate(mu + step)[1], pg.evaluate(mu - step)[1]
        assert_allclose(information, -(up-down)/(2*step), atol=1e-8, rtol=1e-7)
        self.assertTrue(np.all(diagnostics['reliable_positive']))

    def test_zero_mean_derivative_is_regular(self):
        y, gain, readout = np.array([-2., 0., .5]), 1.4, .8
        pg = PoissonGaussian(y, gain, readout)
        _, score, _ = pg.evaluate(np.zeros(3))
        information, _ = pg.observed_information(np.zeros(3))
        r1 = np.exp((gain*y - .5*gain**2)/readout)
        r2 = np.exp((2*gain*y - 2*gain**2)/readout)
        assert_allclose(score, r1-1, atol=2e-15)
        assert_allclose(information, r1*r1-r2, atol=2e-15)

    def test_cache_only_reuses_equal_means_and_does_not_alias(self):
        y, means = np.array([-.3, 2.3, 4.]), np.array([0., 2., 5.])
        pg = PoissonGaussian(y, 1., 1.)
        original, _, _ = pg.evaluate(means)
        original[0] = 100.
        repeat, _, diagnostics = pg.evaluate(means.copy())
        self.assertEqual(diagnostics['terms_evaluated'], 0)
        self.assertEqual(diagnostics['unchanged_pixels_reused'], 3)
        self.assertNotEqual(repeat[0], 100.)
        means[1] += .5
        actual, _, diagnostics = pg.evaluate(means)
        reference = PoissonGaussian(y, 1., 1.).evaluate(means)[0]
        assert_allclose(actual, reference, atol=0, rtol=0)
        self.assertEqual(diagnostics['unchanged_pixels_reused'], 2)
        self.assertFalse(pg.data.flags.writeable)

    def test_information_budget_includes_all_shifted_sums(self):
        with self.assertRaises(LikelihoodBudgetError):
            PoissonGaussian([0.], 1., 1., max_terms_per_evaluation=2).observed_information([0.])
        for budget in (.5, 1.5, np.inf):
            with self.assertRaises(ValueError):
                PoissonGaussian([0.], 1., 1., max_terms_per_pixel=budget)

    def test_exact_pg_information_changes_low_count_uncertainty(self):
        P = np.array([[.6], [.3], [.1]])
        y = np.array([1.2, -.3, 2.4])
        result = fit_pixels(y, P, background=.5, gain=1.4, readout=1.1)
        self.assertTrue(result['converged'])
        diag = result['likelihood_diagnostics']
        self.assertEqual(diag['covariance_information_status'], 'resolved')
        intensity = result['intensities_internal'][0]
        self.assertGreater(intensity, 0.)
        mu = (.5 + P[:, 0] * intensity) / 1.4
        step = 1e-3
        score_up = PoissonGaussian(y, 1.4, 1.1).evaluate(mu+step*P[:, 0]/1.4)[1]
        score_down = PoissonGaussian(y, 1.4, 1.1).evaluate(mu-step*P[:, 0]/1.4)[1]
        curvature = -(P[:, 0]/1.4) @ (score_up-score_down) / (2*step)
        assert_allclose(result['covariance_identifiable_subspace'][0, 0], 1/curvature, rtol=3e-7)
        proxy = 1 / np.sum(P[:, 0]**2 / (1.4*(.5+P[:, 0]*intensity)+1.1))
        self.assertGreater(abs(proxy/result['covariance_identifiable_subspace'][0, 0]-1), .1)

    def test_unresolved_curvature_does_not_claim_observed_covariance(self):
        result = fit_pixels(np.array([-20.]), np.ones((1, 1)), background=0.)
        diag = result['likelihood_diagnostics']
        self.assertEqual(diag['covariance_information_status'], 'unresolved')
        self.assertFalse(diag['covariance_estimate_usable'])
        self.assertEqual(result['profile_intervals'][0]['status'], 'ok')
        self.assertEqual(result['profile_intervals'][0]['lower'], 0.)

    def test_complete_overlap_only_sum_is_identifiable_for_pg(self):
        P = np.array([[.7, .7], [.3, .3]])
        result = fit_pixels(np.array([7., 3.]), P, background=0., readout=.8)
        self.assertEqual(result['rank'], 1)
        self.assertTrue(all(c['status'] == 'unidentifiable' for c in result['profile_intervals']))
        self.assertEqual(result['total_interval']['status'], 'ok')
        total = result['intensities_internal'].sum()
        single = fit_pixels(np.array([7., 3.]), P[:, :1], background=0., readout=.8)
        assert_allclose(total, single['intensities_internal'][0], atol=2e-7)
        assert_allclose(result['total_interval']['upper'], single['total_interval']['upper'], atol=2e-7)

    def test_near_coincident_profiles_preserve_nullspace(self):
        P = np.array([[.7, .7+1e-12], [.3, .3-1e-12]])
        result = fit_pixels(np.array([7., 3.]), P, background=0., rcond=1e-8)
        self.assertEqual(result['rank'], 1)
        self.assertTrue(all(c['status'] == 'unidentifiable' for c in result['profile_intervals']))

    def test_strong_weak_overlap_matches_independent_full_sum_optimum(self):
        axis = np.linspace(-4, 4, 41)
        P = np.column_stack((np.exp(-(axis+.4)**2/.8), np.exp(-(axis-.4)**2/.8)))
        P /= P.sum(axis=0)
        rng = np.random.default_rng(21)
        background, readout = .2, 1.
        truth = np.array([1000., 1.])
        y = rng.poisson(background + P @ truth) + rng.normal(0, np.sqrt(readout), len(axis))
        result = fit_pixels(y, P, background=background, readout=readout)
        frozen_background = result['background_coefficients'][0]
        def independent(I):
            mu = frozen_background + P @ I
            logs = direct_log_density(y, mu, 1., readout)
            shifts = direct_log_density(y-1., mu, 1., readout)
            return -float(logs.sum()), -P.T @ np.expm1(shifts-logs)
        reference = minimize(independent, [900., 10.], jac=True, method='L-BFGS-B',
                             bounds=[(0, None), (0, None)], options={'ftol': 1e-14, 'gtol': 1e-10})
        self.assertTrue(result['converged'])
        assert_allclose(result['intensities_internal'], reference.x, atol=3e-4, rtol=1e-7)
        # Low information may leave the weak component at its boundary. No
        # numerical procedure is allowed to assert recovery of the known truth.
        self.assertEqual(result['profile_intervals'][1]['status'], 'ok')
        self.assertGreater(result['profile_intervals'][1]['upper'], 1.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
