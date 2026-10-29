"""Numerical regression tests, runnable with unittest and NumPy/SciPy only."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import brentq, minimize_scalar
from scipy.special import gammaln, logsumexp, xlogy
from scipy.stats import chi2

SCRIPTS = Path(__file__).resolve().parents[1] / 'skills' / 'ariadne-reduction' / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from constrained_pixels import fit_constrained
from poisson_gaussian import LikelihoodBudgetError, PoissonGaussian


def noise(gain=1., readout=1.):
    return {'source': 'synthetic calibrated detector', 'provenance': {'calibration': 'unit-test'},
            'gain': gain, 'readout_variance': readout, 'offset_removed': True,
            'poisson_counts_verified': True}


def fit(signal, P=None, method='nnls', background=0., calibration=None, variance=1., confidence=.95):
    signal = np.atleast_1d(signal).astype(float)
    if P is None:
        P = np.ones((len(signal), 1)) / len(signal)
    P = np.vstack([P, np.zeros((8, P.shape[1]))])
    y = np.r_[signal, np.full(8, background)]
    return fit_constrained(y, np.full(len(y), variance), P, np.ones(len(y), bool),
                           np.ones((len(y), 1)), calibration, method, 1e-8, confidence)


def reference_pg(y, mu, gain, readout, stop):
    n = np.arange(stop, dtype=float)
    if mu == 0:
        return -.5 * (np.log(2 * np.pi * readout) + y * y / readout)
    return float(logsumexp(n * np.log(mu) - mu - gammaln(n + 1)
                          -.5 * (np.log(2 * np.pi * readout) + (y - gain * n) ** 2 / readout)))


class ConvolutionTests(unittest.TestCase):
    def test_density_and_score_against_wide_sum(self):
        cases = [(-3.2, .02, 2., 1.4, 300), (.8, .7, 1.3, 2., 300),
                 (4.2, 0., .7, 3., 300), (50., .01, 1., .5, 300),
                 (1010.4, 1000., 1., 8., 2000), (1e5 + 1.1, 1e5, 1., .2, 101000)]
        for y, mu, gain, readout, stop in cases:
            with self.subTest(y=y, mu=mu):
                evaluator = PoissonGaussian([y], gain, readout)
                logp, score, diagnostics = evaluator.evaluate([mu])
                expected = reference_pg(y, mu, gain, readout, stop)
                shifted = reference_pg(y - gain, mu, gain, readout, stop)
                assert_allclose(logp[0], expected, atol=3e-10, rtol=0)
                assert_allclose(score[0], np.expm1(shifted - expected), atol=3e-10, rtol=2e-10)
                self.assertLessEqual(diagnostics['max_relative_density_truncation_bound'], 1e-12)
                if 0 < mu < 2000:
                    h = max(mu * 1e-5, 1e-7)
                    fd = (reference_pg(y, mu + h, gain, readout, stop)
                          - reference_pg(y, mu - h, gain, readout, stop)) / (2 * h)
                    assert_allclose(score[0], fd, atol=5e-7, rtol=5e-7)

    def test_reported_tail_bound_against_expanded_sum(self):
        evaluator = PoissonGaussian([20.3], 1., 100., relative_tolerance=.01)
        logp, score, diagnostics = evaluator.evaluate([20.])
        full = reference_pg(20.3, 20., 1., 100., 400)
        shifted = reference_pg(19.3, 20., 1., 100., 400)
        self.assertLessEqual(np.expm1(full - logp[0]), diagnostics['max_relative_density_truncation_bound'] + 1e-14)
        exact_score = np.expm1(shifted - full)
        self.assertLessEqual(abs(score[0] - exact_score), diagnostics['max_absolute_score_truncation_bound'] + 1e-14)

    def test_budget_exhaustion_no_fallback(self):
        with self.assertRaises(LikelihoodBudgetError):
            PoissonGaussian([2.], 1., 2., max_terms_per_pixel=5).evaluate([2.])
        with self.assertRaises(LikelihoodBudgetError):
            PoissonGaussian([0., 0.], 1., 1., max_terms_per_evaluation=2).evaluate([0., 0.])

    def test_invalid_convolution_inputs(self):
        for gain, readout in [(0, 1), (1, 0), (np.nan, 1), (1, np.inf)]:
            with self.assertRaises(ValueError):
                PoissonGaussian([0.], gain, readout)
        evaluator = PoissonGaussian([-2.], 1., 1.)
        for mu in [-1., np.nan, np.inf]:
            with self.assertRaises(ValueError):
                evaluator.evaluate([mu])


class FitTests(unittest.TestCase):
    def test_gaussian_boundary_has_analytic_one_sided_endpoint(self):
        result = fit([-2.])
        self.assertTrue(result['converged'])
        assert_allclose(result['intensities_internal'], [0.], atol=1e-10)
        interval = result['profile_intervals'][0]
        self.assertEqual(interval['status'], 'ok')
        self.assertEqual(interval['lower'], 0.)
        assert_allclose(interval['upper'], -2 + np.sqrt(4 + chi2.ppf(.95, 1)), rtol=1e-8)
        self.assertIn('not exact', interval['calibration'])
        self.assertEqual(result['boundary_components'], [0])

    def test_gaussian_interior_confidence_level(self):
        result = fit([10.], confidence=.90)
        interval = result['profile_intervals'][0]
        distance = np.sqrt(chi2.ppf(.90, 1))
        assert_allclose([interval['lower'], interval['upper']], [10 - distance, 10 + distance], atol=1e-8)

    def test_nuisance_reoptimized_for_component_and_total(self):
        P = np.array([[.8, .1], [.2, .5], [0., .4]])
        signal = P @ [10., 12.]
        result = fit(signal, P)
        self.assertTrue(result['converged'])
        covariance = np.linalg.inv(P.T @ P)
        cutoff = chi2.ppf(.95, 1)
        for i, interval in enumerate(result['profile_intervals']):
            self.assertEqual(interval['status'], 'ok')
            center = [10., 12.][i]
            assert_allclose([interval['lower'], interval['upper']],
                            center + np.array([-1., 1.]) * np.sqrt(cutoff * covariance[i, i]), atol=1e-7)
        total = result['total_interval']
        self.assertEqual(total['status'], 'ok')
        assert_allclose([total['lower'], total['upper']],
                        22 + np.array([-1., 1.]) * np.sqrt(cutoff * covariance.sum()), atol=1e-7)
        frozen_nuisance_width = np.sqrt(cutoff / (P[:, 0] @ P[:, 0]))
        self.assertGreater(result['profile_intervals'][0]['upper'] - 10, frozen_nuisance_width)

    def test_duplicate_columns_only_total_identifiable(self):
        P = np.array([[.7, .7], [.3, .3]])
        result = fit([7., 3.], P)
        self.assertEqual(result['rank'], 1)
        self.assertTrue(all(c['status'] == 'unidentifiable' and c['lower'] is None for c in result['profile_intervals']))
        self.assertEqual(result['total_interval']['status'], 'ok')
        width = np.sqrt(chi2.ppf(.95, 1) / (.7**2 + .3**2))
        assert_allclose([result['total_interval']['lower'], result['total_interval']['upper']],
                        [10 - width, 10 + width], atol=1e-7)

    def test_unequal_collinear_columns_total_not_identifiable(self):
        result = fit([7., 3.], np.array([[.7, .35], [.3, .15]]))
        self.assertEqual(result['total_interval']['status'], 'unidentifiable')
        self.assertIsNone(result['total_interval']['upper'])

    def test_underdetermined_design_has_full_nullspace(self):
        result = fit([4.], np.ones((1, 12)))
        self.assertEqual(result['rank'], 1)
        self.assertEqual(result['nullspace'].shape, (11, 12))
        self.assertTrue(all(c['status'] == 'unidentifiable' for c in result['profile_intervals']))
        self.assertEqual(result['total_interval']['status'], 'ok')

    def test_zero_count_poisson_exact_endpoint(self):
        for method in ['poisson', 'poisson-gaussian']:
            result = fit([0.], method=method, calibration=noise(2., 0.))
            self.assertTrue(result['converged'])
            interval = result['profile_intervals'][0]
            self.assertEqual(interval['lower'], 0.)
            assert_allclose(interval['upper'], chi2.ppf(.95, 1), atol=1e-8)
            self.assertIn('frozen', interval['scope'])

    def test_positive_count_poisson_profile(self):
        result = fit([6.], method='poisson', calibration=noise(2., 0.))
        self.assertTrue(result['converged'])
        assert_allclose(result['intensities_internal'], [6.], atol=1e-7)
        interval = result['profile_intervals'][0]
        self.assertEqual(interval['status'], 'ok')
        for endpoint in ['lower', 'upper']:
            mu = interval[endpoint] / 2
            assert_allclose(2 * (mu - 3 - 3 * np.log(mu / 3)), chi2.ppf(.95, 1), atol=2e-7)

    def test_poisson_disjoint_support_component_lower_root(self):
        result = fit([3., 2.], np.eye(2), method='poisson', calibration=noise(1., 0.))
        self.assertTrue(result['converged'])
        for count, interval in zip([3., 2.], result['profile_intervals']):
            self.assertEqual(interval['status'], 'ok')
            self.assertGreater(interval['lower'], 0.)
            for endpoint in ['lower', 'upper']:
                mu = interval[endpoint]
                assert_allclose(2 * (mu - count - count * np.log(mu / count)),
                                chi2.ppf(.95, 1), atol=2e-7)

    def test_poisson_gaussian_negative_adu_and_boundary(self):
        result = fit([-1.8], method='poisson-gaussian', calibration=noise(1.7, 1.2))
        self.assertTrue(result['converged'])
        self.assertEqual(result['boundary_components'], [0])
        interval = result['profile_intervals'][0]
        self.assertEqual(interval['status'], 'ok')
        self.assertEqual(interval['lower'], 0.)
        log0 = reference_pg(-1.8, 0., 1.7, 1.2, 500)
        root = brentq(lambda t: 2 * (log0 - reference_pg(-1.8, t / 1.7, 1.7, 1.2, 500))
                      - chi2.ppf(.95, 1), 0., 20.)
        assert_allclose(interval['upper'], root, atol=1e-8)
        self.assertTrue(result['likelihood_diagnostics']['negative_adu_allowed'])
        self.assertIsNone(result['reduced_chi2'])

    def test_low_count_pg_fit_independent_scalar_reference(self):
        signal = np.array([-.6, 1.1, 2.4])
        P = np.array([[.6], [.3], [.1]])
        result = fit(signal, P, method='poisson-gaussian', background=.5,
                     calibration=noise(1.4, 1.1))
        def nll(t):
            return -sum(reference_pg(y, (.5 + p * t) / 1.4, 1.4, 1.1, 500)
                        for y, p in zip(signal, P[:, 0]))
        independent = minimize_scalar(nll, bounds=(0., 20.), method='bounded', options={'xatol': 1e-11})
        self.assertTrue(result['converged'])
        assert_allclose(result['intensities_internal'][0], independent.x, atol=2e-6)
        interval = result['total_interval']
        self.assertEqual(interval['status'], 'ok')
        assert_allclose(2 * (nll(interval['upper']) - independent.fun), chi2.ppf(.95, 1), atol=1e-7)

    def test_pg_overlapping_nuisance_and_total(self):
        P = np.array([[.7, .1], [.2, .3], [.1, .6]])
        signal = np.array([2.2, 1.1, -.5])
        result = fit(signal, P, method='poisson-gaussian', background=.2, calibration=noise(1.2, .8))
        self.assertTrue(result['converged'])
        self.assertEqual(result['total_interval']['status'], 'ok')
        self.assertTrue(all(c['status'] == 'ok' for c in result['profile_intervals']))
        cutoff = chi2.ppf(.95, 1)
        def nll(I):
            return -sum(reference_pg(y, mu, 1.2, .8, 500)
                        for y, mu in zip(signal, (.2 + P @ I) / 1.2))
        mle = result['intensities_internal']
        base = nll(mle)
        for i, interval in enumerate(result['profile_intervals']):
            t = interval['upper']
            def objective(z):
                I = np.zeros(2)
                I[i], I[1-i] = t, z
                return nll(I)
            opt = minimize_scalar(objective, bounds=(0., 40.), method='bounded')
            value = min(opt.fun, objective(0.))
            assert_allclose(2 * (value - base), cutoff, atol=3e-7)
        t = result['total_interval']['upper']
        opt = minimize_scalar(lambda z: nll([z, t-z]), bounds=(0., t), method='bounded')
        value = min(opt.fun, nll([0., t]), nll([t, 0.]))
        assert_allclose(2 * (value - base), cutoff, atol=3e-7)

    def test_pg_adu_rescaling_preserves_physical_fit_and_intervals(self):
        signal, P = np.array([-.3, 2.1, 1.]), np.array([[.5], [.3], [.2]])
        first = fit(signal, P, method='poisson-gaussian', background=.4,
                    calibration=noise(1.2, .8), variance=2.)
        factor = 4.
        second = fit(signal * factor, P, method='poisson-gaussian', background=.4 * factor,
                     calibration=noise(1.2 * factor, .8 * factor**2), variance=2. * factor**2)
        assert_allclose(second['intensities_internal'], first['intensities_internal'] * factor, atol=1e-7)
        self.assertTrue(first['converged'] and second['converged'])
        for key in ['lower', 'upper']:
            assert_allclose(second['total_interval'][key], first['total_interval'][key] * factor, atol=1e-7)

    def test_background_and_uninformative_component_rejected(self):
        with self.assertRaisesRegex(ValueError, 'negative expectation'):
            fit([1.], background=-.1)
        with self.assertRaisesRegex(ValueError, 'uninformative'):
            fit([1.], np.zeros((1, 1)))
        y, P = np.array([2.] + [0.] * 8), np.array([[1.]] + [[0.]] * 8)
        P[0] = 0
        with self.assertRaises(ValueError):
            fit_constrained(y, np.ones(9), P, np.ones(9, bool), np.ones((9, 1)),
                            noise(1., 0.), 'poisson', 1e-8)

    def test_no_intervals_if_optimizer_failed(self):
        import constrained_pixels
        original = constrained_pixels._minimize_nonnegative
        def fail(fun, seed):
            x, value, _, error, message = original(fun, seed)
            return x, value, False, error, message
        with patch.object(constrained_pixels, '_minimize_nonnegative', fail):
            result = fit([1.], method='poisson-gaussian', calibration=noise())
        self.assertFalse(result['converged'])
        self.assertEqual(result['total_interval']['status'], 'fit_not_converged')
        self.assertIsNone(result['total_interval']['upper'])

    def test_calibration_and_lattice_validation(self):
        for key, value in [('offset_removed', False), ('source', ''), ('provenance', {}),
                           ('gain', 0), ('gain', np.nan), ('readout_variance', -1),
                           ('readout_variance', np.inf)]:
            calibration = noise()
            calibration[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                fit([0.], method='poisson-gaussian', calibration=calibration)
        calibration = noise()
        calibration.pop('gain')
        with self.assertRaises(ValueError):
            fit([0.], method='poisson-gaussian', calibration=calibration)
        for observed in [-1., .5]:
            with self.assertRaises(ValueError):
                fit([observed], method='poisson-gaussian', calibration=noise(1, 0))
        with self.assertRaises(ValueError):
            fit([1.], method='poisson', calibration=noise())
        with self.assertRaises(ValueError):
            fit([1.], method='arbitrary')
        for confidence in [0, 1, np.nan]:
            with self.assertRaises(ValueError):
                fit([1.], confidence=confidence)

    def test_legacy_positional_interface_and_result_keys(self):
        P = np.r_[1., np.zeros(8)][:, None]
        result = fit_constrained(np.r_[4., np.zeros(8)], np.ones(9), P,
                                 np.ones(9, bool), np.ones((9, 1)), None, 'nnls', 1e-6)
        keys = {'intensities_internal', 'covariance_identifiable_subspace', 'nullspace',
                'column_scales', 'rank', 'singular_values', 'background_coefficients',
                'fitted', 'chi2', 'dof', 'reduced_chi2', 'converged',
                'boundary_components', 'covariance_scope', 'profile_intervals',
                'total_interval', 'likelihood_diagnostics'}
        self.assertEqual(set(result), keys)


if __name__ == '__main__':
    unittest.main(verbosity=2)
