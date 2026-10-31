"""Gaussian profile compression against pixel-space objectives and solves."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skills/ariadne-reduction/scripts'))
from constrained_pixels import _compress_gaussian, _projected_error, fit_constrained


def problem(rows=1600, duplicate=False):
    rng = np.random.default_rng(6817)
    coordinate = np.linspace(-3, 3, rows)
    profiles = np.exp(-.5 * ((coordinate[:, None] - [-.8, .2, 1.1]) / .65)**2)
    profiles /= profiles.sum(axis=0)
    if duplicate:
        profiles[:, 2] = profiles[:, 1]
    signal = profiles @ np.array([2500., 0., 1800.]) + rng.normal(0., .8, rows)
    profiles = np.vstack((profiles, np.zeros((16, 3))))
    variance = np.r_[rng.uniform(.7, 1.8, rows), np.ones(16)]
    data = np.r_[signal + 3., np.full(16, 3.)]
    valid = np.ones(len(data), bool)
    valid[13:19] = False
    return data, variance, profiles, valid, np.ones((len(data), 1))


class GaussianProfileCompressionTests(unittest.TestCase):
    def test_preserves_pixel_quadratic_gradient_and_centered_ratios(self):
        rng = np.random.default_rng(71)
        design = rng.normal(size=(1400, 4))
        design[:, 3] = design[:, 0] + 1e-8 * design[:, 1] + 1e-10 * design[:, 3]
        target = rng.normal(size=len(design))
        compressed, reduced_target = _compress_gaussian(design, target)
        self.assertEqual(compressed.shape, (5, 4))
        reference = np.array([2., -1., .1, .3])
        for x in (np.zeros(4), reference, np.array([-3., 4., 1., .2])):
            residual = design @ x - target
            reduced = compressed @ x - reduced_target
            assert_allclose(reduced @ reduced, residual @ residual, rtol=2e-15)
            assert_allclose(compressed.T @ reduced, design.T @ residual, rtol=3e-14, atol=2e-12)
            shift = x - reference
            expected = (design @ shift) @ (design @ reference - target) + .5 * np.linalg.norm(design @ shift)**2
            actual = (compressed @ shift) @ (compressed @ reference - reduced_target) + .5 * np.linalg.norm(compressed @ shift)**2
            assert_allclose(actual, expected, rtol=3e-14, atol=1e-11)
        # Compression must retain a weak direction rather than select a rank.
        assert_allclose(np.linalg.svd(compressed, compute_uv=False),
                        np.linalg.svd(design, compute_uv=False), rtol=2e-5, atol=1e-14)

    def test_small_and_underdetermined_problems_keep_original_arrays(self):
        for rows, columns in ((30, 3), (130, 40), (2, 8)):
            design, target = np.ones((rows, columns)), np.ones(rows)
            compressed, reduced = _compress_gaussian(design, target)
            self.assertIs(compressed, design)
            self.assertIs(reduced, target)

    def test_full_fit_and_intervals_match_uncompressed_pixel_solver(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate):
                args = problem(duplicate=duplicate)
                with patch('constrained_pixels._compress_gaussian', side_effect=lambda A, b: (A, b)):
                    expected = fit_constrained(*args, None, 'nnls', 1e-8)
                actual = fit_constrained(*args, None, 'nnls', 1e-8)
                self.assertTrue(actual['converged'])
                factorization = actual['likelihood_diagnostics']['likelihood_factorization']
                self.assertEqual(factorization['optimization_observations'], 4)
                self.assertEqual(factorization['gaussian_qr_rows_removed'], 1594 - 4)
                self.assertEqual(actual['rank'], expected['rank'])
                self.assertEqual(actual['boundary_components'], expected['boundary_components'])
                for key in ('fitted', 'chi2', 'covariance_identifiable_subspace', 'singular_values'):
                    assert_allclose(actual[key], expected[key], rtol=2e-10, atol=2e-8)
                if not duplicate:
                    assert_allclose(actual['intensities_internal'], expected['intensities_internal'], atol=2e-6)
                intervals = actual['profile_intervals'] + [actual['total_interval']]
                references = expected['profile_intervals'] + [expected['total_interval']]
                for interval, reference in zip(intervals, references):
                    self.assertEqual(interval['status'], reference['status'])
                    if interval['status'] == 'ok':
                        assert_allclose([interval['lower'], interval['upper']],
                                        [reference['lower'], reference['upper']], rtol=2e-9, atol=2e-6)
                # Check the final KKT directly in the original weighted pixels.
                data, variance, P, valid, basis = args
                weighted = P[valid] / np.sqrt(variance[valid, None])
                units = 1 / np.linalg.norm(weighted, axis=0)
                gradient = (weighted * units).T @ ((actual['fitted'][valid] - data[valid]) / np.sqrt(variance[valid]))
                self.assertLess(_projected_error(actual['intensities_internal'] / units, gradient), 1e-6)

    def test_compressed_single_component_boundary_has_analytic_endpoint(self):
        profile = np.ones(600) / np.sqrt(600.)
        P = np.r_[profile, np.zeros(8)][:, None]
        data = np.r_[-2 * profile, np.zeros(8)]
        result = fit_constrained(data, np.ones(len(data)), P, np.ones(len(data), bool),
                                 np.ones((len(data), 1)), None, 'nnls', 1e-8)
        interval = result['total_interval']
        self.assertTrue(result['converged'])
        self.assertEqual(result['boundary_components'], [0])
        self.assertEqual(interval['status'], 'ok')
        self.assertEqual(interval['lower'], 0.)
        assert_allclose(interval['upper'], -2 + np.sqrt(4 + chi2.ppf(.95, 1)), atol=2e-10)
        assert_allclose(result['chi2'], 4., atol=2e-12)

    def test_boundary_profile_endpoints_against_independent_pixel_search(self):
        data, variance, P, valid, basis = problem(rows=700)
        P = P[:, :2]
        result = fit_constrained(data, variance, P, valid, basis, None, 'nnls', 1e-8)
        self.assertTrue(result['converged'])
        weighted = P[valid] / np.sqrt(variance[valid, None])
        target = (data[valid] - basis[valid] @ result['background_coefficients']) / np.sqrt(variance[valid])
        baseline = np.linalg.norm(weighted @ result['intensities_internal'] - target)**2
        cutoff = chi2.ppf(.95, 1)
        for component, interval in enumerate(result['profile_intervals']):
            self.assertEqual(interval['status'], 'ok')
            for endpoint in ('lower', 'upper'):
                fixed = interval[endpoint]
                other = 1 - component
                def pixel_objective(free):
                    return np.linalg.norm(weighted[:, component] * fixed + weighted[:, other] * free - target)**2
                fit = minimize_scalar(pixel_objective, bounds=(0., 20000.), method='bounded',
                                      options={'xatol': 1e-9})
                delta = min(fit.fun, pixel_objective(0.)) - baseline
                if fixed == 0:
                    self.assertLessEqual(delta, cutoff + 1e-7)
                else:
                    assert_allclose(delta, cutoff, atol=2e-7, rtol=0)


if __name__ == '__main__':
    unittest.main()
