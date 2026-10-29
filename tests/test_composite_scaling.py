"""Composite scale recovery and independent numerical covariance references."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import block_diag

SCRIPTS = Path(__file__).resolve().parents[1] / 'skills' / 'ariadne-reduction' / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from composite_scaling import fit_global


def block(ids, design, x, beta, intensity, known=None, covariance=None):
    design, x = np.asarray(design, float), np.asarray(x, float)
    known = np.ones(len(ids)) if known is None else np.asarray(known, float)
    y = design @ (known * np.exp(x @ beta) * np.asarray(intensity))
    covariance = np.eye(len(y)) if covariance is None else np.asarray(covariance, float)
    return dict(equivalence_ids=ids, design=design, features=x,
                known_factors=known, y=y, covariance=covariance)


def composite_truth(noisy=False):
    beta = np.array([.24, -.16])
    intensity = np.array([18., 7., -1.3])
    ids = ['domain0:1,0,0', 'domain1:1,0,0', 'domain0:0,1,0']
    designs = [np.array([[1., 1., 0.], [0., 1., 1.]]),
               np.array([[1., 0., 1.], [1., 1., 1.]])]
    cov = np.array([[.7, .19], [.19, 1.2]])
    records = [np.array([[1., .3], [-.6, .8], [.4, -.9]]),
               np.array([[-.3, 1.], [.9, .1], [-.8, .4]]),
               np.array([[.7, -.5], [.2, -.9], [1., .2]])]
    blocks = []
    for j, x in enumerate(records):
        for sign in (1., -1.):
            b = block(ids.copy(), designs[j % 2], sign*x, beta, intensity,
                      known=np.array([1.1, .8, 1.4]) + .03*j, covariance=cov)
            blocks.append(b)
    if noisy:
        rng = np.random.default_rng(31)
        for b in blocks:
            b['y'] += rng.multivariate_normal(np.zeros(2), b['covariance']) * .15
    return blocks, beta, intensity


def numerical_jacobian(blocks, theta, ids):
    lookup = {name: j for j, name in enumerate(ids)}
    n = len(ids)

    def predict(value):
        out = []
        for b in blocks:
            ii = [lookup[name] for name in b['equivalence_ids']]
            out.append(b['design'] @ (b['known_factors'] * np.exp(b['features'] @ value[n:]) * value[ii]))
        return np.concatenate(out)

    columns = []
    for j in range(len(theta)):
        step = 2e-5 * max(1., abs(theta[j]))
        direction = np.eye(len(theta))[j] * step
        columns.append((predict(theta+direction)-predict(theta-direction))/(2*step))
    return np.column_stack(columns)


class CompositeScalingTests(unittest.TestCase):
    def test_totals_recover_components_with_real_alternating_steps(self):
        blocks, beta, intensity = composite_truth()
        # Each individual 2x3 block cannot resolve its three contributions.
        self.assertTrue(all(np.linalg.matrix_rank(b['design']) == 2 for b in blocks))
        original = deepcopy(blocks)
        result = fit_global(blocks)
        self.assertEqual(result['status'], 'converged', result['reason'])
        assert_allclose(result['beta'], beta, atol=1e-7)
        assert_allclose(result['I'], intensity, atol=2e-7)
        self.assertEqual(result['equivalence_ids'], blocks[0]['equivalence_ids'])
        self.assertEqual(result['parameter_order'], [('I', name) for name in blocks[0]['equivalence_ids']]
                         + [('beta', 0), ('beta', 1)])
        self.assertLess(result['chi2'], 1e-16)
        self.assertEqual(result['rank'], 5)
        self.assertEqual(result['dof'], 7)
        phases = [t['phase'] for t in result['trace']]
        self.assertIn('fixed_intensity_beta', phases)
        self.assertIn('fixed_beta_wls', phases)
        values = [t['chi2'] for t in result['trace']]
        self.assertTrue(np.all(np.diff(values) <= 0))
        # Independently verify initial and the first fixed-beta WLS objectives.
        aw = np.vstack([np.linalg.solve(np.linalg.cholesky(b['covariance']),
                                       b['design'] * b['known_factors']) for b in blocks])
        yw = np.concatenate([np.linalg.solve(np.linalg.cholesky(b['covariance']), b['y']) for b in blocks])
        initial_i = np.linalg.lstsq(aw, yw, rcond=1e-10)[0]
        assert_allclose(values[0], np.linalg.norm(aw @ initial_i - yw)**2, rtol=1e-12)
        for b, old, fitted, residual in zip(blocks, original, result['fitted'], result['residual']):
            assert_allclose(fitted, b['y'], atol=1e-7)
            assert_allclose(residual, fitted-b['y'])
            for key in ('y', 'covariance', 'features', 'known_factors', 'design'):
                assert_allclose(b[key], old[key], atol=0, rtol=0)
        self.assertLessEqual(result['gradient_norm'], 1e-8)
        self.assertLessEqual(result['relative_chi2_change'], 1e-8)

    def test_signed_weak_negative_and_known_factor_only(self):
        blocks = [block(['A:h', 'B:h'], np.eye(2), np.empty((2, 0)), [], [-.7, 9.],
                        known=[2., .5], covariance=[[2., .3], [.3, 1.]]),
                  block(['A:h', 'B:h'], [[1., 1.]], np.empty((2, 0)), [], [-.7, 9.],
                        known=[1.5, 2.], covariance=[[.4]])]
        result = fit_global(blocks)
        self.assertEqual(result['status'], 'converged')
        assert_allclose(result['I'], [-.7, 9.], atol=1e-12)
        self.assertLess(result['I'][0], 0)
        self.assertEqual(result['beta'].shape, (0,))
        self.assertEqual(result['factor_beta_jacobian'][0].shape, (2, 0))
        assert_allclose(result['factors'][0], [.5, 2.])
        a = np.vstack([b['design']*b['known_factors'] for b in blocks])
        cov = block_diag(*[b['covariance'] for b in blocks])
        expected = np.linalg.inv(a.T @ np.linalg.solve(cov, a))
        assert_allclose(result['joint_covariance'], expected, atol=1e-13)

    def test_shared_beta_does_not_force_domains_to_same_intensity(self):
        ids, truth = ['domain-A:1,2,3', 'domain-B:1,2,3'], [13., 47.]
        blocks = [block(ids.copy(), np.eye(2), np.array([[s], [-s]]), [.3], truth)
                  for s in (-1., 1.)]
        result = fit_global(blocks)
        self.assertEqual(result['status'], 'converged')
        assert_allclose(result['I'], truth, atol=1e-7)
        assert_allclose(result['beta'], [.3], atol=1e-8)
        self.assertEqual(len(result['parameter_order']), 3)

    def test_global_centering_and_repeated_equivalence_contributions(self):
        blocks = []
        for s in (-1., 1.):
            blocks.append(block(['A', 'A', 'B'], [[1., .4, .3], [.2, .6, 1.]],
                                np.array([[1.2+s], [1.2-s], [-2.4]]), [.17], [8., 8., 3.]))
        result = fit_global(blocks)
        self.assertEqual(result['status'], 'converged', result['reason'])
        assert_allclose(result['I'], [8., 3.], atol=1e-7)
        assert_allclose(result['beta'], [.17], atol=1e-8)
        self.assertTrue(result['gauge_diagnostics']['centered'])
        self.assertGreater(np.max(abs(result['gauge_diagnostics']['group_feature_means'])), 1.)

    def test_covariance_and_same_data_crosscov_against_numeric_jacobian(self):
        blocks, _, _ = composite_truth(noisy=True)
        result = fit_global(blocks)
        self.assertEqual(result['status'], 'converged', result['reason'])
        theta = np.r_[result['I'], result['beta']]
        jraw = numerical_jacobian(blocks, theta, result['equivalence_ids'])
        sigma = block_diag(*[b['covariance'] for b in blocks])
        lower = np.linalg.cholesky(sigma)
        jw = np.linalg.solve(lower, jraw)
        covariance = np.linalg.inv(jw.T @ jw)
        response = covariance @ jraw.T @ np.linalg.inv(sigma)
        crosscov = covariance @ jraw.T
        assert_allclose(np.vstack(result['joint_jacobians']), jraw, atol=2e-8, rtol=2e-8)
        assert_allclose(result['joint_covariance'], covariance, atol=2e-9, rtol=2e-8)
        assert_allclose(result['parameter_response'], response, atol=2e-9, rtol=2e-8)
        assert_allclose(np.hstack(result['block_parameter_data_covariance']), crosscov,
                        atol=2e-9, rtol=2e-8)
        assert_allclose(result['parameter_response'] @ sigma, crosscov, atol=2e-9, rtol=2e-8)
        assert_allclose(result['parameter_response'] @ sigma @ result['parameter_response'].T,
                        covariance, atol=2e-9, rtol=2e-8)
        # Full nuisance marginalization must differ from frozen-beta precision.
        frozen = np.linalg.inv(jw[:, :3].T @ jw[:, :3])
        self.assertGreater(np.max(abs(covariance[:3, :3]-frozen)), .01)
        self.assertLess(result['reduced_chi2'], .1)
        self.assertGreater(np.max(abs(result['joint_covariance']-covariance*result['reduced_chi2'])), .1)
        for b, factors, derivative in zip(blocks, result['factors'], result['factor_beta_jacobian']):
            for j in range(2):
                h = 1e-6
                direction = np.eye(2)[j] * h
                plus = 1/(b['known_factors']*np.exp(b['features'] @ (result['beta']+direction)))
                minus = 1/(b['known_factors']*np.exp(b['features'] @ (result['beta']-direction)))
                assert_allclose(derivative[:, j], (plus-minus)/(2*h), atol=2e-10)
            assert_allclose(factors, 1/(b['known_factors']*np.exp(b['features']@result['beta'])))

    def test_rank_deficient_totals_preserve_full_nullspace(self):
        b = block(['A', 'B', 'C'], [[1., 1., 1.]], np.empty((3, 0)), [], [2., 3., 4.])
        result = fit_global([b])
        self.assertEqual(result['status'], 'unresolved')
        self.assertEqual(result['reason'], 'rank_deficient')
        self.assertFalse(result['converged'])
        self.assertTrue(result['optimizer_converged'])
        self.assertEqual(result['rank'], 1)
        self.assertEqual(result['nullspace'].shape, (2, 3))
        self.assertFalse(np.any(result['identifiable_parameters']))
        assert_allclose(result['nullspace'] @ np.ones(3), 0, atol=1e-14)
        assert_allclose(result['joint_covariance'], np.ones((3, 3))/9, atol=1e-14)
        self.assertTrue(result['covariance_identifiable_subspace_only'])

    def test_constant_scale_and_dependent_features_are_not_regularized(self):
        for x in (np.ones((1, 1)), np.zeros((1, 1))):
            b = block(['A'], [[1.]], x, [0.], [5.])
            result = fit_global([b])
            self.assertEqual(result['status'], 'unresolved')
            self.assertEqual(result['rank'], 1)
            self.assertEqual(result['gauge_diagnostics']['centered_feature_rank'], 0)
            self.assertEqual(result['gauge_diagnostics']['beta_gauge_nullspace'].shape, (1, 1))
        blocks = [block(['A'], [[1.]], [[s, 2*s]], [.1, .2], [5.]) for s in (-1., 1.)]
        result = fit_global(blocks)
        self.assertEqual(result['status'], 'unresolved')
        self.assertEqual(result['reason'], 'rank_deficient')
        self.assertEqual(result['rank'], 2)
        self.assertEqual(result['gauge_diagnostics']['feature_rank'], 1)

    def test_full_rank_features_can_still_have_unobservable_beta(self):
        # Centered features with zero signed amplitude carry no scale signal.
        blocks = [block(['A'], [[1.]], [[s]], [.2], [0.]) for s in (-1., 1.)]
        result = fit_global(blocks)
        self.assertEqual(result['gauge_diagnostics']['feature_rank'], 1)
        self.assertEqual(result['rank'], 1)
        self.assertEqual(result['status'], 'unresolved')
        self.assertFalse(result['identifiable_parameters'][-1])

    def test_uncentered_full_rank_features_fail_contract(self):
        blocks = [block(['A'], [[1.]], [[s]], [.2], [5.]) for s in (1., 2., 3.)]
        result = fit_global(blocks)
        self.assertEqual(result['rank'], 2)
        self.assertEqual(result['status'], 'unresolved')
        self.assertEqual(result['reason'], 'uncentered_features')
        self.assertFalse(result['gauge_diagnostics']['centered'])

    def test_explicit_budgets_and_solver_failure_are_unresolved(self):
        blocks, _, _ = composite_truth()
        for kwargs, expected in [({'max_parameters': 4}, 'parameter_budget_exceeded'),
                                 ({'max_measurements': 5}, 'measurement_budget_exceeded'),
                                 ({'max_iterations': 0}, 'iteration_budget_exceeded')]:
            with self.subTest(kwargs=kwargs):
                result = fit_global(blocks, **kwargs)
                self.assertEqual(result['status'], 'unresolved')
                self.assertEqual(result['reason'], expected)
        with patch('composite_scaling.monotonic', side_effect=[0., 1.]):
            result = fit_global(blocks, max_seconds=.1)
        self.assertEqual(result['reason'], 'time_budget_exceeded')
        self.assertEqual(result['status'], 'unresolved')
        result = fit_global(blocks, max_iterations=1)
        self.assertEqual(result['reason'], 'iteration_budget_exceeded')
        self.assertEqual(result['status'], 'unresolved')
        self.assertIsNotNone(result['I'])
        # Solver failures cannot be promoted merely because objective is small.
        from scipy.optimize import least_squares as real_solver
        def failed_solver(*args, **kwargs):
            result = real_solver(*args, **kwargs)
            result.success, result.status = False, 0
            return result
        with patch('composite_scaling.least_squares', side_effect=failed_solver):
            result = fit_global(blocks)
        self.assertEqual(result['reason'], 'beta_solver_failed')
        self.assertEqual(result['status'], 'unresolved')
        self.assertIsNotNone(result['joint_covariance'])
        self.assertTrue(np.all(np.diff([t['chi2'] for t in result['trace']]) <= 0))

    def test_invalid_blocks_rejected(self):
        blocks, _, _ = composite_truth()
        modifications = [('y', [np.nan, 1.]), ('y', [[1., 2.]]), ('y', [1.+1j, 2.]),
                         ('covariance', [[1., 2.], [2., 1.]]),
                         ('covariance', [[0., 0.], [0., 1.]]),
                         ('covariance', [[1., .1], [0., 1.]]),
                         ('covariance', [[1.]]), ('design', np.ones((2, 2))),
                         ('features', np.ones((3, 3))), ('features', np.ones(3)),
                         ('features', np.full((3, 2), np.inf)),
                         ('known_factors', [1., 0., 1.]), ('known_factors', [1., -1., 1.]),
                         ('known_factors', np.ones((3, 1))),
                         ('equivalence_ids', ['A', '', 'B'])]
        for key, value in modifications:
            with self.subTest(key=key, value=value):
                bad = deepcopy(blocks)
                bad[0][key] = value
                with self.assertRaises(ValueError):
                    fit_global(bad)
        for bad in ([], {}, [blocks[0], blocks[0]], [None]):
            with self.assertRaises(ValueError):
                fit_global(bad)
        bad = deepcopy(blocks)
        del bad[0]['y']
        with self.assertRaises(ValueError):
            fit_global(bad)

    def test_invalid_controls_rejected(self):
        blocks, _, _ = composite_truth()
        for kwargs in ({'max_iterations': -1}, {'max_iterations': 1.5}, {'max_parameters': 0},
                       {'max_measurements': True}, {'max_seconds': np.inf}, {'tolerance': np.nan},
                       {'tolerance': 0}, {'rcond': 0}, {'rcond': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fit_global(blocks, **kwargs)


if __name__ == '__main__':
    unittest.main()
