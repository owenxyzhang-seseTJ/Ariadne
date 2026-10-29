"""重复计算消除后的数值等价性、秩判定与缓存边界回归。"""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import qr as real_qr
from scipy.optimize import least_squares as real_least_squares

SCRIPTS = Path(__file__).resolve().parents[1] / 'skills' / 'ariadne-reduction' / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import composite_scaling
import overlap_integrator
from overlap_prepare import template_values
from profile_geometry import gaussian_voxels
from test_composite_scaling import composite_truth


class NumericOptimizationTests(unittest.TestCase):
    def test_gaussian_template_skips_unused_grid(self):
        bounds = (0, 9, 0, 8, 0, 7)
        covariance = np.diag([.2, 1.3, .8]) ** 2
        centre = [.07, 4.2, 3.1]
        expected = gaussian_voxels(covariance, centre, bounds)
        loader = MagicMock(side_effect=AssertionError('Gaussian path loaded a template'))
        grid = MagicMock()
        grid.__getitem__.side_effect = AssertionError('Gaussian path allocated a grid')
        with patch('overlap_prepare.np.mgrid', grid):
            for template in (None, ''):
                actual = template_values({'template': template, 'covariance_px_xyz': covariance},
                                         loader, centre, bounds)
                assert_array_equal(actual, expected)
        loader.assert_not_called()

    def test_empirical_template_mixture_preserves_values(self):
        first = np.arange(1., 28.).reshape(3, 3, 3)
        first /= first.sum()
        second = first[::-1, :, ::-1].copy()
        templates = {'first': first, 'second': second}
        entry = {'template': 'first', 'template_mixture': [
            {'template': 'first', 'weight': .3}, {'template': 'second', 'weight': .7}]}
        expected = np.zeros((5, 5, 5))
        expected[1:4, 1:4, 1:4] = .3 * first + .7 * second
        actual = template_values(entry, templates.__getitem__, [2.5, 2.5, 2.5], (0, 5, 0, 5, 0, 5))
        assert_array_equal(actual, expected)

    @staticmethod
    def position_inputs():
        rng = np.random.default_rng(902)
        pixels = 160
        profiles = rng.uniform(.1, 1., (pixels, 2))
        profiles /= profiles.sum(axis=0)
        background = np.column_stack((np.ones(pixels), np.linspace(-1, 1, pixels) * 1e3,
                                      rng.normal(size=pixels) * 1e-3))
        derivatives = rng.normal(size=(pixels, 6)) * np.logspace(-3, 1, 6)
        intensity = np.array([125., -43.])
        shifts = np.array([[.12, -.07, .03], [-.09, .05, -.02]])
        parameters = np.r_[intensity, [3., .002, -20.], (intensity[:, None] * shifts).ravel()]
        complete = np.column_stack((profiles, background, derivatives))
        data = complete @ parameters + rng.normal(size=pixels) * .01
        variance = rng.uniform(.5, 4., pixels)
        valid = rng.uniform(size=pixels) > .12
        return data, variance, profiles, valid, background, derivatives

    def test_position_reuses_qr_and_matches_full_weighted_fit(self):
        data, variance, profiles, valid, background, derivatives = self.position_inputs()
        with patch('overlap_integrator.qr', wraps=real_qr) as factorize:
            actual = overlap_integrator.solve_position_joint(
                data, variance, profiles, valid, background, derivatives, 1e-10, 1e-7, .3)
        # 一次带主元 nuisance 分解和一次独立完整矩求解，不再重复选定基的 QR。
        self.assertEqual(factorize.call_count, 2)
        self.assertTrue(factorize.call_args_list[0].kwargs['pivoting'])
        self.assertNotIn('pivoting', factorize.call_args_list[1].kwargs)
        complete = np.column_stack((profiles, background, derivatives))
        weighted = complete[valid] / np.sqrt(variance[valid, None])
        expected = np.linalg.lstsq(weighted, data[valid] / np.sqrt(variance[valid]), rcond=1e-10)[0]
        covariance = np.linalg.inv(weighted.T @ weighted)
        assert_allclose(actual['intensities_internal'], expected[:2], rtol=1e-10, atol=1e-9)
        assert_allclose(actual['background_coefficients'], expected[2:5], rtol=1e-9, atol=1e-7)
        assert_allclose(actual['covariance_identifiable_subspace'], covariance[:2, :2], rtol=1e-10)
        assert_allclose(actual['fitted'], complete @ expected, rtol=1e-10, atol=1e-9)
        self.assertEqual(actual['dof'], int(valid.sum()) - complete.shape[1])
        expected_chi2 = np.sum((data[valid] - (complete @ expected)[valid]) ** 2 / variance[valid])
        assert_allclose(actual['chi2'], expected_chi2, rtol=1e-8, atol=1e-12)
        self.assertEqual(actual['position_model']['nuisance_rank'], 9)
        for i, axes in enumerate(actual['position_model']['shifts_xyz']):
            for axis, estimate in enumerate(axes):
                j = 5 + 3 * i + axis
                value = expected[j] / expected[i]
                gradient = np.zeros(len(expected))
                gradient[i], gradient[j] = -value / expected[i], 1 / expected[i]
                self.assertTrue(estimate['identifiable'])
                assert_allclose(estimate['shift_px'], value, rtol=1e-9, atol=1e-10)
                assert_allclose(estimate['variance_px2'], gradient @ covariance @ gradient, rtol=1e-9)

    def test_position_selected_basis_matches_refactorization(self):
        data, variance, profiles, valid, background, derivatives = self.position_inputs()
        # 零列和线性相关列检验 rcond 选择、归一化及主元映射。
        derivatives[:, 4] = 0.
        derivatives[:, 5] = derivatives[:, 0] * 3.
        nuisance = np.column_stack((background, derivatives))
        weighted = nuisance[valid] * (1 / np.sqrt(variance[valid]))[:, None]
        norms = np.linalg.norm(weighted, axis=0)
        nonzero = np.flatnonzero(norms > 1e-15)
        _, r, pivots = real_qr(weighted[:, nonzero] / norms[nonzero], mode='economic', pivoting=True)
        rank = int(np.sum(np.abs(np.diag(r)) > np.abs(np.diag(r)).max() * 1e-8))
        selected = nonzero[pivots[:rank]]
        basis = nuisance[:, selected] / norms[selected]
        expected = overlap_integrator.solve_joint(data, variance, profiles, valid, basis, 1e-8)
        actual = overlap_integrator.solve_position_joint(
            data, variance, profiles, valid, background, derivatives, 1e-8, 1e-6, .3)
        for key in ('intensities_internal', 'covariance_identifiable_subspace', 'fitted',
                    'column_scales', 'singular_values', 'chi2', 'reduced_chi2'):
            assert_allclose(actual[key], expected[key], rtol=2e-11, atol=1e-9)
        coefficients = np.zeros(nuisance.shape[1])
        coefficients[selected] = expected['background_coefficients'] / norms[selected]
        assert_allclose(actual['background_coefficients'], coefficients[:background.shape[1]],
                        rtol=2e-11, atol=1e-9)
        self.assertEqual(actual['rank'], expected['rank'])
        self.assertEqual(actual['dof'], expected['dof'])
        self.assertEqual(actual['position_model']['nuisance_rank'], rank)

    def test_position_retains_background_rank_guard(self):
        data, variance, profiles, valid, _, derivatives = self.position_inputs()
        background = np.ones((len(data), 1))
        derivatives[:] = 0.
        derivatives[:, 0] = 1. + 1e-13 * np.linspace(-1, 1, len(data))
        # rcond 允许保留近相关列时，原有 1e-12 背景秩门槛仍须拒绝回代。
        with self.assertRaisesRegex(ValueError, 'Background basis is rank deficient'):
            overlap_integrator.solve_position_joint(
                data, variance, profiles, valid, background, derivatives, 1e-15, 1e-6, .3)
        with self.assertRaisesRegex(ValueError, 'Background basis is rank deficient'):
            overlap_integrator.solve_joint(data, variance, profiles, valid,
                                           np.column_stack((background, derivatives[:, 0])), 1e-15)

    def test_scaling_callbacks_share_exact_cache_and_copy_parameters(self):
        blocks, _, _ = composite_truth(noisy=True)
        expected = composite_scaling.fit_global(blocks)
        dimensions = set()
        probes = []
        real_exp = np.exp
        with patch('composite_scaling.np.exp', wraps=real_exp) as exponential:
            def solver(fun, initial, jac, **kwargs):
                dimensions.add(len(initial))
                value = initial.copy()
                for coordinate in (0, len(value) - 1):
                    fun(value)
                    before = exponential.call_count
                    jac(value.copy())
                    self.assertEqual(exponential.call_count, before)
                    # 同一个数组原地变化也必须使缓存失效，不能使用近似相等判断。
                    value[coordinate] += 1e-8
                    self.assertTrue(np.allclose(value, initial))
                    before = exponential.call_count
                    residual = fun(value)
                    self.assertEqual(exponential.call_count - before, len(blocks))
                    before = exponential.call_count
                    jac(value.copy())
                    assert_array_equal(fun(value.copy()), residual)
                    self.assertEqual(exponential.call_count, before)
                    probes.append(coordinate)
                return real_least_squares(fun, initial, jac=jac, **kwargs)

            with patch('composite_scaling.least_squares', side_effect=solver):
                actual = composite_scaling.fit_global(blocks)
        self.assertEqual(dimensions, {2, 5})
        self.assertTrue(probes)
        self.assertEqual(actual['status'], expected['status'])
        self.assertEqual(actual['reason'], expected['reason'])
        for key in ('I', 'beta', 'joint_covariance', 'parameter_response', 'gradient', 'chi2'):
            assert_array_equal(actual[key], expected[key])
        self.assertEqual(actual['trace'], expected['trace'])

    def test_scaling_cache_hit_still_checks_time_budget(self):
        blocks, _, _ = composite_truth()
        clock = [0.]
        hit_checked = []
        with patch('composite_scaling.monotonic', side_effect=lambda: clock[0]):
            def expired_solver(fun, initial, jac, **kwargs):
                fun(initial)
                clock[0] = 2.
                try:
                    jac(initial.copy())
                except composite_scaling._StopFit:
                    hit_checked.append(True)
                    raise
                return SimpleNamespace(x=initial, success=True, status=1, nfev=1)

            with patch('composite_scaling.least_squares', side_effect=expired_solver):
                actual = composite_scaling.fit_global(blocks, max_seconds=1.)
        self.assertEqual(hit_checked, [True])
        self.assertEqual(actual['reason'], 'time_budget_exceeded')
        self.assertEqual(actual['status'], 'unresolved')
        self.assertIsNotNone(actual['I'])


if __name__ == '__main__':
    unittest.main()
