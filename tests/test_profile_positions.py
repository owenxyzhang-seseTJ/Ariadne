"""窄峰体素概率与位置 nuisance 的可辨识性回归检查。"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skills/ariadne-reduction/scripts'))
from profile_geometry import gaussian_voxels
from overlap_integrator import solve_joint, solve_position_joint, contrast


class ProfilePositionTests(unittest.TestCase):
    def test_diagonal_mass_and_edges(self):
        bounds = (0, 13, 0, 13, 0, 13)
        for sigma in (.01, .1, .2, 1.):
            values = gaussian_voxels(np.eye(3) * sigma ** 2, [6.37, 6.81, 6.22], bounds)
            self.assertAlmostEqual(values.sum(), 1., places=8)
        edge = gaussian_voxels(np.eye(3), [0, 6, 6], bounds)
        self.assertAlmostEqual(edge.sum(), .5, places=8)
        self.assertTrue(np.all(edge >= 0))

    def test_invalid_covariance(self):
        with self.assertRaises(ValueError):
            gaussian_voxels([[1, .2, 0], [0, 1, 0], [0, 0, 1]], [3, 3, 3], (0, 7, 0, 7, 0, 7))

    def test_position_marginal_covariance(self):
        bounds = (0, 15, 0, 13, 0, 9)
        centres = np.array([[6.2, 6.4, 4.1], [8.3, 6.1, 4.4]])
        cov = np.diag([1.1, 1.2, .8]) ** 2
        templates = np.array([gaussian_voxels(cov, c, bounds) for c in centres])
        P = templates.reshape(2, -1).T
        step = .001
        D = np.column_stack([((gaussian_voxels(cov, c + np.eye(3)[a] * step, bounds)
                               - gaussian_voxels(cov, c - np.eye(3)[a] * step, bounds)) / (2 * step)).ravel()
                             for c in centres for a in range(3)])
        intensity = np.array([1100., 720.])
        shifts = np.array([[.13, -.08, .04], [-.1, .07, -.03]])
        y = 20 + P @ intensity + D @ (intensity[:, None] * shifts).ravel()
        v, valid, B = np.ones(len(y)) * 25, np.ones(len(y), bool), np.ones((len(y), 1))
        result = solve_position_joint(y, v, P, valid, B, D, 1e-9, 1e-7, .3)
        np.testing.assert_allclose(result['intensities_internal'], intensity, atol=1e-7)
        fitted_shifts = [[axis['shift_px'] for axis in axes] for axes in result['position_model']['shifts_xyz']]
        np.testing.assert_allclose(fitted_shifts, shifts, atol=1e-7)
        complete = np.column_stack((P, B, D)) / 5
        expected = np.linalg.inv(complete.T @ complete)[:2, :2]
        np.testing.assert_allclose(result['covariance_identifiable_subspace'], expected, rtol=1e-7)
        fixed = solve_joint(y, v, P, valid, B, 1e-9)
        self.assertTrue(np.all(np.diag(expected) >= np.diag(fixed['covariance_identifiable_subspace'])))

    def test_real_shift_and_degenerate_components(self):
        bounds = (0, 15, 0, 13, 0, 9)
        c = np.array([7.1, 6.2, 4.3])
        cov = np.diag([1.2, 1.3, 1.]) ** 2
        p = gaussian_voxels(cov, c, bounds).ravel()
        step = .001
        D = np.column_stack([((gaussian_voxels(cov, c + np.eye(3)[a] * step, bounds)
                               - gaussian_voxels(cov, c - np.eye(3)[a] * step, bounds)) / (2 * step)).ravel()
                             for a in range(3)])
        y = 15 + 1200 * gaussian_voxels(cov, c + [.12, -.06, .04], bounds).ravel()
        v, valid, B = np.ones(len(y)) * 20, np.ones(len(y), bool), np.ones((len(y), 1))
        result = solve_position_joint(y, v, p[:, None], valid, B, D, 1e-8, 1e-6, .3)
        self.assertLess(abs(result['intensities_internal'][0] / 1200 - 1), .01)
        estimates = [a['shift_px'] for a in result['position_model']['shifts_xyz'][0]]
        np.testing.assert_allclose(estimates, [.12, -.06, .04], atol=.002)
        duplicate = solve_position_joint(y, v, np.column_stack((p, p)), valid, B,
                                         np.tile(D, (1, 2)), 1e-8, 1e-6, .3)
        self.assertFalse(contrast(duplicate, np.array([1., 0]), 1e-6)['identifiable'])
        self.assertTrue(contrast(duplicate, np.ones(2), 1e-6)['identifiable'])


if __name__ == '__main__':
    unittest.main()
