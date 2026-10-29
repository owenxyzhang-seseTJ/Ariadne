"""Conditional-CDF voxel integration against independent 3D quadrature."""
import itertools
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skills/ariadne-reduction/scripts'))
from profile_geometry import gaussian_voxels


_ROTATION = np.linalg.qr(np.array([[.7, -.4, .8], [.6, .8, -.1], [-.3, .5, .7]]))[0]


def _rotated_covariance(widths):
    return _ROTATION @ np.diag(np.square(widths)) @ _ROTATION.T


def _reference_voxel(covariance, centre, lower, order):
    """Integrate the original 3D density, without conditional decomposition.

    NumPy's Legendre rule is independent of the production SciPy rule. Sparse
    coordinates keep the reference bounded to an order**3 scalar workspace.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    points = [lower[axis] + (nodes + 1) / 2 - centre[axis] for axis in range(3)]
    x = points[0][:, None, None]
    y = points[1][None, :, None]
    z = points[2][None, None, :]
    precision = np.linalg.inv(covariance)
    exponent = (precision[0, 0] * x * x + precision[1, 1] * y * y
                + precision[2, 2] * z * z + 2 * precision[0, 1] * x * y
                + 2 * precision[0, 2] * x * z + 2 * precision[1, 2] * y * z)
    weighted = (np.exp(-exponent / 2) * weights[:, None, None]
                * weights[None, :, None] * weights[None, None, :])
    return weighted.sum() / (8 * np.sqrt((2 * np.pi) ** 3 * np.linalg.det(covariance)))


class ConditionalGaussianTests(unittest.TestCase):
    def assert_matches_3d_reference(self, covariance, centre, lowers, atol=1e-11):
        for lower in lowers:
            with self.subTest(lower=lower):
                # Check reference convergence separately from production error.
                reference = _reference_voxel(covariance, centre, lower, 72)
                converged = _reference_voxel(covariance, centre, lower, 96)
                self.assertLess(abs(reference - converged), 2e-12)
                bounds = tuple(edge for lo in lower for edge in (lo, lo + 1))
                actual = gaussian_voxels(covariance, centre, bounds)
                self.assertEqual(actual.shape, (1, 1, 1))
                self.assertTrue(np.isfinite(actual).all())
                self.assertGreaterEqual(actual.item(), 0)
                np.testing.assert_allclose(actual.item(), converged, rtol=2e-9, atol=atol)

    def test_diagonal_and_correlated_limit(self):
        covariance = np.diag([.36, .81, 1.44])
        centre = np.array([.37, .81, .22])
        bounds = (-2, 4, -1, 3, -3, 2)
        probabilities = []
        for axis in range(3):
            lo, hi = bounds[2 * axis:2 * axis + 2]
            edges = (np.arange(lo, hi + 1) - centre[axis]) / np.sqrt(covariance[axis, axis])
            probabilities.append(np.diff(ndtr(edges)))
        expected = (probabilities[2][:, None, None] * probabilities[1][None, :, None]
                    * probabilities[0][None, None, :])
        diagonal = gaussian_voxels(covariance, centre, bounds)
        np.testing.assert_allclose(diagonal, expected, rtol=5e-13, atol=2e-16)
        almost_diagonal = covariance.copy()
        almost_diagonal[0, 1] = almost_diagonal[1, 0] = 1e-13
        actual = gaussian_voxels(almost_diagonal, centre, bounds)
        np.testing.assert_allclose(actual, diagonal, rtol=2e-9, atol=2e-12)

    def test_rotated_covariance_against_3d_density(self):
        self.assert_matches_3d_reference(
            _rotated_covariance([.35, .8, 1.3]), np.array([.37, .81, .22]),
            [(-1, -1, -1), (0, 0, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 1)])

    def test_narrow_rotated_peak_against_3d_density(self):
        covariance = _rotated_covariance([.09, .45, 1.2])
        self.assertGreater(np.max(np.ceil(2 * np.sqrt(np.diag(np.linalg.inv(covariance))))), 16)
        self.assert_matches_3d_reference(
            covariance, np.array([.37, .81, .22]),
            [(-1, -1, -1), (0, 0, 0), (1, 0, 0), (0, 1, 0), (-1, 0, 1)])

    def test_strong_correlation_against_3d_density(self):
        covariance = .36 * np.array([[1, .97, .95], [.97, 1, .96], [.95, .96, 1]])
        self.assert_matches_3d_reference(
            covariance, np.array([.49, .52, .47]),
            [(-1, -1, -1), (0, 0, 0), (1, 0, 0), (0, 1, 0)])

    def test_exact_axis_has_no_quadrature_order_cap(self):
        # X = 4e-6 Y + epsilon, Var(epsilon) = 1e-10. The CDF transition
        # stays smooth in Y although X itself has an extremely small width.
        covariance = np.array([[1.16e-10, 4e-6, 0], [4e-6, 1, 0], [0, 0, 1.5]])
        centre = np.array([0., .2, -.3])
        actual = gaussian_voxels(covariance, centre, (-1, 1, 0, 1, -1, 1))
        p_y = ndtr(.8) - ndtr(-.2)
        p_z = ndtr(1.3 / np.sqrt(1.5)) - ndtr(-.7 / np.sqrt(1.5))
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue((actual >= 0).all())
        self.assertAlmostEqual(actual.sum(), p_y * p_z, delta=2e-11)

    def test_axis_permutations_preserve_zyx(self):
        covariance = _rotated_covariance([.17, .6, 1.1])
        centre = np.array([.37, .81, .22])
        bounds = (-1, 3, -2, 1, 0, 2)
        base_xyz = gaussian_voxels(covariance, centre, bounds).transpose(2, 1, 0)
        for permutation in itertools.permutations(range(3)):
            with self.subTest(permutation=permutation):
                indices = list(permutation)
                permuted_bounds = tuple(edge for axis in indices for edge in bounds[2 * axis:2 * axis + 2])
                actual = gaussian_voxels(covariance[np.ix_(indices, indices)], centre[indices], permuted_bounds)
                expected = base_xyz.transpose(permutation).transpose(2, 1, 0)
                np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-13)

    def test_marginal_cdf_at_support_edges(self):
        covariance = _rotated_covariance([.35, .8, 1.3])
        centre = np.array([.3, .6, -.2])
        for axis in range(3):
            with self.subTest(axis=axis):
                bounds = [-12, 13] * 3
                bounds[2 * axis:2 * axis + 2] = [-1, 2]
                values = gaussian_voxels(covariance, centre, bounds)
                expected = (ndtr((2 - centre[axis]) / np.sqrt(covariance[axis, axis]))
                            - ndtr((-1 - centre[axis]) / np.sqrt(covariance[axis, axis])))
                self.assertTrue((values >= 0).all())
                self.assertAlmostEqual(values.sum(), expected, delta=2e-10)

    def test_positive_and_negative_cdf_tails(self):
        covariance = np.array([[1, .4, 0], [.4, 1.5, 0], [0, 0, 2]])
        centre = np.zeros(3)
        positive = gaussian_voxels(covariance, centre, (9, 10, -14, 15, -12, 13))
        negative = gaussian_voxels(covariance, centre, (-10, -9, -15, 14, -13, 12))
        expected = ndtr(-9.) - ndtr(-10.)
        self.assertGreater(positive.sum(), 0)
        np.testing.assert_allclose(positive.sum(), expected, rtol=2e-9, atol=0)
        np.testing.assert_allclose(negative, positive[::-1, ::-1, ::-1], rtol=2e-12, atol=0)

    def test_unresolved_numerical_axes_are_rejected(self):
        covariance = _rotated_covariance([.004, .006, 1.])
        with self.assertRaisesRegex(ValueError, '64 quadrature nodes per numerical axis'):
            gaussian_voxels(covariance, [0, 0, 0], (0, 1, 0, 1, 0, 1))

    def test_invalid_scientific_inputs(self):
        invalid = [np.diag([1., 0., 1.]), np.diag([1., -1., 1.]),
                   [[1, .2, 0], [0, 1, 0], [0, 0, 1]], np.eye(3) * np.nan,
                   np.eye(2)]
        for covariance in invalid:
            with self.subTest(covariance=covariance):
                with self.assertRaises(ValueError):
                    gaussian_voxels(covariance, [0, 0, 0], (0, 1, 0, 1, 0, 1))
        for centre in ([0, np.inf, 0], [0, 0]):
            with self.subTest(centre=centre):
                with self.assertRaises(ValueError):
                    gaussian_voxels(np.eye(3), centre, (0, 1, 0, 1, 0, 1))


if __name__ == '__main__':
    unittest.main()
