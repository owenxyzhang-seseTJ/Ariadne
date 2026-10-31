"""Independent lattice identities, closest-point searches and weak-domain support."""
import itertools
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.special import logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python/src'))
from ariadne.lattice_math import (PeriodicBudgetError, affine_integer_rank,
                                  periodic_gaussian, reduced_basis)
from ariadne.ranking import evidence


def radii_for(A, sigma, tail=8):
    return np.maximum(1, np.ceil(.5 + tail*sigma*np.linalg.norm(np.linalg.inv(A), axis=1))).astype(int)


def direct_reference(A, fractional, sigma, radius=6):
    offsets = np.asarray(list(itertools.product(range(-radius, radius + 1), repeat=3)))
    r2 = np.sum(((fractional[:, None] - offsets) @ A.T)**2, axis=2)
    values = (np.linalg.slogdet(A)[1] - 1.5*np.log(2*np.pi*sigma**2)
              + logsumexp(-r2/(2*sigma**2), axis=1))
    nearest = offsets[np.argmin(r2, axis=1)]
    return values, np.sqrt(r2.min(axis=1)), nearest


class CrystalStub:
    """The protocol used by evidence; no native DIALS object is required."""
    def __init__(self, matrix):
        self.matrix = matrix

    def get_A(self):
        return self.matrix.ravel()

    def get_unit_cell(self):
        direct = np.linalg.inv(self.matrix).T
        return SimpleNamespace(parameters=lambda: tuple(np.linalg.norm(direct, axis=0)) + (90., 90., 90.))


class LatticePrinciplesTests(unittest.TestCase):
    def test_integer_basis_transform_preserves_lattice_and_volume(self):
        base = np.array([[.03, .008, -.004], [0, .04, .013], [0, 0, .05]])
        shear = np.array([[1, 3500, -11], [0, 1, 17], [0, 0, 1]])
        reduced, transform = reduced_basis(base @ shear)
        self.assertEqual(round(np.linalg.det(transform)), 1)
        np.testing.assert_allclose((base @ shear).astype(np.longdouble) @ transform,
                                   reduced, rtol=2e-12, atol=1e-13)
        np.testing.assert_allclose(abs(np.linalg.det(reduced)), abs(np.linalg.det(base)), rtol=1e-10)
        self.assertLess(np.linalg.cond(reduced), 4)

    def test_orthogonal_real_sum_equals_independent_three_dimensional_sum(self):
        rng = np.random.default_rng(711)
        fraction = rng.uniform(-.5, .5, (93, 3))
        rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        A = rotation @ np.diag([.03, .04, .05])
        result = periodic_gaussian(A, fraction, .001, radii_for(A, .001), 17, 8)
        expected = direct_reference(A, fraction, .001, radius=3)
        for actual, reference in zip(result[:3], expected):
            np.testing.assert_allclose(actual, reference, rtol=2e-12, atol=3e-13)
        self.assertEqual(result[3]['periodic_factorization'], 'orthogonal_1d')
        self.assertEqual(result[3]['periodic_representation'], 'real')

    def test_orthogonal_fourier_sum_matches_long_real_sum(self):
        rng = np.random.default_rng(207)
        fraction = rng.uniform(-.5, .5, (75, 3))
        A = np.diag([.003, .004, .005])
        actual = periodic_gaussian(A, fraction, .003, radii_for(A, .003), 19, 8)
        expected = direct_reference(A, fraction, .003, radius=13)
        for value, reference in zip(actual[:3], expected):
            np.testing.assert_allclose(value, reference, rtol=1e-11, atol=4e-14)
        self.assertEqual(actual[3]['periodic_representation'], 'fourier')

    def test_coupled_metric_real_and_fourier_against_direct_density(self):
        fraction = np.random.default_rng(44).uniform(-.5, .5, (80, 3))
        A = np.array([[.004, .0013, -.0007], [0, .005, .0008], [0, 0, .006]])
        for sigma, representation in ((.0005, 'real'), (.004, 'fourier')):
            actual = periodic_gaussian(A, fraction, sigma, radii_for(A, sigma), 31, 8)
            expected = direct_reference(A, fraction, sigma, radius=10)
            for value, reference in zip(actual[:3], expected):
                np.testing.assert_allclose(value, reference, rtol=2e-12, atol=1e-13)
            self.assertEqual(actual[3]['periodic_representation'], representation)

    def test_density_and_nearest_physical_nodes_are_unimodular_invariant(self):
        base = np.array([[.03, .011, -.004], [0, .04, .014], [0, 0, .05]])
        fractional = np.random.default_rng(824).uniform(-.5, .5, (113, 3))
        expected = direct_reference(base, fractional, .003, radius=5)
        for shear in (np.array([[1, 31, 0], [0, 1, -9], [0, 0, 1]]),
                      np.array([[0, 1, 0], [-1, 0, 7], [0, 0, 1]])):
            A = base @ shear
            fraction = np.linalg.solve(shear, fractional.T).T
            actual = periodic_gaussian(A, fraction, .003, radii_for(A, .003), 23, 8, max_radius=6)
            np.testing.assert_allclose(actual[0], expected[0], rtol=3e-12, atol=2e-12)
            np.testing.assert_allclose(actual[1], expected[1], rtol=2e-12, atol=2e-14)
            np.testing.assert_allclose(actual[2] @ A.T, expected[2] @ base.T, atol=3e-14)

    def test_closest_node_is_not_restricted_to_noise_radius(self):
        A = np.array([[1., 10., 0.], [0., .1, 0.], [0., 0., 1.]])
        fraction = np.array([[.49, .49, .1], [-.47, -.43, -.2]])
        expected = direct_reference(A, fraction, .0001, radius=10)
        actual = periodic_gaussian(A, fraction, .0001, np.ones(3, dtype=int), 7, 8)
        np.testing.assert_allclose(actual[1], expected[1], atol=2e-15)
        np.testing.assert_allclose(actual[2], expected[2])
        self.assertTrue(actual[3]['nearest_lattice_point_certified'])
        self.assertGreater(np.max(np.abs(actual[2])), 1)

    def test_anisotropic_width_uses_axiswise_dual_representations(self):
        A = np.diag([.00005, .02, .03])
        fraction = np.random.default_rng(311).uniform(-.5, .5, (64, 3))
        actual = periodic_gaussian(A, fraction, .001, radii_for(A, .001), 16, 8, max_radius=6)
        # An independent product of long 1D real sums avoids a huge 3D tensor.
        expected = np.zeros(len(fraction))
        for j, length in enumerate(np.diag(A)):
            nodes = np.arange(-201, 202)
            exponent = -.5*((fraction[:, j, None] - nodes)*length/.001)**2
            expected += np.log(length/.001) - .5*np.log(2*np.pi) + logsumexp(exponent, axis=1)
        np.testing.assert_allclose(actual[0], expected, atol=1e-12)
        self.assertEqual(actual[3]['periodic_representation'], 'mixed')
        self.assertLess(actual[3]['periodic_terms'], 20)

    def test_exact_hkl_rank_survives_extreme_integer_shear(self):
        points = np.array([[0, 0, 0], [10**10, 1, 0], [10**10 + 1, 1, 0], [0, 0, 1]])
        self.assertEqual(affine_integer_rank(points), 3)
        self.assertEqual(affine_integer_rank(points[:3]), 2)
        self.assertEqual(affine_integer_rank(points[:2]), 1)
        self.assertEqual(affine_integer_rank(points[:1]), 0)

    def test_weak_minority_geometric_support_survives_equivalent_basis(self):
        rng = np.random.default_rng(1204)
        A = np.diag([.03, .04, .05])
        sigma = .0001
        # Only 18/150 observations belong to this lattice.  Their intensity is
        # deliberately not used: this checks support of weak minority domains,
        # not intensity-dependent spot detection or calibrated phase evidence.
        minority = rng.integers(-4, 5, (18, 3)) @ A.T + rng.normal(0, sigma, (18, 3))
        others = rng.uniform(-.2, .2, (132, 3))
        q = np.vstack((minority, others))
        params = SimpleNamespace(q_sigma=sigma, tail_sigma=8, max_periodic_radius=6,
            score_chunk=32, lattice_fraction=.6, match_sigma=3,
            min_validation_indexed=10, min_log_density_ratio=0)
        expected = evidence(CrystalStub(A), {'rlp': q}, params)
        shear = np.array([[1, 150, -40], [0, 1, 7], [0, 0, 1]])
        actual = evidence(CrystalStub(A @ shear), {'rlp': q}, params)
        self.assertTrue(expected['usable'])
        for key in ('usable', 'validation_indexed', 'unique_hkl', 'hkl_rank'):
            self.assertEqual(actual[key], expected[key])
        np.testing.assert_allclose(actual['mean_log_density_ratio'], expected['mean_log_density_ratio'], atol=2e-10)
        self.assertEqual(actual['hkl_rank_method'], 'exact integer affine rank in reduced basis')

    def test_budget_applies_to_selected_representation(self):
        A = np.diag([.0001, .00012, .00014])
        fraction = np.array([[.13, -.2, .3]])
        result = periodic_gaussian(A, fraction, .003, radii_for(A, .003), 64, 8, max_radius=6)
        self.assertGreater(min(result[3]['periodic_radius_xyz']), 6)
        self.assertEqual(result[3]['periodic_representation'], 'fourier')
        np.testing.assert_allclose(result[0], 0., atol=1e-14)
        with self.assertRaises(PeriodicBudgetError):
            periodic_gaussian(np.diag([.03, .04, .05]), fraction, .005,
                              np.full(3, 7), 64, 8, max_radius=1)

    def test_empty_inputs_and_invalid_geometry_are_explicit(self):
        empty = periodic_gaussian(np.eye(3), np.empty((0, 3)), .01, [1, 1, 1], 10, 8)
        self.assertEqual(empty[0].size, 0)
        for matrix in (np.zeros((3, 3)), np.eye(3)*np.nan):
            with self.assertRaises(ValueError):
                reduced_basis(matrix)
        with self.assertRaisesRegex(ValueError, 'positive'):
            periodic_gaussian(np.eye(3), [[0, 0, 0]], 0., [1, 1, 1], 1, 8)


if __name__ == '__main__':
    unittest.main()
