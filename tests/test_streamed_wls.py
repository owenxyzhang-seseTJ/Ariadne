"""Square-root streaming against independent full weighted least squares."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skills/ariadne-reduction/scripts'))
from overlap_integrator import solve_joint
from overlap_prepare import template_values


def problem(rows=1500):
    rng = np.random.default_rng(713)
    x = np.linspace(-1, 1, rows)
    profiles = rng.uniform(.1, 2, (rows, 3))
    profiles /= profiles.sum(axis=0)
    background = np.column_stack((np.ones(rows), x))
    variance = rng.uniform(.2, 3., rows)
    valid = rng.random(rows) > .13
    data = profiles @ [-30., 200., .03] + background @ [2., -.3] + rng.normal(0, .1, rows)
    return data, variance, profiles, valid, background


class StreamedWLSTests(unittest.TestCase):
    def test_full_wls_reference_and_chunk_boundaries(self):
        y, v, p, valid, b = problem()
        complete = np.column_stack((p, b))
        weighted = complete[valid] / np.sqrt(v[valid, None])
        theta = np.linalg.lstsq(weighted, y[valid] / np.sqrt(v[valid]), rcond=1e-12)[0]
        covariance = np.linalg.inv(weighted.T @ weighted)
        expected_chi2 = np.sum((y[valid] - complete[valid] @ theta)**2 / v[valid])
        for chunk in (7, 97, 4096):
            with self.subTest(chunk=chunk):
                result = solve_joint(y, v, p, valid, b, 1e-10, chunk_rows=chunk)
                np.testing.assert_allclose(result['intensities_internal'], theta[:3], atol=3e-10)
                np.testing.assert_allclose(result['background_coefficients'], theta[3:], atol=2e-13)
                np.testing.assert_allclose(result['covariance_identifiable_subspace'], covariance[:3, :3], rtol=2e-12)
                np.testing.assert_allclose(result['chi2'], expected_chi2, rtol=2e-13)
                self.assertEqual(result['dof'], int(valid.sum())-5)

    def test_modes_preserve_every_profiled_quadratic(self):
        y, v, p, valid, b = problem(900)
        result = solve_joint(y, v, p, valid, b, 1e-10, chunk_rows=23)
        mode = result['information_modes']
        h, z = np.asarray(mode['design']), np.asarray(mode['y'])
        bw = b[valid] / np.sqrt(v[valid, None])
        for intensity in ([0, 0, 0], [-500, 40, .001], [1000, -30, 10]):
            target = (y[valid]-p[valid] @ intensity) / np.sqrt(v[valid])
            beta = np.linalg.lstsq(bw, target, rcond=None)[0]
            expected = np.sum((target-bw @ beta)**2)
            actual = np.sum((z-h @ intensity)**2)+mode['pixel_chi2_constant']
            self.assertAlmostEqual(actual, expected, delta=3e-12*max(1, expected))
        self.assertEqual(mode['valid_pixels'], int(valid.sum()))
        self.assertEqual(mode['nuisance_rank'], 2)

    def test_rank_convention_and_near_duplicate_weak_direction(self):
        y, v, p, valid, b = problem()
        p[:, 2] = p[:, 0]+1e-8*p[:, 2]
        dense = solve_joint(y, v, p, valid, b, 1e-7, chunk_rows=0)
        streamed = solve_joint(y, v, p, valid, b, 1e-7, chunk_rows=111)
        self.assertEqual(streamed['rank'], 2)
        self.assertEqual(streamed['information_modes']['rank'], 3)
        np.testing.assert_allclose(streamed['information_modes']['rank_cutoff'], dense['information_modes']['rank_cutoff'], rtol=3e-15)
        np.testing.assert_allclose(streamed['singular_values'], dense['singular_values'], atol=3e-15)
        np.testing.assert_allclose(streamed['fitted'], dense['fitted'], atol=4e-13)
        p[:, 2] = p[:, 0]
        exact = solve_joint(y, v, p, valid, b, 1e-7, chunk_rows=111)
        self.assertEqual(exact['information_modes']['rank'], 2)

    def test_auto_large_patch_and_rank_guards(self):
        y, v, p, valid, b = problem(9000)
        actual = solve_joint(y, v, p, valid, b, 1e-10)
        self.assertEqual(actual['information_modes']['compression']['chunk_rows'], 4096)
        with self.assertRaisesRegex(ValueError, 'Background basis'):
            solve_joint(y, v, p, valid, np.ones_like(b), 1e-10, chunk_rows=400)
        for chunk in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                solve_joint(y, v, p, valid, b, 1e-10, chunk_rows=chunk)

    def test_empirical_coordinate_chunks_match_full_interpolation(self):
        from scipy.ndimage import map_coordinates
        rng = np.random.default_rng(33)
        template = rng.uniform(size=(7, 9, 5))
        centre = np.array([.7, 1.4, -.8])
        bounds = (-4, 7, -5, 6, -3, 4)
        z, y, x = np.mgrid[-3:4, -5:6, -4:7]
        coordinates = np.stack((z+.5-centre[2], y+.5-centre[1], x+.5-centre[0]))
        coordinates += ((np.array(template.shape)-1)/2)[:, None, None, None]
        expected = map_coordinates(template, coordinates, order=1, mode='constant', cval=0)
        for chunk in (1, 31, 65536):
            actual = template_values({'template':'reference'}, lambda _:template, centre, bounds, chunk_voxels=chunk)
            np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
