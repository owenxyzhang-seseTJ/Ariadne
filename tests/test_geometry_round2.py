"""Independent density, native-prediction derivative and cache regressions."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skills/ariadne-reduction/scripts'))
from centroid_em import (Budget, DialsCrystalAdapter, candidate_support,
                         covariance_factors, expectation, weighted_residual_squares)
from rotation_geometry import compatibility_metrics, crystal_prediction_jacobian


class PairWorkspaceTests(unittest.TestCase):
    def test_blocked_expectation_matches_independent_gaussian_mixture(self):
        rng = np.random.default_rng(8821)
        y, means = rng.normal(size=(17, 3)), rng.normal(size=(43, 3))
        matrix = rng.normal(size=(17, 3, 3))
        covariance = matrix @ matrix.transpose(0, 2, 1) + .1 * np.eye(3)
        white, normalizer = covariance_factors(covariance)
        allowed = rng.uniform(size=(17, 43)) > .3
        allowed[3] = False
        counts, density = np.arange(17) + 43, rng.uniform(1e-6, 1e-3, 17)
        log_components = np.array([[multivariate_normal.logpdf(row, mean=mean, cov=C)
                                   for mean in means] for row, C in zip(y, covariance)])
        log_components += np.log(.93) - np.log(counts)[:, None]
        log_components[~allowed] = -np.inf
        target_log = np.logaddexp(logsumexp(log_components, axis=1), np.log(.07 * density))
        for block in (1, 53, 65536):
            with self.subTest(block=block):
                resp, outlier, actual = expectation(y, means, white, normalizer, allowed,
                                                   counts, density, .07, pair_block_size=block)
                np.testing.assert_allclose(actual, target_log, rtol=3e-14, atol=3e-14)
                np.testing.assert_allclose(resp, np.exp(log_components - target_log[:, None]), rtol=3e-13, atol=3e-15)
                np.testing.assert_allclose(resp.sum(axis=1) + outlier, 1., atol=5e-15)
                self.assertEqual(resp[3].sum(), 0.)

    def test_scan_labels_preserve_all_panel_prior_mass_and_duplicate_keys(self):
        key_a, key_b = (('a', 'b'), (0, 2)), (('c',), (3, 9))
        raw, model = [key_a, key_b, key_a], [key_b, key_a, key_a]
        observed_ids, predicted_ids = np.array([0, 1, 2]), np.array([0, 0, 1, 1, 2])
        panels, pred_panels = np.array([0, 1, 1]), np.array([1, 0, 0, 1, 0])
        allowed, counts = candidate_support(raw, model, observed_ids, predicted_ids, panels, pred_panels)
        scan = np.array([[raw[i] == model[j] for j in predicted_ids] for i in observed_ids])
        np.testing.assert_array_equal(counts, scan.sum(axis=1))
        np.testing.assert_array_equal(allowed, scan & (panels[:, None] == pred_panels))
        self.assertGreater(counts[2], allowed[2].sum())

    def test_weighted_residual_blocks_match_direct_sum(self):
        rng = np.random.default_rng(19)
        y, means, weights = rng.normal(size=(11, 3)), rng.normal(size=(29, 3)), rng.uniform(size=(11, 29))
        reference = np.sum(weights[:, :, None] * (y[:, None] - means[None])**2, axis=(0, 1))
        for block in (1, 37, 65536):
            np.testing.assert_allclose(weighted_residual_squares(y, means, weights, block), reference, rtol=3e-15)

    def test_one_covariance_factor_matches_independent_candidate_logpdf(self):
        rng = np.random.default_rng(811)
        means, y = rng.normal(size=(128, 3)), rng.normal(size=3)
        covariance = np.array([[1., .7, -.3], [.7, 2., .1], [-.3, .1, .5]])
        residual, distance, density = compatibility_metrics(y, means, covariance)
        np.testing.assert_array_equal(residual, y - means)
        np.testing.assert_allclose(density, [multivariate_normal.logpdf(y, mean=p, cov=covariance) for p in means], rtol=1e-13, atol=1e-14)
        reference = np.array([r @ np.linalg.solve(covariance, r) for r in residual])
        np.testing.assert_allclose(distance, reference, rtol=4e-15)

    def test_tangent_ewald_intersection_is_not_regularized_into_finite_derivative(self):
        with self.assertRaisesRegex(ValueError, 'Ewald tangency'):
            crystal_prediction_jacobian([[0., 0., 1.]], [0.], [0., 0., -1.], [1., 0., 0.],
                np.eye(3), np.eye(3), np.eye(3), np.eye(3)[None], np.eye(3)[None])


DIALS = importlib.util.find_spec('dials') is not None


@unittest.skipUnless(DIALS, 'Requires native DIALS environment')
class NativeDerivativeTests(unittest.TestCase):
    @staticmethod
    def configured_adapter():
        import test_centroid_em as fixtures
        from dxtbx.model import ScanFactory
        from scitbx import matrix
        models = fixtures.DialsAdapterTests.experiments()
        fixed = matrix.col((1., 2., 0.)).axis_and_angle_as_r3_rotation_matrix(13., deg=True)
        setting = matrix.col((0., 1., 2.)).axis_and_angle_as_r3_rotation_matrix(-11., deg=True)
        models[0].goniometer.set_fixed_rotation(fixed.elems)
        models[0].goniometer.set_setting_rotation(setting.elems)
        # A third experiment shares the first crystal but belongs to another
        # full turn, exercising exact parameter sharing and branch coordinates.
        extra = copy.deepcopy(models[0])
        extra.crystal = models[0].crystal
        extra.scan = ScanFactory.make_scan((1, 180), [1.]*180, (360., 1.), list(range(180)), deg=True)
        models.append(extra)
        return DialsCrystalAdapter(models, Budget(250, 120), max_predictions=40000)

    def test_analytic_jacobian_matches_native_central_differences(self):
        adapter = self.configured_adapter()
        x = adapter.x0.copy()
        x[[0, 9]] += [.13, -.21]
        x[[3, 12]] *= [1.0001, .9999]
        rows = np.concatenate([np.flatnonzero(adapter.ids == eid)[::251] for eid in (0, 1, 2)])
        actual = adapter.jacobian(x, rows)
        reference = np.zeros_like(actual)
        for parameter in range(len(x)):
            step = 2e-5 if parameter % 9 < 3 else 2e-5 * max(abs(x[parameter]), 1.)
            plus, minus = x.copy(), x.copy()
            plus[parameter] += step
            minus[parameter] -= step
            reference[:, :, parameter] = (adapter.predict(plus)[rows] - adapter.predict(minus)[rows]) / (2 * step)
        scales = np.maximum(np.max(np.abs(reference), axis=(0, 1)), 1e-10)
        np.testing.assert_allclose(actual / scales, reference / scales, rtol=2e-6, atol=3e-7)
        self.assertEqual(adapter.n_crystals, 2)
        self.assertEqual(actual.shape[2], 18)
        self.assertTrue(np.all(adapter.reference[adapter.ids == 2, 2] >= 2 * np.pi))
        np.testing.assert_array_equal(actual[adapter.ids[rows] == 0, :, 9:], 0.)
        np.testing.assert_array_equal(actual[adapter.ids[rows] == 1, :, :9], 0.)

    def test_prediction_cache_is_defensive_and_tracks_instrument_changes(self):
        import test_centroid_em as fixtures
        adapter = DialsCrystalAdapter(fixtures.DialsAdapterTests.experiments(), Budget(50, 120))
        x = adapter.x0.copy()
        original = adapter.predict(x)
        calls = adapter.budget.calls
        original[:] = -999.
        cached = adapter.predict(x)
        self.assertEqual(adapter.budget.calls, calls)
        np.testing.assert_allclose(cached, adapter.reference, atol=1e-8)
        adapter.jacobian(x, np.arange(20))
        self.assertEqual(adapter.budget.calls, calls)
        changed = x.copy()
        changed[0] += .1
        different = adapter.predict(changed)
        self.assertEqual(adapter.budget.calls, calls + 1)
        self.assertGreater(np.max(np.abs(different - cached)), 1e-5)
        adapter.set_values(changed)
        adapter.predict(changed)
        self.assertEqual(adapter.budget.calls, calls + 2)
        panel = adapter.models[0].detector[0]
        origin = np.array(panel.get_origin())
        origin[0] += .001
        panel.set_frame(panel.get_fast_axis(), panel.get_slow_axis(), tuple(origin))
        shifted = adapter.predict(changed)
        self.assertEqual(adapter.budget.calls, calls + 3)
        self.assertGreater(np.max(np.abs(shifted - different)), 1e-5)

    def test_nontriclinic_parameter_blocks_and_multiple_panels(self):
        import test_centroid_em as fixtures
        from dxtbx.model import Crystal
        models = fixtures.DialsAdapterTests.experiments()
        orientation = models[0].crystal.get_U()
        models[0].crystal = Crystal((15., 0., 0.), (0., 15., 0.), (0., 0., 21.), 'P 4')
        models[0].crystal.set_U(orientation)
        second_panel = copy.deepcopy(models[0].detector[0])
        origin = np.array(second_panel.get_origin()) + 150. * np.array(second_panel.get_fast_axis())
        second_panel.set_frame(second_panel.get_fast_axis(), second_panel.get_slow_axis(), tuple(origin))
        models[0].detector.add_panel(second_panel)
        adapter = DialsCrystalAdapter(models, Budget(50, 120), max_predictions=40000)
        self.assertEqual(len(adapter.x0), 14)
        rows = np.concatenate([np.flatnonzero((adapter.ids == eid) & (adapter.panels == panel))[::173]
                               for eid in (0, 1) for panel in (0, 1)])
        self.assertEqual(set(adapter.panels[rows]), {0, 1})
        x = adapter.x0.copy()
        actual = adapter.jacobian(x, rows)
        for parameter in (0, 3, 4, 5, 8, 13):
            # Orientation parameters are milliradians. A 2e-3 mrad step avoids
            # subtracting detector coordinates differing only at 1e-14 mm for
            # a spindle-parallel change whose exact XY derivatives are zero.
            step = 2e-3 if parameter in (0, 5) else 1e-5 * max(abs(x[parameter]), 1.)
            plus, minus = x.copy(), x.copy()
            plus[parameter] += step
            minus[parameter] -= step
            expected = (adapter.predict(plus)[rows] - adapter.predict(minus)[rows]) / (2 * step)
            scale = max(np.max(np.abs(expected)), 1e-10)
            np.testing.assert_allclose(actual[:, :, parameter] / scale, expected / scale, rtol=3e-6, atol=5e-7)


if __name__ == '__main__':
    unittest.main()
