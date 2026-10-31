"""Physical-unit invariance and collective identifiability of Gaussian modes."""
from copy import deepcopy
import json
import tempfile
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from numpy.testing import assert_allclose

SCRIPTS = Path(__file__).resolve().parents[1] / 'skills' / 'ariadne-reduction' / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from composite_scaling import fit_global
from global_scale import prepare, make_corrections, information_observation, scale


def linear_block(design, truth, known=None, covariance=None):
    design = np.asarray(design, float)
    truth = np.asarray(truth, float)
    k = len(truth)
    known = np.ones(k) if known is None else np.asarray(known, float)
    return dict(y=design @ (known * truth), covariance=np.eye(len(design)) if covariance is None else covariance,
                design=design, known_factors=known, features=np.empty((k, 0)),
                equivalence_ids=[f'domain{j}:1,0,0' for j in range(k)])


def mode_inputs(designs, truth, statuses=None):
    truth = np.asarray(truth, float)
    k = len(truth)
    config = dict(schema='ariadne-global-scaling/1', basis='synthetic',
                  geometry_provenance='synthetic known geometry', lp_provenance='known identity',
                  symmetry_provenance='synthetic labels', integration_provenance='known linear profiles',
                  complete_contributor_audit=True, direction_frame='crystal_cartesian_common_setting',
                  scan_order=0, absorption_degree=0,
                  domains={str(j): {'space_group': 'P 1', 'anomalous': True} for j in range(k)}, groups={})
    records = []
    for j, h in enumerate(designs):
        h = np.asarray(h, float)
        rank = len(h)
        name = f'observation{j}'
        contributors = [dict(domain=d, hkl=[1, 0, 0]) for d in range(k)]
        status = statuses[j] if statuses is not None else 'unresolved'
        record = dict(observation_id=name, status=status, quality_flags=[], basis='synthetic',
                      corrected=False, contributors=contributors,
                      intensity_definition='full-profile amplitude; no further partiality division',
                      information_modes=dict(schema='ariadne-information-modes/1', variance_model='fixed',
                                             rank=rank, y=(h @ truth).tolist(), design=h.tolist(),
                                             covariance=np.eye(rank).tolist(), pixel_chi2_constant=3.,
                                             valid_pixels=rank+4, nuisance_rank=1,
                                             likelihood_scope='conditional fixed-profile Gaussian'))
        config['groups'][name] = [dict(c, scan_id='scan', phi_rad=j*.1, lp_forward=1.) for c in contributors]
        if status == 'joint_components':
            covariance = np.linalg.inv(h.T @ h)
            record['components'] = [dict(intensity=float(x)) for x in truth]
            record['covariance_identifiable_subspace'] = covariance.tolist()
        elif status == 'total_only':
            inverse = np.linalg.pinv(h)
            covariance = inverse @ inverse.T
            record['total'] = dict(intensity=float(truth.sum()), variance=float(np.ones(k) @ covariance @ np.ones(k)))
            record['covariance_identifiable_subspace'] = covariance.tolist()
        records.append(record)
    return records, config


class ScalingPrincipleTests(unittest.TestCase):
    def test_extreme_known_factor_units_preserve_signed_observation(self):
        # Exactly solvable independent measurements.  Raw SVD discards the
        # 1e-12 column relative to 1e12 although both measurements have unit SNR.
        truth = np.array([4e12, -3e-12])
        known = np.array([1e-12, 1e12])
        result = fit_global([linear_block(np.eye(2), truth, known)])
        self.assertEqual(result['status'], 'converged', result['reason'])
        self.assertEqual(result['rank'], 2)
        assert_allclose(result['I'], truth, rtol=2e-15, atol=0.)
        assert_allclose(result['joint_covariance'], np.diag(1/known**2), rtol=2e-15, atol=0.)
        assert_allclose(result['parameter_response'], np.diag(1/known), rtol=2e-15, atol=0.)
        self.assertTrue(np.all(result['identifiable_parameters']))
        self.assertLess(result['I'][1], 0.)

    def test_measurement_covariance_reference_under_unit_changes(self):
        rng = np.random.default_rng(19)
        design = rng.normal(size=(14, 3))
        covariance = .2*np.ones((14, 14)) + np.diag(rng.uniform(.7, 1.5, 14))
        known = np.array([1e-11, 2., 1e11])
        truth = np.array([2e11, -.4, 3e-11])
        result = fit_global([linear_block(design, truth, known, covariance)])
        self.assertEqual(result['status'], 'converged')
        # Independent small, well-scaled reference in amplitude units.
        sigma_inv_design = np.linalg.solve(covariance, design)
        amplitude_cov = np.linalg.inv(design.T @ sigma_inv_design)
        expected = amplitude_cov / known[:, None] / known[None, :]
        response = (amplitude_cov @ sigma_inv_design.T) / known[:, None]
        assert_allclose(result['I'], truth, rtol=1e-13, atol=0.)
        assert_allclose(result['joint_covariance'], expected, rtol=1e-12, atol=0.)
        assert_allclose(result['parameter_response'], response, rtol=1e-11, atol=0.)
        assert_allclose(result['parameter_response'] @ covariance,
                        result['block_parameter_data_covariance'][0], rtol=1e-11, atol=0.)

    def test_global_feature_centering_does_not_remove_domain_gauge(self):
        blocks = []
        for _ in range(5):
            b = linear_block(np.eye(2), [1000., .7])
            b['features'] = np.array([[1.], [-1.]])
            blocks.append(b)
        result = fit_global(blocks)
        self.assertTrue(result['gauge_diagnostics']['centered'])
        self.assertEqual(result['gauge_diagnostics']['centered_feature_rank'], 0)
        self.assertEqual(result['rank'], 2)
        self.assertEqual(result['status'], 'unresolved')
        self.assertEqual(result['reason'], 'rank_deficient')
        self.assertFalse(np.any(result['identifiable_parameters']))
        self.assertGreater(result['I'][0]/result['I'][1], 1000.)

    def test_unresolved_modes_become_identifiable_only_collectively(self):
        designs = [np.array([[1., 1., 0.], [0., 1., 2.]]),
                   np.array([[1., 0., 1.], [2., 1., 0.]])]
        truth = np.array([1000., 2., -.1])
        records, config = mode_inputs(designs, truth)
        with patch('global_scale.equivalent_hkl', side_effect=lambda h, *_: h):
            blocks, _, _, _, excluded = prepare(records, config)
        self.assertFalse(excluded)
        self.assertTrue(all(b['observation_representation'] == 'fixed_gaussian_information_modes' for b in blocks))
        for h in designs:
            self.assertEqual(np.linalg.matrix_rank(h), 2)
            # Neither local observation determines the physical total.
            self.assertGreater(np.linalg.norm(h.T @ np.linalg.lstsq(h.T, np.ones(3), rcond=None)[0] - 1), .1)
        fit = fit_global(blocks)
        self.assertEqual(fit['status'], 'converged', fit['reason'])
        assert_allclose(fit['I'], truth, atol=1e-11)
        full_design = np.vstack(designs)
        assert_allclose(fit['joint_covariance'], np.linalg.inv(full_design.T @ full_design), atol=1e-14)
        args = SimpleNamespace(integrated=Path('/tmp/integrated.jsonl'), output=Path('/tmp/scaled'), geometry=Path('/tmp/geometry.json'))
        corrections = make_corrections(blocks, config, fit, args)
        self.assertEqual(corrections['groups'], {})
        self.assertEqual(corrections['fit_only_observations'], ['observation0', 'observation1'])

    def test_pixel_chi_square_constant_and_lost_residual_dimensions_restored(self):
        designs = [np.array([[1., 1., 0.], [0., 1., 2.]]),
                   np.array([[1., 0., 1.], [2., 1., 0.]])]
        records, config = mode_inputs(designs, [10., 2., -.1])
        with tempfile.TemporaryDirectory(prefix='ariadne_modes_') as directory:
            root = Path(directory)
            integrated, geometry = root/'groups.jsonl', root/'geometry.json'
            integrated.write_text(''.join(json.dumps(r)+'\n' for r in records))
            geometry.write_text(json.dumps(config))
            args = SimpleNamespace(integrated=integrated, geometry=geometry, output=root/'result',
                                   max_iterations=100, tolerance=1e-8, rcond=1e-10,
                                   max_parameters=256, max_seconds=30.)
            with patch('global_scale.equivalent_hkl', side_effect=lambda h, *_: h):
                scale(args)
            report = json.loads((args.output/'scaling.json').read_text())
        self.assertEqual(report['status'], 'converged')
        self.assertEqual(report['dof'], 1)
        # 2 groups * (6 pixel rows - 1 nuisance) - 3 shared amplitudes = 7.
        self.assertEqual(report['pixel_profiled_dof'], 7)
        self.assertAlmostEqual(report['pixel_profiled_chi2'], 6., places=12)
        self.assertAlmostEqual(report['pixel_profiled_reduced_chi2'], 6/7, places=12)
        self.assertLess(report['chi2'], 1e-20)

    def test_modes_preserve_measured_component_and_total_corrections(self):
        designs = [np.array([[2., 1.], [.3, 1.4]]), np.array([[1., 1.]])]
        records, config = mode_inputs(designs, [8., -1.], ['joint_components', 'total_only'])
        with patch('global_scale.equivalent_hkl', side_effect=lambda h, *_: h):
            blocks, _, _, _, _ = prepare(records, config)
        fit = fit_global(blocks)
        args = SimpleNamespace(integrated=Path('/tmp/integrated.jsonl'), output=Path('/tmp/scaled'), geometry=Path('/tmp/geometry.json'))
        corrections = make_corrections(blocks, config, fit, args)
        self.assertEqual(set(corrections['groups']), {'observation0', 'observation1'})
        self.assertEqual(corrections['fit_only_observations'], [])
        assert_allclose(corrections['groups']['observation0']['intensity_factor_cross_covariance'], np.zeros((2, 2)))
        self.assertEqual(corrections['groups']['observation1']['total_factor_cross_covariance'], 0.)

    def test_mode_to_legacy_same_data_covariance_with_fitted_scale(self):
        designs = [np.array([[2., .2], [.3, 1.4]]) if j % 3 else np.array([[1., 1.]]) for j in range(9)]
        statuses = ['joint_components' if len(h) == 2 else 'total_only' for h in designs]
        records, config = mode_inputs(designs, [8., -1.], statuses)
        config['scan_order'] = 1
        # Same scan factor for both domains; the weak domain remains signed.
        for j, record in enumerate(records):
            forward = np.exp(.19*(2*j/8-1))
            modes = record['information_modes']
            modes['y'] = (np.asarray(modes['y'])*forward).tolist()
            if record['status'] == 'joint_components':
                for component in record['components']:
                    component['intensity'] *= forward
            else:
                record['total']['intensity'] *= forward
        legacy = deepcopy(records)
        for record in legacy:
            del record['information_modes']
        with patch('global_scale.equivalent_hkl', side_effect=lambda h, *_: h):
            mode_blocks, *_ = prepare(records, config)
            legacy_blocks, *_ = prepare(legacy, config)
        mode_fit, legacy_fit = fit_global(mode_blocks), fit_global(legacy_blocks)
        self.assertEqual(mode_fit['status'], 'converged')
        self.assertEqual(legacy_fit['status'], 'converged')
        assert_allclose(mode_fit['I'], legacy_fit['I'], atol=1e-10)
        assert_allclose(mode_fit['beta'], [.19], atol=1e-12)
        assert_allclose(mode_fit['joint_covariance'], legacy_fit['joint_covariance'], atol=1e-11)
        args = SimpleNamespace(integrated=Path('/tmp/integrated.jsonl'), output=Path('/tmp/scaled'), geometry=Path('/tmp/geometry.json'))
        modes = make_corrections(mode_blocks, config, mode_fit, args)['groups']
        old = make_corrections(legacy_blocks, config, legacy_fit, args)['groups']
        self.assertEqual(set(modes), set(old))
        for name, item in modes.items():
            key = 'intensity_factor_cross_covariance' if 'intensity_factor_cross_covariance' in item else 'total_factor_cross_covariance'
            assert_allclose(item[key], old[name][key], atol=1e-11)
            assert_allclose(item['factor_covariance'], old[name]['factor_covariance'], atol=1e-11)

    def test_nonlinear_feature_units_do_not_change_information_rank(self):
        from test_composite_scaling import composite_truth
        blocks, beta, intensities = composite_truth()
        scales = np.array([1e-12, 1e12])
        for block in blocks:
            block['features'] *= scales
        result = fit_global(blocks)
        self.assertEqual(result['status'], 'converged', result['reason'])
        assert_allclose(result['I'], intensities, atol=1e-10)
        assert_allclose(result['beta']*scales, beta, atol=1e-11)
        self.assertEqual(result['rank'], 5)
        self.assertEqual(result['gauge_diagnostics']['feature_rank'], 2)
        self.assertEqual(result['gauge_diagnostics']['centered_feature_rank'], 2)

    def test_noisy_fit_reaches_strict_stationarity_without_positive_clipping(self):
        from test_composite_scaling import composite_truth
        blocks, _, _ = composite_truth()
        rng = np.random.default_rng(0)
        for block in blocks:
            block['y'] += rng.multivariate_normal(np.zeros(2), block['covariance']) * .3
        fit = fit_global(blocks, max_iterations=12, tolerance=1e-12)
        self.assertEqual(fit['status'], 'converged', fit['reason'])
        self.assertLess(fit['I'][2], 0.)
        self.assertLessEqual(fit['gradient_norm'], 1e-12)
        self.assertTrue(np.all(np.diff([t['chi2'] for t in fit['trace']]) <= 0.))
        theta = np.r_[fit['I'], fit['beta']]
        def residual(value):
            return np.concatenate([np.linalg.solve(np.linalg.cholesky(b['covariance']),
                       b['design'] @ (b['known_factors'] * np.exp(b['features'] @ value[3:]) * value[:3]) - b['y'])
                       for b in blocks])
        # Independent complex-step Jacobian of the original unwhitened block
        # equations checks stationarity against actual observations.
        jacobian = np.column_stack([residual(theta.astype(complex) + 1e-24j*np.eye(5)[j]).imag/1e-24
                                    for j in range(5)])
        rw = residual(theta)
        norm = np.max(np.abs(jacobian.T @ rw)/np.linalg.norm(jacobian, axis=0))/max(1., np.linalg.norm(rw))
        self.assertLess(norm, 1e-12)
        assert_allclose(np.concatenate(fit['whitened_residual']), rw, atol=1e-12, rtol=0.)
        self.assertAlmostEqual(fit['chi2'], rw @ rw, places=12)
        # Independent objective differences also check the residual-Hessian
        # correction used to reject full-rank stationary saddles.
        scales = fit['parameter_scales']
        step = 2e-3
        directions = np.eye(5) * (step/scales)
        def objective(value):
            residuals = residual(value)
            return .5*(residuals @ residuals)
        centre = objective(theta)
        hessian = np.empty((5, 5))
        for j in range(5):
            hessian[j, j] = (objective(theta+directions[j])-2*centre+objective(theta-directions[j]))/step**2
            for k in range(j):
                hessian[j, k] = hessian[k, j] = (
                    objective(theta+directions[j]+directions[k])-objective(theta+directions[j]-directions[k])
                    -objective(theta-directions[j]+directions[k])+objective(theta-directions[j]-directions[k]))/(4*step**2)
        self.assertAlmostEqual(fit['stationary_curvature']['min_eigenvalue'], np.linalg.eigvalsh(hessian)[0], delta=2e-6)

    def test_roundoff_zero_signal_cannot_create_scale_information(self):
        x = np.array([-1., 0., 1.])
        for scale_factor in (1e-12, 1., 1e12):
            with self.subTest(scale=scale_factor):
                values = scale_factor*np.array([-1., 2., -1.])
                blocks = []
                for feature, value in zip(x, values):
                    b = linear_block([[1.]], [value])
                    b['features'] = np.array([[feature]])
                    blocks.append(b)
                fit = fit_global(blocks)
                # beta=0 gives exactly I=0 and no first-order scale information.
                # Tiny WLS cancellation error must not be normalized into rank.
                self.assertEqual(fit['status'], 'unresolved')
                self.assertFalse(fit['converged'])
                self.assertEqual(fit['rank'], 1)
                self.assertFalse(fit['identifiable_parameters'][-1])
                self.assertLessEqual(abs(fit['I'][0]), fit['arithmetic_resolution']['amplitude_resolution_estimate'][0])
                # Independently exhibit a lower objective: stationarity is not
                # sufficient to declare a solution on this singular model.
                gains = np.exp(x)
                alternative_i = np.dot(gains, values)/np.dot(gains, gains)
                alternative_chi2 = np.sum((gains*alternative_i-values)**2)
                self.assertLess(alternative_chi2/scale_factor**2, fit['chi2']/scale_factor**2-.1)
                if 'information_jacobians' in fit:
                    raw = np.vstack(fit['information_jacobians'])
                    assert_allclose(np.hstack(fit['block_parameter_data_covariance']),
                                    fit['joint_covariance'] @ raw.T, atol=1e-14)

    def test_full_rank_stationary_saddle_is_not_accepted_as_minimum(self):
        for magnitude in (1e-12, 1., 1e12):
            with self.subTest(magnitude=magnitude):
                blocks = []
                for x, y in zip((-1., 0., 1.), (1., -1., 1.)):
                    b = linear_block([[1.]], [magnitude*y])
                    b['features'] = np.array([[x]])
                    blocks.append(b)
                fit = fit_global(blocks)
                # At I=magnitude/3,beta=0 the true beta curvature is
                # -2*magnitude**2/9. After normalization by ||J_beta||**2
                # it is exactly -1; full rank and zero gradient are insufficient.
                self.assertEqual(fit['rank'], 2)
                self.assertEqual(fit['status'], 'unresolved')
                self.assertEqual(fit['reason'], 'negative_curvature_stationary_point')
                self.assertAlmostEqual(fit['stationary_curvature']['min_eigenvalue'], -1., places=12)
                gains = np.exp(np.array([-1., 0., 1.]))
                y = magnitude*np.array([1., -1., 1.])
                alternative_i = gains @ y / (gains @ gains)
                self.assertLess(np.sum((gains*alternative_i-y)**2)/magnitude**2,
                                fit['chi2']/magnitude**2)

    def test_independent_tiny_channel_is_not_screened_by_bright_data(self):
        for known_tiny in (1e-100, 1., 1e100):
            with self.subTest(known_tiny=known_tiny):
                truth = np.array([1e10, 1e-30/known_tiny])
                blocks = []
                for x in (-1., 0., 1.):
                    b = linear_block(np.eye(2), truth, known=[1., known_tiny])
                    b['features'] = np.array([[0.], [x]])
                    blocks.append(b)
                fit = fit_global(blocks)
                self.assertEqual(fit['status'], 'converged', fit['reason'])
                self.assertEqual(fit['rank'], 3)
                assert_allclose(fit['I'], truth, rtol=1e-14, atol=0.)
                self.assertTrue(np.all(fit['arithmetic_resolution']['numerically_supported_amplitudes']))
                self.assertEqual(len(fit['arithmetic_resolution']['support_components']), 2)
        blocks = []
        for x in (-1., 0., 1.):
            b = linear_block([[1.]], [1e-100])
            b['features'] = np.array([[x]])
            blocks.append(b)
        fit = fit_global(blocks)
        self.assertEqual(fit['status'], 'converged', fit['reason'])
        self.assertEqual(fit['rank'], 2)
        assert_allclose(fit['I'], [1e-100], rtol=1e-14, atol=0.)

    def test_plugin_irls_modes_do_not_enter_exact_observation_path(self):
        records, config = mode_inputs([np.eye(2)], [2., 1.], ['joint_components'])
        records[0]['information_modes']['variance_model'] = 'irls_plugin'
        with patch('global_scale.equivalent_hkl', side_effect=lambda h, *_: h):
            blocks, _, _, _, _ = prepare(records, config)
        self.assertEqual(blocks[0]['observation_representation'], 'components')
        bad = deepcopy(records[0])
        bad['information_modes']['variance_model'] = 'fixed'
        bad['information_modes']['covariance'] = [[1., 2.], [2., 1.]]
        with self.assertRaisesRegex(ValueError, 'positive definite'):
            information_observation(bad)
        bad = deepcopy(records[0])
        bad['information_modes']['variance_model'] = 'fixed'
        bad['estimator'] = 'poisson'
        with self.assertRaisesRegex(ValueError, 'signed-wls'):
            information_observation(bad)


if __name__ == '__main__':
    unittest.main()
