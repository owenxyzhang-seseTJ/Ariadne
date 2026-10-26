"""Numerical likelihood/EM tests plus a real native DIALS synthetic adapter test.

Run with ariadne-env python -m unittest discover -s tests -p test_centroid_em.py.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.stats import multivariate_normal

SCRIPTS = Path(__file__).resolve().parents[1] / 'skills/ariadne-reduction/scripts'
sys.path.insert(0, str(SCRIPTS))
from centroid_em import (Budget, BudgetExceeded, DialsCrystalAdapter, block_comparison,
                         covariance_factors, expectation, fit_em, run, scan_partition, weighted_targets)


class NumericTests(unittest.TestCase):
    def test_full_covariance_normalized_density(self):
        y = np.array([[.4, -.3, .1], [1., 1., 1.]])
        means = np.array([[0., 0., 0.], [.7, .4, -.1]])
        cov = np.array([[.5, .13, .02], [.13, .3, -.01], [.02, -.01, .04]])
        whitening, norm = covariance_factors(np.array([cov, cov]))
        allowed = np.array([[True, True], [False, True]])
        resp, outlier, ll = expectation(y, means, whitening, norm, allowed,
                                       [2, 2], [1e-3, 1e-3], .1)
        components = np.array([[multivariate_normal.pdf(row, mean=m, cov=cov)
                                for m in means] for row in y]) * allowed * .9 / 2
        density = components.sum(axis=1) + .1e-3
        np.testing.assert_allclose(ll, np.log(density), rtol=1e-12)
        np.testing.assert_allclose(resp, components / density[:, None], rtol=1e-12)
        np.testing.assert_allclose(resp.sum(axis=1) + outlier, 1., atol=1e-14)
        self.assertEqual(resp[1, 0], 0.)

    def test_extra_lattice_dilutes_density_and_unmatched_row_kept(self):
        y = np.array([[0., 0., 0.], [100., 100., 100.]])
        white, norm = covariance_factors(np.tile(np.eye(3), (2, 1, 1)))
        _, _, base = expectation(y, np.zeros((1, 3)), white, norm,
                                 np.ones((2, 1), bool), [1, 1], [1e-6]*2, .1)
        _, _, extra = expectation(y, np.array([[0., 0., 0.], [1000., 0., 0.]]),
                                  white, norm, np.ones((2, 2), bool), [2, 2], [1e-6]*2, .1)
        self.assertLess(extra[0], base[0])
        self.assertAlmostEqual(base[1], np.log(.1e-6))
        self.assertAlmostEqual(extra[1], base[1])

    def test_scan_partition_independent_of_order_and_no_block_leak(self):
        ids = [0]*200
        frames = np.arange(200) + .5
        source = np.arange(200) + 1000
        p, test, keys = scan_partition(ids, frames, source, 10, .2, 7)
        self.assertFalse(set(p['training_source_rows']) & set(p['validation_source_rows']))
        for key in set(keys):
            self.assertEqual(len({bool(test[i]) for i, k in enumerate(keys) if k == key}), 1)
        p2, _, _ = scan_partition(ids, frames[::-1], source[::-1], 10, .2, 7)
        self.assertEqual(set(p['validation_source_rows']), set(p2['validation_source_rows']))
        with self.assertRaises(ValueError):
            scan_partition([0, 0], [1, 2], [4, 4])

    def test_block_uncertainty_and_insufficient_data(self):
        comparison = block_comparison([0]*6, [1, 1, 2, 2, 3, 3],
                                      [(0, i//2) for i in range(6)])
        self.assertAlmostEqual(comparison['mean_log_likelihood_gain'], 2.)
        self.assertAlmostEqual(comparison['block_jackknife_se'], np.sqrt(1/3))
        self.assertEqual(comparison['decision'], 'unresolved')
        small = block_comparison([0, 0], [2, 2], [(0, 1), (0, 2)])
        self.assertIsNone(small['mean_gain_interval_95'])
        self.assertEqual(small['decision'], 'unresolved')

    def test_m_step_sufficient_statistics_preserve_full_quadratic(self):
        rng = np.random.default_rng(8)
        y = rng.normal(size=(9, 3))
        matrices = rng.normal(size=(9, 3, 3))
        covariance = matrices @ matrices.transpose(0, 2, 1) + np.eye(3)
        whitening, _ = covariance_factors(covariance)
        r = rng.uniform(size=(9, 4))
        r[:, 3] = 0.
        r /= 1 + r.sum(axis=1, keepdims=True)
        active, factors, targets = weighted_targets(y, whitening, r)
        self.assertEqual(active.tolist(), [0, 1, 2])
        costs, condensed = [], []
        for _ in range(2):
            means = rng.normal(size=(4, 3))
            white_delta = np.einsum('iab,ijb->ija', whitening, y[:, None] - means[None])
            costs.append(float(np.sum(r[:, :, None] * white_delta**2)))
            residual = np.einsum('jab,jb->ja', factors, means[active] - targets)
            condensed.append(float(np.sum(residual**2)))
        self.assertAlmostEqual(costs[1] - costs[0], condensed[1] - condensed[0], places=12)

    def test_budget_and_invalid_covariance(self):
        budget = Budget(1, 10)
        budget.check(True)
        with self.assertRaises(BudgetExceeded):
            budget.check(True)
        with self.assertRaises(ValueError):
            covariance_factors(np.array([np.diag([1, 1, -1])]))

    def test_em_full_covariance_optimum_and_monotonicity(self):
        class ShiftAdapter:
            x0 = np.array([0.])
            bounds = (np.array([-10.]), np.array([10.]))
            component_crystals = np.array([0])
            n_crystals = 1

            def predict(self, x):
                return np.array([[x[0], 0., 0.]])

            def set_values(self, x):
                self.values = x.copy()

        y = np.array([[1., .5, 0.], [1.4, .3, 0.], [.8, .2, 0.]])
        cov = np.array([[1., .7, 0.], [.7, 1., 0.], [0., 0., 1.]])
        result = fit_em(ShiftAdapter(), y, np.tile(cov, (3, 1, 1)),
                        np.ones((3, 1), bool), np.ones(3), np.full(3, 1e-20),
                        min_support=1, em_iterations=10, max_nfev=20)
        expected = np.mean(y[:, 0] - .7 * y[:, 1])
        self.assertAlmostEqual(result['values'][0], expected, places=6)
        self.assertTrue(np.all(np.diff(result['training_log_likelihood_history']) >= 0))
        self.assertEqual(result['status'], 'converged')


    def test_max_nfev_one_cannot_declare_em_convergence(self):
        class ShiftAdapter:
            x0 = np.array([0.])
            bounds = (np.array([-10.]), np.array([10.]))
            component_crystals = np.array([0])
            n_crystals = 1

            def predict(self, x):
                return np.array([[x[0], 0., 0.]])

            def set_values(self, x):
                self.values = x.copy()

        y = np.array([[1., 0., 0.], [1.2, 0., 0.], [.8, 0., 0.]])
        result = fit_em(ShiftAdapter(), y, np.tile(np.eye(3), (3, 1, 1)),
                        np.ones((3, 1), bool), np.ones(3), np.full(3, 1e-20),
                        min_support=1, em_iterations=3, max_nfev=1)
        self.assertEqual(result['status'], 'unresolved')
        self.assertIn('optimizer limit', result['reason'])
        self.assertFalse(result['optimizer_history'][-1]['success'])
        self.assertGreater(result['optimizer_history'][-1]['optimality'], 1e-6)
        self.assertTrue(result['identifiable'])
        self.assertEqual(result['training_log_likelihood_history'][0],
                         result['training_log_likelihood_history'][-1])


DIALS = importlib.util.find_spec('dials') is not None


@unittest.skipUnless(DIALS, 'Requires native DIALS environment')
class DialsAdapterTests(unittest.TestCase):
    @staticmethod
    def experiments():
        from dxtbx.model import BeamFactory, Crystal, DetectorFactory, GoniometerFactory, ScanFactory
        from dxtbx.model.experiment_list import Experiment, ExperimentList
        from dxtbx.imageset import ImageSetFactory
        from scitbx import matrix
        beam = BeamFactory.simple(1.)
        detector = DetectorFactory.simple('PAD', 150., (75., 75.), '+x', '-y',
                                           (.15, .15), (1000, 1000), (0, 1e6))
        goniometer = GoniometerFactory.single_axis()
        scan = ScanFactory.make_scan((1, 180), [1.]*180, (0., 1.), list(range(180)), deg=True)
        imageset = ImageSetFactory.make_sequence('/tmp/ariadne_synthetic_####.cbf',
                                                  list(range(1, 181)), check_format=False,
                                                  beam=beam, detector=detector,
                                                  goniometer=goniometer, scan=scan)
        imageset.set_beam(beam)
        imageset.set_detector(detector)
        imageset.set_goniometer(goniometer)
        imageset.set_scan(scan)
        crystal = Crystal((15., 0., 0.), (0., 18., 0.), (0., 0., 21.), 'P 1')
        crystal.set_U(matrix.col((1., 2., 3.)).axis_and_angle_as_r3_rotation_matrix(17., deg=True))
        second = copy.deepcopy(crystal)
        second.rotate_around_origin((0., 1., 0.), 11., deg=True)
        return ExperimentList([Experiment(beam=beam, detector=detector, goniometer=goniometer,
                                          scan=scan, imageset=imageset, crystal=c)
                               for c in (crystal, second)])

    def test_native_prediction_whitening_joint_refinement(self):
        models = self.experiments()
        budget = Budget(450, 120)
        adapter = DialsCrystalAdapter(models, budget, max_predictions=20000)
        truth = adapter.predict(adapter.x0)
        np.testing.assert_allclose(truth, adapter.reference, atol=1e-8)
        selected = []
        for eid in (0, 1):
            rows = np.flatnonzero(adapter.ids == eid)
            selected.extend(rows[np.linspace(0, len(rows)-1, 35, dtype=int)])
        selected = np.asarray(selected)
        y = truth[selected]
        # Physically perturb all crystal orientations and cell metrics, then fit
        # the same full candidate list to genuinely predicted single-source rows.
        adapter.x0 = adapter.x0.copy()
        adapter.x0[[0, 9]] += .4
        adapter.x0[[3, 12]] *= 1.0003
        initial = adapter.predict(adapter.x0)
        cov = np.array([[.02, .004, 1e-5], [.004, .018, -1e-5], [1e-5, -1e-5, 1e-6]])
        covariance = np.tile(cov, (len(y), 1, 1))
        instrument_before = [(e.beam.to_dict(), e.detector.to_dict(), e.goniometer.to_dict(),
                              e.scan.to_dict()) for e in adapter.models]
        result = fit_em(adapter, y, covariance, np.ones((len(y), len(truth)), bool),
                        np.full(len(y), len(truth)), np.full(len(y), 1e-9),
                        min_support=20, em_iterations=5, max_nfev=12)
        final = adapter.predict(result['values'])
        initial_rms = np.sqrt(np.mean((initial[selected] - y)**2, axis=0))
        final_rms = np.sqrt(np.mean((final[selected] - y)**2, axis=0))
        self.assertTrue(np.all(final_rms < initial_rms * .01), (initial_rms, final_rms, result['reason']))
        self.assertTrue(np.all(np.diff(result['training_log_likelihood_history']) >= 0))
        self.assertTrue(result['supported'])
        self.assertTrue(result['identifiable'])
        instrument_after = [(e.beam.to_dict(), e.detector.to_dict(), e.goniometer.to_dict(),
                             e.scan.to_dict()) for e in adapter.models]
        self.assertEqual(instrument_before, instrument_after)
        print('DIALS_SYNTHETIC', json.dumps({'predictions': len(truth), 'parameters': len(adapter.x0),
              'initial_rms_mm_mm_rad': initial_rms.tolist(), 'final_rms_mm_mm_rad': final_rms.tolist(),
              'support': result['effective_crystal_support'], 'rank': result['jacobian_rank'],
              'likelihood_history': result['training_log_likelihood_history'],
              'status': result['status'], 'prediction_calls': budget.calls}))

    def test_cli_output_composite_exclusion_and_holdout_comparison(self):
        from dials.array_family import flex
        from dxtbx.model.experiment_list import ExperimentList
        models = self.experiments()
        observed = ExperimentList([models[0]])
        adapter = DialsCrystalAdapter(models, Budget(10, 120))
        rows = np.concatenate([np.flatnonzero(adapter.ids == eid)[::80] for eid in (0, 1)])
        rows = rows[:200]
        truth = adapter.reference[rows]
        table = flex.reflection_table()
        table['id'] = flex.int(len(rows), -1)
        table['panel'] = flex.size_t(adapter.panels[rows].tolist())
        table['xyzobs.mm.value'] = flex.vec3_double([tuple(p) for p in truth])
        table['xyzobs.mm.variance'] = flex.vec3_double(len(rows), (.02, .02, 1e-6))
        table['xyzobs.px.value'] = adapter.table['xyzcal.px'].select(flex.size_t(rows.tolist()))
        source = np.arange(len(rows)) + 100
        table['scxrd_source_row'] = flex.size_t(source.tolist())
        _, test, _ = scan_partition(np.zeros(len(rows)), np.asarray(table['xyzobs.px.value'])[:, 2], source, 10, .3, 42)
        training_rows = source[~test].tolist()
        with tempfile.TemporaryDirectory(prefix='ariadne_em_test_') as directory:
            root = Path(directory)
            models.as_file(str(root / 'models.expt'))
            observed.as_file(str(root / 'observed.expt'))
            # A second predeclared hypothesis with one unsupported additional
            # lattice exercises probability normalization in a real CLI run.
            alternative = copy.deepcopy(models)
            extra = copy.deepcopy(alternative[0])
            extra.crystal = copy.deepcopy(extra.crystal)
            extra.crystal.rotate_around_origin((0., 0., 1.), 7., deg=True)
            alternative.append(extra)
            alternative.as_file(str(root / 'alternative.expt'))
            table.as_file(str(root / 'spots.refl'))
            evidence = {'schema': 'ariadne-soft-compatibility/1', 'models_expt': str(root / 'models.expt'),
                        'source_spots': str(root / 'spots.refl'), 'observed_expt': str(root / 'observed.expt'),
                        'calibration': {'model_covariance_mm_rad': np.zeros((3, 3)).tolist()}}
            (root / 'compatibility.json').write_text(json.dumps(evidence))
            selection = {'single_source_source_rows': source[1:].tolist(),
                         'composite_source_rows': [int(source[0])], 'provenance': 'synthetic known single-source truth',
                         'candidate_generation_source_rows': training_rows, 'calibration_source_rows': training_rows}
            (root / 'selection.json').write_text(json.dumps(selection))
            args = SimpleNamespace(compatibility=root / 'compatibility.json', models_expt=root / 'models.expt',
                    spots=root / 'spots.refl', output=root / 'output', centroid_selection=root / 'selection.json',
                    block_frames=10, holdout_fraction=.3, seed=42, min_anchors=5, max_hypotheses=3,
                    compare_models=[root / 'alternative.expt'], max_prediction_calls=180, max_seconds=120,
                    max_predictions=20000, max_pairs=1500000, orientation_bound_mrad=1.,
                    cell_relative_bound=.001, outlier_fraction=.05, em_iterations=2, max_iterations=2,
                    em_tolerance=1e-6)
            run(args)
            report = json.loads((args.output / 'joint_refinement.json').read_text())
            self.assertEqual(report['composite_source_rows_excluded'], [int(source[0])])
            self.assertNotIn(int(source[0]), report['training_source_rows'] + report['validation_source_rows'])
            self.assertEqual(len(report['evaluations']), 2)
            self.assertTrue(report['upstream_training_provenance_declared'])
            self.assertTrue((args.output / 'refined.expt').exists())
            self.assertTrue((args.output / 'hypothesis_1_refined.expt').exists())
            training = flex.reflection_table.from_file(str(args.output / 'training.refl'))
            self.assertEqual(len(training), len(report['training_source_rows']))
            self.assertEqual(len(set(report['training_source_rows'])), len(training))
            self.assertLess(report['evaluations'][1]['against_first_hypothesis']['log_likelihood_gain'], 0.)
            print('DIALS_CLI', json.dumps({'rows': len(rows), 'training': len(training),
                  'heldout': len(report['validation_source_rows']),
                  'extra_lattice_comparison': report['evaluations'][1]['against_first_hypothesis'],
                  'prediction_calls': report['budgets']['prediction_calls']}))
            # Changing final-only holdout centroids cannot alter fitted geometry.
            from dxtbx.model.experiment_list import ExperimentListFactory
            original_A = [e.crystal.get_A() for e in
                          ExperimentListFactory.from_json_file(str(args.output / 'refined.expt'), check_format=False)]
            shifted = np.asarray(table['xyzobs.mm.value']).copy()
            shifted[test, 0] += .1
            table['xyzobs.mm.value'] = flex.vec3_double([tuple(p) for p in shifted])
            table.as_file(str(root / 'spots.refl'))
            args.output = root / 'changed_holdout'
            run(args)
            from dxtbx.model.experiment_list import ExperimentListFactory
            changed = ExperimentListFactory.from_json_file(str(args.output / 'refined.expt'), check_format=False)
            for before, after in zip(original_A, changed):
                np.testing.assert_array_equal(before, after.crystal.get_A())
            # A real native optimizer exhausted at one evaluation must remain
            # unresolved in the top-level report and every comparison decision.
            shifted[:, 0] += .1
            table['xyzobs.mm.value'] = flex.vec3_double([tuple(p) for p in shifted])
            table.as_file(str(root / 'spots.refl'))
            args.max_iterations = 1
            args.output = root / 'optimizer_limit'
            run(args)
            limited = json.loads((args.output / 'joint_refinement.json').read_text())
            self.assertEqual(limited['status'], 'unresolved')
            self.assertFalse(limited['evaluations'][0]['optimizer_history'][-1]['success'])
            for evaluation in limited['evaluations']:
                self.assertEqual(evaluation['status'], 'unresolved')
                self.assertEqual(evaluation['refinement_gain']['decision'], 'unresolved')
                self.assertEqual(evaluation['against_first_hypothesis']['decision'], 'unresolved')


if __name__ == '__main__':
    unittest.main()
