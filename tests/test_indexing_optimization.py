"""固定旧版、独立晶格求和与实际 DIALS 排序器的数值回归。"""
import itertools
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
import zipfile

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python/src'))
from ariadne import ranking, spots
try:
    from dials.algorithms.indexing import model_evaluation
    from dials.array_family import flex
    from dxtbx.model import Crystal, Scan
    from dxtbx.model.experiment_list import Experiment, ExperimentList
except ImportError as exc:
    raise unittest.SkipTest('Requires native DIALS/dxtbx runtime') from exc
from unittest.mock import patch

# 开发目录使用明确的 overlay；已安装插件直接使用安装后模块。
DIALS_SOURCE = ROOT.parent / 'vendor/dials-3.30.1/src/dials/algorithms/indexing'
if DIALS_SOURCE.exists():
    source = DIALS_SOURCE / 'model_evaluation.py'
    exec(compile(source.read_text(), str(source), 'exec'), model_evaluation.__dict__)


def baseline(name, path):
    module = ModuleType(name)
    with zipfile.ZipFile(ROOT / 'benchmarks/baseline-2b9bd04.zip') as archive:
        exec(compile(archive.read(path), path, 'exec'), module.__dict__)
    return module


old_ranking = baseline('old_ranking', 'python/src/ariadne/ranking.py')
old_spots = baseline('old_spots', 'python/src/ariadne/spots.py')


class IndexingOptimizationTests(unittest.TestCase):
    def test_cached_holdout_preserves_candidate_training(self):
        n = 2400
        reflections = flex.reflection_table()
        reflections['imageset_id'] = flex.int(np.repeat([0, 1], n // 2))
        reflections['xyzobs.px.value'] = flex.vec3_double(np.column_stack((np.zeros(n), np.zeros(n), np.arange(n) % 600)))
        reflections['row'] = flex.int(range(n))
        params = SimpleNamespace(seed=42, holdout_block_frames=20, holdout_fraction=.2,
                                 min_training=25, min_validation=20)
        selection = ranking.holdout_selection(reflections, params)
        for candidate in range(4):
            reflections['id'] = flex.int(np.where((np.arange(n) + candidate) % 5 == 0, -1, 0))
            expected = old_ranking.refinement_partition(reflections, params)
            actual = ranking.refinement_partition(reflections, params, selection)
            for old, new in zip(expected, actual):
                np.testing.assert_array_equal(np.asarray(old['row']), np.asarray(new['row']))

    def test_periodic_real_and_fourier_against_direct_sum(self):
        rng = np.random.default_rng(17)
        fractional = rng.uniform(-.5, .5, (350, 3))
        for A, sigma, representation in [
            (np.diag([.03, .04, .05]), .001, 'real'),
            (np.diag([.003, .004, .005]), .003, 'fourier'),
            (np.array([[.004, .0013, -.0007], [0, .005, .0008], [0, 0, .006]]), .004, 'fourier'),
        ]:
            radii = np.maximum(1, np.ceil(.5 + 8 * sigma * np.linalg.norm(np.linalg.inv(A), axis=1)).astype(int))
            actual, residual, nearest, info = ranking.periodic_gaussian(A, fractional, sigma, radii, 73, 8)
            self.assertEqual(info['periodic_representation'], representation)
            offsets = np.asarray(list(itertools.product(*[range(-r, r + 1) for r in radii])))
            r2 = np.sum(((fractional[:, None, :] - offsets) @ A.T)**2, axis=2)
            expected = (np.log(abs(np.linalg.det(A))) - 1.5 * np.log(2 * np.pi * sigma**2)
                        + logsumexp(-.5 * r2 / sigma**2, axis=1))
            np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=2e-13)
            np.testing.assert_allclose(residual, np.sqrt(r2.min(axis=1)), atol=2e-17)
            np.testing.assert_array_equal(nearest, offsets[np.argmin(r2, axis=1)])

    def test_evidence_outputs_match_legacy_score(self):
        rng = np.random.default_rng(81)
        params = SimpleNamespace(q_sigma=.003,tail_sigma=6,max_periodic_radius=6,
            score_chunk=67,lattice_fraction=.6,match_sigma=3,min_validation_indexed=10,min_log_density_ratio=0)
        for lengths, sigma in [([30.,40.,50.],.001),([250.,200.,166.6666667],.003)]:
            crystal = Crystal((lengths[0],0,0),(0,lengths[1],0),(0,0,lengths[2]),'P 1')
            A = np.asarray(crystal.get_A()).reshape(3,3)
            q = rng.integers(-5,6,(250,3)) @ A.T + rng.normal(0,sigma,(250,3))
            reflections = flex.reflection_table()
            reflections['rlp'] = flex.vec3_double(q)
            params.q_sigma=sigma
            actual = ranking.evidence(crystal,reflections,params)
            expected = old_ranking.evidence(crystal,reflections,params)
            for key in ['usable','validation_rows','validation_indexed','unique_hkl','hkl_rank','periodic_radius_xyz','periodic_images']:
                self.assertEqual(actual[key],expected[key])
            for key in ['mean_log_density_ratio','validation_fraction','residual_p95_inv_angstrom']:
                np.testing.assert_allclose(actual[key],expected[key],atol=1e-10,rtol=1e-12)

    def test_threshold_masks_are_exact(self):
        rng = np.random.default_rng(54)
        image = rng.poisson(3., (130, 150)).astype(float)
        image[48:52, 71:75] += 50
        mask = rng.random(image.shape) > .07
        image[0, 0] = np.nan
        for model in ('poisson', 'poisson_gaussian'):
            for iterations in (0, 2):
                settings = dict(gain=1., offset=0., readout_variance=0. if model == 'poisson' else 4.,
                                noise_model=model, background_width=21, exclusion_width=7,
                                scales=[1, 3, 5], score_threshold=4., support_sigma=1.,
                                min_valid_fraction=.5, background_iterations=iterations)
                np.testing.assert_array_equal(spots.propose(image, mask, **settings),
                                              old_spots.propose(image, mask, **settings))

    def test_native_candidate_loop_shares_selection_without_mutation(self):
        from dials.algorithms.indexing import lattice_search
        from libtbx import easy_mp
        if DIALS_SOURCE.exists():
            source = DIALS_SOURCE / 'lattice_search/__init__.py'
            exec(compile(source.read_text(), str(source), 'exec'), lattice_search.__dict__)
        search = object.__new__(lattice_search.LatticeSearch)
        config = SimpleNamespace(solution_scorer='filter', filter=SimpleNamespace(
            check_doubled_cell=True, likelihood_cutoff=.8, volume_cutoff=1.25, n_indexed_cutoff=.9),
            sys_absent_threshold=0, max_refine=5)
        search.params = SimpleNamespace(basis_vector_combinations=config,
            known_symmetry=SimpleNamespace(space_group=None), nproc=1)
        search.all_params = None
        search.d_min = 2.
        search.hkl_offset = None
        search.experiments = ExperimentList([Experiment(scan=Scan(image_range=(1, 800), oscillation=(0., 1.)))])
        n = 800
        table = flex.reflection_table()
        table['id'] = flex.int(np.where(np.arange(n) % 7, -1, 1))
        table['imageset_id'] = flex.int(n, 0)
        table['xyzobs.mm.value'] = flex.vec3_double(np.column_stack((np.zeros(n), np.zeros(n), np.arange(n) * np.pi / 180)))
        table['rlp'] = flex.vec3_double(np.column_stack((np.linspace(.1, .9, n), np.zeros(n), np.zeros(n))))
        table['row'] = flex.int(range(n))
        search.reflections = table
        expected = np.flatnonzero((np.asarray(table['id']) == -1) & (np.linspace(.1, .9, n) < .5) & (np.arange(n) <= 360))
        seen = []
        def index(experiments, reflections):
            np.testing.assert_array_equal(np.asarray(reflections['row']), expected)
            self.assertTrue(np.all(np.asarray(reflections['id']) == -1))
            reflections['id'] = flex.int(len(reflections), 0)
            reflections.set_flags(flex.bool(len(reflections), True), reflections.flags.indexed)
            seen.append(reflections)
        search.index_reflections = index
        crystals = [Crystal((30+i*.1,0,0),(0,35,0),(0,0,40),'P 1') for i in range(5)]
        def evaluate(experiments, reflections):
            return model_evaluation.Result(.9,experiments[0].crystal,(.01,.02,.03),len(reflections),1.,(0,0,0))
        evaluator = SimpleNamespace(evaluate=evaluate)
        with patch.object(model_evaluation,'ModelEvaluation',return_value=evaluator), patch.object(
                easy_mp,'parallel_map',side_effect=lambda fn, args, **kwargs:[fn(*item) for item in args]):
            crystal, count = search.choose_best_orientation_matrix(crystals)
        self.assertIs(crystal,crystals[0])
        self.assertEqual(count,len(expected))
        self.assertEqual(len(seen),5)
        np.testing.assert_array_equal(np.asarray(table['id']),np.where(np.arange(n)%7,-1,1))

    def test_all_seven_doubled_cells_keep_native_decisions(self):
        from cctbx import sgtbx, uctbx
        from scitbx import matrix
        group = sgtbx.space_group_info('P1')
        cell = group.any_compatible_unit_cell(volume=1000)
        parameters = cell.parameters()
        for multiples in itertools.product((1,2),repeat=3):
            if multiples == (1,1,1):
                continue
            doubled = uctbx.unit_cell(tuple(parameters[i]*multiples[i] for i in range(3))+parameters[3:])
            crystals = [Crystal(matrix.sqr(c.fractionalization_matrix()).transpose(),group.group()) for c in (cell,doubled)]
            results = [model_evaluation.Result(.9,c,(.02,.03,.01),100,.8,(0,0,0)) for c in crystals]
            filtered = model_evaluation.filter_doubled_cell(results)
            self.assertEqual(len(filtered),1)
            self.assertIs(filtered[0],results[0])

    def test_rank_append_extend_preserves_order_and_best(self):
        results = []
        for i in range(12):
            crystal = Crystal((30 + i * .2, 0, 0), (0, 35, 0), (0, 0, 40), 'P 1')
            results.append(model_evaluation.Result(.91 + i * .002, crystal,
                           (.02 + i * .001, .03, .01), 1000 + i * 20, .8 + i * .01,
                           (0, 0, 0), {'usable': True, 'mean_log_density_ratio': i / 10}))
        for factory in (model_evaluation.ModelRankWeighted, model_evaluation.ModelRankFilter, ranking.EvidenceRank):
            old, new = factory(), factory()
            for result in results:
                old.append(result)
            new.extend(results)
            self.assertEqual(len(old), len(new))
            self.assertIs(old.best_model(), new.best_model())
            self.assertEqual(str(old), str(new))


if __name__ == '__main__':
    unittest.main()
