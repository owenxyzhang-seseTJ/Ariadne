"""Mechanistic overlap graph theorem and ridge-free candidate selection."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'skills/ariadne-reduction/scripts'))
from overlap_observability import observability, select_observations, unit_pair_sum_graph


def pair_design(n, edges):
    result = np.zeros((len(edges), n))
    for index, (u, v) in enumerate(edges):
        result[index, [u, v]] = 1.
    return result


class OverlapObservabilityTests(unittest.TestCase):
    def test_three_domains_chain_has_alternating_null_mode(self):
        pairs = [[0, 1], [1, 2]]
        H = pair_design(3, pairs)
        graph = unit_pair_sum_graph(3, pairs, [.3, 7.])
        self.assertEqual(graph['rank'], 2)
        self.assertEqual(graph['nullity'], 1)
        null = np.asarray(graph['nullspace'])[0]
        np.testing.assert_allclose(H @ null, 0., atol=1e-15)
        np.testing.assert_allclose(np.abs(null), 1/np.sqrt(3))
        np.testing.assert_allclose(graph['signless_laplacian'], H.T @ np.diag([.3, 7.]) @ H)
        self.assertEqual(observability(H)['individual_identifiable'], [False, False, False])

    def test_triangle_recovers_three_individual_intensities(self):
        pairs = [[0, 1], [1, 2], [2, 0]]
        H = pair_design(3, pairs)
        graph = unit_pair_sum_graph(3, pairs)
        self.assertEqual(graph['rank'], 3)
        self.assertEqual(graph['nullity'], 0)
        self.assertTrue(graph['components'][0]['contains_odd_cycle'])
        truth = np.array([5., -2., 17.])
        np.testing.assert_allclose(np.linalg.solve(H, H @ truth), truth)
        self.assertEqual(observability(H)['individual_identifiable'], [True]*3)

    def test_four_domains_even_cycle_needs_odd_cycle_or_general_coefficients(self):
        pairs = [[0, 1], [1, 2], [2, 3], [3, 0]]
        H = pair_design(4, pairs)
        self.assertEqual(unit_pair_sum_graph(4, pairs)['rank'], 3)
        self.assertEqual(observability(H)['rank'], 3)
        self.assertEqual(unit_pair_sum_graph(4, pairs + [[0, 2]])['rank'], 4)
        # Physical component factors can break the alternating ambiguity even
        # in a graph containing only an even cycle.  The generic matrix path
        # must not impose the unit-coefficient graph theorem here.
        H[0, 0] = 1.7
        self.assertEqual(observability(H)['rank'], 4)

    def test_disconnected_and_isolated_graph_components(self):
        pairs = [[0, 1], [1, 2], [2, 0], [3, 4]]
        graph = unit_pair_sum_graph(6, pairs)
        self.assertEqual(graph['rank'], 4)
        self.assertEqual(graph['nullity'], 2)
        self.assertEqual(len(graph['components']), 3)
        H = pair_design(6, pairs)
        np.testing.assert_allclose(H @ np.asarray(graph['nullspace']).T, 0., atol=1e-15)

    def test_graph_rank_agrees_with_independent_dense_svd(self):
        rng = np.random.default_rng(521)
        for n in range(2, 9):
            edges = [(i, j) for i in range(n) for j in range(i+1, n) if rng.uniform() < .38]
            graph = unit_pair_sum_graph(n, edges)
            H = pair_design(n, edges)
            rank = np.linalg.matrix_rank(H) if len(edges) else 0
            self.assertEqual(graph['rank'], rank)

    def test_rank_gain_precedes_repeat_measurement_precision(self):
        H = pair_design(3, [[0, 1], [1, 2]])
        candidates = [
            {'id': 'precise-repeat', 'design': [[1, 1, 0]], 'variance': [.01]},
            {'id': 'weak-third-pair', 'design': [[1, 0, 1]], 'variance': [1e8]},
        ]
        selection = select_observations(H, candidates, budget=1)
        self.assertEqual(selection['selected_ids'], ['weak-third-pair'])
        self.assertEqual(selection['initial_rank'], 2)
        self.assertEqual(selection['final_rank'], 3)
        self.assertEqual(selection['remaining_nullity'], 0)
        chosen = selection['steps'][0]['selected']
        self.assertEqual(chosen['rank_gain'], 1)
        self.assertLess(chosen['weakest_new_direction_singular_value'], 1e-3)

    def test_new_direction_strength_breaks_equal_rank_gain_ties(self):
        H = pair_design(3, [[0, 1], [1, 2]])
        candidates = [
            {'id': 'noisy', 'design': [[1, 0, 1]], 'variance': [16.]},
            {'id': 'clear', 'design': [[1, 0, 1]], 'variance': [1.]},
        ]
        selection = select_observations(H, candidates, budget=1)
        self.assertEqual(selection['selected_ids'], ['clear'])

    def test_full_rank_uses_information_gain_without_ridge(self):
        candidates = [
            {'id': 'weak', 'design': [[1, 0]], 'variance': [100.]},
            {'id': 'strong', 'design': [[0, 1]], 'variance': [1.]},
        ]
        selection = select_observations(np.eye(2), candidates, budget=1)
        self.assertEqual(selection['selected_ids'], ['strong'])
        self.assertEqual(selection['initial_rank'], selection['final_rank'])
        np.testing.assert_allclose(selection['steps'][0]['selected']['log_pseudodeterminant_gain'], np.log(2), atol=1e-14)

    def test_covariance_whitening_matches_information_matrix(self):
        H = np.array([[1, 2, 0], [0, 1, 1], [1, 0, 2.]])
        covariance = np.array([[2., .3, .1], [.3, 1.5, -.2], [.1, -.2, 3.]])
        scales = np.array([2., .5, 3.])
        result = observability(H, covariance, parameter_scales=scales)
        information = (H*scales).T @ np.linalg.solve(covariance, H*scales)
        expected = np.linalg.eigvalsh(information)[::-1]
        np.testing.assert_allclose(np.square(result['singular_values']), expected, rtol=2e-14)
        np.testing.assert_allclose(result['log_pseudodeterminant_information'], np.linalg.slogdet(information)[1], atol=2e-14)

    def test_multi_package_greedy_is_stable_and_preserves_fixed_threshold(self):
        H = np.empty((0, 3))
        candidates = [
            {'id': 'a', 'design': [[1, 0, 0]]},
            {'id': 'duplicate-a', 'design': [[1, 0, 0]]},
            {'id': 'b', 'design': [[0, 1, 0]]},
            {'id': 'c', 'design': [[0, 0, 1]]},
        ]
        result = select_observations(H, candidates, budget=3)
        self.assertEqual(result['selected_ids'], ['a', 'b', 'c'])
        self.assertEqual(result, select_observations(H, candidates, budget=3))
        self.assertEqual([step['selected']['rank_gain'] for step in result['steps']], [1, 1, 1])
        self.assertEqual(result['final_rank'], 3)

    def test_compressed_design_selection_matches_dense_information_gain(self):
        rng = np.random.default_rng(728)
        H = rng.normal(size=(125, 4))
        variance = rng.uniform(.5, 3., len(H))
        candidates = [{'id': str(j), 'design': rng.normal(size=(80, 4)),
                       'variance': rng.uniform(.2, 2., 80)} for j in range(4)]
        result = select_observations(H, candidates, variance, budget=1)
        current_information = H.T @ (H / variance[:, None])
        expected = []
        for candidate in candidates:
            G, v = candidate['design'], candidate['variance']
            expected.append(np.linalg.slogdet(current_information + G.T @ (G/v[:, None]))[1]
                            - np.linalg.slogdet(current_information)[1])
        self.assertEqual(result['selected_ids'], [str(np.argmax(expected))])
        for actual, reference in zip(result['steps'][0]['alternatives'], expected):
            np.testing.assert_allclose(actual['log_pseudodeterminant_gain'], reference, atol=2e-14)

    def test_zero_information_and_zero_budget_do_not_create_measurements(self):
        candidates = [{'id': 'zero', 'design': [[0, 0]]}]
        result = select_observations([[0, 0]], candidates, budget=2)
        self.assertEqual(result['selected_ids'], [])
        self.assertEqual(result['remaining_nullity'], 2)
        result = select_observations(np.eye(2), [{'id': 'a', 'design': [[1, 0]]}], budget=0)
        self.assertEqual(result['selected_ids'], [])

    def test_invalid_noise_and_candidate_identity_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'positive'):
            observability(np.eye(2), [1, 0])
        with self.assertRaisesRegex(ValueError, 'positive definite'):
            observability(np.eye(2), [[1, 2], [2, 1]])
        with self.assertRaisesRegex(ValueError, 'unique'):
            select_observations([[1, 0]], [{'id': 'a', 'design': [[0, 1]]},
                                            {'id': 'a', 'design': [[0, 1]]}])
        with self.assertRaisesRegex(ValueError, 'distinct'):
            unit_pair_sum_graph(2, [[0, 0]])


if __name__ == '__main__':
    unittest.main()
