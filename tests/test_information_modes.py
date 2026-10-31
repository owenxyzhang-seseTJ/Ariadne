"""Physical observation identities and global recovery of overlapped twins."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from numpy.testing import assert_allclose

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skills/ariadne-reduction/scripts'))
from overlap_integrator import solve_joint, contrast
from overlap_prepare import (connected_boxes, support_union_mask, canonical_frame_identities,
                             register_frame_partition)
from composite_scaling import fit_global


def profiles():
    x = np.linspace(-5, 5, 120)
    a, b = np.exp(-.5 * ((x + 1.6) / .6)**2), np.exp(-.5 * ((x - 1.4) / .7)**2)
    a /= a.sum()
    b /= b.sum()
    return a, b


def mode_block(result, labels):
    modes = result['information_modes']
    return dict(y=modes['y'], covariance=modes['covariance'], design=modes['design'],
                equivalence_ids=labels, features=np.empty((len(labels), 0)),
                known_factors=np.ones(len(labels)))


class InformationModeTests(unittest.TestCase):
    def test_preserves_arbitrary_profiled_pixel_likelihood_and_information(self):
        rng = np.random.default_rng(927)
        n, k = 170, 4
        P = rng.uniform(0, 1, (n, k))
        P /= P.sum(axis=0)
        B = np.column_stack((np.ones(n), np.linspace(-1, 1, n)))
        v = rng.uniform(.5, 3, n)
        y = P @ np.array([1000., 10., -.4, 2.]) + B @ [2., .4] + rng.normal(size=n) * np.sqrt(v)
        valid = rng.uniform(size=n) > .08
        result = solve_joint(y, v, P, valid, B, 1e-8)
        modes = result['information_modes']
        z, H = np.asarray(modes['y']), np.asarray(modes['design'])
        for amplitude in rng.normal(size=(8, k)) * [1000, 10, 5, 20]:
            beta = np.linalg.lstsq(B[valid] / np.sqrt(v[valid, None]),
                                   (y[valid] - P[valid] @ amplitude) / np.sqrt(v[valid]), rcond=None)[0]
            original = np.sum((y[valid] - P[valid] @ amplitude - B[valid] @ beta)**2 / v[valid])
            compressed = np.sum((z - H @ amplitude)**2) + modes['pixel_chi2_constant']
            assert_allclose(compressed, original, rtol=3e-14, atol=1e-10)
        complete = np.column_stack((P, B))[valid] / np.sqrt(v[valid, None])
        expected = np.linalg.inv(complete.T @ complete)[:k, :k]
        assert_allclose(np.linalg.inv(H.T @ H), expected, rtol=3e-13)

    def test_modes_retain_weak_directions_beyond_publication_cutoff(self):
        a, b = profiles()
        P = np.column_stack((a, (1 - 1e-7) * a + 1e-7 * b))
        y, v, valid, B = 2 + P @ [500., 2.], np.ones(len(a)), np.ones(len(a), bool), np.ones((len(a), 1))
        conservative = solve_joint(y, v, P, valid, B, 1e-3)
        detailed = solve_joint(y, v, P, valid, B, 1e-10)
        self.assertEqual(conservative['rank'], 1)
        self.assertEqual(conservative['information_modes']['rank'], 2)
        self.assertEqual(conservative['information_modes'], detailed['information_modes'])

    def test_three_domain_overlap_changes_recover_all_components(self):
        a, b = profiles()
        truth = np.array([1200., 12., .8])
        designs = [np.column_stack((a, a, b)), np.column_stack((a, b, b))]
        blocks, totals = [], []
        labels = ['domain-A:1,2,3', 'domain-B:1,2,3', 'domain-C:1,2,3']
        for P in designs:
            y = .6 + P @ truth
            result = solve_joint(y, y, P, np.ones(len(y), bool), np.ones((len(y), 1)), 1e-8)
            self.assertEqual(result['rank'], 2)
            self.assertTrue(contrast(result, np.ones(3), 1e-7)['identifiable'])
            self.assertFalse(all(contrast(result, np.eye(3)[j], 1e-7)['identifiable'] for j in range(3)))
            blocks.append(mode_block(result, labels))
            total = contrast(result, np.ones(3), 1e-7)
            totals.append(dict(y=[total['intensity']], covariance=[[total['variance']]],
                               design=np.ones((1, 3)), equivalence_ids=labels,
                               features=np.empty((3, 0)), known_factors=np.ones(3)))
        actual = fit_global(blocks)
        self.assertEqual(actual['status'], 'converged')
        self.assertEqual(actual['rank'], 3)
        assert_allclose(actual['I'], truth, rtol=1e-10, atol=1e-10)
        discarded = fit_global(totals)
        self.assertEqual(discarded['status'], 'unresolved')
        self.assertEqual(discarded['rank'], 1)

    def test_four_domain_pair_graph_gains_rank_only_with_third_geometry(self):
        a, b = profiles()
        truth = np.array([10000., 120., 11., 1.])
        layouts = [np.column_stack((a, a, b, b)), np.column_stack((a, b, a, b)), np.column_stack((a, b, b, a))]
        blocks = []
        for P in layouts:
            y = 1 + P @ truth
            result = solve_joint(y, y, P, np.ones(len(y), bool), np.ones((len(y), 1)), 1e-8)
            self.assertEqual(result['information_modes']['rank'], 2)
            blocks.append(mode_block(result, ['A', 'B', 'C', 'D']))
        unresolved = fit_global(blocks[:2])
        self.assertEqual(unresolved['rank'], 3)
        self.assertEqual(unresolved['status'], 'unresolved')
        recovered = fit_global(blocks)
        self.assertEqual(recovered['rank'], 4)
        self.assertEqual(recovered['status'], 'converged')
        assert_allclose(recovered['I'], truth, rtol=1e-9, atol=2e-9)

    def test_no_measurement_can_resolve_identical_profiles_in_every_scan(self):
        a, _ = profiles()
        P = np.column_stack((a, a, a))
        blocks = []
        for value in [14., 12., 15., 11.]:
            y = 1 + value * a
            result = solve_joint(y, np.ones(len(y)), P, np.ones(len(y), bool), np.ones((len(y), 1)), 1e-8)
            blocks.append(mode_block(result, ['A', 'B', 'C']))
        result = fit_global(blocks)
        self.assertEqual(result['rank'], 1)
        self.assertEqual(result['status'], 'unresolved')
        self.assertFalse(np.any(result['identifiable_parameters']))

    def test_negative_weak_measurements_remain_signed(self):
        a, b = profiles()
        P = np.column_stack((a, b))
        y = 2 + P @ [70., -3.]
        result = solve_joint(y, np.ones(len(y)), P, np.ones(len(y), bool), np.ones((len(y), 1)), 1e-8)
        fitted = fit_global([mode_block(result, ['A', 'B'])])
        assert_allclose(fitted['I'], [70., -3.], atol=2e-12)

    def test_union_ownership_prevents_hull_pixel_duplication(self):
        boxes = [(0, 4, 0, 1, 0, 1), (0, 1, 0, 4, 0, 1), (2, 3, 2, 3, 0, 1)]
        groups = connected_boxes(boxes)
        self.assertEqual(sorted(map(len, groups)), [1, 2])
        full = (0, 4, 0, 4, 0, 1)
        masks = [support_union_mask(full, [boxes[i] for i in group]) for group in groups]
        self.assertLessEqual(np.stack(masks).sum(axis=0).max(), 1)
        self.assertEqual(np.stack(masks).sum(), 8)
        # The L-shaped first component's rectangular hull contains (2,2).
        l_shape = next(mask for mask in masks if mask.sum() == 7)
        self.assertFalse(l_shape[0, 2, 2])

    def test_margin_and_clipping_preserve_single_group_behavior(self):
        mask = support_union_mask((0, 6, 0, 6, 0, 3), [(1, 5, 1, 5, 1, 2)], margin=1)
        self.assertTrue(mask.all())

    def test_file_aliases_and_multi_image_indices_preserve_raw_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.h5'
            alias = Path(directory) / 'alias.h5'
            path.write_bytes(b'identity fixture only')
            alias.symlink_to(path)
            first = canonical_frame_identities([str(path)] * 2, [0, 1], True)
            second = canonical_frame_identities([str(alias)] * 2, [0, 1], True)
            self.assertEqual(first, second)
            self.assertNotEqual(first[0], first[1])
            owners = {}
            register_frame_partition(owners, first, 0, 'full_scan')
            register_frame_partition(owners, second, 0, 'full_scan')
            with self.assertRaisesRegex(ValueError, 'overlapping imageset partitions'):
                register_frame_partition(owners, second[1:], 0, 'sliced_scan')
            register_frame_partition(owners, second[1:], 1, 'independent_panel')

    def test_single_image_slice_index_is_not_a_new_raw_frame(self):
        first = canonical_frame_identities(['/nonexistent/one.cbf'], [0], False)
        second = canonical_frame_identities(['/nonexistent/one.cbf'], [25], False)
        self.assertEqual(first, second)


if __name__ == '__main__':
    unittest.main()
