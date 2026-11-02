"""原生似然和 EM 的参考对照、预算边界与完整重叠可辨识性。"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'skills/ariadne-reduction/scripts'))
import poisson_gaussian as pg
import centroid_em as em
import composite_scaling as scaling


@pytest.mark.parametrize('shape', [(0, 3, 2), (5, 0, 0), (13, 41, 7)])
def test_scaling_design_repeated_ids_and_cancellation(shape):
    if scaling._native_scaling_design is None:
        pytest.skip('optional native scaling kernel is unavailable')
    rows, columns, count = shape
    rng = np.random.default_rng(171)
    design = rng.normal(size=(rows, columns))
    gain = np.exp(rng.normal(size=columns))
    indices = rng.integers(max(count, 1), size=columns)
    if columns >= 3:
        design[:, :3] = [1e20, 1., -1e20]
        gain[:3] = 1.
        indices[:3] = 0
    with patch.object(scaling, '_native_scaling_design', None):
        expected = scaling._intensity_design(design, gain, indices, count)
    np.testing.assert_array_equal(scaling._intensity_design(design, gain, indices, count), expected)
from constrained_pixels import fit_constrained


@pytest.mark.parametrize('gain,variance', [(1., .2), (2.5, 6.), (1e-12, 1e-24), (1e8, 1e16)])
def test_pg_density_score_curvature_cache_and_concurrent_calls(gain, variance):
    native = pg._native_pg_sums
    if native is None:
        pytest.skip('optional native likelihood is unavailable')
    mu = np.array([0., 1e-6, .1, 1., 10., 1000., 1e5, 1e12])
    y = gain*np.array([-2., .3, -.5, 2., 11., 1001., 1e5+.1, 1e12+.01])
    with patch.object(pg, '_native_pg_sums', None):
        reference = pg.PoissonGaussian(y, gain, variance)
        expected = reference.evaluate(mu)
        curvature = reference.observed_information(mu)
    def run(_):
        actual = pg.PoissonGaussian(y, gain, variance)
        result = actual.evaluate(mu)
        info = actual.observed_information(mu)
        cached = actual.evaluate(mu)
        assert cached[2]['terms_evaluated'] == 0
        assert cached[2]['unchanged_pixels_reused'] == len(mu)
        return result, info
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(run, range(4)))
    for actual, info in results:
        for index in (0, 1):
            np.testing.assert_allclose(actual[index], expected[index], rtol=2e-10, atol=3e-10)
        np.testing.assert_allclose(info[0], curvature[0], rtol=2e-8, atol=3e-10)
        np.testing.assert_array_equal(info[1]['reliable_positive'], curvature[1]['reliable_positive'])
        assert actual[2]['max_relative_density_truncation_bound'] <= 1e-12


@pytest.mark.parametrize('native', [False, True])
def test_pg_budgets_zero_means_and_integer_limit(native):
    if native and pg._native_pg_sums is None:
        pytest.skip('optional native likelihood is unavailable')
    with patch.object(pg, '_native_pg_sums', pg._native_pg_sums if native else None):
        with pytest.raises(pg.LikelihoodBudgetError):
            pg.PoissonGaussian([2.], 1., 2., max_terms_per_pixel=5).evaluate([2.])
        with pytest.raises(pg.LikelihoodBudgetError):
            pg.PoissonGaussian([0., 0.], 1., 1., max_terms_per_evaluation=3).evaluate([0., 0.])
        result = pg.PoissonGaussian([0., 0.], 1., 1., max_terms_per_evaluation=4).evaluate([0., 0.])
        assert result[2]['terms_evaluated'] == 4
        with pytest.raises(ValueError, match='integer range'):
            pg.PoissonGaussian([float(2**52)], 1., 1.).evaluate([1.])


@pytest.mark.parametrize('shape,block', [((9, 11), 1), ((13, 23), 7), ((11, 0), 5), ((0, 3), 5), ((65, 129), 100)])
def test_complete_candidate_em_matches_reference_including_empty_and_forbidden(shape, block):
    native = em._native_expectation
    if native is None:
        pytest.skip('optional native centroid kernel is unavailable')
    rng = np.random.default_rng(713)
    n, m = shape
    y = rng.normal(size=(n, 3))
    means = rng.normal(size=(m, 3))
    matrices = rng.normal(size=(n, 3, 3))
    white, normalizer = em.covariance_factors(matrices@matrices.transpose(0, 2, 1)+np.eye(3)*.01)
    allowed = rng.random((n, m)) > .4
    allowed[:1] = False
    if m > 1:
        means[1] = means[0]  # 完全重叠候选仍保留两份责任度。
        allowed[:, 1] = allowed[:, 0]
    parameters = y, means, white, normalizer, allowed, np.full(n, max(m, 1)), np.full(n, 1e-8), .05
    expected = None
    with patch.object(em, '_native_expectation', None):
        expected = em.expectation(*parameters, pair_block_size=block)
    actual = em.expectation(*parameters, pair_block_size=block)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-13)
    np.testing.assert_allclose(actual[0].sum(axis=1)+actual[1], 1., atol=2e-14)
    if m > 1:
        np.testing.assert_array_equal(actual[0][:, 0], actual[0][:, 1])


def test_complete_overlap_pg_does_not_invent_individual_information():
    if pg._native_pg_sums is None:
        pytest.skip('optional native likelihood is unavailable')
    profile = np.array([.1, .2, .3, .4])
    profiles = np.vstack((np.column_stack((profile, profile)), np.zeros((8, 2))))
    y = np.r_[[-.3, 2.1, 3.7, .9], np.full(8, .2)]
    args = (y, np.ones(len(y)), profiles, np.ones(len(y), bool), np.ones((len(y), 1)),
            dict(source='synthetic test', gain=1., readout_variance=1., offset_removed=True),
            'poisson-gaussian', 1e-8)
    with patch.object(pg, '_native_pg_sums', None):
        expected = fit_constrained(*args)
    actual = fit_constrained(*args)
    assert actual['rank'] == expected['rank'] == 1
    assert all(x['status'] == 'unidentifiable' for x in actual['profile_intervals'])
    assert actual['total_interval']['status'] == expected['total_interval']['status'] == 'ok'
    for key in ('lower', 'upper'):
        np.testing.assert_allclose(actual['total_interval'][key], expected['total_interval'][key], rtol=2e-8, atol=2e-8)
