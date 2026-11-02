"""分组顺序、int64 边界与周期候选完整性，对照原始 Python 循环。"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'skills/ariadne-reduction/scripts'))
import overlap_prepare as overlap
import domain_finder as domain


def window_reference(values, width, period=0):
    left = best_left = best_right = 0
    for right in range(len(values)):
        while values[right]-values[left] > width or (period > 0 and right-left+1 > period):
            left += 1
        if (period == 0 or left < period) and right-left > best_right-best_left:
            best_left, best_right = left, right
    return best_left, best_right


def common_reference(gaps, tolerance):
    gaps = np.sort(np.asarray(gaps, dtype=float))
    if not len(gaps):
        return None
    left, right = window_reference(gaps, 2*tolerance)
    return float(np.median(gaps[left:right+1]))


def periodic_reference(values, spacing, tolerance, minimum):
    if spacing <= 1e-12 or len(values) < minimum:
        return np.empty(0, dtype=int)
    tolerance = min(tolerance, .1*spacing)
    remainders = np.mod(values, spacing)
    order = np.argsort(remainders)
    expanded = np.r_[remainders[order], remainders[order]+spacing]
    n = len(values)
    left, right = window_reference(expanded, 2*tolerance, n)
    chosen = order[np.arange(left, right+1) % n]
    if len(chosen) < minimum:
        return np.empty(0, dtype=int)
    phase = float(np.median(expanded[left:right+1])) % spacing
    nodes = np.rint((values[chosen]-phase)/spacing).astype(int)
    if len(np.unique(nodes)) < minimum:
        return np.empty(0, dtype=int)
    return chosen


def require_native():
    if overlap._native_connected_boxes is None or domain._native_densest_window is None:
        pytest.skip('optional native grouping kernels are unavailable')


def box_cases():
    limits = np.iinfo(np.int64)
    return [
        ('empty', np.empty((0, 6), dtype=np.int64)),
        ('touching', np.array([[0, 2, 0, 2, 0, 2], [2, 4, 0, 2, 0, 2],
                               [0, 2, 2, 4, 0, 2], [0, 2, 0, 2, 2, 4]])),
        ('duplicate_z_chain', np.array([[4, 7, 0, 2, 1, 3], [0, 3, 0, 2, 1, 3],
                                        [2, 5, 0, 2, 1, 3], [50, 51, 0, 2, 1, 3]])),
        ('multiple_components', np.array([[0, 4, 0, 4, 3, 8], [50, 54, 1, 4, 0, 4],
                                           [1, 5, 1, 4, 0, 4], [51, 55, 1, 4, 1, 5],
                                           [100, 101, 0, 1, 0, 1]])),
        ('all_overlap', np.tile([0, 10, 0, 10, 0, 10], (64, 1))),
        ('extreme_int64', np.array([[limits.min, limits.min+2, 0, 3, limits.min, limits.min+5],
                                    [limits.min+1, limits.min+3, 1, 4, limits.min+1, limits.min+6],
                                    [limits.max-2, limits.max, 0, 3, limits.max-3, limits.max],
                                    [limits.max-1, limits.max, 1, 4, limits.max-2, limits.max],
                                    [2**53, 2**53+1, 0, 1, 0, 1],
                                    [2**53+1, 2**53+2, 0, 1, 0, 1]], dtype=np.int64)),
        ('degenerate', np.array([[0, 2, 0, 2, 3, 2], [0, 2, 0, 2, 1, 4],
                                 [1, 1, 0, 2, 0, 4], [0, 2, 0, 2, 2, 2]])),
    ]


@pytest.mark.parametrize('name,boxes', box_cases(), ids=lambda x: x if isinstance(x, str) else None)
def test_connected_boxes_exact_order(name, boxes):
    require_native()
    expected = overlap._connected_boxes_python(boxes)
    assert overlap.connected_boxes(boxes) == expected
    assert overlap.connected_boxes(boxes.tolist()) == expected
    assert overlap._native_connected_boxes(boxes[:, ::-1][:, ::-1]) == expected
    assert sorted(i for group in expected for i in group) == list(range(len(boxes)))


def test_connected_boxes_random_stable_union_order_and_strides():
    require_native()
    rng = np.random.default_rng(1719)
    for n in (1, 2, 13, 100, 500):
        for _ in range(8):
            starts = rng.integers(-5, 12, (n, 3))
            ends = starts+rng.integers(-1, 8, (n, 3))
            boxes = np.stack((starts, ends), axis=2).reshape(n, 6)
            boxes = np.asfortranarray(boxes)[::-1]
            expected = overlap._connected_boxes_python(boxes)
            assert overlap.connected_boxes(boxes) == expected
            assert overlap._native_connected_boxes(boxes) == expected


@pytest.mark.parametrize('boxes', [np.array([[0., 2., 0., 2., 0., 2.]]),
    [[0, 2, 0, 2, 0, 2, 99]], [[0, 2, 0, 2, 0, 2]],
    [[2**80, 2**80+2, 0, 2, 0, 2]], [[0, 2, 0, 2, 0, 2], [1, 3, 0, 2, 0, 2]]])
def test_boxes_fallback_and_optional_import(boxes):
    expected = overlap._connected_boxes_python(boxes)
    with patch.object(overlap, '_native_connected_boxes', None):
        assert overlap.connected_boxes(boxes) == expected
    if np.asarray(boxes).dtype.kind not in 'i' or np.asarray(boxes).shape[1] != 6:
        large = np.tile(np.asarray(boxes), (64, 1))
        with patch.object(overlap, '_native_connected_boxes', side_effect=AssertionError('unexpected dispatch')):
            assert overlap.connected_boxes(large) == overlap._connected_boxes_python(large)


@pytest.mark.parametrize('values,width,period,expected', [
    ([], 0., 0, (0, 0)), ([1., 1., 2., 2.], 0., 0, (0, 1)),
    ([0., .2, 1., 1.2], .2, 0, (0, 1)),
    ([.01, .5, .99, 1.01, 1.5, 1.99], .03, 3, (2, 3)),
    ([0., 0., 0., 1., 1., 1.], 10., 3, (0, 2)),
])
def test_window_ties_boundaries_and_circular_wrap(values, width, period, expected):
    require_native()
    values = np.asarray(values, dtype=float)
    assert window_reference(values, width, period) == expected
    assert domain._native_densest_window(values, width, period) == expected
    assert domain._densest_window(values, width, period) == expected


def test_window_random_indices_and_final_scientific_outputs():
    require_native()
    rng = np.random.default_rng(8113)
    for n in (1, 3, 20, 200, 2000):
        values = np.sort(rng.uniform(0, 1, n))
        for width in (0., .001, .05, 2.):
            assert domain._native_densest_window(values, width, 0) == window_reference(values, width)
            expanded = np.r_[values, values+1.]
            assert domain._native_densest_window(expanded, width, n) == window_reference(expanded, width, n)
        for tolerance in (0., .001, .03, .5):
            assert domain.common_gap(values, tolerance) == common_reference(values, tolerance)
            nodes = np.arange(n)+rng.choice([-.01, .01, .5], n)
            for minimum in (1, 3):
                np.testing.assert_array_equal(domain.periodic_subset(nodes, 1., tolerance, minimum),
                                              periodic_reference(nodes, 1., tolerance, minimum))


def test_window_fallback_unusual_width_semantics():
    for width in (np.nan, np.inf, -np.inf, -.1):
        values = np.repeat([0., .1, 1.], 32)
        with patch.object(domain, '_native_densest_window', side_effect=AssertionError('unexpected dispatch')):
            try:
                expected = window_reference(values, width)
            except IndexError:
                with pytest.raises(IndexError):
                    domain._densest_window(values, width)
            else:
                assert domain._densest_window(values, width) == expected
    with patch.object(domain, '_native_densest_window', None):
        assert domain.common_gap([0., .1, .5], .06) == common_reference([0., .1, .5], .06)
        values = np.array([-.01, 1.01, 1.99, 3.01, 9.5])
        np.testing.assert_array_equal(domain.periodic_subset(values, 1., .03, 3), periodic_reference(values, 1., .03, 3))


def test_small_inputs_avoid_unprofitable_dispatch():
    with patch.object(overlap, '_native_connected_boxes', side_effect=AssertionError('small boxes dispatched')):
        assert overlap.connected_boxes(np.tile([0, 2, 0, 2, 0, 2], (16, 1))) == [list(range(16))]
    with patch.object(domain, '_native_densest_window', side_effect=AssertionError('small window dispatched')):
        assert domain._densest_window(np.array([0., 0., 1.]), 0.) == (0, 1)
        assert domain.common_gap([0., .1, .5], .06) == common_reference([0., .1, .5], .06)


def test_native_validation_and_concurrent_calls():
    require_native()
    with pytest.raises(ValueError):
        overlap._native_connected_boxes(np.zeros((2, 6), dtype=float))
    for values, width, period in [(np.array([1., 0.]), 1., 0), (np.array([np.nan]), 1., 0),
                                  (np.array([0.]), -.1, 0), (np.array([0., 1.]), 1., 2)]:
        with pytest.raises(ValueError):
            domain._native_densest_window(values, width, period)
    boxes = box_cases()[2][1]
    expected = overlap._connected_boxes_python(boxes)
    def run(_):
        assert overlap._native_connected_boxes(boxes) == expected
        assert domain._native_densest_window(np.array([0., 0., 1.]), 0., 0) == (0, 1)
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(run, range(32)))
