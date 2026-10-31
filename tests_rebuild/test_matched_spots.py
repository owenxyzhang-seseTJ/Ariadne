"""Compiled matched-filter proposals against the NumPy reference, and their accuracy."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from ariadne import spots


def frame(seed, shape, background=3., readout=0., centre=None):
    rng = np.random.default_rng(seed)
    yy, xx = np.indices(shape)
    cy, cx = centre if centre is not None else (shape[0]/2, shape[1]/2)
    radius = np.hypot(yy-cy, xx-cx)
    expectation = background*(1+.5*np.exp(-radius/40))*(1+.2*xx/shape[1])
    truth = []
    for _ in range(max(1, shape[0]*shape[1]//900)):
        y0, x0 = rng.uniform(1, shape[0]-1), rng.uniform(1, shape[1]-1)
        g = np.exp(-.5*((yy-y0)**2+(xx-x0)**2)/rng.uniform(.6, 1.6)**2)
        expectation += rng.uniform(20, 400)*g/g.sum()
        truth.append((y0, x0))
    image = rng.poisson(expectation).astype(float)
    if readout:
        image += rng.normal(0, np.sqrt(readout), shape)
    mask = rng.random(shape) > .02
    mask[:, shape[1]//3] = False
    return image, mask, radius, np.array(truth)


@pytest.mark.parametrize('use_bins', [False, True])
@pytest.mark.parametrize('readout', [0., 4.])
@pytest.mark.parametrize('calibrate,local_cells', [(True, 4), (True, 0), (False, 2)])
def test_cpp_and_reference_agree_without_mutating_inputs(monkeypatch, use_bins, readout, calibrate, local_cells):
    if spots._native_matched is None:
        pytest.skip('optional C++ extension is not built')
    for seed, shape in enumerate([(5, 7), (64, 80), (131, 119), (200, 257)]):
        image, mask, radius, _ = frame(seed, shape, readout=readout)
        image = 2.*image+10.
        image[1, 1] = np.nan
        image[3, 4] += 5000.   # zinger
        bins = spots.radial_bins(radius, mask) if use_bins else None
        image, mask = image[:, ::-1], mask[:, ::-1]   # strided arrays cross the C boundary
        bins = None if bins is None else bins[:, ::-1]
        before, mask_before = image.copy(), mask.copy()
        settings = dict(gain=2., offset=10., readout_variance=4.*readout, bins=bins, tile=16,
                        calibrate=calibrate, local_cells=local_cells)
        native = spots.propose_matched(image, mask, **settings)
        with monkeypatch.context() as patch:
            patch.setattr(spots, '_native_matched', None)
            reference = spots.propose_matched(image, mask, **settings)
        np.testing.assert_array_equal(native, reference)
        np.testing.assert_array_equal(image, before)
        np.testing.assert_array_equal(mask, mask_before)


def test_compiled_kernel_has_no_cross_call_state():
    if spots._native_matched is None:
        pytest.skip('optional C++ extension is not built')
    image, mask, radius, _ = frame(7, (96, 128))
    bins = spots.radial_bins(radius, mask)
    first = spots.propose_matched(image, mask, gain=1., bins=bins)
    masks = [np.zeros(image.shape, bool), mask, np.ones(image.shape, bool), mask]
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda m: spots.propose_matched(image, m, gain=1., bins=bins), masks))
    assert not results[0].any()
    np.testing.assert_array_equal(results[1], first)
    np.testing.assert_array_equal(results[3], first)
    small = spots.propose_matched(image[:40, :50], mask[:40, :50], gain=1.)
    assert small.shape == (40, 50)
    np.testing.assert_array_equal(spots.propose_matched(image, mask, gain=1., bins=bins), first)


def test_finds_isolated_spots_and_rejects_zingers():
    image, mask, radius, truth = frame(11, (256, 256), background=2.)
    mask[:] = True
    image[100, 37] += 3000.
    out = spots.propose_matched(image, mask, gain=1., bins=spots.radial_bins(radius, mask))
    assert not out[99:102, 36:39].any()
    from scipy.ndimage import center_of_mass, label
    groups, n = label(out)
    found = np.array(center_of_mass(out, groups, range(1, n+1))).reshape(-1, 2)
    distance = np.hypot(*(found[:, None, :]-truth[None, :, :]).transpose(2, 0, 1))
    assert (distance.min(0) < 2).mean() > .8
    assert (distance.min(1) < 3).mean() > .9


@pytest.mark.parametrize('bad', [dict(gain=0.), dict(tile=15), dict(background_rounds=0),
                                 dict(readout_variance=-1.), dict(component_min_size=0),
                                 dict(bins=-np.ones((8, 8), int)), dict(bins=np.zeros((8, 7), int)),
                                 dict(readout_variance=np.inf), dict(clip=np.inf), dict(contrast=np.inf),
                                 dict(false_alarms=0.), dict(peak_threshold=np.nan), dict(local_cells=-1),
                                 dict(zinger_sigma=-1.), dict(zinger_fraction=np.inf),
                                 dict(bins=np.full((8, 8), 2**32, np.int64))])
def test_invalid_settings_are_rejected(bad):
    settings = dict(gain=1.)
    settings.update(bad)
    with pytest.raises(ValueError):
        spots.propose_matched(np.ones((8, 8)), np.ones((8, 8), bool), **settings)


def test_radial_bins_default_width_is_one_pixel_step():
    yy, xx = np.indices((50, 60))
    radius = np.hypot(yy-10.3, xx-20.7)
    bins = spots.radial_bins(radius, np.ones(radius.shape, bool))
    assert bins.dtype == np.int32 and bins.min() == 0
    assert abs(bins.max()-(radius.max()-radius.min())) <= 1.5


@pytest.mark.parametrize('calibrate', [False, True])
def test_noise_diagnostics_agree_with_reference(monkeypatch, calibrate):
    if spots._native_matched is None:
        pytest.skip('optional C++ extension is not built')
    image, mask, radius, _ = frame(31, (400, 512), background=20.)
    # 共享相邻像素的信号，检验相关噪声而非只检验独立 Poisson 情形。
    image = .75*image+.25*np.roll(image, 1, axis=1)
    bins = spots.radial_bins(radius, mask)
    native_info, reference_info = {}, {}
    native = spots.propose_matched(image, mask, gain=1., bins=bins, calibrate=calibrate, info=native_info)
    with monkeypatch.context() as patch:
        patch.setattr(spots, '_native_matched', None)
        reference = spots.propose_matched(image, mask, gain=1., bins=bins, calibrate=calibrate, info=reference_info)
    np.testing.assert_array_equal(native, reference)
    for key in ('gain', 'offset', 'readout_variance', 'kappa', 'kappa_inf', 'thresholds'):
        np.testing.assert_allclose(native_info[key], reference_info[key], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(list(native_info['correlation'].values()),
                               list(reference_info['correlation'].values()), rtol=1e-7, atol=1e-9)
    assert all(value >= 0 for value in native_info['stage_ms'].values())


def test_mask_cache_tracks_small_changes(monkeypatch):
    if spots._native_matched is None:
        pytest.skip('optional C++ extension is not built')
    image, mask, radius, _ = frame(81, (131, 151))
    bins = spots.radial_bins(radius, mask)
    for y, x in ((1, 1), (63, 75), (129, 149)):
        mask[y:y+2, x:x+2] ^= True
        native = spots.propose_matched(image, mask, gain=1., bins=bins, component_min_size=6)
        with monkeypatch.context() as patch:
            patch.setattr(spots, '_native_matched', None)
            reference = spots.propose_matched(image, mask, gain=1., bins=bins, component_min_size=6)
        np.testing.assert_array_equal(native, reference)
        assert not native[~mask].any()


def test_parallel_rows_preserve_decisions_and_noise_model(monkeypatch):
    if spots._native_matched is None:
        pytest.skip('optional C++ extension is not built')
    image, mask, radius, _ = frame(48, (512, 513), background=15.)
    bins = spots.radial_bins(radius, mask)
    monkeypatch.setenv('ARIADNE_MATCHED_THREADS', '1')
    info = {}
    expected = spots.propose_matched(image, mask, gain=1., bins=bins, info=info)
    monkeypatch.setenv('ARIADNE_MATCHED_THREADS', '4')
    def detect(i):
        details = {}
        out = spots.propose_matched(image, mask, gain=1., bins=bins, info=details)
        return out, details
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(detect, range(4)))
    for out, details in results:
        np.testing.assert_array_equal(out, expected)
        for key in ('gain', 'offset', 'readout_variance', 'kappa', 'kappa_inf', 'thresholds'):
            np.testing.assert_allclose(details[key], info[key], rtol=1e-7, atol=1e-9)
